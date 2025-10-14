import json
import os
from typing import List, Dict, Tuple
from pathlib import Path

import torch
from PIL import Image
import io
import base64

from transformers import CLIPProcessor, CLIPModel
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage

import numpy as np


class CLIPEmbeddings(Embeddings):
    """Custom CLIP embeddings for multimodal content"""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of text documents"""
        embeddings = []
        for text in texts:
            inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            embeddings.append(text_features.cpu().numpy().flatten().tolist())
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text"""
        return self.embed_documents([text])[0]
    
    def embed_image(self, image_base64: str) -> List[float]:
        """Embed a single image from base64"""
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features.cpu().numpy().flatten().tolist()
    
    def embed_multimodal(self, text: str, images: List[str]) -> List[float]:
        """Embed combined text and images"""
        text_emb = self.embed_query(text)
        
        if not images:
            return text_emb
        
        image_embs = [self.embed_image(img) for img in images]
        
        # Average text and image embeddings
        all_embs = [text_emb] + image_embs
        combined = np.mean(all_embs, axis=0)
        # Normalize
        combined = combined / np.linalg.norm(combined)
        return combined.tolist()


class MultimodalRAGPipeline:
    """Complete pipeline for multimodal RAG with CLIP embeddings"""
    
    def __init__(self, api_key: str = None):
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        
        self.clip_embeddings = CLIPEmbeddings()
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.vectorstore = None
    
    def partition_document(self, file_path: str):
        """Extract elements from PDF using unstructured"""
        print(f"Partitioning document: {file_path}")
        elements = partition_pdf(
            filename=file_path,
            strategy="hi_res",
            infer_table_structure=True,
            extract_image_block_types=["Image"],
            extract_image_block_to_payload=True
        )
        print(f"Extracted {len(elements)} elements")
        return elements
    
    def create_chunks_by_title(self, elements):
        """Create chunks using title-based chunking"""
        print("Creating chunks by title...")
        chunks = chunk_by_title(
            elements,
            combine_text_under_n_chars=500,
            max_characters=3000,
            new_after_n_chars=2400
        )
        print(f"Created {len(chunks)} chunks")
        return chunks
    
    def separate_content_types(self, chunk) -> Dict:
        """Analyze and separate different content types in a chunk"""
        content_data = {
            'text': chunk.text,
            'tables': [],
            'images': [],
            'types': ['text']
        }
        
        for element in chunk.metadata.orig_elements:
            element_type = type(element).__name__
            
            if element_type == 'Table':
                if 'table' not in content_data['types']:
                    content_data['types'].append('table')
                table_html = getattr(element.metadata, 'text_as_html', None)
                if table_html:
                    content_data['tables'].append(table_html)
            
            elif element_type == 'Image':
                if 'image' not in content_data['types']:
                    content_data['types'].append('image')
                image_base64 = getattr(element.metadata, 'image_base64', None)
                if image_base64:
                    content_data['images'].append(image_base64)
        
        return content_data
    
    def create_ai_summary(self, text: str, tables: List[str], images: List[str]) -> str:
        """Create AI-enhanced summary of multimodal content"""
        prompt_text = f"""You are an expert content summarizer. Analyze the provided text, tables, and images 
and create a clear, concise, and structured summary.

CONTENT TO ANALYZE:
-------------------
TEXT CONTENT:
{text}

"""
        if tables:
            prompt_text += "TABLES:\n"
            for i, table in enumerate(tables):
                prompt_text += f"Table {i+1}:\n{table}\n\n"
        
        if images:
            prompt_text += "IMAGES are provided (as base64). Analyze them for context.\n\n"
        
        prompt_text += """
YOUR TASK:
1. Provide a well-structured summary (max ~500 words).
2. Capture the main ideas, trends, and insights across text, tables, and images.
3. For tables with numeric data, highlight key figures and patterns.
4. For images, describe what they show and their relevance.
5. End with "SEARCHABLE KEYWORDS" – 5-7 keywords or phrases for finding this content.

OUTPUT FORMAT:
---------------
Summary:
[your summary here]

Searchable Keywords:
[keywords/phrases here]
"""
        
        message_content = [{"type": "text", "text": prompt_text}]
        
        for image_base64 in images:
            message_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
            })
        
        try:
            message = HumanMessage(content=message_content)
            response = self.llm.invoke([message])
            return response.content
        except Exception as e:
            print(f"AI summary failed: {e}")
            return f"{text[:300]}..."
    
    def process_chunks(self, chunks) -> List[Document]:
        """Process chunks and create enhanced documents"""
        print("Processing chunks and creating summaries...")
        langchain_documents = []
        
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}")
            
            content_data = self.separate_content_types(chunk)
            
            # Create AI summary for multimodal content
            if content_data["tables"] or content_data['images']:
                print(f"  - Multimodal content detected: {content_data['types']}")
                enhanced_content = self.create_ai_summary(
                    content_data['text'],
                    content_data['tables'],
                    content_data['images']
                )
            else:
                enhanced_content = content_data['text']
            
            # Create document with metadata
            doc = Document(
                page_content=enhanced_content,
                metadata={
                    "chunk_id": i,
                    "content_types": json.dumps(content_data['types']),
                    "original_content": json.dumps({
                        "raw_text": content_data['text'],
                        "tables_html": content_data["tables"],
                        "images_base64": content_data['images']
                    })
                }
            )
            
            langchain_documents.append(doc)
        
        print(f"Processed {len(langchain_documents)} documents")
        return langchain_documents
    
    def create_vector_store(self, docs: List[Document], persist_directory: str = "chroma_clip_db"):
        """Create ChromaDB vector store with CLIP embeddings"""
        print("Creating vector store with CLIP embeddings...")
        
        # Prepare documents with multimodal embeddings
        texts = []
        embeddings_list = []
        metadatas = []
        
        for doc in docs:
            texts.append(doc.page_content)
            metadatas.append(doc.metadata)
            
            # Extract images for multimodal embedding
            original_content = json.loads(doc.metadata["original_content"])
            images = original_content.get("images_base64", [])
            
            # Create multimodal embedding
            embedding = self.clip_embeddings.embed_multimodal(doc.page_content, images)
            embeddings_list.append(embedding)
        
        # Create vector store
        self.vectorstore = Chroma.from_texts(
            texts=texts,
            embedding=self.clip_embeddings,
            metadatas=metadatas,
            persist_directory=persist_directory,
            collection_metadata={"hnsw:space": "cosine"}
        )
        
        print(f"Vector store created and persisted to {persist_directory}")
        return self.vectorstore
    
    def generate_answer(self, chunks: List[Document], query: str) -> str:
        """Generate final answer using retrieved chunks"""
        print("Generating answer...")
        
        prompt_text = f"""You are an expert assistant. Analyze the provided documents 
(text, tables, and images) and give a clear, accurate, and well-structured answer 
to the user query.

USER QUERY:
{query}

RETRIEVED CONTENT:
"""
        
        for i, chunk in enumerate(chunks):
            prompt_text += f"\n--- Document {i+1} ---\n"
            
            original_data = json.loads(chunk.metadata["original_content"])
            raw_text = original_data["raw_text"]
            prompt_text += f"TEXT:\n{raw_text}\n\n"
            
            if original_data["tables_html"]:
                prompt_text += "TABLES:\n"
                for j, table in enumerate(original_data["tables_html"]):
                    prompt_text += f"Table {j+1}:\n{table}\n\n"
        
        prompt_text += """
INSTRUCTIONS:
1. Use the documents as your main source of truth.
2. If images are provided, describe what they show and connect with the text.
3. Be clear and concise, avoid copying raw text.
4. If information is insufficient, state clearly what's missing.
5. Provide a well-structured answer with relevant details.

ANSWER:
"""
        
        message_content = [{"type": "text", "text": prompt_text}]
        
        # Add images to the message
        for chunk in chunks:
            original_data = json.loads(chunk.metadata["original_content"])
            images = original_data.get("images_base64", [])
            for img_b64 in images:
                message_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                })
        
        try:
            message = HumanMessage(content=message_content)
            response = self.llm.invoke([message])
            return response.content
        except Exception as e:
            print(f"Answer generation failed: {e}")
            return "An error occurred while generating the answer. Please try again."
    
    def query(self, query_text: str, k: int = 3) -> str:
        """Query the RAG pipeline"""
        if not self.vectorstore:
            return "Vector store not initialized. Please process a document first."
        
        print(f"\nQuerying: {query_text}")
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        chunks = retriever.invoke(query_text)
        
        print(f"Retrieved {len(chunks)} relevant chunks")
        answer = self.generate_answer(chunks, query_text)
        return answer
    
    def process_pdf(self, file_path: str, persist_directory: str = "chroma_clip_db"):
        """Run complete pipeline for a PDF file"""
        print("="*60)
        print("STARTING MULTIMODAL RAG PIPELINE WITH CLIP")
        print("="*60)
        
        elements = self.partition_document(file_path)
        chunks = self.create_chunks_by_title(elements)
        docs = self.process_chunks(chunks)
        self.create_vector_store(docs, persist_directory)
        
        print("="*60)
        print("PIPELINE COMPLETE - Ready for queries!")
        print("="*60)
        return self


# Example usage
if __name__ == "__main__":
    # Initialize pipeline
    api_key = os.getenv("OPENAI_API_KEY")  # or pass directly
    pipeline = MultimodalRAGPipeline(api_key=api_key)
    
    # Process PDF
    pdf_path = "imagenet.pdf"
    pipeline.process_pdf(pdf_path)
    
    # Query the system
    query = "What makes this model's results better than other approaches?"
    answer = pipeline.query(query, k=3)
    print("\n" + "="*60)
    print("ANSWER:")
    print("="*60)
    print(answer)
    
    # Additional queries
    query2 = "What are the key findings from the tables?"
    answer2 = pipeline.query(query2, k=3)
    print("\n" + "="*60)
    print("ANSWER 2:")
    print("="*60)
    print(answer2)
