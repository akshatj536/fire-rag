import json
from typing import List
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

from .config import settings

load_dotenv()

def partition_document(file_path):
    """Extract Documents from pdf using unstructured"""
    elements = partition_pdf(
        filename=file_path,
        strategy="hi_res",
        infer_table_structure=True,
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True
    )
    return elements

def create_chunks_by_title(elements):
    """Create chunking by using title as the main differentiator"""
    chunks = chunk_by_title(
        elements,
        combine_text_under_n_chars=500,
        max_characters=3000,
        new_after_n_chars=2400
    )
    return chunks

def separate_content_types(chunk):
    '''Analyze what types of content are there in a chunk'''
    content_data = {
        'text': chunk.text,
        'tables': [],
        'images': [],
        'types': ['text']
    }
    for element in chunk.metadata.orig_elements:
        element_type = type(element).__name__
        if element_type == 'Table':
            content_data['types'].append('table')
            table_html = getattr(element.metadata, 'text_as_html', "table not found")
            content_data['tables'].append(table_html)
        elif element_type == 'Image':
            content_data['types'].append('image')
            image_base64 = getattr(element.metadata, 'image_base64', "image not found")
            content_data['images'].append(image_base64)
    content_data['types'] = list(set(content_data['types']))
    return content_data

def created_ai_summary(text: str, tables: List[str], images: List[str]) -> str:
    """Create AI enhanced summary"""
    try:
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        prompt_text = f"""
You are an expert content summarizer. Your job is to analyze the provided text, tables, and images
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
            prompt_text += "IMAGES are also provided (as base64). Analyze them if relevant.\n\n"

        prompt_text += """YOUR TASK:
        1. Provide a well-structured summary (max ~500 words).
        2. Capture the main ideas, trends, and insights across text, tables, and images.
        3. If data is numeric (from tables), highlight key figures and patterns.
        4. If images are included, mention what they add to the context.
        5. End with a "SEARCHABLE DESCRIPTION" – 3-5 keywords or short phrases someone might use to find this content.

        OUTPUT FORMAT:
        ---------------
        Summary:
        [write summary here]

        Searchable Description:
        [keywords/phrases here]
        """
        message_content = [{"type": "text", "text": prompt_text}]
        for image_base64 in images:
            message_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
            })
        message = HumanMessage(content=message_content)
        response = llm.invoke([message])
        return response.content
    except Exception as e:
        return f"{text[:300]}..."

def summarize_chunks(chunks):
    langchain_documents = []
    for chunk in chunks:
        content_data = separate_content_types(chunk)
        if content_data["tables"] or content_data['images']:
            try:
                enhanced_content = created_ai_summary(
                    content_data['text'],
                    content_data['tables'],
                    content_data['images']
                )
            except Exception as e:
                enhanced_content = content_data['text']
        else:
            enhanced_content = content_data['text']

        doc = Document(
            page_content=enhanced_content,
            metadata={
                "original_content": json.dumps({
                    "raw_text": content_data['text'],
                    "tables_html": content_data["tables"],
                    "images_base64": content_data['images']
                })
            }
        )
        langchain_documents.append(doc)
    return langchain_documents

def create_vector_store(docs):
    """Create and persist ChromaDB vector store"""
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embedding_model,
        persist_directory=settings.CHROMA_DB_PATH,
        collection_metadata={"hnsw:space": "cosine"}
    )
    return vectorstore

def generate_final_answer(chunks, query):
    """generate final answer to the query"""
    try:
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        prompt_text = f"""You are an expert assistant. Your job is to read the provided documents
        (text, tables, and images if available) and give a clear, accurate, and well-structured
        answer to the user query.

        USER QUERY:
        {query}

        CONTENT TO ANALYZE:"""
        for i, chunk in enumerate(chunks):
            prompt_text += f"---Document{i+1} ---\n"
            raw_text = json.loads(chunk.metadata["original_content"])["raw_text"]
            prompt_text += f"TEXT:\n{raw_text}\n\n"
            if json.loads(chunk.metadata["original_content"])["tables_html"]:
                prompt_text += "TABLES: \n"
                for j, table in enumerate(json.loads(chunk.metadata["original_content"])["tables_html"]):
                    prompt_text += f"Tables {j+1}: \n {table} \n \n"
            prompt_text += "\n"
            prompt_text += """INSTRUCTIONS:
            1. Use the documents as your main source of truth.
            2. If images are provided, describe what they show and connect it with the text.
            3. Summarize and explain clearly, avoid copying raw text.
            4. Be concise but thorough.
            5. If data is not provided in the retrieved documents the just say that I dont have enough information

            FINAL ANSWER:"""

        message_content = [{"type": "text", "text": prompt_text}]
        for chunk in chunks:
            if "original_content" in chunk.metadata:
                original_data = json.loads(chunk.metadata["original_content"])
                images_base64 = original_data.get("images_base64", [])
                for image_base64 in images_base64:
                    message_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                    })
        message = HumanMessage(content=message_content)
        response = llm.invoke([message])
        return response.content

    except Exception as e:
        return "Problem occured, retry"

def process_pdf(file_path):
    """Runs the complete pipeline for a given PDF file and returns the retriever"""
    elements = partition_document(file_path)
    chunks = create_chunks_by_title(elements)
    docs = summarize_chunks(chunks)
    db = create_vector_store(docs)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    return retriever

if __name__ == "__main__":
    # This is an example of how to run the pipeline
    # You can run this script directly to test the pipeline
    # Make sure to have a .env file with your OPENAI_API_KEY
    # and a file named 'imagenet.pdf' in the root directory
    retriever = process_pdf("imagenet.pdf")
    query = "what does this model give results so much better than any other approaches"
    chunks = retriever.invoke(query)
    answer = generate_final_answer(chunks, query)
    print("Answer:", answer)