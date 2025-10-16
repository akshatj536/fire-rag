import streamlit as st
import os
import tempfile
from pathlib import Path
import sys

# Add src directory to path if needed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pipeline import MultimodalRAGPipeline

# Page configuration
st.set_page_config(
    page_title="Multimodal RAG Chatbot 💬",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        height: 3rem;
        font-size: 1rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "document_processed" not in st.session_state:
    st.session_state.document_processed = False

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Key input
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Enter your OpenAI API key",
        value=os.getenv("OPENAI_API_KEY", "")
    )
    
    # Model selection
    k_retrieval = st.slider(
        "Number of chunks to retrieve",
        min_value=1,
        max_value=10,
        value=3,
        help="How many document chunks to retrieve for each query"
    )
    
    st.divider()
    
    # Status indicators
    st.subheader("📊 Status")
    if st.session_state.document_processed:
        st.success("✅ Document loaded")
    else:
        st.info("📄 No document loaded")
    
    st.metric("Chat Messages", len(st.session_state.chat_history))
    
    st.divider()
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()
    
    # Reset button
    if st.button("🔄 Reset Pipeline"):
        st.session_state.pipeline = None
        st.session_state.document_processed = False
        st.session_state.chat_history = []
        st.rerun()

# Main content
st.markdown('<p class="main-header">📄 Multimodal RAG Chatbot</p>', unsafe_allow_html=True)
st.markdown("Upload a PDF document and ask questions about its content, including text, tables, and images!")

# File upload section
st.subheader("1️⃣ Upload Document")
uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"],
    help="Upload a PDF document to analyze"
)

if uploaded_file:
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.info(f"📎 **File:** {uploaded_file.name} ({uploaded_file.size / 1024:.2f} KB)")
    
    with col2:
        process_button = st.button("🚀 Process PDF", use_container_width=True)
    
    if process_button:
        if not api_key:
            st.error("⚠️ Please enter your OpenAI API key in the sidebar")
        else:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                temp_file_path = tmp_file.name
            
            try:
                with st.spinner("🔍 Processing PDF... This may take a few minutes."):
                    # Initialize pipeline
                    if st.session_state.pipeline is None:
                        st.session_state.pipeline = MultimodalRAGPipeline(api_key=api_key)
                    
                    # Progress indicators
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("Extracting content from PDF...")
                    progress_bar.progress(25)
                    
                    # Process the document
                    st.session_state.pipeline.process_pdf(
                        temp_file_path,
                        persist_directory=f"chroma_db_{uploaded_file.name}"
                    )
                    
                    progress_bar.progress(100)
                    status_text.text("Processing complete!")
                    
                    st.session_state.document_processed = True
                    st.success("✅ Document processed successfully! You can now ask questions below.")
                    
            except Exception as e:
                st.error(f"❌ Error processing document: {str(e)}")
                st.exception(e)
            
            finally:
                # Clean up temporary file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)

# Chat interface
if st.session_state.document_processed and st.session_state.pipeline:
    st.divider()
    st.subheader("2️⃣ Ask Questions")
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("### 💬 Conversation History")
        for i, chat in enumerate(st.session_state.chat_history):
            with st.container():
                st.markdown(f"**Q{i+1}:** {chat['question']}")
                st.markdown(f"**A{i+1}:** {chat['answer']}")
                st.divider()
    
    # Query input
    with st.form(key="query_form", clear_on_submit=True):
        query = st.text_area(
            "Enter your question:",
            height=100,
            placeholder="Ask anything about the document...",
            help="Type your question about the uploaded document"
        )
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            submit_button = st.form_submit_button("🔍 Search", use_container_width=True)
    
    if submit_button and query.strip():
        with st.spinner("🤖 Analyzing document and generating answer..."):
            try:
                # Query the pipeline
                answer = st.session_state.pipeline.query(query, k=k_retrieval)
                
                # Add to chat history
                st.session_state.chat_history.append({
                    "question": query,
                    "answer": answer
                })
                
                # Display the answer
                st.markdown("### 💡 Answer")
                st.markdown(answer)
                
                # Rerun to update chat history display
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error generating answer: {str(e)}")
                st.exception(e)
    
    elif submit_button:
        st.warning("⚠️ Please enter a question first.")

else:
    if not st.session_state.document_processed:
        st.info("👆 Upload and process a PDF document to start asking questions!")

# Footer
st.divider()
st.markdown("""
    <div style='text-align: center; color: gray; font-size: 0.8rem;'>
        <p>Powered by CLIP embeddings, ChromaDB, and GPT-4</p>
        <p>Supports multimodal content: text, tables, and images</p>
    </div>
""", unsafe_allow_html=True)
