import streamlit as st
from .pipe import process_pdf, generate_final_answer
import os
from .config import settings

st.set_page_config(page_title="RAG Chatbot 💬", layout="centered")
st.title("📄 Multimodal RAG Chatbot")

# 1️⃣ File uploader
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    # Save uploaded file temporarily
    temp_file_path = settings.TEMP_FILE_PATH
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("✅ File uploaded successfully!")

    # 2️⃣ Load and process PDF
    if st.button("Process PDF"):
        with st.spinner("🔍 Processing and creating vector store... (may take a minute)"):
            retriever = process_pdf(temp_file_path)
        st.session_state["retriever"] = retriever
        st.success("✅ Document processed successfully! You can now ask questions.")

# 3️⃣ Query input
if "retriever" in st.session_state:
    query = st.text_input("💬 Ask me anything about the document:")
    if st.button("Submit Query"):
        if query.strip():
            with st.spinner("🤖 Thinking..."):
                chunks = st.session_state["retriever"].invoke(query)
                answer = generate_final_answer(chunks, query)
            st.write("**Answer:**")
            st.write(answer)
        else:
            st.warning("Please enter a question first.")
