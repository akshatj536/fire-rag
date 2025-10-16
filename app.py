import streamlit as st
from pipe import process_pdf, generate_final_answer  # import from your pipeline.py
import os

st.set_page_config(page_title="RAG Chatbot 💬", layout="centered")
st.title("📄 Multimodal RAG Chatbot")

# 1️⃣ File uploader
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    # Save uploaded file temporarily
    temp_file_path = os.path.join("temp.pdf")
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

"""
FROM python:3.13.5-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
COPY src/ ./src/

RUN pip3 install -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "src/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
"""