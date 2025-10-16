# Multimodal RAG Chatbot

A chatbot that can answer questions about a PDF document, using a multimodal Retrieval-Augmented Generation (RAG) pipeline.

## Features

- **PDF Processing:** Upload and process PDF documents.
- **Multimodal RAG Pipeline:** Uses CLIP embeddings to understand both text and images in the PDF.
- **Streamlit Web Interface:** A user-friendly web interface for uploading PDFs and asking questions.
- **Dockerized:** Comes with a Dockerfile for easy and consistent deployment.

## Getting Started

### Prerequisites

- Python 3.12
- Docker

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/akshatj536/fire-rag.git
   cd fire-rag
   ```

2. **Create a virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Streamlit App

1. **Set up your environment variables:**
   Create a `.env` file in the root of the project and add your OpenAI API key:
   ```
   OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
   ```

2. **Run the Streamlit app:**
   ```bash
   streamlit run src/app.py
   ```

### Running the Pipeline Directly

You can also run the RAG pipeline directly from the command line:

```bash
python src/pipe_clip.py
```

## Configuration

The following environment variables can be configured in the `.env` file:

- `OPENAI_API_KEY`: Your OpenAI API key.
- `TEMP_FILE_PATH`: The path to store the temporary PDF file (defaults to `temp.pdf`).
- `CHROMA_DB_PATH`: The path to the ChromaDB for the simple pipeline (defaults to `db/chroma_db`).
- `CHROMA_CLIP_DB_PATH`: The path to the ChromaDB for the CLIP pipeline (defaults to `db/chroma_clip_db`).

## Running Tests

To run the test suite, use `pytest`:

```bash
pytest
```