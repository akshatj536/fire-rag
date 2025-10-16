# Multimodal RAG Chatbot

**fire-rag** is a multimodal Retrieval-Augmented Generation (RAG) project. A chatbot that can answer questions about a PDF document, using a multimodal Retrieval-Augmented Generation (RAG) pipeline.

The core idea of this project is to use **summarization of multimodal data** for building embeddings and enabling retrieval. Instead of directly embedding raw data from different modalities, the data is first summarized into a compact representation. These summaries are then embedded into a vector space, making it easier to perform retrieval and generate context-aware responses.  

Additionally, **fire-rag** integrates **CLIP (Contrastive Language–Image Pretraining)** embeddings within the RAG framework to bridge text and visual data effectively.

## Features

- **PDF Processing:** Upload and process PDF documents.
- **Multimodal RAG Pipeline:** Uses CLIP embeddings to understand both text and images in the PDF.
- **Streamlit Web Interface:** A user-friendly web interface for uploading PDFs and asking questions.
- **Dockerized:** Comes with a Dockerfile for easy and consistent deployment.

## Architecture

- **Multimodal Input**: Works with text, images, and other modalities.  
- **Summarization First**: Summarizes multimodal content into concise, information-dense representations.  
- **Embedding & Retrieval**: Creates embeddings using both summarization-based and CLIP-based models.  
- **RAG Pipeline**: Combines retrieval and generation for context-aware answers.  
- **CLIP Model Integration**: Uses CLIP to generate joint embeddings for text–image pairs, enriching the retrieval process.  

### Project Flow

1. **Data Ingestion** – Multimodal data (text, images, videos, etc.) is collected.  
2. **Summarization** – Each piece of data is summarized into a compact textual or structured form.  
3. **Embedding** – Two types of embeddings are created:
   - **Text Summarization Embeddings** (using language models)  
   - **CLIP Embeddings** (for image and text alignment)  
4. **Retrieval** – Given a query, relevant embeddings (textual or visual) are retrieved.  
5. **Generation** – Retrieved results are passed to the RAG generator to produce a coherent, context-aware response.  

### CLIP Model Implementation

Below is a simplified representation of how **CLIP** is integrated into the RAG pipeline:

```python
from transformers import CLIPProcessor, CLIPModel
import torch

# Load pretrained CLIP model
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

# Example: Compute CLIP embeddings for an image-text pair
inputs = processor(
    text=["a photo of a fire truck"],
    images=["fire_truck.jpg"],
    return_tensors="pt",
    padding=True
)

# Generate embeddings
outputs = model(**inputs)
image_embeds = outputs.image_embeds
text_embeds = outputs.text_embeds

# Combine embeddings for multimodal retrieval
combined_embeds = torch.cat((text_embeds, image_embeds), dim=1)
```

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