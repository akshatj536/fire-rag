import pytest
from src.pipe_clip import MultimodalRAGPipeline
import os

def test_process_pdf_clip():
    """
    Tests the process_pdf method of the MultimodalRAGPipeline from src.pipe_clip
    """
    # Use a small, real PDF for testing to ensure the pipeline works end-to-end
    file_path = "imagenet.pdf"
    
    # Check if the test file exists
    assert os.path.exists(file_path), f"Test file not found at {file_path}"
    
    # Initialize and run the processing pipeline
    pipeline = MultimodalRAGPipeline()
    processed_pipeline = pipeline.process_pdf(file_path)
    
    # Assert that the pipeline object is returned
    assert processed_pipeline is not None, "process_pdf should return the pipeline object"
    assert processed_pipeline.vectorstore is not None, "process_pdf should create a vectorstore"
