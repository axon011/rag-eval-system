import io
import pytest
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient


class MockPDF:
    def __init__(self, num_pages=2):
        self.num_pages = num_pages
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass
    
    def __len__(self):
        return self.num_pages


class MockPage:
    def get_text(self):
        return "This is a sample PDF document for testing.\n\nIt contains multiple paragraphs of text.\n\nUseful for testing the RAG pipeline."


@pytest.fixture
def mock_pdf_document():
    with patch('fitz.open') as mock_open:
        mock_doc = MagicMock()
        mock_doc.__len__ = lambda self: 2
        
        mock_page1 = MagicMock()
        mock_page1.get_text.return_value = "This is a sample PDF document for testing.\n\nIt contains multiple paragraphs of text.\n\nUseful for testing the RAG pipeline."
        
        mock_page2 = MagicMock()
        mock_page2.get_text.return_value = "Second page content.\n\nMore information about the document.\n\nAdditional context for retrieval."
        
        mock_doc.__getitem__ = lambda self, key: [mock_page1, mock_page2][key]
        
        mock_open.return_value = mock_doc
        
        yield mock_doc


@pytest.fixture
def mock_rag_pipeline():
    with patch('app.routes.ingest.RAGPipeline') as mock_pipeline:
        mock_instance = MagicMock()
        mock_instance.ingest_documents.return_value = {
            "status": "success",
            "chunks_indexed": 10,
            "embed_model": "nomic-embed-text"
        }
        mock_pipeline.return_value = mock_instance
        
        yield mock_instance


def test_ingest_pdf_success(mock_pdf_document, mock_rag_pipeline):
    from app.routes.ingest import extract_text_from_pdf, chunk_text
    
    pdf_bytes = b"%PDF-1.4 mock content"
    text = extract_text_from_pdf(pdf_bytes)
    
    assert text is not None
    assert isinstance(text, str)


def test_chunk_text():
    from app.routes.ingest import chunk_text
    
    text = "This is a test document. " * 100
    
    chunks = chunk_text(text)
    
    assert len(chunks) > 0
    assert all(isinstance(chunk, str) for chunk in chunks)
    assert all(len(chunk) <= 600 for chunk in chunks)


def test_ingest_endpoint_validation():
    from app.routes.ingest import router
    
    assert router is not None
    assert router.prefix == "/ingest"


def test_chunk_text_small():
    from app.routes.ingest import chunk_text
    
    text = "Short text."
    
    chunks = chunk_text(text)
    
    assert len(chunks) >= 1
    assert "Short text." in chunks[0]


def test_chunk_text_with_newlines():
    from app.routes.ingest import chunk_text
    
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    
    chunks = chunk_text(text)
    
    assert len(chunks) > 0
