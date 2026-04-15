import os
import fitz
from typing import List, Optional
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter


CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "64"))


class PDFLoader:
    """Load and extract text from PDF files."""

    def __init__(
        self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )

    def load(self, file_bytes: bytes) -> str:
        """Extract text from PDF bytes."""
        text = ""
        doc = fitz.open(stream=file_bytes, filetype="pdf")

        for page_num in range(len(doc)):
            page = doc[page_num]
            text += page.get_text() + "\n\n"

        doc.close()
        return text

    def load_and_chunk(self, file_bytes: bytes) -> List[str]:
        """Extract text and return chunks."""
        text = self.load(file_bytes)
        return self.splitter.split_text(text)


class MarkdownLoader:
    """Load and extract text from Markdown files."""

    def __init__(
        self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n## ", "\n\n### ", "\n\n", "\n", " ", ""],
        )

    def load(self, file_bytes: bytes) -> str:
        """Extract text from Markdown bytes."""
        return file_bytes.decode("utf-8")

    def load_and_chunk(self, file_bytes: bytes) -> List[str]:
        """Extract text and return chunks."""
        text = self.load(file_bytes)
        return self.splitter.split_text(text)


def get_loader(
    file_extension: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
):
    """Factory function to get appropriate loader based on file extension."""
    ext = file_extension.lower()

    if ext == ".pdf":
        return PDFLoader(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif ext in [".md", ".markdown", ".txt"]:
        return MarkdownLoader(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
