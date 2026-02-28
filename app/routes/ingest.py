import os
import io
from fastapi import APIRouter, UploadFile, File, HTTPException
from langchain.text_splitter import RecursiveCharacterTextSplitter
import fitz

from app.models.schemas import IngestResponse

router = APIRouter(prefix="/ingest", tags=["Ingest"])

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "64"))


def extract_text_from_pdf(file_bytes: bytes) -> str:
    text = ""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text += page.get_text()
    
    doc.close()
    return text


def chunk_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = splitter.split_text(text)
    return chunks


@router.post("/", response_model=IngestResponse)
async def ingest_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        contents = await file.read()
        
        text = extract_text_from_pdf(contents)
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from PDF")
        
        chunks = chunk_text(text)
        
        from app.core.pipeline import RAGPipeline
        pipeline = RAGPipeline()
        
        result = pipeline.ingest_documents(chunks)
        
        return IngestResponse(**result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


@router.get("/health")
async def ingest_health():
    return {"status": "ingest service healthy"}
