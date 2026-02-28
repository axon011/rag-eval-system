# Agent.md - Production RAG System with Evaluation Harness

## Project Overview

- **Project Name**: rag-eval-system
- **Goal**: Build a production-grade RAG Q&A API over PDF documents with full evaluation pipeline
- **Stack**: Ollama, LangChain, Qdrant, FastAPI, RAGAs, MLflow, Docker, GitHub Actions
- **Package Manager**: uv
- **Status**: ✅ Implemented

---

## Prerequisites Installed

### 1. Docker & docker-compose
```bash
# Installed via apt
sudo apt-get install docker.io docker-compose
```

### 2. Ollama Models
```bash
# Models pulled and running in Docker
docker exec ollama ollama pull llama3.2      # 2GB
docker exec ollama ollama pull nomic-embed-text  # 274MB
```

---

## Project Structure

```
rag-eval-system/
├── AGENT.md                      # This file
├── README.md                      # Project documentation
├── pyproject.toml                # uv project config
├── requirements.txt              # Python dependencies
├── docker-compose.yml            # All services (Ollama, Qdrant, MLflow, FastAPI)
├── Dockerfile                    # FastAPI container
├── .env.example                  # Environment template
├── sample.pdf                    # Test PDF document
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI entrypoint
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── ingest.py            # POST /ingest - upload & index PDFs
│   │   └── query.py             # POST /query - ask question, get answer + sources
│   ├── core/
│   │   ├── __init__.py
│   │   ├── embedder.py          # Ollama nomic-embed-text wrapper
│   │   ├── retriever.py         # Qdrant hybrid retrieval (dense + BM25 + RRF)
│   │   ├── generator.py         # LLM answer generation (Ollama llama3.2)
│   │   └── pipeline.py          # Full RAG chain: retrieve → rerank → generate
│   └── models/
│       ├── __init__.py
│       └── schemas.py           # Pydantic request/response schemas
├── eval/
│   ├── __init__.py
│   ├── dataset.py               # Build Q&A eval dataset from docs
│   └── run_eval.py             # Run RAGAs evaluation suite
├── mlflow_tracking/
│   ├── __init__.py
│   └── log_experiment.py       # Log RAGAs scores + config to MLflow
├── tests/
│   ├── __init__.py
│   ├── test_ingest.py
│   └── test_query.py
└── .github/
    └── workflows/
        └── ci.yml              # GitHub Actions: lint + pytest
```

---

## Implementation Status

### Phase 1: Project Initialization ✅

- [x] Initialize uv project
- [x] Create pyproject.toml with all dependencies
- [x] Create requirements.txt for compatibility
- [x] Create .env.example template

### Phase 2: Infrastructure Setup ✅

- [x] Write docker-compose.yml with services:
  - ollama (port 11434)
  - qdrant (port 6333)
  - mlflow (port 5000)
  - fastapi (port 8000)
- [x] Write Dockerfile for FastAPI app
- [x] Verify all services start: `docker-compose up -d`

### Phase 3: Core RAG Components ✅

- [x] **app/core/embedder.py**: Ollama embedding wrapper using langchain_community.embeddings.OllamaEmbeddings with nomic-embed-text
- [x] **app/core/retriever.py**: 
  - Dense retrieval from Qdrant (top-k=5)
  - BM25 sparse retrieval using rank_bm25
  - Reciprocal Rank Fusion (RRF) for hybrid results
- [x] **app/core/pipeline.py**: Full RAG chain with query rewriting, retrieval, and generation
- [x] **app/core/generator.py**: LLM prompt construction and answer generation using Ollama llama3.2
- [x] **app/models/schemas.py**: Pydantic models for request/response validation

### Phase 4: API Routes ✅

- [x] **app/routes/ingest.py**:
  - POST /ingest endpoint
  - Accept multipart/form-data PDF upload
  - Extract text using PyMuPDF (fitz)
  - Chunk with RecursiveCharacterTextSplitter (chunk_size=512, overlap=64)
  - Embed and upsert to Qdrant collection "documents"
- [x] **app/routes/query.py**:
  - POST /query endpoint
  - Query rewriting step
  - Hybrid retrieval
  - Return {answer, sources, latency_ms}
- [x] **app/main.py**: FastAPI app entrypoint with route registration

### Phase 5: Evaluation Pipeline ✅

- [x] **eval/dataset.py**: Create 10 Q&A pairs from PDF corpus (ground truth)
- [x] **eval/run_eval.py**: Run RAGAs metrics:
  - faithfulness
  - answer_relevancy
  - context_recall
  - context_precision
  - Save results to eval/results/run_YYYY-MM-DD.json
- [x] **mlflow_tracking/log_experiment.py**: Log parameters and metrics to MLflow

### Phase 6: Testing & CI/CD ✅

- [x] **tests/test_ingest.py**: Test PDF ingestion with sample 2-page PDF
- [x] **tests/test_query.py**: Test query endpoint
- [x] **.github/workflows/ci.yml**: 
  - Trigger on push to main/PR
  - Steps: checkout → setup Python 3.11 → uv sync → pytest
- [x] **README.md**: Architecture diagram, setup instructions, curl examples

---

## Running Services

```bash
# Start all services
cd /home/axon/Downloads/RAG/rag-eval-system
docker-compose up -d

# Check status
docker ps

# View logs
docker logs rag-fastapi
docker logs rag-ollama
docker logs rag-qdrant
docker logs rag-mlflow
```

---

## Service URLs

| Service | URL |
|---------|-----|
| FastAPI | http://localhost:8000 |
| FastAPI Docs | http://localhost:8000/docs |
| Qdrant Dashboard | http://localhost:6333/dashboard |
| MLflow UI | http://localhost:5000 |
| Ollama API | http://localhost:11434 |

---

## API Usage

### Ingest PDF
```bash
curl -X POST "http://localhost:8000/ingest/" \
  -F "file=@sample.pdf"
```

### Query
```bash
curl -X POST "http://localhost:8000/query/" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this document about?"}'
```

---

## Dependencies

```
fastapi==0.115.0
uvicorn[standard]==0.30.0
langchain==0.2.16
langchain-community==0.2.16
qdrant-client==1.10.1
ragas==0.1.21
mlflow==2.15.0
pymupdf==1.24.9
rank-bm25==0.2.2
pydantic==2.8.0
python-multipart==0.0.9
httpx==0.27.0
pytest==8.3.2
python-dotenv==1.0.1
tqdm==4.66.0
datasets==2.18.0
```

---

## Notes

- Use uv for all Python package management: `uv sync`, `uv add`, `uv run`
- Commit after each milestone
- Use sample PDF (10-page research paper) for testing
- Name MLflow experiments: rag-chunk256-dense, rag-chunk512-hybrid, etc.
- Keep eval dataset in eval/dataset.json

---

## Known Issues

- Sample PDF needs to be a valid PDF file (not manually created)
- Docker socket permission may require: `sudo chmod 666 /var/run/docker.sock`
- May need to restart containers after code changes: `docker-compose restart`
