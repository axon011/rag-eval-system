# Production RAG System with Evaluation Harness

A production-grade RAG (Retrieval-Augmented Generation) Q&A API system with full evaluation pipeline using RAGAs metrics and MLflow experiment tracking.

![CI](https://github.com/your-username/rag-eval-system/workflows/CI/badge.svg)
![Python](https://img.shields.io/badge/python-3.11+-blue)
![Docker](https://img.shields.io/badge/docker-ready-blue)

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client (curl/UI)                         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FastAPI (Port 8000)                         │
│  ┌─────────────┐              ┌─────────────┐                │
│  │ /ingest     │              │ /query      │                │
│  │ (PDF → Qdrant)             │ (Q&A)       │                │
│  └──────┬──────┘              └──────┬──────┘                │
└─────────┼─────────────────────────────┼─────────────────────────┘
          │                             │
          ▼                             ▼
┌──────────────────┐         ┌──────────────────────────────┐
│  PyMuPDF         │         │  RAG Pipeline                │
│  (PDF → Text)    │         │  ┌────────────────────────┐  │
└──────────────────┘         │  │ 1. Query Rewrite        │  │
                            │  │ 2. Hybrid Retrieval    │  │
                            │  │    - Dense (Qdrant)    │  │
                            │  │    - Sparse (BM25)     │  │
                            │  │ 3. Reciprocal Rank     │  │
                            │  │    Fusion (RRF)       │  │
                            │  │ 4. Generate Answer     │  │
                            │  │    (Ollama LLM)       │  │
                            │  └────────────────────────┘  │
                            └──────────────────────────────┘
          │                             │
          ▼                             ▼
┌──────────────────┐         ┌──────────────────────────────┐
│  Qdrant          │         │  Ollama                       │
│  (Vector Store)  │         │  - nomic-embed-text (embed)  │
│                  │         │  - llama3.2 (generation)     │
└──────────────────┘         └──────────────────────────────┘
```

## Features

- **PDF Ingestion**: Upload and index PDF documents with automatic text extraction and chunking
- **Hybrid Retrieval**: Combines dense (vector) and sparse (BM25) search with Reciprocal Rank Fusion
- **Query Rewriting**: Improves retrieval by rephrasing user questions
- **RAGAs Evaluation**: Measures faithfulness, answer relevancy, context recall, and precision
- **MLflow Tracking**: Log and compare experiments with different configurations
- **Docker Ready**: Full containerization with docker-compose

## Quick Start

### Prerequisites

- Docker & docker-compose
- Python 3.11+
- uv (optional, for local development)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/rag-eval-system.git
cd rag-eval-system
```

2. Start all services:
```bash
docker-compose up -d
```

3. Verify services:
```bash
# Ollama
curl http://localhost:11434/api/tags

# Qdrant Dashboard
open http://localhost:6333/dashboard

# MLflow
open http://localhost:5000

# FastAPI
curl http://localhost:8000/health
```

### Usage

#### 1. Ingest a PDF

```bash
curl -X POST "http://localhost:8000/ingest/" \
  -F "file=@your-document.pdf"
```

Response:
```json
{
  "status": "success",
  "chunks_indexed": 42,
  "embed_model": "nomic-embed-text"
}
```

#### 2. Query the system

```bash
curl -X POST "http://localhost:8000/query/" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic of the document?"}'
```

Response:
```json
{
  "answer": "The document discusses...",
  "sources": [
    {
      "text": "First few lines of source...",
      "score": 0.92,
      "methods": ["dense", "sparse"]
    }
  ],
  "retrieved_chunks": 5,
  "retrieval_mode": "hybrid",
  "rewritten_query": "main topic discussed in document",
  "latency_ms": 1250.5
}
```

## Configuration

Environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | http://ollama:11434 | Ollama API endpoint |
| `OLLAMA_MODEL` | llama3.2 | LLM model for generation |
| `EMBED_MODEL` | nomic-embed-text | Embedding model |
| `QDRANT_HOST` | qdrant | Qdrant server host |
| `QDRANT_PORT` | 6333 | Qdrant server port |
| `CHUNK_SIZE` | 512 | Text chunk size |
| `CHUNK_OVERLAP` | 64 | Chunk overlap |
| `TOP_K` | 5 | Number of chunks to retrieve |
| `RETRIEVAL_MODE` | hybrid | dense, sparse, or hybrid |

## Evaluation

Run RAGAs evaluation:

```bash
python -m eval.run_eval
```

Results are saved to `eval/results/run_YYYY-MM-DD.json` and logged to MLflow.

### RAGAs Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Faithfulness | > 0.75 | Is answer grounded in context? |
| Answer Relevancy | > 0.80 | Does answer address the question? |
| Context Recall | > 0.70 | Did retrieval fetch needed chunks? |
| Context Precision | > 0.75 | Were retrieved chunks relevant? |

## MLflow Experiments

Compare different configurations:

```python
from mlflow_tracking.log_experiment import MLflowTracker

tracker = MLflowTracker()
tracker.log_experiment(
    config={"chunk_size": 512, "retrieval_mode": "hybrid"},
    metrics={"faithfulness": 0.85, "answer_relevancy": 0.82}
)
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_ingest.py -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html
```

## Project Structure

```
rag-eval-system/
├── app/
│   ├── main.py              # FastAPI entrypoint
│   ├── routes/
│   │   ├── ingest.py       # PDF upload & indexing
│   │   └── query.py        # Q&A endpoint
│   ├── core/
│   │   ├── embedder.py    # Ollama embeddings
│   │   ├── retriever.py   # Hybrid retrieval
│   │   ├── generator.py   # LLM generation
│   │   └── pipeline.py    # Full RAG chain
│   └── models/
│       └── schemas.py     # Pydantic models
├── eval/
│   ├── dataset.py          # Evaluation dataset
│   └── run_eval.py         # RAGAs evaluation
├── mlflow_tracking/
│   └── log_experiment.py   # MLflow logging
├── tests/
│   ├── test_ingest.py
│   └── test_query.py
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── pyproject.toml
├── .env.example
└── README.md
```

## Tech Stack

| Tool | Purpose |
|------|---------|
| Ollama | Local LLM + embeddings |
| Qdrant | Vector database |
| LangChain | RAG orchestration |
| FastAPI | REST API |
| RAGAs | Evaluation metrics |
| MLflow | Experiment tracking |
| Docker | Containerization |

## License

MIT License
