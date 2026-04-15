# Production RAG System with Evaluation Harness

A production-grade RAG (Retrieval-Augmented Generation) Q&A API system with full evaluation pipeline using RAGAs metrics and MLflow experiment tracking. Features a modern web UI with support for multiple LLM providers.

![CI](https://github.com/axon011/rag-eval-system/workflows/CI/badge.svg)
![Python](https://img.shields.io/badge/python-3.11+-blue)
![Docker](https://img.shields.io/badge/docker-ready-blue)

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Web UI (Port 8000)                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │   Query    │  │   Upload    │  │  Settings   │            │
│  │   Tab      │  │   Tab       │  │    Tab      │            │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘            │
└─────────┼────────────────┼────────────────┼───────────────────┘
          │                │                │
          ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FastAPI (Port 8000)                         │
│  ┌─────────────┐              ┌─────────────┐                  │
│  │ /ingest     │              │ /query      │                  │
│  │ (PDF/MD)    │              │ (Q&A)       │                  │
│  └──────┬──────┘              └──────┬──────┘                  │
└─────────┼─────────────────────────────┼─────────────────────────┘
          │                             │
          ▼                             ▼
┌──────────────────┐         ┌──────────────────────────────┐
│  Loaders         │         │  RAG Pipeline              │
│  (PDF/MD/TXT)   │         │  ┌────────────────────────┐  │
└──────────────────┘         │  │ 1. Query Rewrite      │  │
                              │  │ 2. Hybrid Retrieval   │  │
                              │  │    - Dense (Qdrant)  │  │
                              │  │    - Sparse (BM25)   │  │
                              │  │ 3. Reciprocal Rank   │  │
                              │  │    Fusion (RRF)     │  │
                              │  │ 4. Generate Answer  │  │
                              │  │    (Ollama/OpenAI/  │  │
                              │  │    Anthropic)      │  │
                              │  └────────────────────────┘  │
                              └──────────────────────────────┘
          │                             │
          ▼                             ▼
┌──────────────────┐         ┌──────────────────────────────┐
│  Qdrant          │         │  LLM Providers              │
│  (Vector Store)  │         │  - Ollama (local)          │
│                  │         │  - OpenAI (GPT-4o, etc.)   │
└──────────────────┘         │  - Anthropic (Claude)       │
                              └──────────────────────────────┘
```

## Features

- **Hybrid Retrieval**: Combines dense vector search (Qdrant) with sparse BM25 keyword matching via Reciprocal Rank Fusion (RRF) — singleton retriever persists BM25 index across requests
- **RAGAs Evaluation**: Automated quality benchmarking with faithfulness, answer relevancy, context recall, and precision metrics on real evaluation datasets
- **MLflow Integration**: Evaluation results auto-logged with config params and metrics for experiment comparison
- **Multi-Provider LLM**: Switch between Ollama (local), OpenAI, Anthropic, and OpenRouter
- **Modern Web UI**: Chat interface with query, upload, and settings tabs
- **No-Docker Mode**: Run locally with in-memory Qdrant + sentence-transformers embeddings (no servers needed)
- **18 Tests Passing**: Loaders, retriever (BM25/RRF), schemas, pipeline, evaluation dataset validation
- **PDF/Markdown/TXT Ingestion**: Automatic text extraction, chunking (512 chars, 64 overlap), and indexing
- **Query Rewriting**: LLM-powered query expansion for better retrieval
- **Caching**: LRU caching for embeddings and queries with TTL

## Quick Start

### Option A: Run Without Docker (Recommended for Development)

No Docker, no Ollama, no external servers needed. Uses in-memory Qdrant + sentence-transformers for embeddings + any OpenAI-compatible API for generation.

```bash
git clone https://github.com/axon011/rag-eval-system.git
cd rag-eval-system
pip install -r requirements.txt
cp .env.local .env    # Uses in-memory Qdrant + sentence-transformers
python -m uvicorn app.main:app --port 8000
```

**`.env.local` config:**
```env
QDRANT_MODE=memory
EMBED_PROVIDER=sentence-transformers
EMBED_MODEL=all-MiniLM-L6-v2
EMBED_DIM=384
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o
LLM_API_KEY=your-key-here
```

### Option B: Run With Docker (Production)

Full containerized setup with persistent Qdrant, Ollama, and MLflow.

**Prerequisites:** Docker & docker-compose, 8GB+ RAM

```bash
git clone https://github.com/axon011/rag-eval-system.git
cd rag-eval-system
docker-compose up -d
```

3. Wait for services to initialize (first run will pull Docker images and Ollama models)

4. Verify services:
```bash
# Web UI
open http://localhost:8000

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

#### Web UI (Recommended)

Open http://localhost:8000 in your browser:

1. **Query Tab**: Ask questions about your uploaded documents
2. **Upload Tab**: Drag & drop PDF, Markdown, or TXT files
3. **Settings Tab**: Configure LLM provider, model, and retrieval options

#### API Examples

##### 1. Ingest a PDF

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

##### 2. Query the system

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

##### 3. Query with custom settings (OpenAI)

```bash
curl -X POST "http://localhost:8000/query/" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the main topic?",
    "provider": "openai",
    "api_key": "sk-...",
    "model": "gpt-4o",
    "retrieval_mode": "hybrid",
    "max_chunks": 5
  }'
```

## Web UI Features

### Query Tab
- Real-time chat interface
- Copy answer to clipboard
- Retry failed requests
- Message history with stats
- Keyboard shortcuts:
  - `Ctrl+Enter` - Send message
  - `Ctrl+Shift+C` - Clear chat
  - `Ctrl+L` - Go to Upload tab
  - `Escape` - Cancel request

### Upload Tab
- Drag & drop support
- Multiple file formats (PDF, MD, TXT)
- Progress indicator
- Chunk count display

### Settings Tab
- **LLM Provider**: Ollama (local), OpenAI, or Anthropic
- **API Key**: For external providers (stored in localStorage)
- **Model**: Configurable per request
- **Retrieval Mode**: Hybrid, Dense (semantic), or Sparse (BM25)
- **Max Chunks**: Number of documents to retrieve (1-20)

## Demo

### Web UI

```
┌─────────────────────────────────────────────────────────────┐
│  RAG Q&A System                                              │
│  Production Retrieval-Augmented Generation with Evaluation    │
├─────────────────────────────────────────────────────────────┤
│  [Query]  [Upload]  [Settings]     ● FastAPI ● Ollama ...   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ 🤖 What are the key benefits of RAG systems?          ││
│  └─────────────────────────────────────────────────────────┘│
│                                                              │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ RAG systems provide several key benefits:              ││
│  │                                                         ││
│  │ • Improved accuracy by grounding responses in          ││
│  │   source documents                                     ││
│  │ • Reduced hallucinations through context retrieval      ││
│  │ • Ability to cite sources for verification              ││
│  │                                                         ││
│  │ Sources:                                                ││
│  │ [Source 1] 92.5% │ [Source 2] 87.3% │ [Source 3] 81% ││
│  │ Latency: 1.2s | Mode: hybrid                           ││
│  └─────────────────────────────────────────────────────────┘│
│                                                              │
│  ┌────────────────────────────────────────┐  [Send]        │
│  │ Ask a question...                      │                 │
│  └────────────────────────────────────────┘                 │
└─────────────────────────────────────────────────────────────┘
```

### Example API Usage

```bash
# Upload a document
curl -X POST "http://localhost:8000/ingest/" \
  -F "file=@document.pdf"

# Response:
# {"status":"success","chunks_indexed":42,"embed_model":"nomic-embed-text"}

# Ask a question
curl -X POST "http://localhost:8000/query/" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main findings?"}'

# Response:
# {
#   "answer": "The main findings indicate that...",
#   "sources": [{"text": "...", "score": 0.92}],
#   "retrieved_chunks": 5,
#   "latency_ms": 1250.5,
#   "retrieval_mode": "hybrid"
# }
```

## Technical Deep Dive

### Hybrid Retrieval Architecture

The retriever uses a singleton pattern to persist the BM25 index across requests. On startup, it rebuilds the index from chunks stored in Qdrant — no data loss on restart.

```
Query → Embed (sentence-transformers or Ollama)
  ├── Dense: Qdrant cosine similarity → top-k by semantic match
  ├── Sparse: BM25 keyword scoring (lowercase tokenized) → top-k by term frequency
  └── RRF Fusion: score = Σ 1/(60+rank) across both lists → merged top-k
```

**Why hybrid beats dense-only:** Dense search catches paraphrases ("car" → "automobile"), while BM25 catches exact keywords ("error 4201" → "4201"). RRF combines both without needing calibrated scores.

### Evaluation Pipeline

Real evaluation dataset (10 Q&A pairs on LangGraph/RAG topics), scored with RAGAS metrics, results auto-logged to MLflow:

```bash
python -m eval.run_eval
# Computes: faithfulness, answer_relevancy, context_recall, context_precision
# Logs to: MLflow (experiment tracking) + eval/results/run_YYYY-MM-DD.json
```

### Key Engineering Decisions

| Decision | Why |
|----------|-----|
| Singleton Retriever | BM25 index lives in memory — must persist across HTTP requests |
| In-memory Qdrant mode | Development without Docker; `QDRANT_MODE=memory` env var |
| sentence-transformers | Local embeddings (384-dim, all-MiniLM-L6-v2) — no Ollama needed |
| Dynamic vector dimensions | Collection created with actual embedding size, not hardcoded 768 |
| Lowercase BM25 tokenization | Case-insensitive keyword matching |
| qdrant-client v1.17+ compat | Uses `query_points()` with fallback to `search()` for older versions |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI or health info |
| `/health` | GET | Health check |
| `/ingest/` | POST | Upload PDF/MD/TXT file |
| `/ingest/pdf` | POST | Upload PDF only |
| `/ingest/markdown` | POST | Upload Markdown only |
| `/ingest/cache/stats` | GET | Get cache statistics |
| `/ingest/cache/clear` | POST | Clear all caches |
| `/query/` | POST | Ask a question |
| `/query/config` | GET | Get pipeline configuration |
| `/query/cache/stats` | GET | Get query cache stats |

## Configuration

Environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | http://ollama:11434 | Ollama API endpoint |
| `LLM_MODEL` | llama3.2 | Default LLM model |
| `EMBED_MODEL` | nomic-embed-text | Embedding model |
| `QDRANT_HOST` | qdrant | Qdrant server host |
| `QDRANT_PORT` | 6333 | Qdrant server port |
| `QDRANT_COLLECTION` | documents | Collection name |
| `CHUNK_SIZE` | 512 | Text chunk size |
| `CHUNK_OVERLAP` | 64 | Chunk overlap |
| `TOP_K` | 5 | Default chunks to retrieve |
| `RETRIEVAL_MODE` | hybrid | dense, sparse, or hybrid |
| `QUERY_CACHE_ENABLED` | true | Enable query caching |
| `MLFLOW_TRACKING_URI` | http://mlflow:5000 | MLflow server |
| `QDRANT_MODE` | server | `memory` for no-Docker mode |
| `EMBED_PROVIDER` | ollama | `sentence-transformers` for local embeddings |
| `EMBED_DIM` | 768 | Embedding dimension (384 for MiniLM, 768 for nomic) |
| `LLM_PROVIDER` | ollama | `openai`, `anthropic`, `openrouter` |
| `LLM_API_KEY` | - | API key for external LLM providers |

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
│   │   ├── generator.py   # LLM generation (multi-provider)
│   │   ├── pipeline.py    # Full RAG chain
│   │   └── loaders.py     # Document loaders
│   ├── models/
│   │   └── schemas.py     # Pydantic models
│   ├── cache/
│   │   ├── query_cache.py  # LRU query cache
│   │   └── embedding_cache.py # Embedding cache
│   └── workers/           # Async workers
├── ui/
│   └── index.html        # Web UI
├── eval/
│   ├── dataset.py        # Evaluation dataset
│   └── run_eval.py       # RAGAs evaluation
├── mlflow_tracking/
│   └── log_experiment.py  # MLflow logging
├── tests/
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
| Ollama | Local LLM + embeddings (llama3.2, nomic-embed-text) |
| OpenAI | External LLM provider (GPT-4o, etc.) |
| Anthropic | External LLM provider (Claude) |
| Qdrant | Vector database |
| LangChain | RAG orchestration |
| FastAPI | REST API |
| RAGAs | Evaluation metrics |
| MLflow | Experiment tracking |
| Docker | Containerization |

## Cloud Deployment

### Option 1: DigitalOcean Droplet (Recommended)

1. **Create a Droplet**:
   - Go to https://digitalocean.com
   - Create a new Droplet (4GB+ RAM, Ubuntu 22.04)
   - Add your SSH key

2. **SSH into the server**:
   ```bash
   ssh root@your-droplet-ip
   ```

3. **Install Docker**:
   ```bash
   curl -fsSL https://get.docker.com | sh
   sudo usermod -aG docker $USER
   ```

4. **Clone and start**:
   ```bash
   git clone https://github.com/axon011/rag-eval-system.git
   cd rag-eval-system
   docker-compose up -d
   ```

5. **Configure Firewall** (optional):
   ```bash
   ufw allow 8000/tcp
   ufw allow 22/tcp
   ufw enable
   ```

6. **Access the app**:
   - Web UI: `http://your-droplet-ip:8000`
   - Qdrant: `http://your-droplet-ip:6333`
   - MLflow: `http://your-droplet-ip:5000`

**Cost**: ~$24/month for 4GB RAM

---

### Option 2: AWS EC2

1. **Launch Instance**:
   - Go to EC2 Console
   - Launch t3.large or larger (8GB RAM recommended)
   - Use Ubuntu 22.04 AMI

2. **Install Docker**:
   ```bash
   sudo apt update
   sudo apt install -y docker.io
   sudo systemctl start docker
   sudo usermod -aG docker ubuntu
   ```

3. **Configure Security Group**:
   - Open ports: 8000, 6333, 5000, 11434

4. **Deploy**:
   ```bash
   git clone https://github.com/axon011/rag-eval-system.git
   cd rag-eval-system
   docker-compose up -d
   ```

5. **Access**: `http://your-ec2-public-ip:8000`

**Cost**: ~$83/month (on-demand), ~$30/month (spot)

---

### Option 3: Google Cloud Platform

1. **Create VM**:
   - Go to GCP Console
   - Create VM with e2-standard-4 (4 vCPU, 16GB)
   - Allow HTTP/HTTPS traffic

2. **Install Docker**:
   ```bash
   sudo apt update && sudo apt install -y docker.io
   sudo systemctl start docker
   ```

3. **Deploy**:
   ```bash
   git clone https://github.com/axon011/rag-eval-system.git
   cd rag-eval-system
   docker-compose up -d
   ```

4. **Configure Firewall**:
   - Add firewall rules for ports 8000, 6333, 5000, 11434

**Cost**: ~$67/month

---

### Option 4: Using External LLM (No GPU Required)

For cheaper cloud deployment, use OpenAI instead of local Ollama:

1. **Edit `.env`**:
   ```bash
   # Use OpenAI instead of Ollama
   LLM_PROVIDER=openai
   OPENAI_API_KEY=sk-your-key-here
   OPENAI_MODEL=gpt-4o
   ```

2. **Remove Ollama** from docker-compose.yml (optional)

3. **Deploy to Render/Railway** (cheaper than VM):
   - Connect GitHub repo
   - Set environment variables
   - Deploy

---

### Quick Demo: ngrok (Local to Public URL)

To share your local instance temporarily:

1. **Install ngrok**:
   ```bash
   # Windows
   winget install ngrok

   # Linux/Mac
   brew install ngrok
   ```

2. **Tunnel port 8000**:
   ```bash
   ngrok http 8000
   ```

3. **Share the URL**: Anyone can access your local RAG system via the ngrok URL

---

### Production Considerations

1. **Use OpenAI/Anthropic** instead of Ollama for cloud (no GPU needed)
2. **Add authentication** to FastAPI endpoints
3. **Use Qdrant Cloud** (https://qdrant.tech) for managed vector DB
4. **Set up domain** with Nginx + SSL (Let's Encrypt)
5. **Configure backups** for Qdrant data

## License

MIT License
