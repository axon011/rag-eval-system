# Project: RAG Eval System - Production RAG Q&A with Evaluation Harness

## Overview
RAG Q&A system that lets users upload PDF/Markdown/TXT documents, index them into Qdrant vector DB, and ask natural language questions answered by an LLM grounded in retrieved document chunks. Includes hybrid retrieval (dense + BM25 + RRF), RAGAs evaluation metrics, and MLflow experiment tracking.

## Architecture

```
User uploads PDF/MD/TXT
         ↓
    Document Loader (PyMuPDF / plain text)
         ↓
    Chunking (RecursiveCharacterTextSplitter, 512 chars, 64 overlap)
         ↓
    Embedding (Ollama nomic-embed-text / OpenAI)
         ↓
    Qdrant Vector DB (dense storage)
    + BM25 In-Memory Index (sparse - BROKEN, see issues)
         ↓
User asks question → Query Rewrite (LLM) → Hybrid Retrieval (RRF) → LLM Generation → Answer
```

## Tech Stack
- **API**: FastAPI (port 8000), serves single-page HTML UI
- **Vector DB**: Qdrant (dense retrieval, cosine similarity)
- **Sparse retrieval**: BM25 (rank-bm25 library, in-memory)
- **Fusion**: Reciprocal Rank Fusion (dense + sparse)
- **Embeddings**: Ollama (nomic-embed-text) or OpenAI
- **LLM**: Ollama (llama3.2), OpenAI, Anthropic, or OpenRouter
- **Evaluation**: RAGAs (faithfulness, answer_relevancy, context_recall, context_precision)
- **Experiment tracking**: MLflow
- **Orchestration**: Docker Compose (FastAPI, Ollama, Qdrant, MLflow)
- **Caching**: In-memory LRU with TTL (embeddings + query responses)
- **Package Manager**: uv

## Project Structure
```
rag-eval-system/
├── app/
│   ├── main.py                # FastAPI entrypoint, CORS, static files
│   ├── routes/
│   │   ├── ingest.py          # POST /ingest - upload & index docs
│   │   └── query.py           # POST /query - Q&A endpoint
│   ├── core/
│   │   ├── embedder.py        # Ollama/OpenAI embedding wrapper
│   │   ├── retriever.py       # Hybrid retrieval (dense + BM25 + RRF)
│   │   ├── generator.py       # LLM generation (multi-provider)
│   │   ├── pipeline.py        # Full RAG chain: rewrite → retrieve → generate
│   │   └── loaders.py         # Document loaders (PDF/MD/TXT)
│   ├── models/
│   │   └── schemas.py         # Pydantic request/response models
│   ├── cache/
│   │   ├── query_cache.py     # LRU query cache (IGNORES provider/model)
│   │   └── embedding_cache.py # Embedding cache
│   ├── workers/               # Empty - async embedding planned but not implemented
│   └── logging_config.py      # JSON structured logging (NEVER WIRED UP)
├── ui/
│   └── index.html             # Web UI
├── eval/
│   ├── dataset.py             # 10 Q&A pairs - ALL PLACEHOLDER DATA
│   └── run_eval.py            # RAGAs evaluation runner
├── mlflow_tracking/
│   └── log_experiment.py      # MLflow logging
├── tests/
│   ├── test_ingest.py         # BROKEN - imports non-existent functions
│   └── test_query.py          # Query tests
├── .github/workflows/
│   └── ci.yml                 # CI (triggers on main/develop, repo uses master)
├── docker-compose.yml         # 4 services: fastapi, ollama, qdrant, mlflow
├── Dockerfile
├── requirements.txt
├── pyproject.toml             # MISSING deps: cachetools, python-json-logger, aiofiles
├── .env.example
├── API.md
├── AGENT.md                   # Original build plan with phases
└── README.md
```

## Endpoints

| Method | Path | Description | Status |
|--------|------|-------------|--------|
| GET | / | Web UI | Working |
| GET | /health | Health check | Working |
| POST | /ingest/ | Upload PDF/MD/TXT | Working (dense only) |
| POST | /ingest/pdf | Upload PDF only | Working |
| POST | /ingest/markdown | Upload Markdown only | Working |
| POST | /query/ | Ask questions | Working (dense only, BM25 broken) |
| GET | /query/config | Pipeline configuration | Working |
| GET | /ingest/cache/stats | Cache statistics | Working |
| POST | /ingest/cache/clear | Clear caches | Working |

## Running Services

| Service | URL |
|---------|-----|
| FastAPI + Web UI | http://localhost:8000 |
| FastAPI Docs | http://localhost:8000/docs |
| Qdrant Dashboard | http://localhost:6333/dashboard |
| MLflow UI | http://localhost:5000 |
| Ollama API | http://localhost:11434 |

## Common Commands
```bash
# Start all services
docker-compose up -d

# Rebuild after code changes
docker-compose up -d --build fastapi

# Pull Ollama models (required after first docker-compose up)
docker exec ollama ollama pull llama3.2
docker exec ollama ollama pull nomic-embed-text

# View logs
docker logs rag-fastapi

# Run tests (many will fail - see issues)
uv run pytest tests/ -v
```

---

## Known Issues & Required Fixes

### P0 — Critical (core features broken)

1. **Hybrid retrieval (BM25) is completely non-functional**
   - `RAGPipeline()` is instantiated fresh per request in both `ingest.py:38` and `query.py:31`
   - BM25 index (`self.chunks`, `self.bm25_index`) is stored as instance variables on `Retriever`
   - New pipeline = new retriever = empty BM25 index every time
   - Sparse retrieval always returns `[]`, hybrid mode silently degrades to dense-only
   - **This is the project's headline feature and it does not work**
   - **Fix**: Use a shared singleton `RAGPipeline` (like FastAPI dependency injection), or persist BM25 index to disk/Redis, or rebuild from Qdrant on startup
   - Files: `app/routes/query.py`, `app/routes/ingest.py`, `app/core/retriever.py`, `app/core/pipeline.py`

2. **Evaluation dataset is 100% placeholder**
   - `eval/dataset.py:20-72` has 10 Q&A pairs all containing `"[extract from your document]"` as answers and `"The relevant context from the document"` as context
   - RAGAs evaluation with this data produces meaningless metrics
   - **Fix**: Upload real documents, generate actual Q&A pairs, populate with real ground truth
   - File: `eval/dataset.py`

3. **Tests are broken — will fail with ImportError**
   - `tests/test_ingest.py` lines 60, 70, 89, 100 import `extract_text_from_pdf` and `chunk_text` from `app.routes.ingest` — these functions don't exist (code uses `get_loader()` from `app.core.loaders`)
   - **Fix**: Rewrite test imports to match actual codebase
   - File: `tests/test_ingest.py`

### P1 — High (security, correctness, deployment)

4. **Hardcoded vector dimension (768)**
   - `retriever.py:37` hardcodes `size=768` — only works for `nomic-embed-text`
   - Switching to OpenAI `text-embedding-3-small` (1536 dims) silently breaks
   - **Fix**: Make dimension configurable, or detect from embedding model
   - File: `app/core/retriever.py`

5. **Query cache ignores provider/model/retrieval_mode**
   - `query_cache.py:36` hashes only the question text
   - Same question with different LLMs returns cached result from whichever ran first
   - **Fix**: Include full request params (provider, model, retrieval_mode) in cache key
   - File: `app/cache/query_cache.py`

6. **Missing dependencies in pyproject.toml**
   - `cachetools`, `python-json-logger`, `aiofiles` are in `requirements.txt` but not `pyproject.toml`
   - `uv sync` will miss them, causing import failures
   - **Fix**: Add to pyproject.toml `[project.dependencies]`
   - File: `pyproject.toml`

7. **No document metadata in chunks**
   - Chunks stored in Qdrant only have `text` and `chunk_index` as payload
   - No source filename, page number, or section heading
   - Makes it impossible to cite which document/page an answer came from
   - **Fix**: Add `source_file`, `page_number` to Qdrant payload
   - Files: `app/routes/ingest.py`, `app/core/retriever.py`

8. **CORS misconfiguration**
   - `main.py:16-22` uses `allow_origins=["*"]` with `allow_credentials=True`
   - Per CORS spec, `*` with credentials is invalid — browsers reject it
   - **Fix**: Either restrict origins or remove `allow_credentials=True`
   - File: `app/main.py`

9. **Container runs as root, no HEALTHCHECK**
   - Dockerfile has no `USER` directive or `HEALTHCHECK`
   - **Fix**: Add non-root user and health check
   - File: `Dockerfile`

10. **No Ollama model pre-pull automation**
    - After `docker-compose up`, first query fails because `llama3.2` and `nomic-embed-text` aren't downloaded
    - **Fix**: Add a startup script or init container that pulls models
    - File: `docker-compose.yml` or new `scripts/init.sh`

### P2 — Medium (code quality, NLP concerns)

11. **BM25 tokenization is too naive**
    - `retriever.py:68` tokenizes with `chunk.split()` — no lowercasing, no punctuation stripping, no stopword removal
    - `"Document."` and `"document"` are different tokens
    - **Fix**: Add `.lower().strip()`, punctuation removal, optionally stemming
    - File: `app/core/retriever.py`

12. **Chunk size of 512 characters is small**
    - 512 chars ≈ 80-100 tokens. Can break coherent passages mid-sentence.
    - Standard RAG chunk size is 1000-1500 chars (or 256-512 tokens)
    - **Fix**: Increase `CHUNK_SIZE` to 1000-1500, or use token-based splitting
    - File: `app/routes/ingest.py` or config

13. **No document deduplication**
    - Uploading the same PDF twice re-embeds and inserts all chunks with new UUIDs
    - Leads to duplicated context in retrieval results
    - **Fix**: Hash document content, check before re-indexing
    - File: `app/routes/ingest.py`

14. **Query rewriting uses the expensive generation LLM**
    - `pipeline.py:43` calls the same LLM for rewriting and generation — doubles cost for OpenAI/Anthropic
    - **Fix**: Option to use a cheaper/faster model for rewriting, or make rewriting optional
    - File: `app/core/pipeline.py`

15. **No connection pooling or retry logic**
    - Qdrant and Ollama clients created fresh per request (pipeline is per-request)
    - No retry, no circuit breaker — raw 500 if Qdrant is temporarily unreachable
    - **Fix**: Shared client instances via dependency injection
    - Files: `app/core/retriever.py`, `app/core/embedder.py`

16. **Logging config is dead code**
    - `app/logging_config.py` defines JSON logger with file rotation but is never imported
    - Hardcoded path `/app/logs/rag.log` — directory doesn't exist in Docker image
    - **Fix**: Wire up in `main.py` or delete
    - File: `app/logging_config.py`, `app/main.py`

17. **Embedder ignores constructor `base_url` parameter**
    - `embedder.py:16` always overwrites `self.base_url` with `os.getenv(...)`, ignoring the passed argument
    - **Fix**: Use `base_url or os.getenv(...)` pattern
    - File: `app/core/embedder.py`

18. **CI triggers on main/develop but repo uses master**
    - CI workflow never actually runs
    - **Fix**: Change branch to `master` or rename branch
    - File: `.github/workflows/ci.yml`

19. **RAGAs column names may be outdated**
    - `run_eval.py` uses `question`, `answer`, `contexts`, `ground_truth` columns
    - RAGAs v0.1.x uses these, but newer versions use `user_input`, `response`, `retrieved_contexts`, `reference`
    - Pinned `ragas==0.1.21` should work, but upgrading will break
    - File: `eval/run_eval.py`

---

## Improvement Action Plan

| # | Priority | Action | Files |
|---|----------|--------|-------|
| 1 | P0 | Fix BM25 persistence — shared singleton RAGPipeline | `query.py`, `ingest.py`, `pipeline.py`, `retriever.py` |
| 2 | P0 | Create real eval dataset from uploaded documents | `eval/dataset.py` |
| 3 | P0 | Fix broken test imports | `tests/test_ingest.py` |
| 4 | P1 | Make vector dimension configurable | `app/core/retriever.py` |
| 5 | P1 | Include model/provider in query cache key | `app/cache/query_cache.py` |
| 6 | P1 | Add document metadata to chunks | `ingest.py`, `retriever.py` |
| 7 | P1 | Sync pyproject.toml dependencies | `pyproject.toml` |
| 8 | P1 | Fix CORS config | `app/main.py` |
| 9 | P1 | Add Dockerfile HEALTHCHECK + non-root user | `Dockerfile` |
| 10 | P2 | Fix BM25 tokenization (lowercase, strip punctuation) | `app/core/retriever.py` |
| 11 | P2 | Increase chunk size to 1000-1500 chars | config or `ingest.py` |
| 12 | P2 | Add document deduplication | `app/routes/ingest.py` |
| 13 | P2 | Wire up logging config or delete dead code | `app/logging_config.py` |
| 14 | P2 | Fix CI branch trigger | `.github/workflows/ci.yml` |

---

## Unimplemented Phases (from AGENT.md)

| Phase | Feature | Status |
|-------|---------|--------|
| 7 | Markdown Support | Partial (loader exists, routes exist) |
| 8 | Async Embedding Pre-computation | Not implemented (empty workers/) |
| 9 | Query Caching | Partial (cache module exists, but ignores model/provider) |
| 10 | JSON Structured Logging | Partial (config exists, never wired up) |
| 11 | Performance Optimization | Not implemented (no connection pooling, no p95 tracking) |

---

## Interview Weak Points
- **"Does hybrid retrieval actually work?"** — No. BM25 state is lost between requests. Dense-only in practice.
- **"Show me an eval run with real data"** — Can't. Dataset is all placeholders. Must populate before demo.
- **"Do your tests pass?"** — No. Broken imports in test_ingest.py. Fix before showing.
- **"How do you handle document provenance?"** — Currently can't cite source file or page number.
- **"What happens if I switch to OpenAI embeddings?"** — Silently breaks (hardcoded 768 dims).

## Notes
- Use uv for Python package management: `uv sync`, `uv add`, `uv run`
- Name MLflow experiments: rag-chunk256-dense, rag-chunk512-hybrid, etc.
- Docker socket permission may require: `sudo chmod 666 /var/run/docker.sock`
- After `docker-compose up`, must manually pull Ollama models before first query
