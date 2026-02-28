# RAG Q&A System - API Endpoints

## Base URL
```
http://localhost:8000
```

## Interactive Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## Endpoints

### 1. Health Check

**GET** `/health`

Check if the API is running.

**Response:**
```json
{
  "status": "healthy",
  "services": {
    "api": "healthy"
  }
}
```

---

### 2. Upload PDF

**POST** `/ingest/`

Upload and index a PDF document.

| Parameter | Type | Description |
|-----------|------|-------------|
| `file` | File (multipart/form-data) | PDF file to upload |

**Example:**
```bash
curl -X POST "http://localhost:8000/ingest/" \
  -F "file=@document.pdf"
```

**Response:**
```json
{
  "status": "success",
  "chunks_indexed": 42,
  "embed_model": "nomic-embed-text"
}
```

---

### 3. Query

**POST** `/query/`

Ask a question about uploaded documents.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `question` | string | Yes | The question to ask |
| `rewrite_query` | boolean | No | Whether to rewrite the query for better retrieval (default: true) |

**Example:**
```bash
curl -X POST "http://localhost:8000/query/" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic of the document?"}'
```

**Response:**
```json
{
  "answer": "The document discusses machine learning...",
  "sources": [
    {
      "text": "First few lines of source text...",
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

---

### 4. Get Query Config

**GET** `/query/config`

Get current RAG pipeline configuration.

**Response:**
```json
{
  "embed_model": "nomic-embed-text",
  "llm_model": "llama3.2",
  "retrieval_mode": "hybrid",
  "top_k": 5
}
```

---

### 5. Ingest Health

**GET** `/ingest/health`

Check ingest service status.

**Response:**
```json
{
  "status": "ingest service healthy"
}
```

---

## Python Client Example

```python
import requests

BASE_URL = "http://localhost:8000"

# Upload PDF
def upload_pdf(file_path):
    with open(file_path, 'rb') as f:
        response = requests.post(
            f"{BASE_URL}/ingest/",
            files={"file": f}
        )
    return response.json()

# Query
def ask_question(question):
    response = requests.post(
        f"{BASE_URL}/query/",
        json={"question": question}
    )
    return response.json()

# Usage
result = upload_pdf("document.pdf")
print(result)

answer = ask_question("What is this about?")
print(answer["answer"])
```

---

## JavaScript Client Example

```javascript
const API_BASE = 'http://localhost:8000';

// Upload PDF
async function uploadPDF(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await fetch(`${API_BASE}/ingest/`, {
        method: 'POST',
        body: formData
    });
    return response.json();
}

// Query
async function askQuestion(question) {
    const response = await fetch(`${API_BASE}/query/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question })
    });
    return response.json();
}
```
