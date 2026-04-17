FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

# Pre-download the default sentence-transformers model so container starts fast
# on Azure Container Apps / App Service (avoids startup-timeout on first request).
ARG EMBED_MODEL=all-MiniLM-L6-v2
ENV HF_HOME=/app/.cache/huggingface
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBED_MODEL}')"

COPY app /app/app
COPY ui /app/ui
COPY eval /app/eval
COPY mlflow_tracking /app/mlflow_tracking
COPY tests /app/tests

# Create logs directory and switch to non-root user
RUN mkdir -p /app/logs && \
    useradd -m -u 1000 -s /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
