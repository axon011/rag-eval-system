FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv

COPY requirements.txt .
RUN uv pip install --system -r requirements.txt

COPY app /app/app
COPY ui /app/ui
COPY eval /app/eval
COPY mlflow_tracking /app/mlflow_tracking
COPY tests /app/tests

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
