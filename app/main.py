import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.routes import ingest, query
from app.models.schemas import HealthResponse

app = FastAPI(
    title="RAG Evaluation System",
    description="Production RAG System with Evaluation Harness",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve UI static files
UI_PATH = os.getenv("UI_PATH", "/app/ui")


@app.get("/")
async def root():
    ui_file = os.path.join(UI_PATH, "index.html")
    if os.path.exists(ui_file):
        return FileResponse(ui_file)
    return HealthResponse(
        status="ok",
        services={
            "api": "healthy",
            "ollama": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            "qdrant": f"{os.getenv('QDRANT_HOST', 'localhost')}:{os.getenv('QDRANT_PORT', '6333')}",
            "mlflow": os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        },
    )


app.include_router(ingest.router)
app.include_router(query.router)


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="healthy", services={"api": "healthy"})
