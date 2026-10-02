"""FastAPI endpoints for Egyptian Civil Code Q&A."""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from src.api.schemas import AskRequest, AskResponse, HealthResponse
from src.rag.generator import run_rag_pipeline
from src.rag.retriever import get_retriever_resources

app = FastAPI(title="Egyptian Civil Code Q&A", version="1.0.0")


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    client, _, params = get_retriever_resources()
    collection = params["qdrant"]["collection_name"]
    count = client.get_collection(collection).points_count
    return HealthResponse(status="healthy", documents_indexed=count)


@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest) -> AskResponse:
    try:
        result = await run_in_threadpool(run_rag_pipeline, request.question)
        return AskResponse(answer=result["answer"], sources=result["sources"])
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail="Could not answer the question"
        ) from exc


frontend_dir = Path(__file__).resolve().parents[2] / "frontend"
app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
