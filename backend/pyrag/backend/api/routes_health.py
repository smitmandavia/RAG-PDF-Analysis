"""
PyRAG — Health Check Route
"""

from fastapi import APIRouter
from models.schemas import HealthResponse
from db.database import get_documents_count, get_total_chunks
from config import EMBEDDING_MODEL, LLM_PROVIDER

router = APIRouter(tags=["health"])


@router.get("/api/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        documents_count=get_documents_count(),
        total_chunks=get_total_chunks(),
        embedding_model=EMBEDDING_MODEL,
        llm_provider=LLM_PROVIDER
    )
