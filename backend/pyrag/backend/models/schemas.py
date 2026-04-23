"""
PyRAG — Pydantic Schemas
"""

from pydantic import BaseModel
from typing import Optional
from datetime import datetime


# ── Document Models ──

class DocumentInfo(BaseModel):
    id: str
    filename: str
    page_count: int
    chunk_count: int
    file_size: int  # bytes
    uploaded_at: str
    status: str  # "processing", "ready", "error"
    title: Optional[str] = None
    summary: Optional[list[str]] = None
    key_terms: Optional[list[str]] = None


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]
    total: int


class DocumentProfileResponse(BaseModel):
    document_id: str
    filename: str
    title: str
    summary: list[str]
    key_terms: list[str]
    sections: list[dict]
    date_mentions: list[str]
    stats: dict


# ── Chat Models ──

class ChatRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5
    document_ids: Optional[list[str]] = None  # Filter to specific docs
    use_history: Optional[bool] = True
    history_limit: Optional[int] = 4


class SourceChunk(BaseModel):
    document_name: str
    document_id: str
    page_number: int
    chunk_text: str
    similarity_score: float
    section_title: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    question: str
    processing_time: float  # seconds


# ── Chunk Preview ──

class ChunkPreview(BaseModel):
    chunk_index: int
    page_number: int
    text: str
    token_count: int
    section_title: Optional[str] = None


class ChunkListResponse(BaseModel):
    document_id: str
    document_name: str
    chunks: list[ChunkPreview]
    total: int


# ── Health ──

class HealthResponse(BaseModel):
    status: str
    documents_count: int
    total_chunks: int
    embedding_model: str
    llm_provider: str
