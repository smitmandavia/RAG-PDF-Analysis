"""
PyRAG — Document API Routes
"""

import os
import tempfile
from fastapi import APIRouter, UploadFile, File, HTTPException
from models.schemas import (
    DocumentInfo,
    DocumentListResponse,
    DocumentProfileResponse,
    ChunkPreview,
    ChunkListResponse,
)
from services.ingestion import ingest_document
from services.vector_store import delete_document_chunks, get_document_chunks
from db.database import get_all_documents, get_document, delete_document, get_document_profile
from config import MAX_UPLOAD_SIZE, STORAGE_DIR

router = APIRouter(prefix="/api/documents", tags=["documents"])


def _to_document_info(doc: dict) -> DocumentInfo:
    profile = get_document_profile(doc["id"]) or {}
    return DocumentInfo(
        id=doc['id'],
        filename=doc['filename'],
        page_count=doc['page_count'],
        chunk_count=doc['chunk_count'],
        file_size=doc['file_size'],
        uploaded_at=doc['uploaded_at'],
        status=doc['status'],
        title=profile.get("title"),
        summary=(profile.get("summary") or [])[:2],
        key_terms=(profile.get("key_terms") or [])[:8]
    )


@router.post("/upload", response_model=list[DocumentInfo])
async def upload_documents(files: list[UploadFile] = File(...)):
    """Upload one or more PDF files for ingestion."""
    results = []

    for file in files:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(400, f"Only PDF files are supported. Got: {file.filename}")

        # Validate file size
        content = await file.read()
        if len(content) > MAX_UPLOAD_SIZE:
            raise HTTPException(400, f"File too large: {file.filename} ({len(content) / 1024 / 1024:.1f} MB). Max: {MAX_UPLOAD_SIZE / 1024 / 1024:.0f} MB")

        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            doc_id = ingest_document(tmp_path, file.filename)
            doc = get_document(doc_id)

            results.append(_to_document_info(doc))
        except Exception as e:
            raise HTTPException(500, f"Failed to process {file.filename}: {str(e)}")
        finally:
            os.unlink(tmp_path)

    return results


@router.get("", response_model=DocumentListResponse)
async def list_documents():
    """List all uploaded documents."""
    docs = get_all_documents()
    items = [_to_document_info(d) for d in docs]
    return DocumentListResponse(documents=items, total=len(items))


@router.delete("/{doc_id}")
async def remove_document(doc_id: str):
    """Delete a document and its chunks."""
    doc = get_document(doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")

    # Remove vectors
    delete_document_chunks(doc_id)

    # Remove file
    file_path = STORAGE_DIR / f"{doc_id}.pdf"
    if file_path.exists():
        file_path.unlink()

    # Remove from database
    delete_document(doc_id)

    return {"message": f"Document '{doc['filename']}' deleted", "id": doc_id}


@router.get("/{doc_id}/profile", response_model=DocumentProfileResponse)
async def document_profile(doc_id: str):
    """Return the document intelligence profile generated during ingestion."""
    doc = get_document(doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")

    profile = get_document_profile(doc_id)
    if not profile:
        raise HTTPException(404, "Document profile not found. Re-upload the document to generate one.")

    return DocumentProfileResponse(**profile)


@router.get("/{doc_id}/chunks", response_model=ChunkListResponse)
async def preview_chunks(doc_id: str):
    """Preview the chunks for a document (useful for debugging)."""
    doc = get_document(doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")

    chunks = get_document_chunks(doc_id)

    return ChunkListResponse(
        document_id=doc_id,
        document_name=doc['filename'],
        chunks=[
            ChunkPreview(
                chunk_index=c['metadata']['chunk_index'],
                page_number=c['metadata']['page_number'],
                text=c['text'],
                token_count=c['metadata']['token_count'],
                section_title=c['metadata'].get('section_title')
            )
            for c in chunks
        ],
        total=len(chunks)
    )
