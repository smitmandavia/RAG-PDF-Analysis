"""
PyRAG — Document API Routes
"""

import os
import tempfile
from fastapi import APIRouter, UploadFile, File, HTTPException
from models.schemas import DocumentInfo, DocumentListResponse, ChunkPreview, ChunkListResponse
from services.ingestion import ingest_document
from services.vector_store import delete_document_chunks, get_document_chunks
from services.chunker import estimate_tokens
from db.database import get_all_documents, get_document, delete_document
from config import MAX_UPLOAD_SIZE, STORAGE_DIR

router = APIRouter(prefix="/api/documents", tags=["documents"])


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

            results.append(DocumentInfo(
                id=doc['id'],
                filename=doc['filename'],
                page_count=doc['page_count'],
                chunk_count=doc['chunk_count'],
                file_size=doc['file_size'],
                uploaded_at=doc['uploaded_at'],
                status=doc['status']
            ))
        except Exception as e:
            raise HTTPException(500, f"Failed to process {file.filename}: {str(e)}")
        finally:
            os.unlink(tmp_path)

    return results


@router.get("", response_model=DocumentListResponse)
async def list_documents():
    """List all uploaded documents."""
    docs = get_all_documents()
    items = [
        DocumentInfo(
            id=d['id'],
            filename=d['filename'],
            page_count=d['page_count'],
            chunk_count=d['chunk_count'],
            file_size=d['file_size'],
            uploaded_at=d['uploaded_at'],
            status=d['status']
        )
        for d in docs
    ]
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
                token_count=c['metadata']['token_count']
            )
            for c in chunks
        ],
        total=len(chunks)
    )
