"""
PyRAG — Ingestion Pipeline
Orchestrates: PDF → Text → Chunks → Embeddings → Vector Store
"""

import uuid
import shutil
import traceback
from pathlib import Path

from config import STORAGE_DIR
from services.pdf_parser import extract_text_from_pdf, get_page_count
from services.chunker import chunk_pages
from services.embedder import embed_texts
from services.vector_store import add_chunks
from db.database import insert_document, update_document


def ingest_document(file_path: str, original_filename: str) -> str:
    """
    Full ingestion pipeline for a PDF document.
    Returns the document ID.
    """
    doc_id = uuid.uuid4().hex[:12]
    file_size = Path(file_path).stat().st_size

    # Save to permanent storage
    stored_path = STORAGE_DIR / f"{doc_id}.pdf"
    shutil.copy2(file_path, stored_path)

    # Register in database
    insert_document(doc_id, original_filename, str(stored_path), file_size)

    try:
        print(f"\n  -- Ingesting: {original_filename} --")

        # Step 1: Extract text
        print(f"  [1/4] Extracting text...")
        pages = extract_text_from_pdf(str(stored_path))
        page_count = get_page_count(str(stored_path))
        print(f"        -> {page_count} pages, {len(pages)} with text")

        if not pages:
            raise ValueError("No text could be extracted from this PDF. It might be scanned/image-only.")

        # Step 2: Chunk
        print(f"  [2/4] Chunking text...")
        chunks = chunk_pages(pages, doc_id, original_filename)
        print(f"        -> {len(chunks)} chunks created")

        # Step 3: Embed
        print(f"  [3/4] Generating embeddings...")
        chunk_texts = [c.text for c in chunks]
        embeddings = embed_texts(chunk_texts)
        print(f"        -> {len(embeddings)} embeddings ({len(embeddings[0])}d vectors)")

        # Step 4: Store
        print(f"  [4/4] Storing in vector database...")
        add_chunks(doc_id, original_filename, chunks, embeddings)
        print("        -> Done!")

        # Update database
        update_document(doc_id,
                        page_count=page_count,
                        chunk_count=len(chunks),
                        status="ready")

        print(f"  [ok] {original_filename} ingested successfully")
        print(f"    ID: {doc_id} | Pages: {page_count} | Chunks: {len(chunks)}")

        return doc_id

    except Exception as e:
        error_msg = str(e)
        print(f"  [error] Error ingesting {original_filename}: {error_msg}")
        traceback.print_exc()
        update_document(doc_id, status="error", error_message=error_msg)
        raise
