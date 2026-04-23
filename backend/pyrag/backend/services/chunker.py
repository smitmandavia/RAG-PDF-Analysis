"""
PyRAG — Smart Text Chunker
Recursive splitting strategy that respects document structure.
"""

import re
from typing import NamedTuple
from config import CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_SIZE


class Chunk(NamedTuple):
    text: str
    page_number: int
    chunk_index: int
    token_count: int


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English."""
    return len(text) // 4


def chunk_pages(pages: list, doc_id: str, doc_name: str) -> list[Chunk]:
    """
    Take page-level text and produce overlapping chunks.

    Strategy:
    1. Concatenate all pages but track page boundaries
    2. Split by sections/headings first
    3. Then by paragraphs
    4. Then by sentences
    5. Merge small chunks, split large ones
    6. Add overlap between consecutive chunks
    """
    # Build a unified text with page markers
    segments = []
    for page in pages:
        paragraphs = _split_paragraphs(page.text)
        for para in paragraphs:
            if para.strip():
                segments.append({
                    'text': para.strip(),
                    'page': page.page_number
                })

    # Group segments into chunks of target size
    chunks = []
    current_text = ""
    current_page = segments[0]['page'] if segments else 1
    chunk_index = 0

    for seg in segments:
        candidate = (current_text + "\n\n" + seg['text']).strip() if current_text else seg['text']
        candidate_tokens = estimate_tokens(candidate)

        if candidate_tokens > CHUNK_SIZE and current_text:
            # Current chunk is full — save it
            if estimate_tokens(current_text) >= MIN_CHUNK_SIZE:
                chunks.append(Chunk(
                    text=current_text.strip(),
                    page_number=current_page,
                    chunk_index=chunk_index,
                    token_count=estimate_tokens(current_text)
                ))
                chunk_index += 1

            # Start new chunk with overlap
            overlap_text = _get_overlap(current_text, CHUNK_OVERLAP)
            current_text = (overlap_text + "\n\n" + seg['text']).strip() if overlap_text else seg['text']
            current_page = seg['page']
        else:
            current_text = candidate
            if not chunks:
                current_page = seg['page']

    # Don't forget the last chunk
    if current_text.strip() and estimate_tokens(current_text) >= MIN_CHUNK_SIZE:
        chunks.append(Chunk(
            text=current_text.strip(),
            page_number=current_page,
            chunk_index=chunk_index,
            token_count=estimate_tokens(current_text)
        ))

    # Handle edge case: if a single segment is too large, split it further
    final_chunks = []
    for chunk in chunks:
        if chunk.token_count > CHUNK_SIZE * 2:
            sub_chunks = _split_large_chunk(chunk, CHUNK_SIZE)
            for i, sc in enumerate(sub_chunks):
                final_chunks.append(Chunk(
                    text=sc,
                    page_number=chunk.page_number,
                    chunk_index=len(final_chunks),
                    token_count=estimate_tokens(sc)
                ))
        else:
            final_chunks.append(Chunk(
                text=chunk.text,
                page_number=chunk.page_number,
                chunk_index=len(final_chunks),
                token_count=chunk.token_count
            ))

    return final_chunks


def _split_paragraphs(text: str) -> list[str]:
    """Split text by paragraph boundaries."""
    # Try double newlines first
    paragraphs = re.split(r'\n\s*\n', text)
    if len(paragraphs) > 1:
        return [p.strip() for p in paragraphs if p.strip()]

    # Fall back to single newlines for densely formatted text
    return [p.strip() for p in text.split('\n') if p.strip()]


def _get_overlap(text: str, overlap_tokens: int) -> str:
    """Get the last N tokens worth of text for chunk overlap."""
    target_chars = overlap_tokens * 4
    if len(text) <= target_chars:
        return text

    # Try to break at a sentence boundary
    tail = text[-target_chars:]
    sentence_break = tail.find('. ')
    if sentence_break > 0:
        return tail[sentence_break + 2:]

    return tail


def _split_large_chunk(chunk: Chunk, target_size: int) -> list[str]:
    """Split an oversized chunk by sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', chunk.text)
    sub_chunks = []
    current = ""

    for sentence in sentences:
        candidate = (current + " " + sentence).strip() if current else sentence
        if estimate_tokens(candidate) > target_size and current:
            sub_chunks.append(current.strip())
            current = sentence
        else:
            current = candidate

    if current.strip():
        sub_chunks.append(current.strip())

    return sub_chunks
