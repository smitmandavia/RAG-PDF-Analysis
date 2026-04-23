"""
PyRAG — Smart Text Chunker
Recursive splitting strategy that respects document structure.
"""

import re
from typing import NamedTuple
from config import CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_SIZE

COMMON_HEADINGS = {
    "summary", "professional summary", "experience", "professional experience",
    "work experience", "education", "skills", "technical skills", "projects",
    "certifications", "achievements", "contact", "publications", "awards",
    "technical stack", "timeline", "overview", "background", "conclusion"
}


class Chunk(NamedTuple):
    text: str
    page_number: int
    chunk_index: int
    token_count: int
    section_title: str


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
        current_section = "Overview"
        for para in paragraphs:
            if para.strip():
                is_heading = _looks_like_heading(para)
                if is_heading:
                    current_section = _normalize_heading(para)
                segments.append({
                    'text': para.strip(),
                    'page': page.page_number,
                    'section': current_section,
                    'is_heading': is_heading
                })

    # Group segments into chunks of target size
    chunks = []
    current_text = ""
    current_page = segments[0]['page'] if segments else 1
    current_section = segments[0]['section'] if segments else "Overview"
    chunk_index = 0

    for seg in segments:
        if (
            seg.get('is_heading')
            and current_text
            and estimate_tokens(current_text) >= max(MIN_CHUNK_SIZE // 2, 20)
        ):
            chunks.append(Chunk(
                text=current_text.strip(),
                page_number=current_page,
                chunk_index=chunk_index,
                token_count=estimate_tokens(current_text),
                section_title=current_section
            ))
            chunk_index += 1
            current_text = ""
            current_page = seg['page']
            current_section = seg['section']

        had_current_text = bool(current_text)
        candidate = (current_text + "\n\n" + seg['text']).strip() if current_text else seg['text']
        candidate_tokens = estimate_tokens(candidate)

        if candidate_tokens > CHUNK_SIZE and current_text:
            # Current chunk is full — save it
            if estimate_tokens(current_text) >= MIN_CHUNK_SIZE:
                chunks.append(Chunk(
                    text=current_text.strip(),
                    page_number=current_page,
                    chunk_index=chunk_index,
                    token_count=estimate_tokens(current_text),
                    section_title=current_section
                ))
                chunk_index += 1

            # Start new chunk with overlap
            overlap_text = _get_overlap(current_text, CHUNK_OVERLAP)
            current_text = (overlap_text + "\n\n" + seg['text']).strip() if overlap_text else seg['text']
            current_page = seg['page']
            current_section = seg['section']
        else:
            current_text = candidate
            if not had_current_text:
                current_page = seg['page']
                current_section = seg['section']

    # Don't forget the last chunk
    if current_text.strip() and estimate_tokens(current_text) >= MIN_CHUNK_SIZE:
        chunks.append(Chunk(
            text=current_text.strip(),
            page_number=current_page,
            chunk_index=chunk_index,
            token_count=estimate_tokens(current_text),
            section_title=current_section
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
                    token_count=estimate_tokens(sc),
                    section_title=chunk.section_title
                ))
        else:
            final_chunks.append(Chunk(
                text=chunk.text,
                page_number=chunk.page_number,
                chunk_index=len(final_chunks),
                token_count=chunk.token_count,
                section_title=chunk.section_title
            ))

    return final_chunks


def _split_paragraphs(text: str) -> list[str]:
    """Split text by paragraph boundaries."""
    # Try double newlines first
    paragraphs = re.split(r'\n\s*\n', text)
    if len(paragraphs) <= 1:
        paragraphs = text.split('\n')

    results = []
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        results.extend(_split_heading_prefix(paragraph))
    return results


def _looks_like_heading(text: str) -> bool:
    first_line = text.strip().splitlines()[0].strip()
    if not first_line or len(first_line) > 90:
        return False
    if first_line.startswith(("-", "*", "#")):
        return False

    words = re.findall(r"[A-Za-z][A-Za-z0-9&/+-]*", first_line)
    if not 1 <= len(words) <= 9:
        return False

    normalized = _normalize_heading(first_line).lower()
    if normalized in COMMON_HEADINGS:
        return True

    letters = [char for char in first_line if char.isalpha()]
    if letters and sum(1 for char in letters if char.isupper()) / len(letters) >= 0.72:
        return True

    return first_line.endswith(":") and len(words) <= 6


def _normalize_heading(text: str) -> str:
    heading = re.sub(r"\s+", " ", text.strip().strip(":")).strip()
    return heading[:80] or "Overview"


def _split_heading_prefix(text: str) -> list[str]:
    normalized = " ".join(text.split())
    lowered = normalized.lower()
    for heading in sorted(COMMON_HEADINGS, key=len, reverse=True):
        if lowered.startswith(f"{heading} "):
            rest = normalized[len(heading):].strip(" :-")
            if len(rest) >= 25:
                return [_normalize_heading(heading), rest]
    return [normalized]


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
