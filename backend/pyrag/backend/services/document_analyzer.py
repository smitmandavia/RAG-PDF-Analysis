"""
Document profile generation for PyRAG.
Builds deterministic, local metadata that improves retrieval and gives the UI
a useful view into what was understood during ingestion.
"""

import re
from collections import Counter, OrderedDict
from pathlib import Path
from services.chunker import estimate_tokens, _looks_like_heading, _normalize_heading, _split_paragraphs


STOPWORDS = {
    "about", "above", "after", "again", "against", "also", "and", "are",
    "because", "been", "between", "both", "built", "can", "could", "data",
    "did", "does", "each", "from", "had", "has", "have", "into", "its",
    "more", "most", "not", "our", "over", "per", "such", "than", "that",
    "the", "their", "then", "there", "these", "this", "through", "under",
    "using", "was", "were", "when", "where", "which", "while", "with",
    "within", "work", "your"
}

DATE_PATTERNS = [
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}\b",
    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
    r"\b(?:19|20)\d{2}\b",
    r"\bPresent\b",
]


def analyze_document(doc_id: str, filename: str, pages: list, chunks: list) -> dict:
    text = "\n\n".join(page.text for page in pages)
    paragraphs = _paragraphs(text)
    title = _extract_title(filename, paragraphs)
    key_terms = _extract_key_terms(text)
    summary = _extract_summary(paragraphs, key_terms, title)
    sections = _summarize_sections(pages, chunks)
    date_mentions = _extract_dates(text)

    return {
        "document_id": doc_id,
        "filename": filename,
        "title": title,
        "summary": summary,
        "key_terms": key_terms,
        "sections": sections,
        "date_mentions": date_mentions,
        "stats": {
            "pages_with_text": len(pages),
            "chunks": len(chunks),
            "estimated_tokens": estimate_tokens(text),
            "characters": len(text),
        },
    }


def _paragraphs(text: str) -> list[str]:
    return [" ".join(part.split()) for part in _split_paragraphs(text or "")]


def _extract_title(filename: str, paragraphs: list[str]) -> str:
    for paragraph in paragraphs[:5]:
        candidate = paragraph.strip()
        if 3 <= len(candidate) <= 90 and len(candidate.split()) <= 12:
            return candidate
    return Path(filename).stem.replace("_", " ").replace("-", " ").strip().title()


def _extract_key_terms(text: str, limit: int = 14) -> list[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9+#./-]{2,}", text or "")
    normalized = []
    display = {}
    for token in tokens:
        key = token.lower().strip(".-/")
        if key in STOPWORDS or len(key) < 3:
            continue
        normalized.append(key)
        display.setdefault(key, token.strip(".,;:"))

    counts = Counter(normalized)
    return [display[key] for key, _ in counts.most_common(limit)]


def _extract_summary(paragraphs: list[str], key_terms: list[str], title: str) -> list[str]:
    term_set = {term.lower() for term in key_terms[:10]}
    scored = []
    for index, paragraph in enumerate(paragraphs):
        if paragraph == title or _looks_like_heading(paragraph) or len(paragraph) < 45:
            continue
        words = {word.lower().strip(".,;:") for word in paragraph.split()}
        score = len(words & term_set) + min(len(paragraph) / 260, 1.0)
        scored.append((score, index, _clip_sentence(paragraph, 210)))

    top = sorted(scored, key=lambda item: item[0], reverse=True)[:4]
    top.sort(key=lambda item: item[1])
    return [item[2] for item in top] or [_clip_sentence(paragraph, 210) for paragraph in paragraphs[:3]]


def _summarize_sections(pages: list, chunks: list) -> list[dict]:
    sections = OrderedDict()
    current_title = "Overview"

    for page in pages:
        for paragraph in _split_paragraphs(page.text):
            if _looks_like_heading(paragraph):
                current_title = _normalize_heading(paragraph)
                sections.setdefault(
                    current_title,
                    {
                        "title": current_title,
                        "first_page": page.page_number,
                        "chunk_count": 0,
                        "token_count": 0,
                    }
                )
                continue

            current = sections.setdefault(
                current_title,
                {
                    "title": current_title,
                    "first_page": page.page_number,
                    "chunk_count": 0,
                    "token_count": 0,
                }
            )
            current["token_count"] += estimate_tokens(paragraph)
            current["first_page"] = min(current["first_page"], page.page_number)

    for chunk in chunks:
        name = getattr(chunk, "section_title", "Overview") or "Overview"
        if name in sections:
            sections[name]["chunk_count"] += 1

    if sections:
        return list(sections.values())[:20]

    sections = OrderedDict()
    for chunk in chunks:
        name = getattr(chunk, "section_title", "Overview") or "Overview"
        current = sections.setdefault(
            name,
            {
                "title": name,
                "first_page": chunk.page_number,
                "chunk_count": 0,
                "token_count": 0,
            }
        )
        current["chunk_count"] += 1
        current["token_count"] += chunk.token_count
        current["first_page"] = min(current["first_page"], chunk.page_number)

    return list(sections.values())[:20]


def _extract_dates(text: str, limit: int = 16) -> list[str]:
    found = OrderedDict()
    for pattern in DATE_PATTERNS:
        for match in re.finditer(pattern, text or "", flags=re.IGNORECASE):
            value = " ".join(match.group(0).split())
            found.setdefault(value, None)
    return list(found.keys())[:limit]


def _clip_sentence(text: str, max_chars: int) -> str:
    text = " ".join((text or "").split())
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "..."
