"""
PyRAG — PDF Parser Service
Extracts text from PDFs with page-level tracking.
"""

import fitz  # PyMuPDF
import re
from pathlib import Path
from typing import NamedTuple


class BlockText(NamedTuple):
    text: str
    bbox: tuple[float, float, float, float]
    block_index: int


class PageText(NamedTuple):
    page_number: int
    text: str
    blocks: list[BlockText]


def extract_text_from_pdf(file_path: str) -> list[PageText]:
    """
    Extract text from a PDF file, page by page.
    Returns a list of PageText(page_number, text).
    """
    doc = fitz.open(file_path)
    pages = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        blocks = _extract_blocks(page)
        text = "\n\n".join(block.text for block in blocks)

        if not text.strip():
            text = _clean_text(page.get_text("text", sort=True))

        if text.strip():
            pages.append(PageText(page_number=page_num + 1, text=text, blocks=blocks))

    doc.close()
    return pages


def get_page_count(file_path: str) -> int:
    """Get the number of pages in a PDF."""
    doc = fitz.open(file_path)
    count = len(doc)
    doc.close()
    return count


def _clean_text(text: str) -> str:
    """Clean extracted text — fix encoding issues, normalize whitespace."""
    # Replace common problematic characters
    text = text.replace('\x00', '')
    text = text.replace('\ufeff', '')

    # Normalize whitespace but preserve paragraph breaks
    lines = text.split('\n')
    cleaned_lines = []

    for line in lines:
        line = line.strip()
        # Skip lines that are just page numbers or headers
        if re.match(r'^\d+$', line):
            continue
        if line:
            cleaned_lines.append(line)

    text = '\n'.join(cleaned_lines)

    # Collapse multiple blank lines into one
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Fix hyphenated line breaks (e.g., "docu-\nment" → "document")
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)

    return text.strip()


def _extract_blocks(page) -> list[BlockText]:
    """Extract text blocks in visual reading order."""
    blocks = []
    for index, block in enumerate(page.get_text("blocks", sort=True)):
        if len(block) < 5:
            continue

        text = _clean_text(block[4])
        if not text:
            continue

        block_type = block[6] if len(block) > 6 else 0
        if block_type != 0:
            continue

        blocks.append(
            BlockText(
                text=text,
                bbox=(float(block[0]), float(block[1]), float(block[2]), float(block[3])),
                block_index=index
            )
        )
    return blocks
