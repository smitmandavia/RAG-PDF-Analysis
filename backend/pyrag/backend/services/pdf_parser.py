"""
PyRAG — PDF Parser Service
Extracts text from PDFs with page-level tracking.
"""

import fitz  # PyMuPDF
import re
from pathlib import Path
from typing import NamedTuple


class PageText(NamedTuple):
    page_number: int
    text: str


def extract_text_from_pdf(file_path: str) -> list[PageText]:
    """
    Extract text from a PDF file, page by page.
    Returns a list of PageText(page_number, text).
    """
    doc = fitz.open(file_path)
    pages = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text")

        # Clean up the text
        text = _clean_text(text)

        if text.strip():
            pages.append(PageText(page_number=page_num + 1, text=text))

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
