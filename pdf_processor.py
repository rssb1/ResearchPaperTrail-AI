"""
pdf_processor.py
-----------------
Extracts text from PDFs (page by page, via PyMuPDF) and splits that text
into overlapping chunks that carry paper-title + page-number metadata.

No LLM or embedding calls happen here — this module is pure text handling,
which makes it easy to unit test deterministically (see tests/test_basic.py).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import List

import fitz  # PyMuPDF


class PDFExtractionError(Exception):
    """Raised when a PDF cannot be opened or contains no extractable text."""


@dataclass
class Chunk:
    """A single retrievable unit of text with full provenance metadata."""

    chunk_id: str
    paper_title: str
    page_number: int  # 1-indexed, matches what a human would see in a PDF viewer
    text: str
    embedding: "list[float] | None" = field(default=None, repr=False)


def extract_pages(file_bytes: bytes, filename: str) -> List[tuple[int, str]]:
    """
    Extract text from a PDF, page by page.

    Returns a list of (page_number, text) tuples, 1-indexed.
    Raises PDFExtractionError if the file can't be opened or has no text at all.
    """
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as e:
        raise PDFExtractionError(f"Could not open '{filename}' as a PDF: {e}") from e

    if doc.page_count == 0:
        raise PDFExtractionError(f"'{filename}' has no pages.")

    pages = []
    for i in range(doc.page_count):
        try:
            page_text = doc[i].get_text("text") or ""
        except Exception:
            page_text = ""
        pages.append((i + 1, page_text.strip()))
    doc.close()

    if not any(text for _, text in pages):
        raise PDFExtractionError(
            f"'{filename}' contains no extractable text. "
            "It may be a scanned/image-only PDF (OCR is not supported in this MVP)."
        )

    return pages


def chunk_text(
    text: str,
    chunk_size: int = 900,
    overlap: int = 150,
) -> List[str]:
    """
    Split a single page's text into overlapping character-based chunks.

    Chunking is intentionally scoped to one page at a time (see chunk_pages
    below) so every chunk can be attributed to exactly one page number —
    there is no ambiguity about which page a citation points to.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = end - overlap  # step forward, re-including the overlap window
        if start < 0:
            start = 0
    return chunks


def build_chunks(
    paper_title: str,
    pages: List[tuple[int, str]],
    chunk_size: int = 900,
    overlap: int = 150,
    min_chunk_chars: int = 40,
) -> List[Chunk]:
    """
    Turn a paper's extracted pages into a flat list of Chunk objects,
    each tagged with the paper's title and the page it came from.
    """
    chunks: List[Chunk] = []
    for page_number, page_text in pages:
        for piece in chunk_text(page_text, chunk_size=chunk_size, overlap=overlap):
            if len(piece) < min_chunk_chars:
                continue  # skip near-empty fragments (headers, page numbers, etc.)
            chunks.append(
                Chunk(
                    chunk_id=str(uuid.uuid4()),
                    paper_title=paper_title,
                    page_number=page_number,
                    text=piece,
                )
            )
    return chunks
