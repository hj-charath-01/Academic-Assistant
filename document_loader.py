"""
document_loader.py
──────────────────
Handles loading and text-extraction from uploaded academic files.
Supported formats: PDF, DOCX, TXT, MD

Each loader returns a list of "chunks" — dicts with keys:
    text   : the raw text content
    source : original filename
    page   : page / section number (1-based)
"""

import re
import io
from pathlib import Path
from typing import List, Dict, Any

import PyPDF2
import docx


# ── Typing alias ───────────────────────────────────────────────────────────────
Chunk = Dict[str, Any]   # {"text": str, "source": str, "page": int}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    """Remove excessive whitespace and non-printable characters."""
    text = re.sub(r"[^\x20-\x7E\n]", " ", text)   # strip non-ASCII junk
    text = re.sub(r"\n{3,}", "\n\n", text)          # collapse blank lines
    text = re.sub(r"[ \t]{2,}", " ", text)          # collapse spaces
    return text.strip()


def _split_into_chunks(
    text: str,
    source: str,
    page: int,
    chunk_size: int = 800,
    overlap: int = 100,
) -> List[Chunk]:
    """
    Split a long text string into overlapping chunks for embedding.

    Args:
        text       : raw text from one page / section
        source     : filename
        page       : page or section index
        chunk_size : target character length per chunk
        overlap    : character overlap between consecutive chunks

    Returns:
        List of Chunk dicts
    """
    chunks: List[Chunk] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end].strip()
        if len(chunk_text) > 50:           # skip near-empty chunks
            chunks.append({
                "text":   chunk_text,
                "source": source,
                "page":   page,
            })
        start += chunk_size - overlap      # slide window with overlap
    return chunks


# ── Per-format loaders ─────────────────────────────────────────────────────────

def load_pdf(file_bytes: bytes, filename: str) -> List[Chunk]:
    """Extract text page-by-page from a PDF file."""
    chunks: List[Chunk] = []
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))

    for page_num, page in enumerate(reader.pages, start=1):
        raw  = page.extract_text() or ""
        text = _clean(raw)
        if text:
            chunks.extend(_split_into_chunks(text, filename, page_num))

    return chunks


def load_docx(file_bytes: bytes, filename: str) -> List[Chunk]:
    """Extract text paragraph-by-paragraph from a DOCX file."""
    doc    = docx.Document(io.BytesIO(file_bytes))
    chunks: List[Chunk] = []
    current_section: List[str] = []
    section_num = 1

    for para in doc.paragraphs:
        stripped = para.text.strip()
        if not stripped:
            continue

        # Treat Heading styles as section boundaries
        if para.style.name.startswith("Heading"):
            if current_section:
                combined = _clean(" ".join(current_section))
                chunks.extend(_split_into_chunks(combined, filename, section_num))
                section_num += 1
                current_section = []
            current_section.append(stripped)
        else:
            current_section.append(stripped)

    # Flush remaining paragraphs
    if current_section:
        combined = _clean(" ".join(current_section))
        chunks.extend(_split_into_chunks(combined, filename, section_num))

    return chunks


def load_txt(file_bytes: bytes, filename: str) -> List[Chunk]:
    """Load plain text or Markdown files and split into chunks."""
    text       = file_bytes.decode("utf-8", errors="replace")
    text       = _clean(text)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: List[Chunk] = []
    buffer     = ""
    section_num = 1

    for para in paragraphs:
        buffer += " " + para
        if len(buffer) >= 800:
            chunks.extend(_split_into_chunks(buffer.strip(), filename, section_num))
            buffer = ""
            section_num += 1

    if buffer.strip():
        chunks.extend(_split_into_chunks(buffer.strip(), filename, section_num))

    return chunks


# ── Public API ─────────────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


def load_document(file_bytes: bytes, filename: str) -> List[Chunk]:
    """
    Dispatch to the correct loader based on file extension.

    Args:
        file_bytes : raw bytes of the uploaded file
        filename   : original filename

    Returns:
        List of Chunk dicts ready for embedding

    Raises:
        ValueError : if the file format is not supported
    """
    ext = Path(filename).suffix.lower()

    if ext == ".pdf":
        return load_pdf(file_bytes, filename)
    elif ext == ".docx":
        return load_docx(file_bytes, filename)
    elif ext in {".txt", ".md"}:
        return load_txt(file_bytes, filename)
    else:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )