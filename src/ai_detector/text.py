# Local text cleaning, chunking, and document extraction helpers.
from __future__ import annotations

import re
import zipfile
from html import unescape
from pathlib import Path
from xml.etree import ElementTree


WORD_RE = re.compile(r"\b[\w'-]+\b", re.UNICODE)
WHITESPACE_RE = re.compile(r"[ \t\r\f\v]+")
BLANK_LINES_RE = re.compile(r"\n{3,}")
DOCX_TEXT_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t"


class TextExtractionError(RuntimeError):
    pass


# Clean pasted or extracted text into a stable shape before detection.
def clean_text(text: str) -> str:
    # Normalize copied text, document text, and extracted PDF text into one stable shape.
    text = unescape(text or "")
    text = text.replace("\ufeff", "")
    text = text.replace("\u200b", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = WHITESPACE_RE.sub(" ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()


# Count word-like tokens for length gates and chunking.
def count_words(text: str) -> int:
    return len(WORD_RE.findall(text))


# Split long documents into overlapping chunks for detector models.
def chunk_text(text: str, target_words: int = 380, overlap_words: int = 60, max_chunks: int = 24) -> list[str]:
    # Overlap keeps context around chunk boundaries, which helps transformer detectors.
    words = WORD_RE.findall(text)
    if not words:
        return []
    if len(words) <= target_words:
        return [text.strip()]

    chunks: list[str] = []
    step = max(1, target_words - overlap_words)
    for start in range(0, len(words), step):
        end = min(start + target_words, len(words))
        chunk_words = words[start:end]
        if len(chunk_words) < 30 and chunks:
            break
        chunks.append(" ".join(chunk_words))
        if end >= len(words) or len(chunks) >= max_chunks:
            break
    return chunks


# Extract local text from supported document paths.
def load_text_from_path(path: str | Path) -> str:
    # Keep supported formats small and local; no cloud extraction or upload happens here.
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if not file_path.exists():
        raise TextExtractionError(f"File not found: {file_path}")

    if suffix in {".txt", ".md", ".markdown", ".csv", ".tsv", ".json"}:
        return clean_text(file_path.read_text(encoding="utf-8", errors="replace"))
    if suffix == ".docx":
        return clean_text(_read_docx(file_path))
    if suffix == ".pdf":
        return clean_text(_read_pdf(file_path))

    raise TextExtractionError(f"Unsupported file type: {suffix or '(no extension)'}")


# Read text from the main XML document inside a .docx file.
def _read_docx(path: Path) -> str:
    try:
        with zipfile.ZipFile(path) as docx:
            xml = docx.read("word/document.xml")
    except (KeyError, zipfile.BadZipFile) as exc:
        raise TextExtractionError("Could not read this .docx file.") from exc

    root = ElementTree.fromstring(xml)
    parts = [node.text or "" for node in root.iter(DOCX_TEXT_NS)]
    return "\n".join(part for part in parts if part.strip())


# Read text from each page of a PDF when pypdf is installed.
def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise TextExtractionError("PDF support requires installing the optional pypdf dependency.") from exc

    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n\n".join(pages)
