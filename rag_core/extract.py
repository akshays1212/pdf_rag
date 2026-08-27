import os
import tempfile
from typing import List

from .models import PageText


_DOCLING_CONVERTER = None


def _read_file_bytes(file_path_or_bytes):
    """Return bytes from a filesystem path, file-like object, or raw bytes."""
    if isinstance(file_path_or_bytes, (bytes, bytearray)):
        return bytes(file_path_or_bytes)
    if isinstance(file_path_or_bytes, (str, os.PathLike)):
        with open(file_path_or_bytes, "rb") as f:
            return f.read()
    if hasattr(file_path_or_bytes, "read"):
        position = None
        try:
            position = file_path_or_bytes.tell()
        except Exception:
            pass
        data = file_path_or_bytes.read()
        if position is not None:
            try:
                file_path_or_bytes.seek(position)
            except Exception:
                pass
        return data
    raise TypeError("Unsupported input type")


# ── PDF via Docling ──────────────────────────────────────────────────

def _extract_pages_with_docling(file_path: str) -> List[PageText]:
    """Extract page text (including table structure) from a PDF using Docling."""
    from docling.document_converter import DocumentConverter

    global _DOCLING_CONVERTER
    if _DOCLING_CONVERTER is None:
        _DOCLING_CONVERTER = DocumentConverter()

    result = _DOCLING_CONVERTER.convert(file_path)
    document = getattr(result, "document", result)

    if hasattr(document, "pages") and document.pages:
        pages: List[PageText] = []
        for idx, page in enumerate(document.pages, start=1):
            text = ""
            if hasattr(page, "export_to_markdown"):
                text = (page.export_to_markdown() or "").strip()
            elif hasattr(page, "text"):
                text = (page.text or "").strip()
            if text:
                pages.append(PageText(page_number=idx, text=text))
        if pages:
            return pages

    # fallback — full document markdown
    text = ""
    if hasattr(document, "export_to_markdown"):
        text = (document.export_to_markdown() or "").strip()
    if not text:
        return []
    return [PageText(page_number=1, text=text)]


def extract_text_from_pdf(file_path_or_bytes) -> List[PageText]:
    """Extract text and tables from a PDF using Docling."""
    if isinstance(file_path_or_bytes, (str, os.PathLike)):
        return _extract_pages_with_docling(str(file_path_or_bytes))

    pdf_bytes = _read_file_bytes(file_path_or_bytes)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name
    try:
        return _extract_pages_with_docling(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


# ── DOCX via python-docx ─────────────────────────────────────────────

def _extract_pages_from_docx(file_path: str) -> List[PageText]:
    """Extract text and tables from a DOCX file, grouped into virtual pages."""
    from docx import Document as DocxDocument

    doc = DocxDocument(file_path)
    pages: List[PageText] = []
    current_lines: List[str] = []
    page_number = 1
    para_count = 0
    paras_per_page = 30  # virtual page size

    def flush_page():
        nonlocal page_number, para_count
        text = "\n".join(current_lines).strip()
        if text:
            pages.append(PageText(page_number=page_number, text=text))
        current_lines.clear()
        page_number += 1
        para_count = 0

    # paragraphs
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        if para.style.name.startswith("Heading"):
            current_lines.append(f"\n## {text}")
        else:
            current_lines.append(text)
        para_count += 1
        if para_count >= paras_per_page:
            flush_page()

    # tables — always kept as whole markdown table blocks
    for table in doc.tables:
        rows = []
        for i, row in enumerate(table.rows):
            cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
            rows.append("| " + " | ".join(cells) + " |")
            if i == 0:
                # add markdown separator after header row
                rows.append("| " + " | ".join(["---"] * len(cells)) + " |")
        if rows:
            current_lines.append("\n" + "\n".join(rows) + "\n")
            para_count += 1
            if para_count >= paras_per_page:
                flush_page()

    # flush remaining content
    flush_page()

    return pages


def extract_text_from_docx(file_path_or_bytes) -> List[PageText]:
    """Extract text and tables from a DOCX file."""
    if isinstance(file_path_or_bytes, (str, os.PathLike)):
        return _extract_pages_from_docx(str(file_path_or_bytes))

    docx_bytes = _read_file_bytes(file_path_or_bytes)
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
        tmp.write(docx_bytes)
        tmp_path = tmp.name
    try:
        return _extract_pages_from_docx(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


# ── Auto-detect ──────────────────────────────────────────────────────

def extract_text(file_path_or_bytes, filename: str = "") -> List[PageText]:
    """
    Auto-detect file type from extension and extract text.
    Pass filename when file_path_or_bytes is bytes (e.g. from Streamlit uploader).
    """
    ext = ""
    if isinstance(file_path_or_bytes, (str, os.PathLike)):
        ext = os.path.splitext(str(file_path_or_bytes))[1].lower()
    elif filename:
        ext = os.path.splitext(filename)[1].lower()

    if ext == ".docx":
        return extract_text_from_docx(file_path_or_bytes)
    else:
        # default to PDF
        return extract_text_from_pdf(file_path_or_bytes)