import os
import tempfile
from typing import List
from .injection_guard import sanitize_text
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

def _get_page_num_from_item(item) -> int:
    """Extract page number from a Docling document item via prov metadata."""
    try:
        # Docling stores provenance as item.prov — list of PageItem
        if hasattr(item, "prov") and item.prov:
            prov = item.prov[0] if isinstance(item.prov, list) else item.prov
            if hasattr(prov, "page_no"):
                return int(prov.page_no)
            if hasattr(prov, "page"):
                return int(prov.page)
            if hasattr(prov, "page_number"):
                return int(prov.page_number)
        # fallback — check origin
        if hasattr(item, "origin") and item.origin:
            origin = item.origin
            for attr in ("page_no", "page", "page_number"):
                if hasattr(origin, attr):
                    return int(getattr(origin, attr))
    except Exception:
        pass
    return 1  # default


def _extract_pages_with_docling(file_path: str) -> List[PageText]:
    """Extract per-page text from PDF using Docling item provenance."""
    from docling.document_converter import DocumentConverter

    global _DOCLING_CONVERTER
    if _DOCLING_CONVERTER is None:
        _DOCLING_CONVERTER = DocumentConverter()

    result = _DOCLING_CONVERTER.convert(file_path)
    document = getattr(result, "document", result)

    print(f"[DEBUG-Extract] Docling converting document...")

    page_map = {}  

    items_tried = 0
    items_with_page = 0

    item_sources = []

    if hasattr(document, "iterate_items"):
        try:
            item_sources = list(document.iterate_items())
            print(f"[DEBUG-Extract] iterate_items() returned {len(item_sources)} items")
        except Exception as e:
            print(f"[DEBUG-Extract] iterate_items() failed: {e}")

    if not item_sources and hasattr(document, "texts"):
        item_sources = list(document.texts or [])
        print(f"[DEBUG-Extract] Using document.texts: {len(item_sources)} items")

    if not item_sources and hasattr(document, "body") and document.body:
        item_sources = list(document.body)
        print(f"[DEBUG-Extract] Using document.body: {len(item_sources)} items")

    for item in item_sources:
        
        if isinstance(item, tuple):
            item = item[0]

        items_tried += 1
        page_num = _get_page_num_from_item(item)
        if page_num != 1:
            items_with_page += 1

        
        text = ""
        try:
            if hasattr(item, "export_to_markdown"):
                text = item.export_to_markdown() or ""
            elif hasattr(item, "text"):
                text = item.text or ""
            elif hasattr(item, "orig"):
                text = item.orig or ""
        except Exception:
            pass

        text = text.strip()
        if text:
            if page_num not in page_map:
                page_map[page_num] = []
            page_map[page_num].append(text)

    print(f"[DEBUG-Extract] Items tried: {items_tried}, with page info: {items_with_page}")
    print(f"[DEBUG-Extract] Pages found via items: {sorted(page_map.keys())}")

    if len(page_map) > 1:
        pages = []
        for page_num in sorted(page_map.keys()):
            combined = "\n\n".join(page_map[page_num])
            pages.append(PageText(page_number=page_num, text=combined))
            print(f"[DEBUG-Extract] Page {page_num}: {len(combined)} chars")
        print(f"[DEBUG-Extract] ✓ Docling extracted {len(pages)} pages via items")
        return pages

    print(f"[DEBUG-Extract] Trying full markdown with page marker splitting...")
    full_text = ""
    try:
        if hasattr(document, "export_to_markdown"):
            full_text = document.export_to_markdown() or ""
    except Exception:
        pass

    if full_text:
        pages = _split_markdown_by_page_markers(full_text)
        if len(pages) > 1:
            print(f"[DEBUG-Extract] ✓ Split markdown into {len(pages)} pages via markers")
            return pages

    if hasattr(document, "pages") and document.pages:
        print(f"[DEBUG-Extract] Trying document.pages export: {len(document.pages)} pages")
        pages = []
        for idx, page in enumerate(document.pages, start=1):
            text = ""

            for method in ("export_to_markdown", "export_to_text", "get_text"):
                if hasattr(page, method):
                    try:
                        text = getattr(page, method)() or ""
                        if text.strip():
                            break
                    except Exception:
                        pass
            if not text and hasattr(page, "text"):
                text = page.text or ""
            text = text.strip()
            if text:
                pages.append(PageText(page_number=idx, text=text))
                print(f"[DEBUG-Extract] Page {idx}: {len(text)} chars")

        if pages:
            print(f"[DEBUG-Extract] ✓ Docling extracted {len(pages)} pages via .pages")
            return pages

    if full_text:
        print(f"[DEBUG-Extract] Fallback: splitting full text into virtual pages...")
        pages = _split_into_virtual_pages(full_text)
        print(f"[DEBUG-Extract] ✓ Created {len(pages)} virtual pages from full text")
        return pages

    print(f"[DEBUG-Extract] ✗ No text extracted")
    return []


def _split_markdown_by_page_markers(text: str) -> List[PageText]:
    """Split markdown text by page break markers if present."""
    import re
    patterns = [
        r'\f',                         
        r'<!-- pagebreak -->',
        r'<!-- page-break -->',
        r'\n---\n',                     
    ]
    for pattern in patterns:
        parts = re.split(pattern, text, flags=re.IGNORECASE)
        if len(parts) > 1:
            pages = []
            for i, part in enumerate(parts, start=1):
                part = part.strip()
                if part:
                    pages.append(PageText(page_number=i, text=part))
            return pages
    return [PageText(page_number=1, text=text.strip())]


def _split_into_virtual_pages(text: str, chars_per_page: int = 3000) -> List[PageText]:
    """
    Last resort — split large text into virtual pages by character count,
    breaking only at paragraph boundaries.
    """
    paragraphs = text.split("\n\n")
    pages = []
    current = []
    current_len = 0
    page_num = 1

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if current_len + len(para) > chars_per_page and current:
            pages.append(PageText(
                page_number=page_num,
                text="\n\n".join(current)
            ))
            page_num += 1
            current = [para]
            current_len = len(para)
        else:
            current.append(para)
            current_len += len(para)

    if current:
        pages.append(PageText(
            page_number=page_num,
            text="\n\n".join(current)
        ))

    return pages if pages else [PageText(page_number=1, text=text.strip())]


def _extract_pages_with_pypdf(file_path: str) -> List[PageText]:
    """Extract pages using PyPDF2 - reliable for actual PDF page numbers."""
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(file_path)
        pages: List[PageText] = []

        for page_num, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text()
                if text and text.strip():
                    pages.append(PageText(page_number=page_num, text=text.strip()))
            except Exception as e:
                print(f"[DEBUG-Extract] PyPDF2 Page {page_num} error: {e}")

        if pages:
            print(f"[DEBUG-Extract] ✓ PyPDF2 extracted {len(pages)} pages")
        return pages

    except ImportError:
        print(f"[DEBUG-Extract] PyPDF2 not installed - install with: pip install PyPDF2")
        return []
    except Exception as e:
        print(f"[DEBUG-Extract] PyPDF2 failed: {e}")
        return []


def extract_text_from_pdf(file_path_or_bytes) -> List[PageText]:
    """Extract text from PDF. Priority: PyPDF2 → Docling → virtual pages."""
    file_path = None
    temp_file_created = False

    if isinstance(file_path_or_bytes, (str, os.PathLike)):
        file_path = str(file_path_or_bytes)
    else:
        print(f"[DEBUG-Extract] Input is bytes, creating temp file...")
        pdf_bytes = _read_file_bytes(file_path_or_bytes)
        print(f"[DEBUG-Extract] Read {len(pdf_bytes)} bytes")
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_bytes)
            file_path = tmp.name
            temp_file_created = True
            print(f"[DEBUG-Extract] Temp file created: {file_path}")

    try:
        print(f"[DEBUG-Extract] Attempting PyPDF2 extraction...")
        pages = _extract_pages_with_pypdf(file_path)
        if pages:
            return pages

        print(f"[DEBUG-Extract] PyPDF2 returned no pages, trying Docling...")
        pages = _extract_pages_with_docling(file_path)
        if pages:
            return pages

        print(f"[DEBUG-Extract] ✗ FAILED: Both extractors returned no pages")
        return []

    finally:
        if temp_file_created:
            try:
                os.remove(file_path)
                print(f"[DEBUG-Extract] Temp file cleaned up")
            except OSError as e:
                print(f"[DEBUG-Extract] Failed to delete temp file: {e}")


# ── DOCX via python-docx ─────────────────────────────────────────────

def _extract_pages_from_docx(file_path: str) -> List[PageText]:
    """Extract text and tables from a DOCX file, grouped into virtual pages."""
    from docx import Document as DocxDocument

    doc = DocxDocument(file_path)
    pages: List[PageText] = []
    current_lines: List[str] = []
    page_number = 1
    para_count = 0
    paras_per_page = 30

    def flush_page():
        nonlocal page_number, para_count
        text = "\n".join(current_lines).strip()
        if text:
            pages.append(PageText(page_number=page_number, text=text))
        current_lines.clear()
        page_number += 1
        para_count = 0

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

    for table in doc.tables:
        rows = []
        for i, row in enumerate(table.rows):
            cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
            rows.append("| " + " | ".join(cells) + " |")
            if i == 0:
                rows.append("| " + " | ".join(["---"] * len(cells)) + " |")
        if rows:
            current_lines.append("\n" + "\n".join(rows) + "\n")
            para_count += 1
            if para_count >= paras_per_page:
                flush_page()

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
    """Auto-detect file type, extract text, and sanitize against prompt injections."""
    ext = ""
    if isinstance(file_path_or_bytes, (str, os.PathLike)):
        ext = os.path.splitext(str(file_path_or_bytes))[1].lower()
    elif filename:
        ext = os.path.splitext(filename)[1].lower()

    if ext == ".docx":
        raw_pages = extract_text_from_docx(file_path_or_bytes)
    else:
        raw_pages = extract_text_from_pdf(file_path_or_bytes)

    # Sanitize all extracted pages in one pass before returning
    sanitized_pages = []
    for p in raw_pages:
        clean_text, redactions = sanitize_text(p.text)
        if redactions:
            print(f"[Security] Page {p.page_number}: removed {len(redactions)} injection patterns")
        sanitized_pages.append(PageText(page_number=p.page_number, text=clean_text))

    return sanitized_pages