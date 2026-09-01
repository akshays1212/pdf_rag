from .chunking import chunk_pages
from .generation import SYSTEM_PROMPT, build_prompt, generate_answer
from .models import Chunk, PageText
from .vector_store import VectorStore
from .pipeline import build_store_from_files, build_store_from_pdf, answer_question
from .extract import extract_text, extract_text_from_pdf, extract_text_from_docx
from .relevance_gate import check_relevance, DEFAULT_THRESHOLD  # ← add

__all__ = [
    "PageText",
    "Chunk",
    "extract_text",              
    "extract_text_from_pdf",
    "extract_text_from_docx",   
    "chunk_pages",
    "VectorStore",
    "SYSTEM_PROMPT",
    "build_prompt",
    "generate_answer",
    "build_store_from_files",   
    "build_store_from_pdf",
    "answer_question",
    "check_relevance",
    "DEFAULT_THRESHOLD",
]