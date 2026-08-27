from .chunking import chunk_pages
from .extract import extract_text
from .generation import generate_answer
from .vector_store import VectorStore
from .reranker import rerank  # ← add

import os
from dotenv import load_dotenv
from langsmith import traceable

load_dotenv()


def build_store_from_files(
    files,
    collection_name: str = "doc",
    chunk_size: int = 1200,
    overlap: int = 120,
    max_chunks: int = 1200,
):
    """Extract, chunk, and index multiple PDF/DOCX files into one vector store."""
    if not isinstance(files, list):
        files = [files]

    all_pages = []
    all_chunks = []

    for file_entry in files:
        if isinstance(file_entry, tuple):
            file_data, filename = file_entry
        else:
            file_data = file_entry
            filename = str(file_entry) if isinstance(file_entry, (str, os.PathLike)) else ""

        pages = extract_text(file_data, filename=filename)
        chunks = chunk_pages(pages, chunk_size=chunk_size, overlap=overlap, max_chunks=max_chunks)
        all_pages.extend(pages)
        all_chunks.extend(chunks)

    all_chunks = all_chunks[:max_chunks]

    store = VectorStore(collection_name=collection_name)
    if all_chunks:
        store.add_chunks(all_chunks)

    return store, all_pages, all_chunks


def build_store_from_pdf(file_path_or_bytes, **kwargs):
    return build_store_from_files(file_path_or_bytes, **kwargs)


@traceable(name="hybrid-retrieval", run_type="retriever")
def _retrieve(store: VectorStore, question: str, candidate_k: int = 20):
    results = store.hybrid_query(question, top_k=candidate_k)
    return [
        {
            "page_content": r["text"],
            "metadata": {
                "page_number": r["page_number"],
                "retrieval_source": r["retrieval_source"],  # ← pass through
                "semantic_rank": r["semantic_rank"],
                "keyword_rank": r["keyword_rank"],
                "rrf_score": r["rrf_score"],
            }
        }
        for r in results
    ]


@traceable(name="reranking", run_type="retriever")
def _rerank(question: str, retrieved, top_k: int = 5):
    """Rerank candidates using cross-encoder and return top_k."""
    chunks = [
        {
            "text": r["page_content"],
            "page_number": r["metadata"]["page_number"],
            "retrieval_source": r["metadata"].get("retrieval_source", "unknown"),  # ← preserve
            "semantic_rank": r["metadata"].get("semantic_rank", -1),               # ← preserve
            "keyword_rank": r["metadata"].get("keyword_rank", -1),                 # ← preserve
            "rrf_score": r["metadata"].get("rrf_score", 0.0),                      # ← preserve
        }
        for r in retrieved
    ]
    reranked = rerank(question, chunks, top_k=top_k)

    return [
        {
            "page_content": r["text"],
            "metadata": {
                "page_number": r["page_number"],
                "retrieval_source": r["retrieval_source"],  # ← pass through
                "semantic_rank": r["semantic_rank"],         # ← pass through
                "keyword_rank": r["keyword_rank"],           # ← pass through
                "rrf_score": r["rrf_score"],                 # ← pass through
                "rerank_score": r["rerank_score"],
            }
        }
        for r in reranked
    ]

@traceable(name="answer-generation", run_type="llm")
def _generate(question: str, retrieved, model: str):
    chunks = [
        {"text": r["page_content"], "page_number": r["metadata"]["page_number"]}
        for r in retrieved
    ]
    return generate_answer(question, chunks, model=model)


@traceable(name="hybrid-rag-pipeline", run_type="chain")
def answer_question(
    store: VectorStore,
    question: str,
    candidate_k: int = 20,
    top_k: int = 5,
    model: str = "qwen3:8b",
):
    candidates = _retrieve(store, question, candidate_k=candidate_k)
    reranked = _rerank(question, candidates, top_k=top_k)
    answer = _generate(question, reranked, model)

    final_chunks = [
        {
            "text": r["page_content"],
            "page_number": r["metadata"]["page_number"],
            "retrieval_source": r["metadata"].get("retrieval_source", "unknown"),
            "rrf_score": r["metadata"].get("rrf_score", 0.0),
            "rerank_score": r["metadata"].get("rerank_score", 0.0),
        }
        for r in reranked
    ]

    # ← also return raw candidates so UI can show pre-rerank breakdown
    candidate_chunks = [
        {
            "text": r["page_content"],
            "page_number": r["metadata"]["page_number"],
            "retrieval_source": r["metadata"].get("retrieval_source", "unknown"),
            "semantic_rank": r["metadata"].get("semantic_rank", -1),
            "keyword_rank": r["metadata"].get("keyword_rank", -1),
            "rrf_score": r["metadata"].get("rrf_score", 0.0),
        }
        for r in candidates
    ]

    return answer, final_chunks, candidate_chunks  # ← now returns 3 values