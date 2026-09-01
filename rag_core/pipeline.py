from .chunking import chunk_pages
from .extract import extract_text
from .generation import generate_answer
from .vector_store import VectorStore
from .reranker import rerank
from .relevance_gate import check_relevance, DEFAULT_THRESHOLD

import os


def build_store_from_files(
    files,
    collection_name: str = "doc",
    chunk_size: int = 1200,
    overlap: int = 120,
    max_chunks: int = 1200,
):
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


def _retrieve(store: VectorStore, question: str, candidate_k: int = 20):
    results = store.hybrid_query(question, top_k=candidate_k)
    return [
        {
            "page_content": r["text"],
            "metadata": {
                "page_number": r["page_number"],
                "retrieval_source": r["retrieval_source"],
                "semantic_rank": r["semantic_rank"],
                "keyword_rank": r["keyword_rank"],
                "rrf_score": r["rrf_score"],
            }
        }
        for r in results
    ]


def _rerank(question: str, retrieved, top_k: int = 5):
    chunks = [
        {
            "text": r["page_content"],
            "page_number": r["metadata"]["page_number"],
            "retrieval_source": r["metadata"].get("retrieval_source", "unknown"),
            "semantic_rank": r["metadata"].get("semantic_rank", -1),
            "keyword_rank": r["metadata"].get("keyword_rank", -1),
            "rrf_score": r["metadata"].get("rrf_score", 0.0),
        }
        for r in retrieved
    ]
    reranked = rerank(question, chunks, top_k=top_k)

    return [
        {
            "page_content": r["text"],
            "metadata": {
                "page_number": r["page_number"],
                "retrieval_source": r["retrieval_source"],
                "semantic_rank": r["semantic_rank"],
                "keyword_rank": r["keyword_rank"],
                "rrf_score": r["rrf_score"],
                "rerank_score": r["rerank_score"],
            }
        }
        for r in reranked
    ]


def _check_relevance(
    question: str,
    reranked,
    threshold: float = DEFAULT_THRESHOLD,
):
    """Deterministic relevance check — no LLM involved."""
    # build list with rerank_score for gate check
    scored_chunks = [
        {"rerank_score": r["metadata"].get("rerank_score", float("-inf"))}
        for r in reranked
    ]
    is_relevant, best_score = check_relevance(scored_chunks, threshold=threshold)
    return is_relevant, best_score


def _generate(question: str, retrieved, model: str):
    chunks = [
        {"text": r["page_content"], "page_number": r["metadata"]["page_number"]}
        for r in retrieved
    ]
    answer = generate_answer(question, chunks, model=model)
    return answer


def answer_question(
    store: VectorStore,
    question: str,
    candidate_k: int = 20,
    top_k: int = 5,
    model: str = "qwen3:8b",
    relevance_threshold: float = DEFAULT_THRESHOLD,
):
    # step 1 — hybrid retrieval
    candidates = _retrieve(store, question, candidate_k=candidate_k)

    # step 2 — rerank
    reranked = _rerank(question, candidates, top_k=top_k)

    # step 3 — relevance gate (deterministic, no LLM)
    is_relevant, best_score = _check_relevance(question, reranked, threshold=relevance_threshold)

    NOT_FOUND = "Not found in document."

    if not is_relevant:
        # Build candidate_chunks even when not relevant (for display consistency)
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
        return NOT_FOUND, [], candidate_chunks

    # step 4 — generate answer only if relevant
    answer = _generate(question, reranked, model)

    # Build final chunks with all source metadata for deterministic citations
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

    # Debug: print page numbers
    print(f"[DEBUG] Final chunks page numbers: {[c['page_number'] for c in final_chunks]}")
    print(f"[DEBUG] Reranked data: {[(r['metadata'].get('page_number'), r['page_content'][:50]) for r in reranked]}")

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

    # Build deterministic source attribution from actual retrieved chunks
    # Extract unique pages from final_chunks
    source_pages = sorted(set(chunk["page_number"] for chunk in final_chunks))
    print(f"[DEBUG-Pipeline] Final source pages extracted: {source_pages}")
    print(f"[DEBUG-Pipeline] Chunk page numbers detail: {[(i, c['page_number']) for i, c in enumerate(final_chunks)]}")
    source_text = f" [Sources: Pages {', '.join(map(str, source_pages))}]" if source_pages else ""

    # Return answer with backend-generated source metadata
    return answer + source_text, final_chunks, candidate_chunks