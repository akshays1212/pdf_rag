import os
import logging
from typing import List, Dict

from .chunking import chunk_pages
from .extract import extract_text
from .generation import generate_answer
from .vector_store import VectorStore
from .reranker import rerank
from .relevance_gate import check_relevance, DEFAULT_THRESHOLD
from .injection_guard import (
    validate_user_query,
    scan_chunks,
    scan_llm_output,
    detect_injection_regex,  # ← add for regex-only chunk scanning
)

logger = logging.getLogger(__name__)


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
    scored_chunks = [
        {"rerank_score": r["metadata"].get("rerank_score", float("-inf"))}
        for r in reranked
    ]
    is_relevant, best_score = check_relevance(scored_chunks, threshold=threshold)
    return is_relevant, best_score


def _generate(question: str, retrieved, model: str, chat_history: List[Dict] = None):
    chunks = [
        {"text": r["page_content"], "page_number": r["metadata"]["page_number"]}
        for r in retrieved
    ]
    answer = generate_answer(
        question,
        chunks,
        model=model,
        chat_history=chat_history,
    )
    return answer


def _resolve_question(question: str, chat_history: List[Dict] = None) -> str:
    """Rewrite vague follow-up questions into standalone questions for retrieval."""
    if not chat_history:
        return question

    followup_triggers = [
        "what about", "how about", "and ", "what of",
        "same for", "compare with", "versus", "vs ",
        "explain in detail", "explain more", "tell me more",
        "elaborate", "more details", "can you explain",
        "what was", "what were", "previous", "last question",
        "give more", "expand on", "in detail", "more about",
    ]
    q_lower = question.lower().strip()

    is_followup = (
        any(q_lower.startswith(t) for t in followup_triggers)
        or any(q_lower == t.strip() for t in followup_triggers)
        or len(q_lower.split()) <= 4
    )

    if not is_followup:
        return question

    last_q = chat_history[-1]["question"]
    resolved = f"{last_q} {question}"
    logger.debug(f"[History] Resolved follow-up: '{question}' → '{resolved}'")
    return resolved


def answer_question(
    store: VectorStore,
    question: str,
    candidate_k: int = 20,
    top_k: int = 5,
    model: str = "gemini-2.0-flash-lite",
    relevance_threshold: float = DEFAULT_THRESHOLD,
    chat_history: List[Dict] = None,
):
    # ── step 0 — validate user query (FULL ML + regex + synonym) ─────
    # validate_user_query uses complete 4-layer detection:
    # regex → normalize → llama guard → synonym → subtle
    is_safe, reason = validate_user_query(question)
    if not is_safe:
        logger.warning(f"[Security] Query blocked: {reason}")
        return (
            "Your query could not be processed for security reasons.",
            [], []
        )

    # step 1 — resolve follow-up questions using history
    resolved_question = _resolve_question(question, chat_history)

    # step 2 — hybrid retrieval
    candidates = _retrieve(store, resolved_question, candidate_k=candidate_k)

    # ── step 2.5 — scan retrieved chunks (REGEX ONLY — fast) ─────────
    # chunks are already sanitized at upload time by extract.py
    # using full ML here = 20 Llama Guard calls per query = too slow
    # regex-only is sufficient for already-sanitized chunks
    safe_candidates, flagged = scan_chunks(candidates, use_ml=False)  # ← regex only
    if flagged:
        logger.warning(
            f"[Security] {len(flagged)} chunks flagged and removed. "
            f"Pages: {[c.get('page_number') for c in flagged]}"
        )
    # if all chunks flagged fall back to original candidates
    # (avoid empty context causing false Not Found)
    candidates_to_rerank = safe_candidates if safe_candidates else candidates

    # step 3 — rerank
    reranked = _rerank(resolved_question, candidates_to_rerank, top_k=top_k)

    # step 4 — relevance gate
    is_relevant, best_score = _check_relevance(
        resolved_question, reranked, threshold=relevance_threshold
    )

    logger.debug(
        f"[Relevance] question='{resolved_question[:60]}' | "
        f"threshold={relevance_threshold} | "
        f"best_score={best_score:.4f} | "
        f"is_relevant={is_relevant}"
    )

    NOT_FOUND = "Not found in document."

    if not is_relevant:
        logger.info(
            f"[Pipeline] Relevance gate blocked — "
            f"best_score={best_score:.4f} < threshold={relevance_threshold}"
        )
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

    # step 5 — generate with history
    answer = _generate(question, reranked, model, chat_history=chat_history)

    # ── step 5.5 — scan LLM output (REGEX ONLY — fast) ───────────────
    # output scanning uses regex only — ML too slow here
    # catches any injection that slipped through into LLM output
    answer, was_modified = scan_llm_output(answer)
    if was_modified:
        logger.warning("[Security] LLM output was sanitized by injection guard")

    # build final chunks
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

    source_pages = sorted(set(chunk["page_number"] for chunk in final_chunks))
    source_text = (
        f" [Sources: Pages {', '.join(map(str, source_pages))}]"
        if source_pages else ""
    )

    logger.info(
        f"[Pipeline] Answer generated | "
        f"pages={source_pages} | "
        f"chunks_used={len(final_chunks)} | "
        f"was_modified={was_modified}"
    )

    return answer + source_text, final_chunks, candidate_chunks