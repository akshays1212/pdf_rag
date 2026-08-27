from typing import List, Dict
from sentence_transformers import CrossEncoder

_RERANKER = None
_RERANKER_MODEL = None


def get_reranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> CrossEncoder:
    """Load reranker once and cache it."""
    global _RERANKER, _RERANKER_MODEL
    if _RERANKER is None or _RERANKER_MODEL != model_name:
        _RERANKER = CrossEncoder(model_name)
        _RERANKER_MODEL = model_name
    return _RERANKER


def rerank(
    question: str,
    chunks: List[Dict],
    top_k: int = 5,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> List[Dict]:
    """
    Score each chunk against the question using a cross-encoder.
    Returns top_k chunks sorted by relevance score descending.

    chunks: list of {"text": ..., "page_number": ...}
    """
    if not chunks:
        return []

    reranker = get_reranker(model_name)

    # cross-encoder takes (query, passage) pairs
    pairs = [(question, chunk["text"]) for chunk in chunks]
    scores = reranker.predict(pairs)

    # attach score to each chunk
    scored = [
        {**chunk, "rerank_score": float(score)}
        for chunk, score in zip(chunks, scores)
    ]

    # sort by score descending, return top_k
    scored.sort(key=lambda x: x["rerank_score"], reverse=True)
    return scored[:top_k]