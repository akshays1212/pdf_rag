from typing import List, Dict, Tuple

# > 0  clearly relevant
# -2 to 0  borderline
# < -2  likely not relevant

DEFAULT_THRESHOLD = -2.0


def check_relevance(
    reranked_chunks: List[Dict],
    threshold: float = DEFAULT_THRESHOLD,
) -> Tuple[bool, float]:
    """
    Check if any reranked chunk passes the relevance threshold.

    Returns:
        (is_relevant, best_score)
        is_relevant: True if at least one chunk passes threshold
        best_score: highest rerank score among all chunks
    """
    if not reranked_chunks:
        return False, float("-inf")

    best_score = reranked_chunks[0].get("rerank_score", float("-inf"))

    return best_score >= threshold, best_score