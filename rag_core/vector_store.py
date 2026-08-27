from typing import List
import os
import uuid

from annotated_types import doc
import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from .models import Chunk


class VectorStore:
    """Thin wrapper around ChromaDB + BM25 for hybrid search."""

    _embedder = None
    _embedder_model = None

    def __init__(self, collection_name: str = "doc"):
        model_name = os.getenv("RAG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        if VectorStore._embedder is None or VectorStore._embedder_model != model_name:
            VectorStore._embedder = SentenceTransformer(model_name)
            VectorStore._embedder_model = model_name
        self.embedder = VectorStore._embedder

        self.client = chromadb.EphemeralClient()
        try:
            self.client.delete_collection(collection_name)
        except Exception:
            pass
        self.collection = self.client.create_collection(collection_name)

        # BM25 state — built after add_chunks
        self.bm25 = None
        self.bm25_corpus = []      # raw texts in insertion order
        self.bm25_metadata = []    # page_number per text

    def add_chunks(self, chunks: List[Chunk]):
        if not chunks:
            return

        normalized = []
        for chunk in chunks:
            text = ""
            page_number = 1
            chunk_id = str(uuid.uuid4())

            if hasattr(chunk, "text"):
                text = (chunk.text or "").strip()
                page_number = getattr(chunk, "page_number", 1)
                chunk_id = str(getattr(chunk, "id", chunk_id))
            elif hasattr(chunk, "page_content"):
                text = (chunk.page_content or "").strip()
                metadata = getattr(chunk, "metadata", {}) or {}
                page_number = metadata.get("page", metadata.get("page_number", 1))
                chunk_id = str(metadata.get("id", chunk_id))
            elif isinstance(chunk, dict):
                text = (chunk.get("text") or chunk.get("page_content") or "").strip()
                page_number = chunk.get("page_number", chunk.get("page", 1))
                chunk_id = str(chunk.get("id", chunk_id))

            if text:
                normalized.append((chunk_id, text, int(page_number)))

        if not normalized:
            return

        texts = [text for _, text, _ in normalized]
        embeddings = self.embedder.encode(
            texts,
            normalize_embeddings=True,
            batch_size=64,
            show_progress_bar=False,
        ).tolist()

        self.collection.add(
            ids=[chunk_id for chunk_id, _, _ in normalized],
            embeddings=embeddings,
            documents=texts,
            metadatas=[{"page_number": page_number} for _, _, page_number in normalized],
        )

        # build BM25 index over all inserted texts
        self.bm25_corpus = texts
        self.bm25_metadata = [page_number for _, _, page_number in normalized]
        tokenized = [text.lower().split() for text in texts]
        self.bm25 = BM25Okapi(tokenized)

    def query(self, question: str, top_k: int = 5):
        """Semantic-only search (kept for backward compatibility)."""
        query_embedding = self.embedder.encode([question], normalize_embeddings=True).tolist()
        results = self.collection.query(query_embeddings=query_embedding, n_results=top_k)
        return [
            {"text": doc, "page_number": meta["page_number"]}
            for doc, meta in zip(results["documents"][0], results["metadatas"][0])
        ]
    def hybrid_query(self, question: str, top_k: int = 5, semantic_weight: float = 0.6):
        fetch_k = max(top_k * 3, 20)

        # ── Semantic results ──────────────────────────────────────────
        query_embedding = self.embedder.encode([question], normalize_embeddings=True).tolist()
        sem_results = self.collection.query(
        query_embeddings=query_embedding,
        n_results=min(fetch_k, self.collection.count()),
        )
        sem_docs = sem_results["documents"][0]
        sem_metas = sem_results["metadatas"][0]
        sem_ranked = {doc: rank for rank, doc in enumerate(sem_docs)}

        # ── BM25 results ──────────────────────────────────────────────
        bm25_ranked = {}
        if self.bm25 is not None:
            tokens = question.lower().split()
            scores = self.bm25.get_scores(tokens)
            bm25_order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            for rank, idx in enumerate(bm25_order[:fetch_k]):
                bm25_ranked[self.bm25_corpus[idx]] = rank

        # ── RRF Fusion ────────────────────────────────────────────────
        k_rrf = 60
        all_docs = set(sem_ranked.keys()) | set(bm25_ranked.keys())

        def rrf_score(doc):
            sem_rank = sem_ranked.get(doc, fetch_k)
            bm25_rank = bm25_ranked.get(doc, fetch_k)
            sem_score = semantic_weight / (k_rrf + sem_rank)
            bm25_score = (1 - semantic_weight) / (k_rrf + bm25_rank)
            return sem_score + bm25_score

        ranked_docs = sorted(all_docs, key=rrf_score, reverse=True)[:top_k]

        # ── tag retrieval source ──────────────────────────────────────
        text_to_page = {}
        for doc, meta in zip(sem_docs, sem_metas):
            text_to_page[doc] = meta["page_number"]
        for idx, text in enumerate(self.bm25_corpus):
            if text not in text_to_page:
                text_to_page[text] = self.bm25_metadata[idx]

        results = []
        for doc in ranked_docs:
            in_semantic = doc in sem_ranked
            in_keyword  = doc in bm25_ranked

            # tag based on which search found it
            if in_semantic and in_keyword:
                source = "hybrid"
            elif in_semantic:
                source = "semantic"
            else:
                source = "keyword"

            results.append({
                "text": doc,
                "page_number": text_to_page.get(doc, 1),
                "retrieval_source": source,               # ← new field
                "semantic_rank": sem_ranked.get(doc, -1), # ← rank in semantic results
                "keyword_rank": bm25_ranked.get(doc, -1), # ← rank in BM25 results
                "rrf_score": rrf_score(doc),              # ← final fusion score
            })

        return results