"""Streamlit entry point for the modular PDF/DOCX RAG app.

Run with:
    streamlit run streamlit_pdf_app.py
"""

import re
import streamlit as st
from rag_core import build_store_from_files, answer_question


def sanitize_collection_name(filename: str) -> str:
    base_name = filename.rsplit(".", 1)[0] if "." in filename else filename
    cleaned = re.sub(r"[^a-zA-Z0-9._-]", "_", base_name).strip("._-")
    if len(cleaned) < 3:
        cleaned = f"doc_{cleaned or 'file'}"
    return cleaned


st.set_page_config(page_title="Ask My PDF", page_icon="📄", layout="centered")
st.title("📄 Ask My Documents")
st.caption("Upload PDF or DOCX files, then ask questions grounded in those documents.")

# ── session state ────────────────────────────────────────────────────
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "processed_filenames" not in st.session_state:
    st.session_state.processed_filenames = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ── file uploader ────────────────────────────────────────────────────
uploaded_files = st.file_uploader(
    "Upload PDF or DOCX files",
    type=["pdf", "docx"],
    accept_multiple_files=True,
)

if uploaded_files:
    current_names = sorted([f.name for f in uploaded_files])

    if current_names != st.session_state.processed_filenames:
        collection_name = sanitize_collection_name(uploaded_files[0].name)
        st.info(f"Processing {len(uploaded_files)} file(s)... this may take a moment.")

        try:
            with st.spinner("Reading and indexing documents..."):
                files = [(f.read(), f.name) for f in uploaded_files]
                store, pages, chunks = build_store_from_files(
                    files,
                    collection_name=collection_name,
                )

            if pages and chunks:
                st.session_state.vector_store = store
                st.session_state.processed_filenames = current_names
                st.session_state.chat_history = []
                st.success(
                    f"✅ Indexed {len(uploaded_files)} file(s) — "
                    f"{len(pages)} pages → {len(chunks)} chunks."
                )
                st.write("BM25 initialized:", store.bm25 is not None)
                st.write("BM25 corpus size:", len(store.bm25_corpus))
            else:
                st.warning("No readable content found. Please try different files.")

        except Exception as exc:
            st.error("One or more files could not be processed. Please try different files.")
            st.code(str(exc))

# ── show which files are loaded ──────────────────────────────────────
if st.session_state.processed_filenames:
    with st.expander("📁 Loaded files"):
        for name in st.session_state.processed_filenames:
            st.markdown(f"- `{name}`")

# ── chat interface ───────────────────────────────────────────────────
if st.session_state.vector_store is None:
    st.info("Upload a PDF or DOCX file to begin.")
else:
    # render chat history
    for entry in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(entry["question"])
        with st.chat_message("assistant"):
            st.markdown(entry["answer"])

            with st.expander("📊 Final Chunks (after reranking)"):
                for i, chunk in enumerate(entry["retrieved"]):
                    source = chunk.get("retrieval_source", "unknown")
                    color = {"semantic": "🔵", "keyword": "🟢", "hybrid": "🟡"}
                    st.markdown(
                        f"{color.get(source, '⚪')} **Chunk {i+1}** | "
                        f"`{source}` | Page {chunk['page_number']} | "
                        f"RRF: `{chunk.get('rrf_score', 0):.4f}` | "
                        f"Rerank: `{chunk.get('rerank_score', 0):.4f}`"
                    )
                    st.caption(chunk["text"][:300] + "...")

            with st.expander("🔍 Candidate Pool (before reranking)"):  # ← inside entry loop ✅
                for i, chunk in enumerate(entry["candidates"]):         # ← uses entry ✅
                    source = chunk.get("retrieval_source", "unknown")
                    color = {"semantic": "🔵", "keyword": "🟢", "hybrid": "🟡"}
                    st.markdown(
                        f"{color.get(source, '⚪')} **Candidate {i+1}** | "
                        f"`{source}` | Page {chunk['page_number']} | "
                        f"Sem rank: `{chunk.get('semantic_rank', 'N/A')}` | "
                        f"KW rank: `{chunk.get('keyword_rank', 'N/A')}`"
                    )
                    st.caption(chunk["text"][:200] + "...")

    # ── chat input ───────────────────────────────────────────────────
    question = st.chat_input("Ask a question about your documents...")
    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching and generating answer..."):
                answer, retrieved, candidates = answer_question(
                    st.session_state.vector_store,
                    question,
                    candidate_k=20,
                    top_k=5,
                )

            st.markdown(answer)

            with st.expander("📊 Final Chunks (after reranking)"):
                for i, chunk in enumerate(retrieved):
                    source = chunk.get("retrieval_source", "unknown")
                    color = {"semantic": "🔵", "keyword": "🟢", "hybrid": "🟡"}
                    st.markdown(
                        f"{color.get(source, '⚪')} **Chunk {i+1}** | "
                        f"`{source}` | Page {chunk['page_number']} | "
                        f"RRF: `{chunk.get('rrf_score', 0):.4f}` | "
                        f"Rerank: `{chunk.get('rerank_score', 0):.4f}`"
                    )
                    st.caption(chunk["text"][:300] + "...")

            # with st.expander("🔍 Candidate Pool (before reranking)"):
            #     for i, chunk in enumerate(candidates):                  # ← uses candidates ✅
            #         source = chunk.get("retrieval_source", "unknown")
            #         color = {"semantic": "🔵", "keyword": "🟢", "hybrid": "🟡"}
            #         st.markdown(
            #             f"{color.get(source, '⚪')} **Candidate {i+1}** | "
            #             f"`{source}` | Page {chunk['page_number']} | "
            #             f"Sem rank: `{chunk.get('semantic_rank', 'N/A')}` | "
            #             f"KW rank: `{chunk.get('keyword_rank', 'N/A')}`"
            #         )
            #         st.caption(chunk["text"][:200] + "...")

        # save to history after chat message block
        st.session_state.chat_history.append({
            "question": question,
            "answer": answer,
            "retrieved": retrieved,
            "candidates": candidates,   # ← saved ✅
        })