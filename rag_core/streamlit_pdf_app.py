"""Alternative Streamlit entry point for the modular PDF RAG app.

Run with:
    streamlit run streamlit_pdf_app.py
"""

import re

import streamlit as st

from rag_core import build_store_from_pdf, generate_answer


def sanitize_collection_name(filename: str) -> str:
    base_name = filename.rsplit(".", 1)[0] if "." in filename else filename
    cleaned = re.sub(r"[^a-zA-Z0-9._-]", "_", base_name).strip("._-")
    if len(cleaned) < 3:
        cleaned = f"pdf_{cleaned or 'doc'}"
    return cleaned


st.set_page_config(page_title="Ask My PDF", page_icon="📄", layout="centered")
st.title("📄 Ask My PDF")
st.caption("Upload a PDF first, then ask questions grounded in that document.")

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "processed_filename" not in st.session_state:
    st.session_state.processed_filename = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None and uploaded_file.name != st.session_state.processed_filename:
    collection_name = sanitize_collection_name(uploaded_file.name)
    st.info("Processing PDF... this can take a moment on the first upload while the embedding model loads.")
    try:
        with st.spinner("Reading and indexing the document..."):
            store, pages, chunks = build_store_from_pdf(
                uploaded_file,
                collection_name=collection_name,
            )

        if pages and chunks:
            st.session_state.vector_store = store
            st.session_state.processed_filename = uploaded_file.name
            st.session_state.chat_history = []
            st.success(f"Indexed {len(pages)} pages into {len(chunks)} chunks.")
        else:
            st.warning("No readable content was found in this PDF. Please try a different file.")
    except Exception as exc:
        st.error("The PDF could not be processed by Docling. Please try a different file.")
        st.code(str(exc))

if st.session_state.vector_store is None:
    st.info("Upload a PDF to begin.")
else:
    for question, answer, sources in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            st.write(answer)

    question = st.chat_input("Ask a question about this PDF...")
    if question:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching and generating answer..."):
                retrieved = st.session_state.vector_store.query(question, top_k=5)
                answer = generate_answer(question, retrieved)
            st.write(answer)
        st.session_state.chat_history.append((question, answer, retrieved))
