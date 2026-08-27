# Ask My PDF — local RAG bot

A minimal, fully local RAG (Retrieval-Augmented Generation) app:
upload a PDF, ask questions, get answers grounded in the document with page citations.

- **Embeddings:** BGE-M3 (via `sentence-transformers`, runs on CPU)
- **Vector store:** ChromaDB (in-memory, no server needed)
- **Generator LLM:** Qwen (hosted on Groq's API — fast, free-tier friendly, no local GPU needed)
- **Frontend:** Streamlit

## 1. Get a free Groq API key

1. Sign up at https://console.groq.com (no credit card needed)
2. Create a key at https://console.groq.com/keys
3. Export it in your shell:

```bash
export GROQ_API_KEY=gsk_your_key_here
```

Note: Groq doesn't host a plain "qwen3:8b" — the app defaults to
`qwen/qwen3.6-27b`, the closest current equivalent (newer and larger, still
fast/free on Groq). If you'd rather avoid Qwen's preview status on Groq, swap
the `model` argument in `generate_answer()` (in `rag_pipeline.py`) to
`"openai/gpt-oss-20b"`.

**Trade-off to know:** since this calls a hosted API, your document chunks
leave your machine and go to Groq for each question. Fine for testing or
personal documents — worth reconsidering if you're indexing sensitive
company files. (If that matters to you, the earlier Ollama-based local setup
avoids this entirely — happy to give you a version that lets you toggle
between the two.)

## 2. Set up the Python environment

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

The first time you run the app, `sentence-transformers` will download the
BGE-M3 model (~2.2GB) from Hugging Face — this only happens once, it's cached
after that.

## 3. Run it

```bash
streamlit run app.py
```

This opens the app in your browser (usually http://localhost:8501). Upload a
PDF, wait for it to index, then ask questions in the chat box.

## How it works

```
PDF upload
   │
   ▼
extract_text_from_pdf()   → text + table-aware markdown (Docling)
   │
   ▼
chunk_pages()              → overlapping ~800-char chunks, tagged with page number
   │
   ▼
VectorStore.add_chunks()   → BGE-M3 embeds each chunk → stored in ChromaDB
   │
   ▼
[user asks a question]
   │
   ▼
VectorStore.query()        → embeds the question, finds top-5 most similar chunks
   │
   ▼
generate_answer()          → sends question + chunks to Qwen3 via Ollama
   │
   ▼
Answer shown in chat, with an expandable "source excerpts" panel per message
```

## Known limitations (v1 — PDF only)

- **Extraction now uses Docling.** It generally performs better on both text
   and table-heavy PDFs than plain text extractors and keeps more structure in
   markdown output.
- **Very complex layouts can still degrade.** Multi-column scans, rotated text,
   or low-quality images may reduce extraction quality.
- **No re-ranking step.** Retrieval is single-stage (embed + cosine similarity
  top-5). For larger/noisier documents, add a reranker
  (e.g. `Qwen3-Reranker`) between retrieval and generation for better
  precision.
- **In-memory vector store.** Re-uploading resets the index; nothing persists
  across app restarts. Switch `chromadb.EphemeralClient()` to
  `chromadb.PersistentClient(path="./chroma_db")` if you want persistence.

## Extending to DOCX / Excel later

Keep `app.py` and the chunking/embedding/generation logic untouched — just add
new extraction functions in `rag_pipeline.py`:

- **DOCX:** `python-docx` — iterate `document.paragraphs` and `document.tables`
- **Excel:** `openpyxl` or `pandas` — one chunk per row or per logical table
  block, so numeric lookups stay accurate (don't let a 1000-row sheet become
  one giant chunk)

Then in `app.py`, branch on file extension and route to the right extractor
before calling the same `chunk_pages` → `VectorStore` → `generate_answer` flow.
