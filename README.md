# Ask My Documents — RAG System

A production-grade RAG (Retrieval-Augmented Generation) app:

Upload PDF or DOCX files, ask questions, and get answers grounded in the document with page citations and full retrieval transparency.

* **Embeddings:** `all-MiniLM-L6-v2` (via `sentence-transformers`, runs on CPU)
* **Keyword search:** BM25 (via `rank-bm25`, runs locally)
* **Vector store:** ChromaDB (in-memory, no server needed)
* **Reranker:** `ms-marco-MiniLM-L-6-v2` cross-encoder (runs on CPU)
* **Generator LLM:** Gemini 2.0 Flash-Lite via Google Gemini API
* **Document extraction:** PyPDF2 + Docling + python-docx
* **Injection guard:** DeBERTa v2 + Llama Guard 3 1B + Regex + Synonym detection
* **Observability:** Langfuse + LangSmith
* **Frontend:** Streamlit + HTML/JS
* **API:** FastAPI

## 1. Get a Google Gemini API Key

1. Get an API key from [Google AI Studio](https://aistudio.google.com).
2. Set the key as an environment variable.

### Windows PowerShell

```powershell
$env:GOOGLE_API_KEY="your_api_key_here"
```

### Linux / macOS

```bash
export GOOGLE_API_KEY=your_api_key_here
```

You can also place the key in a `.env` file:

```env
# ── LLM ───────────────────────────────────────────────────
GOOGLE_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-2.0-flash-lite

# ── Observability ──────────────────────────────────────────
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com

LANGCHAIN_API_KEY=ls__...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=hybrid-rag

# ── Embeddings ─────────────────────────────────────────────
RAG_EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

Make sure your application loads the `.env` file using `python-dotenv`.

The app uses:

```text
gemini-2.0-flash-lite
```

as the generation model.

> **Privacy note:** `all-MiniLM-L6-v2` embeddings, BM25 search, reranking, and injection detection all run locally. Retrieved document chunks are sent to Google's Gemini API when generating an answer. Avoid uploading sensitive or confidential documents unless this is acceptable for your use case. For fully local deployment, switch generation to Ollama (`qwen3:8b`).

## 2. Install Ollama (for Llama Guard injection detection)

Download from [ollama.com](https://ollama.com) and pull the guard model:

```bash
ollama pull llama-guard3:1b
```

Llama Guard acts as layer 4 of the injection guard. The system degrades gracefully if Ollama is unavailable — layers 1–3 remain active.

## 3. Set up the Python environment

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

The first time you run the application, the following models are downloaded from Hugging Face and cached locally:

* `all-MiniLM-L6-v2` — ~90 MB (embeddings)
* `ms-marco-MiniLM-L-6-v2` — ~90 MB (reranker)
* `deberta-v3-small-prompt-injection-v2` — ~180 MB (injection guard)

## 4. Run the application

### Streamlit UI

```bash
streamlit run streamlit_pdf_app.py
```

Opens at:

```text
http://localhost:8501
```

### FastAPI + HTML frontend

```bash
python api.py
```

Opens at:

```text
http://localhost:8000
```

API docs (Swagger UI) at:

```text
http://localhost:8000/docs
```

### Quick injection guard test

```bash
python test_guard_quick.py
```

Upload a PDF or DOCX, wait for it to be indexed, and then ask questions about its contents.

## How it works

```text
PDF / DOCX upload
        │
        ▼
Text Extraction
  ├── PyPDF2 (primary — fast, accurate page numbers)
  └── Docling (fallback — OCR via RapidOCR, tables, scanned PDFs)
  └── python-docx (DOCX files)
        │
        ▼
Injection Sanitization
  └── Regex patterns removed from extracted text before indexing
        │
        ▼
Structure-Aware Chunking
  ├── Split on headings (##, ###)
  ├── Tables kept whole — never split
  ├── Paragraphs accumulated up to 1200 chars
  └── Each chunk tagged with page number + content type
        │
        ▼
Dual Indexing
  ├── ChromaDB (semantic vectors via all-MiniLM-L6-v2)
  └── BM25 index (keyword search via rank-bm25)
        │
        ▼
User asks a question
        │
        ▼
Injection Guard — user query (5 layers)
  ├── L1: Regex on original + normalized text
  ├── L2: Synonym detection
  ├── L3: DeBERTa v2 ML classifier (~50ms)
  ├── L4: Llama Guard 3 1B (~500ms, final safety net)
  └── L5: Subtle pattern fallback
        │
        ▼
Query Resolution
  └── Follow-up detection → resolved question used for retrieval
        │
        ▼
Hybrid Search (top 20 candidates)
  ├── Semantic search (ChromaDB cosine similarity)
  ├── BM25 keyword search
  └── RRF fusion (60% semantic + 40% keyword)
        │
        ▼
Injection Guard — retrieved chunks (regex only, fast)
        │
        ▼
Cross-Encoder Reranking
  └── ms-marco-MiniLM-L-6-v2 scores each (query, chunk) pair
  └── Top 5 chunks selected from 20 candidates
        │
        ▼
Relevance Gate (deterministic — no LLM call)
  └── best rerank score < -2.0 → return "Not found in document."
        │
        ▼
generate_answer()
  └── Gemini 2.0 Flash-Lite with conversation history (last 5 turns)
        │
        ▼
Injection Guard — LLM output (regex only, fast)
        │
        ▼
Source Attribution (deterministic — not by LLM)
  └── Page numbers extracted from retrieved chunks
        │
        ▼
Grounded answer + [Sources: Pages X, Y, Z]
```

## Retrieval and Generation

### 1. Document processing

**PDF:** PyPDF2 is tried first for fast, accurate per-page extraction. If it returns no content (scanned PDFs, image-based), Docling takes over with RapidOCR and table extraction. If per-page Docling export fails, the full document is split into virtual pages of ~3000 characters at paragraph boundaries.

**DOCX:** python-docx extracts paragraphs with heading detection and tables as markdown pipe format, grouped into virtual pages of ~30 paragraphs.

### 2. Chunking

The extracted content is split using structure-aware chunking — not fixed character counts.

* Headings (`##`, `###`) define section boundaries
* Tables are always kept as one whole chunk — never split mid-row
* Paragraphs are accumulated until `chunk_size=1200`
* Each chunk retains its page number and content type (`section`, `table`, `text`)

### 3. Embedding

`all-MiniLM-L6-v2` converts each chunk into a 384-dimensional vector.

These embeddings are stored in ChromaDB alongside a BM25 index built over the same corpus.

### 4. Hybrid retrieval

When the user asks a question, both search methods run in parallel:

* **Semantic search** — ChromaDB cosine similarity (60% weight)
* **BM25 keyword search** — exact term matching (40% weight)

Results are fused using Reciprocal Rank Fusion (RRF) and the top 20 candidates are returned. Each chunk is tagged `semantic`, `keyword`, or `hybrid` based on which search found it.

### 5. Reranking

The cross-encoder `ms-marco-MiniLM-L-6-v2` scores each `(question, chunk)` pair together — much more accurate than bi-encoder similarity alone. The top 5 chunks are selected from the 20 candidates.

### 6. Relevance gate

Before calling the LLM, the best rerank score is checked against a threshold (`-2.0`). If no chunk is relevant enough, `Not found in document.` is returned instantly without calling the LLM — preventing hallucination.

### 7. Generation

The top 5 chunks are passed along with the user's question and conversation history to:

```text
Gemini 2.0 Flash-Lite
```

Gemini generates an answer using only the retrieved document context.

The application then displays the answer along with the source pages.

## Injection Guard — 5 Layers

| Layer | Method | Speed | Catches |
| ----- | ------ | ----- | ------- |
| L1 | Regex (original + normalized) | ~0.1ms | Explicit patterns, leet speak, zero-width chars, dot-splitting |
| L2 | Synonym detection | ~0.1ms | "discard directives", paraphrase attacks |
| L3 | DeBERTa v2 ML | ~50ms | Semantic attacks, novel phrasing, role switching |
| L4 | Llama Guard 3 1B | ~500ms | Final safety net — rare edge cases |
| L5 | Subtle pattern fallback | ~0.1ms | Last resort |

Applied at three points in every query:

* **User query** — full 5-layer detection
* **Retrieved chunks** — regex only (already sanitized at upload)
* **LLM output** — regex only (fast output sanitization)

## Conversation History

Follow-up questions are resolved automatically before retrieval:

```text
Q1: "What is the stock amount of item X?"
Q2: "What about item Y?"
     ↓ resolved to:
    "What is the stock amount of item X? What about item Y?"
```

The last 5 turns are injected into the LLM context as `user` / `assistant` pairs.

## Known limitations

* **In-memory vector store:** The ChromaDB index and BM25 index are lost when the application restarts. To persist the vector database, replace:

```python
chromadb.EphemeralClient()
```

with:

```python
chromadb.PersistentClient(path="./chroma_db")
```

* **Gemini dependency:** Generation requires internet access and Google API quota. Switch to Ollama + `qwen3:8b` for fully local deployment.
* **Ollama required for L4:** Llama Guard layer 4 requires Ollama running locally. Layers 1–3 remain active if Ollama is unavailable.
* **Scanned PDF accuracy:** Complex layouts, rotated text, or very low quality scans may reduce extraction quality despite OCR fallback.
* **Session memory:** Sessions are stored in-memory and lost on restart.

## Project Structure

```text
rag_file/
├── api.py                     ← FastAPI REST backend
├── streamlit_pdf_app.py       ← Streamlit UI
├── test_guard_quick.py        ← Injection guard quick test
├── test_injection_guard.py    ← Full injection guard test suite
├── requirements.txt
├── .env
└── rag_core/
    ├── __init__.py
    ├── models.py              ← PageText, Chunk dataclasses
    ├── extract.py             ← PDF/DOCX extraction
    ├── chunking.py            ← Structure-aware chunking
    ├── vector_store.py        ← ChromaDB + BM25 hybrid store
    ├── reranker.py            ← Cross-encoder reranking
    ├── relevance_gate.py      ← Deterministic relevance check
    ├── injection_guard.py     ← 5-layer injection detection
    ├── generation.py          ← Gemini generation + history
    └── pipeline.py            ← Full RAG pipeline orchestration
```

## Technology Stack

| Component | Technology |
| --------- | ---------- |
| Frontend | Streamlit + HTML/JS |
| API | FastAPI + Uvicorn |
| Document Parsing | PyPDF2 + Docling + python-docx |
| Embeddings | all-MiniLM-L6-v2 (sentence-transformers) |
| Keyword Search | BM25 (rank-bm25) |
| Vector Database | ChromaDB |
| Reranker | ms-marco-MiniLM-L-6-v2 (cross-encoder) |
| LLM | Gemini 2.0 Flash-Lite |
| Injection Guard | DeBERTa v2 + Llama Guard 3 1B |
| Observability | Langfuse + LangSmith |
| Backend | Python 3.10+ |
| LLM API | Google Gemini API |