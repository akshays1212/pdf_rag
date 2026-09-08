# Ask My PDF — Local RAG Bot

A minimal RAG (Retrieval-Augmented Generation) app:

Upload a PDF, ask questions, and get answers grounded in the document with page citations.

* **Embeddings:** BGE-M3 (via `sentence-transformers`, runs on CPU)
* **Vector store:** ChromaDB (in-memory, no server needed)
* **Generator LLM:** Gemini 2.0 Flash-Lite via Google Gemini API
* **Document extraction:** Docling
* **Frontend:** Streamlit

## 1. Get a Google Gemini API Key

1. Get an API key from Google AI Studio.
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
GOOGLE_API_KEY=your_api_key_here
```

Make sure your application loads the `.env` file using `python-dotenv`.

The app uses:

```text
gemini-3.5-flash-lite
```

as the generation model.

> **Privacy note:** BGE-M3 embeddings and ChromaDB run locally, but the retrieved document chunks are sent to Google's Gemini API when generating an answer. Avoid uploading sensitive or confidential documents unless this is acceptable for your use case.

## 2. Set up the Python environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

The first time you run the application, `sentence-transformers` will download the BGE-M3 model (~2.2 GB) from Hugging Face.

The model is cached locally, so it does not need to be downloaded again on subsequent runs.

## 3. Run the application

```bash
streamlit run app.py
```

The application will normally open at:

```text
http://localhost:8501
```

Upload a PDF, wait for it to be indexed, and then ask questions about its contents.

## How it works

```text
PDF upload
    │
    ▼
extract_text_from_pdf()
    │
    ▼
Docling extraction
    │
    ▼
chunk_pages()
    │
    ▼
Overlapping ~800-character chunks
with page numbers
    │
    ▼
VectorStore.add_chunks()
    │
    ▼
BGE-M3 embeddings
    │
    ▼
ChromaDB
    │
    ▼
User asks a question
    │
    ▼
VectorStore.query()
    │
    ▼
Retrieve top relevant chunks
    │
    ▼
generate_answer()
    │
    ▼
Gemini 2.0 Flash-Lite
    │
    ▼
Grounded answer + page citations
```

## Retrieval and Generation

### 1. Document processing

Docling extracts text from the PDF while preserving useful document structure such as headings, paragraphs, and tables.

### 2. Chunking

The extracted content is divided into overlapping chunks of approximately 800 characters.

Each chunk retains its page number so that the retrieved information can be traced back to the original PDF.

### 3. Embedding

BGE-M3 converts each chunk into a numerical vector representation.

These embeddings are stored in ChromaDB.

### 4. Retrieval

When the user asks a question, the question is also converted into an embedding.

ChromaDB performs similarity search and retrieves the most relevant chunks.

### 5. Generation

The retrieved chunks are passed along with the user's question to:

```text
Gemini 2.0 Flash-Lite
```

Gemini generates an answer using the retrieved document context.

The application then displays the answer along with the relevant source excerpts and page numbers.

## Known limitations — v1

* **PDF only:** The current version processes PDF files.
* **Complex layouts:** Multi-column documents, rotated text, or low-quality scanned PDFs may reduce extraction quality.
* **No re-ranking:** Retrieval currently uses embedding similarity only. A cross-encoder reranker can be added later to improve retrieval precision.
* **In-memory vector store:** The current ChromaDB index is lost when the application restarts.

To persist the vector database, replace:

```python
chromadb.EphemeralClient()
```

with:

```python
chromadb.PersistentClient(path="./chroma_db")
```

## Extending to DOCX / Excel

The same RAG pipeline can be extended to additional document types.

### DOCX

Use `python-docx` to extract:

* Paragraphs
* Headings
* Tables

### Excel

Use `openpyxl` or `pandas` to extract:

* Worksheets
* Rows
* Tables
* Cell values

For large spreadsheets, avoid putting the entire sheet into one chunk. Create chunks based on rows or logical table blocks so that numerical lookups remain accurate.

The overall pipeline can remain the same:

```text
Document
    ↓
Extraction
    ↓
Chunking
    ↓
BGE-M3 Embedding
    ↓
ChromaDB
    ↓
Similarity Search
    ↓
Gemini 2.0 Flash-Lite
    ↓
Answer + Sources
```

## Technology Stack

| Component        | Technology            |
| ---------------- | --------------------- |
| Frontend         | Streamlit             |
| Document Parsing | Docling               |
| Embeddings       | BGE-M3                |
| Vector Database  | ChromaDB              |
| LLM              | Gemini 2.0 Flash-Lite |
| Backend Logic    | Python                |
| LLM API          | Google Gemini API     |
