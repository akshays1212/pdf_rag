from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import os

from rag_core import build_store_from_files, answer_question
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Glaxm Vault", version="1.0.0")

# ── CORS — allow browser to call API ──────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── serve frontend ────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse("frontend/index.html")


# ── in-memory session store ───────────────────────────────────────────
# key: session_id → {store, chat_history}
sessions = {}


# ── request/response models ───────────────────────────────────────────
class QuestionRequest(BaseModel):
    session_id: str
    question: str
    candidate_k: int = 20
    top_k: int = 5
    model: str = "qwen3:8b"


class ChunkInfo(BaseModel):
    text: str
    page_number: int
    retrieval_source: str
    rrf_score: float
    rerank_score: float


class QuestionResponse(BaseModel):
    answer: str
    session_id: str
    final_chunks: List[ChunkInfo]
    history_turns: int


class UploadResponse(BaseModel):
    session_id: str
    filenames: List[str]
    pages: int
    chunks: int
    message: str


# ── endpoints ─────────────────────────────────────────────────────────

@app.post("/upload", response_model=UploadResponse)
async def upload_files(files: List[UploadFile] = File(...)):
    """Upload one or more PDF/DOCX files and index them."""
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    # validate file types
    for f in files:
        ext = os.path.splitext(f.filename)[1].lower()
        if ext not in (".pdf", ".docx"):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {f.filename}. Only PDF and DOCX allowed."
            )

    # read files
    file_tuples = []
    for f in files:
        contents = await f.read()
        file_tuples.append((contents, f.filename))

    # build vector store
    try:
        store, pages, chunks = build_store_from_files(file_tuples)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process files: {str(e)}")

    # create session
    import uuid
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "store": store,
        "chat_history": [],
        "filenames": [f.filename for f in files],
    }

    return UploadResponse(
        session_id=session_id,
        filenames=[f.filename for f in files],
        pages=len(pages),
        chunks=len(chunks),
        message=f"Successfully indexed {len(files)} file(s)",
    )


@app.post("/ask", response_model=QuestionResponse)
def ask_question(req: QuestionRequest):
    """Ask a question against the uploaded documents."""
    if req.session_id not in sessions:
        raise HTTPException(
            status_code=404,
            detail="Session not found. Please upload files first."
        )

    session = sessions[req.session_id]
    store = session["store"]
    chat_history = session["chat_history"]

    # build history to pass — last 5 turns
    MAX_HISTORY_TURNS = 5
    history_to_pass = chat_history[-MAX_HISTORY_TURNS:]

    try:
        answer, final_chunks, candidates = answer_question(
            store,
            req.question,
            candidate_k=req.candidate_k,
            top_k=req.top_k,
            model=req.model,
            chat_history=history_to_pass,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    # save to session history
    session["chat_history"].append({
        "question": req.question,
        "answer": answer
    })

    return QuestionResponse(
        answer=answer,
        session_id=req.session_id,
        final_chunks=[
            ChunkInfo(
                text=c["text"],
                page_number=c.get("page_number", 0),
                retrieval_source=c.get("retrieval_source", "unknown"),
                rrf_score=c.get("rrf_score", 0.0),
                rerank_score=c.get("rerank_score", 0.0),
            )
            for c in final_chunks
        ],
        history_turns=len(session["chat_history"]),
    )


@app.get("/session/{session_id}")
def get_session_info(session_id: str):
    """Get session info — filenames and chat history length."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    session = sessions[session_id]
    return {
        "session_id": session_id,
        "filenames": session["filenames"],
        "history_turns": len(session["chat_history"]),
        "chat_history": session["chat_history"],
    }


@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    """Clear session and free memory."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    del sessions[session_id]
    return {"message": "Session cleared"}


@app.get("/health")
def health():
    return {"status": "ok", "sessions_active": len(sessions)}


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)