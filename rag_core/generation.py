from typing import List
import ollama

SYSTEM_PROMPT = """You are a document Q&A assistant.

Use ONLY the provided context chunks to answer the question.

Rules:
- Return ONLY content that directly answers the user's question.
- Do not add introductions, extra explanation, background, or assumptions.
- If the exact answer is not in context, reply exactly: Not found in document.
- Preserve numbers and table values exactly as written in context.
- If context contains a table, present it as clean labeled key-value pairs, NOT as markdown table syntax with | symbols.
- If the user asks for information "in tabular form", "as a table", or "as a list", 
  that is a FORMAT request — find the relevant content first, then present it 
  in the requested format. Do not return Not found just because no table exists in the document.

Response format:
1) Answer: <direct answer only, formatted as user requested if specified>
"""


def build_prompt(question: str, retrieved_chunks) -> str:

    def _chunk_to_context(chunk):
        if isinstance(chunk, dict):
            page = chunk.get("page") or chunk.get("page_number") or "?"
            text = chunk.get("text") or chunk.get("page_content") or ""
            return page, text

        if hasattr(chunk, "page_content"):
            metadata = getattr(chunk, "metadata", {}) or {}
            page = metadata.get("page") or metadata.get("page_number") or "?"
            text = chunk.page_content or ""
            return page, text

        if hasattr(chunk, "text"):
            page = getattr(chunk, "page_number", "?")
            text = chunk.text or ""
            return page, text

        return "?", ""

    question = question.strip()
    context_block = "\n\n".join(
        f"[Page {page}]\n{text}"
        for chunk in retrieved_chunks
        for page, text in [_chunk_to_context(chunk)]
        if text
    )
    if not context_block:
        context_block = "[No retrieved context]"
    return f"Context:\n{context_block}\n\nQuestion: {question}\n\nAnswer:"


def generate_answer(question: str, retrieved_chunks, model: str = "qwen3:8b") -> str:
    """Call the local Ollama server with a Qwen model."""
    prompt = build_prompt(question, retrieved_chunks)
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return response["message"]["content"]