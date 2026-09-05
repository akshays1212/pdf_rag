from typing import List, Dict
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
- DO NOT add source citations like "Source: page X" to your answer. 
  The backend will handle all source attribution automatically from the retrieved chunks.
- Use conversation history to understand follow-up questions and references like
  "what about X", "and Y?", "compare with Z" — resolve them using prior context.

Response format:
- Provide ONLY the answer text. No "Source:" or citation info.
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

def generate_answer(
    question: str,
    retrieved_chunks,
    model: str = "qwen3:8b",
    chat_history: List[Dict] = None,
) -> str:
    prompt = build_prompt(question, retrieved_chunks)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if chat_history:
        print(f"[DEBUG-History] Injecting {len(chat_history)} history turns into LLM")
        for turn in chat_history:
            messages.append({"role": "user", "content": turn["question"]})
            messages.append({"role": "assistant", "content": turn["answer"]})
    else:
        print(f"[DEBUG-History] No history to inject")

    messages.append({"role": "user", "content": prompt})

    print(f"[DEBUG-History] Total messages sent to LLM: {len(messages)}")

    response = ollama.chat(model=model, messages=messages)
    answer_text = response["message"]["content"].strip()

    if "Source:" in answer_text:
        answer_text = answer_text.split("Source:")[0].strip()

    return answer_text