import re
from typing import List
from langchain_core.documents import Document as LCDocument  # ← rename to avoid conflict


def structure_aware_split(text: str, page_number: int, max_chunk_size: int = 1200) -> List[LCDocument]:
    chunks = []
    sections = re.split(r'(?=^#{1,4}\s)', text, flags=re.MULTILINE)

    for section in sections:
        section = section.strip()
        if not section:
            continue

        if len(section) <= max_chunk_size:
            chunks.append(LCDocument(
                page_content=section,
                metadata={"page": page_number, "type": "section"}
            ))
            continue

        blocks = _split_into_blocks(section)
        current_chunk = ""

        for block in blocks:
            block = block.strip()
            if not block:
                continue

            if _is_table(block):
                if current_chunk:
                    chunks.append(LCDocument(
                        page_content=current_chunk.strip(),
                        metadata={"page": page_number, "type": "text"}
                    ))
                    current_chunk = ""
                chunks.append(LCDocument(
                    page_content=block,
                    metadata={"page": page_number, "type": "table"}
                ))
                continue

            if len(current_chunk) + len(block) + 2 <= max_chunk_size:
                current_chunk += "\n\n" + block
            else:
                if current_chunk:
                    chunks.append(LCDocument(
                        page_content=current_chunk.strip(),
                        metadata={"page": page_number, "type": "text"}
                    ))
                current_chunk = block

        if current_chunk:
            chunks.append(LCDocument(
                page_content=current_chunk.strip(),
                metadata={"page": page_number, "type": "text"}
            ))

    return chunks


def _split_into_blocks(text: str) -> List[str]:
    blocks = []
    current = []
    in_table = False

    for line in text.split("\n"):
        is_table_line = line.strip().startswith("|")

        if is_table_line and not in_table:
            if current:
                blocks.append("\n".join(current))
                current = []
            in_table = True
            current.append(line)
        elif not is_table_line and in_table:
            blocks.append("\n".join(current))
            current = [line]
            in_table = False
        elif line.strip() == "" and not in_table:
            if current:
                blocks.append("\n".join(current))
                current = []
        else:
            current.append(line)

    if current:
        blocks.append("\n".join(current))

    return blocks


def _is_table(block: str) -> bool:
    lines = [l for l in block.strip().split("\n") if l.strip()]
    table_lines = sum(1 for l in lines if l.strip().startswith("|"))
    return table_lines >= 2


def chunk_pages(pages, chunk_size: int = 1200, overlap: int = 120, max_chunks: int = 1200):
    """Structure-aware chunking — keeps headings, paragraphs, and tables intact."""
    print(f"[DEBUG-Chunking] Input pages: {len(pages)}")
    for i, p in enumerate(pages[:3]):  # Debug first 3 pages
        page_num = p.page_number if hasattr(p, "page_number") else 1
        text_preview = (p.text if hasattr(p, "text") else p.page_content)[:100]
        print(f"[DEBUG-Chunking] Page {i}: page_number={page_num}, text_preview={text_preview}")
    
    all_chunks = []
    for page in pages:
        page_text = page.text if hasattr(page, "text") else page.page_content  # ← handle both types
        page_num = page.page_number if hasattr(page, "page_number") else 1      # ← handle both types
        print(f"[DEBUG-Chunking] Processing page_number={page_num}")
        page_chunks = structure_aware_split(page_text, page_number=page_num, max_chunk_size=chunk_size)
        print(f"[DEBUG-Chunking] Created {len(page_chunks)} chunks for page {page_num}")
        all_chunks.extend(page_chunks)
        if len(all_chunks) >= max_chunks:
            break
    print(f"[DEBUG-Chunking] Total chunks: {len(all_chunks)}")
    return all_chunks[:max_chunks]
