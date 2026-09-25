"""
utils.py — Shared utility functions for TriMem-AR modules
"""
import re

def chunk_long_text(user_text, asst_text):
    """Split long assistant responses into smaller chunks for SSA/SSU retrieval.
    EM-LLM-inspired: split at content boundaries (paragraphs, list items, sentences).
    Each chunk preserves user context for reference."""
    MIN_CHUNK = 80
    MAX_CHUNK = 600
    asst = asst_text.strip()
    user_prefix = user_text[:150]  # keep enough user context

    # Short enough: keep as one
    if len(asst) < MAX_CHUNK:
        return [f"[user] {user_text} [assistant] {asst}"]

    # 1) Paragraph split by double newlines
    paras = [p.strip() for p in re.split(r'\n\s*\n', asst) if len(p.strip()) >= MIN_CHUNK]
    if len(paras) > 1:
        return [f"[user] {user_prefix} [assistant] {p}" for p in paras]

    # 2) List item split (n. ..., - ..., * ...)
    items = re.findall(r'(?:^|\n)\s*(?:\d+\.|[-•*])\s*(.*?)(?=\n\s*(?:\d+\.|[-•*]|\Z)|\Z)', asst, re.DOTALL)
    items = [item.strip() for item in items if len(item.strip()) >= MIN_CHUNK]
    if len(items) > 1:
        return [f"[user] {user_prefix} [assistant] {item}" for item in items]

    # 3) Split by single newlines (line-based)
    lines = [l.strip() for l in asst.split('\n') if len(l.strip()) >= MIN_CHUNK]
    if len(lines) > 1:
        return [f"[user] {user_prefix} [assistant] {l}" for l in lines]

    # 4) Fallback: split at ~600 char boundaries
    chunks = []
    remaining = asst
    first = True
    while remaining:
        size = min(MAX_CHUNK, len(remaining)) if first else min(400, len(remaining))
        first = False
        chunk = remaining[:size]
        remaining = remaining[size:]
        if len(chunk) >= MIN_CHUNK:
            chunks.append(f"[user] {user_prefix} [assistant] {chunk}")
    return chunks if chunks else [f"[user] {user_text} [assistant] {asst}"]
