"""chunking, embedding, re-ranking, answer generation."""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

EMBED_MODEL = "gemini-embedding-001"
GEMINI_MODEL = "gemini-2.5-flash"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
SPLIT_CHARS = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

_client: genai.Client | None = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        load_dotenv()
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise EnvironmentError(
                "GEMINI_API_KEY is not set.\n"
                "  PowerShell: $env:GEMINI_API_KEY = 'your-key'\n"
                "  Or add it to your .env file."
            )
        _client = genai.Client(api_key=key)
    return _client


# Chunking

def _split(text: str, sep: str, size: int, overlap: int) -> list[str]:
    parts = text.split(sep) if sep else list(text)
    chunks = []
    current = ""
    for part in parts:
        piece = part + sep if sep else part
        if len(current) + len(piece) <= size:
            current += piece
        else:
            if current.strip():
                chunks.append(current.strip())
            current = (current[-overlap:] if overlap else "") + piece
    if current.strip():
        chunks.append(current.strip())
    return chunks


def _recursive_split(text: str, seps: list[str], size: int, overlap: int) -> list[str]:
    if not text.strip():
        return []
    if len(text) <= size:
        return [text.strip()]
    sep, rest = seps[0], seps[1:]
    pieces = _split(text, sep, size, overlap)
    result = []
    for piece in pieces:
        if len(piece) > size and rest:
            result.extend(_recursive_split(piece, rest, size, overlap))
        else:
            result.append(piece)
    return result


def chunk_text(text: str, filename: str, page_number: int) -> list[dict[str, Any]]:
    """Split a page of text into overlapping chunks with metadata attached."""
    raw = _recursive_split(text, SPLIT_CHARS, CHUNK_SIZE, CHUNK_OVERLAP)
    chunks = []
    for i, chunk in enumerate(raw):
        if not chunk:
            continue
        chunk_id = str(uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"{filename}::p{page_number}::c{i}::{chunk[:64]}"
        ))
        chunks.append({
            "id": chunk_id,
            "text": chunk,
            "metadata": {
                "filename": filename,
                "page_number": page_number,
                "chunk_index": i,
            },
        })
    return chunks


# Embeddings

def get_embeddings(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    """Batch-embed texts using text-embedding-004."""
    client = get_client()
    all_vectors = []
    for i, text in enumerate(texts):
        result = client.models.embed_content(
            model=EMBED_MODEL,
            contents=text,
            config=types.EmbedContentConfig(task_type=task_type),
        )
        all_vectors.append(result.embeddings[0].values)
        # brief pause every 20 texts to respect rate limits
        if i > 0 and i % 20 == 0:
            time.sleep(0.5)
    return all_vectors


# Re-ranking

RERANK_PROMPT = """\
You are a relevance ranker. Given a question and {n} numbered excerpts, \
return ONLY a JSON array of the 3 integer indices (0-based) of the most relevant excerpts.

Output format: [2, 7, 4]  — no prose, no markdown, just the array.

QUESTION: {query}

EXCERPTS:
{excerpts}
"""


def re_rank_chunks(query: str, chunks: list[dict], verbose: bool = False) -> list[dict]:
    """Ask Gemini to pick the 3 most relevant chunks from a candidate list."""
    client = get_client()
    if len(chunks) <= 3:
        return chunks

    excerpts = "\n\n".join(
        f"[{i}] {c['text'].replace(chr(10), ' ')}" for i, c in enumerate(chunks)
    )
    prompt = RERANK_PROMPT.format(n=len(chunks), query=query, excerpts=excerpts)

    if verbose:
        print("\n[re-ranker] Prompt sent to Gemini:\n")
        print(prompt)

    response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    raw = response.text.strip()
    raw = re.sub(r"```[a-z]*\n?", "", raw).strip()

    if verbose:
        print(f"\n[re-ranker] Response: {raw}\n")

    try:
        indices = json.loads(raw)
        if not isinstance(indices, list):
            raise ValueError
        seen: set[int] = set()
        valid = []
        for idx in indices:
            if isinstance(idx, int) and 0 <= idx < len(chunks) and idx not in seen:
                valid.append(idx)
                seen.add(idx)
        indices = valid[:3]
    except (json.JSONDecodeError, ValueError):
        found = [int(x) for x in re.findall(r"\d+", raw) if int(x) < len(chunks)]
        indices = found[:3]

    if not indices:
        indices = list(range(min(3, len(chunks))))

    return [chunks[i] for i in indices]


# Answer generation

SYSTEM_PROMPT = """\
You are a document analyst. Your job is to answer questions using only the source excerpts provided.

Rules:
- Answer using ONLY the provided excerpts.
- If the answer isn't in the excerpts, say: "I don't have enough information in the provided documents to answer this."
- Do not use outside knowledge, even if you're confident in it.
"""

ANSWER_TEMPLATE = """\
SOURCE EXCERPTS:
{excerpts}

---

QUESTION: {query}
"""


def generate_answer(query: str, chunks: list[dict]) -> dict[str, Any]:
    """Generate a grounded answer from the top-3 re-ranked chunks."""
    client = get_client()

    parts = []
    for i, c in enumerate(chunks, 1):
        m = c["metadata"]
        parts.append(f"--- Excerpt {i} | {m['filename']} | Page {m['page_number']} ---\n{c['text']}")

    prompt = ANSWER_TEMPLATE.format(excerpts="\n\n".join(parts), query=query)

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
    )

    seen: set[tuple] = set()
    sources = []
    for c in chunks:
        m = c["metadata"]
        key = (m["filename"], m["page_number"])
        if key not in seen:
            seen.add(key)
            sources.append({"filename": m["filename"], "page_number": m["page_number"]})

    return {"answer": response.text.strip(), "sources": sources}
