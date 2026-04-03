""" indexing pipeline: parse → chunk → embed → store."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import chromadb
from pypdf import PdfReader
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from src.core import chunk_text, get_embeddings

console = Console()

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".rag_db")
COLLECTION_NAME = "rag_documents"


def get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=DB_PATH)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _parse_pdf(path: Path) -> list[tuple[str, int]]:
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((text, i))
    return pages


def _parse_text(path: Path) -> list[tuple[str, int]]:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    return [(text, 1)] if text else []


def ingest_file(file_path: str) -> dict[str, Any]:
    """Parse a file, chunk it, embed it, and upsert to ChromaDB."""
    path = Path(file_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    filename = path.name

    console.print(f"\n[bold cyan]Parsing:[/bold cyan] {filename}")

    if suffix == ".pdf":
        pages = _parse_pdf(path)
    elif suffix in (".txt", ".md", ".rst"):
        pages = _parse_text(path)
    else:
        raise ValueError(f"Unsupported file type '{suffix}'. Use .pdf, .txt, .md, or .rst")

    if not pages:
        raise ValueError(f"No extractable text found in '{filename}'.")

    console.print(f"  Pages:  {len(pages)}")

    # Chunking
    all_chunks: list[dict[str, Any]] = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("  Chunking...", total=len(pages))
        for text, page_num in pages:
            all_chunks.extend(chunk_text(text, filename=filename, page_number=page_num))
            progress.advance(task)

    console.print(f"  Chunks: {len(all_chunks)}")

    if not all_chunks:
        raise ValueError("Chunking produced no output — file may be image-only.")

    # Embedding
    texts = [c["text"] for c in all_chunks]
    with console.status("  Embedding...", spinner="dots"):
        vectors = get_embeddings(texts, task_type="RETRIEVAL_DOCUMENT")

    # Upsert to ChromaDB
    collection = get_collection()
    collection.upsert(
        ids=[c["id"] for c in all_chunks],
        embeddings=vectors,
        documents=texts,
        metadatas=[c["metadata"] for c in all_chunks],
    )

    return {
        "filename": filename,
        "chunks_ingested": len(all_chunks),
        "pages_processed": len(pages),
    }
