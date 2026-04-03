"""query pipeline: embed → retrieve → re-rank → generate."""

from __future__ import annotations

from typing import Any

import chromadb
from rich.console import Console

from src.core import generate_answer, get_embeddings, re_rank_chunks
from src.ingest import DB_PATH, COLLECTION_NAME

console = Console()

TOP_K = 10


def _get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        return client.get_collection(name=COLLECTION_NAME)
    except Exception:
        raise RuntimeError(
            "No documents indexed yet.\n"
            "Run:  python main.py ingest <your_file.pdf>"
        )


def run_query(query: str, verbose: bool = False) -> dict[str, Any]:
    """Run the full 4-step RAG pipeline and return the answer + sources."""

    # Step A — embed the query
    with console.status("  Embedding query...", spinner="dots"):
        query_vector = get_embeddings([query], task_type="RETRIEVAL_QUERY")[0]

    # Step B — retrieve top-K from ChromaDB
    collection = _get_collection()
    total = collection.count()
    if total == 0:
        raise RuntimeError("Index is empty. Ingest some documents first.")

    n = min(TOP_K, total)
    with console.status(f"  Searching {total} chunks...", spinner="dots"):
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )

    candidates = [
        {"text": doc, "metadata": meta, "score": round(1 - dist, 4)}
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]

    if verbose:
        console.print(f"\n[bold yellow]Top-{n} retrieved chunks:[/bold yellow]")
        for i, c in enumerate(candidates):
            m = c["metadata"]
            preview = c["text"][:100].replace("\n", " ")
            console.print(f"  [{i}] {m['filename']} p.{m['page_number']} (sim={c['score']})\n      {preview}...")

    # Step C — re-rank to top-3
    with console.status("  Re-ranking...", spinner="dots"):
        top3 = re_rank_chunks(query, candidates, verbose=verbose)

    if verbose:
        console.print("\n[bold green]Top-3 after re-ranking:[/bold green]")
        for i, c in enumerate(top3):
            m = c["metadata"]
            console.print(f"  [{i+1}] {m['filename']} p.{m['page_number']}")
        console.print()

    # Step D — generate grounded answer
    with console.status("  Generating answer...", spinner="dots"):
        return generate_answer(query, top3)
