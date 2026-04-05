"""Memory management using ChromaDB."""

from datetime import datetime
from typing import Optional
import chromadb
from sentence_transformers import SentenceTransformer

from . import config


class ResearchMemory:
    """Vector store for research findings that persist across sessions."""

    def __init__(self):
        """Initialize ChromaDB and embeddings model."""
        self.client = chromadb.PersistentClient(
            path=str(config.RESEARCH_MEMORY_DIR)
        )
        self.embedder = SentenceTransformer(config.EMBEDDING_MODEL)
        self.collection = self.client.get_or_create_collection(
            name=config.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": config.CHROMA_DISTANCE_METRIC},
        )

    def recall(self, query: str, top_k: int = 3) -> list[dict]:
        """Retrieve past research findings related to this query."""
        if self.collection.count() == 0:
            return []
        
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
        )
        
        findings = []
        if results["documents"] and len(results["documents"]) > 0:
            for doc, metadata, distance in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                findings.append({
                    "content": doc,
                    "source": metadata.get("source", "unknown"),
                    "timestamp": metadata.get("timestamp", "unknown"),
                    "similarity": 1 - distance,  # Convert distance to similarity
                })
        
        return findings

    def save(self, content: str, source: str, topic: str) -> None:
        """Save a research finding to the vector store."""
        timestamp = datetime.now().isoformat()
        doc_id = f"{source}_{timestamp}".replace("/", "_").replace(":", "-")[:100]
        
        self.collection.add(
            documents=[content],
            metadatas=[{
                "source": source,
                "topic": topic,
                "timestamp": timestamp,
            }],
            ids=[doc_id],
        )

    def clear(self) -> None:
        """Delete all findings and reset memory."""
        self.client.delete_collection(name=config.CHROMA_COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=config.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": config.CHROMA_DISTANCE_METRIC},
        )

    def get_statistics(self) -> dict:
        """Return basic stats about the memory store."""
        count = self.collection.count()
        return {
            "total_findings": count,
            "memory_path": str(config.RESEARCH_MEMORY_DIR),
        }
