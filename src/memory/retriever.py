"""
BrainOS — Memory Retriever
Converts a user query into a context block for the LLM.
"""

from __future__ import annotations

from src.config import settings
from src.memory.vector_store import vector_store
from src.processing.embedder import embedding_engine
from src.utils.logger import log


class MemoryRetriever:
    def __init__(self, top_k: int = settings.top_k_retrieval):
        self.top_k = top_k

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        """
        Embed query → search FAISS → return ranked metadata dicts.
        Each dict has: content, input_type, source, timestamp, score
        """
        k = top_k or self.top_k
        log.debug(f"Retrieving top-{k} for query: '{query[:80]}'")

        query_vec = embedding_engine.embed_one(query)
        results = vector_store.search(query_vec, top_k=k)

        log.debug(f"Retrieved {len(results)} memory chunks.")
        return results

    def build_context(self, query: str, top_k: int | None = None) -> tuple[str, list[dict]]:
        """
        Returns (context_string, source_list) for use in an LLM prompt.
        """
        results = self.retrieve(query, top_k)

        if not results:
            return "No relevant memories found.", []

        lines = ["### Retrieved Memory Chunks\n"]
        for i, r in enumerate(results, 1):
            ts = r.get("timestamp", "")[:19]  # trim microseconds
            src = r.get("source", "")
            itype = r.get("input_type", "text")
            score = r.get("score", 0.0)
            content = r.get("content", "")

            lines.append(
                f"[{i}] ({itype} | {ts} | src: {src or 'direct'} | score: {score:.3f})\n{content}\n"
            )

        context = "\n".join(lines)
        return context, results


# Singleton
memory_retriever = MemoryRetriever()
