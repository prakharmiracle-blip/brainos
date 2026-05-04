"""
BrainOS — Brain Chat Agent
RAG-based Q&A: retrieve relevant memories → build prompt → LLM answer.
Answers are GROUNDED in stored data only.
"""

from __future__ import annotations

from src.memory.retriever import memory_retriever
from src.utils.llm_client import llm_client
from src.utils.logger import log
from src.utils.models import AskResponse

SYSTEM_PROMPT = """You are BrainOS, a personal AI memory assistant.
Your ONLY job is to answer the user's question using the memory chunks provided below.

Rules:
1. ONLY use information from the provided memory chunks — do NOT use any outside knowledge.
2. If the memory chunks don't contain a relevant answer, say: "I don't have any notes about that yet."
3. Be concise and direct.
4. Always cite which memory entry you're referencing (use the numbered [1], [2] format).
5. Never hallucinate or make up information.
"""


class BrainChatAgent:

    def ask(self, query: str, top_k: int = 5) -> AskResponse:
        """
        Main entry point: query → RAG → answer.
        """
        log.info(f"Brain query: '{query[:100]}'")

        context, sources = memory_retriever.build_context(query, top_k=top_k)

        if not sources:
            return AskResponse(
                answer="I don't have any notes about that yet. Try adding some memories first!",
                sources=[],
                query=query,
            )

        user_prompt = f"""User Question: {query}

{context}

Please answer the question based ONLY on the memory chunks above."""

        answer = llm_client.chat(
            system=SYSTEM_PROMPT,
            user=user_prompt,
            max_tokens=1024,
        )

        log.info(f"Generated answer ({len(answer)} chars).")
        return AskResponse(
            answer=answer,
            sources=[
                {
                    "content": s.get("content", "")[:200],
                    "input_type": s.get("input_type"),
                    "source": s.get("source", ""),
                    "timestamp": s.get("timestamp", ""),
                    "score": round(s.get("score", 0.0), 4),
                }
                for s in sources
            ],
            query=query,
        )


# Singleton
brain_chat_agent = BrainChatAgent()
