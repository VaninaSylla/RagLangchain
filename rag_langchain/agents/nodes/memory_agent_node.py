"""Node memory_agent : recherche sémantique dans l'historique conversationnel.

Injecte les messages pertinents (k=3 par défaut) dans state["mem_results"]
pour que router / synthesis puissent exploiter le contexte long-terme.
"""
from __future__ import annotations

from rag_langchain.memory import MemoryStore


def memory_agent_node(state: dict) -> dict:
    """Retourne jusqu'à 3 messages historiques sémantiquement proches."""
    question = state.get("cleaned_question") or state["question"]
    conv_id = state.get("conversation_id")
    if not conv_id:
        return {"mem_results": []}

    memory = MemoryStore()
    try:
        results = memory.semantic_search(question, k=3, conversation_id=conv_id)
    except Exception:
        return {"mem_results": []}

    mem_results = [
        {
            "content": m.content,
            "role": m.role,
            "route": m.route,
            "timestamp": m.timestamp,
        }
        for m, _ in results
    ]
    return {"mem_results": mem_results}