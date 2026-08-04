"""Node doc_agent : retrieval + reranking sur ChromaDB."""
from __future__ import annotations

from rag_langchain.core.rag_chain import retrieve_and_rerank
from rag_langchain.core.ingestion import get_vectorstore


def doc_agent_node(state: dict) -> dict:
    cleaned = state.get("cleaned_question") or state["question"]
    try:
        vs = get_vectorstore()
        docs = retrieve_and_rerank(
            cleaned, vs,
            source_filter=state.get("selected_doc_filters") or None,
        )
        return {"doc_results": [
            {"content": d.page_content, "metadata": dict(d.metadata)} for d in docs
        ]}
    except Exception as e:
        return {"doc_results": [], "error": f"doc_agent: {e}"}