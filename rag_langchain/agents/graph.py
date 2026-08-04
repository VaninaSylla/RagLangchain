"""Construction du StateGraph LangGraph multi-agents.

Flux :
    START → parse → memory → router → (conditional)
        - conversation         → synthesis → END
        - documents            → doc_agent  → synthesis → END
        - database             → db_agent   → synthesis → END
        - mixed                → planner    → doc_agent → synthesis → END
                                                    ↘ db_agent ↗
        - auto_investigate     → planner    → doc_agent → synthesis → END
                                                    ↘ db_agent ↗
        - unknown              → synthesis (no_info) → END
"""
from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from .state import AgentState
from .nodes.parse_node import parse_node
from .nodes.memory_agent_node import memory_agent_node
from .nodes.router_node import router_node
from .nodes.planner_node import planner_node
from .nodes.doc_agent_node import doc_agent_node
from .nodes.db_agent_node import db_agent_node
from .nodes.synthesis_node import synthesis_node


def _route_after_router(state: AgentState) -> str:
    return state.get("route", "unknown")


def build_graph():
    g = StateGraph(AgentState)

    g.add_node("parse", parse_node)
    g.add_node("memory", memory_agent_node)
    g.add_node("router", router_node)
    g.add_node("planner", planner_node)
    g.add_node("doc_agent", doc_agent_node)
    g.add_node("db_agent", db_agent_node)
    g.add_node("synthesis", synthesis_node)

    g.add_edge(START, "parse")
    g.add_edge("parse", "memory")
    g.add_edge("memory", "router")

    g.add_conditional_edges(
        "router",
        _route_after_router,
        {
            "conversation": "synthesis",
            "documents": "doc_agent",
            "database": "db_agent",
            "mixed": "planner",
            "auto_investigate": "planner",
            "unknown": "synthesis",
        },
    )
    g.add_edge("planner", "doc_agent")
    g.add_edge("planner", "db_agent")
    g.add_edge("doc_agent", "synthesis")
    g.add_edge("db_agent", "synthesis")
    g.add_edge("synthesis", END)

    return g.compile()


def run_question(question: str, *, history: list | None = None,
                 conversation_id: str = "", source_filter: list[str] | None = None) -> dict:
    """Exécute le graph sur une question et retourne l'état final."""
    graph = build_graph()
    init: AgentState = {
        "question": question,
        "cleaned_question": "",
        "history": history or [],
        "groups": [],
        "tags": [],
        "references": {},
        "route": "unknown",
        "selected_connectors": [],
        "selected_doc_filters": source_filter or [],
        "doc_results": [],
        "db_results": [],
        "answer": "",
        "sources": [],
        "conversation_id": conversation_id,
        "error": None,
        "refresh_attempted": False,
    }
    return graph.invoke(init)