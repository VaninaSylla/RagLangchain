"""État partagé entre les nodes du LangGraph orchestrateur."""
from __future__ import annotations

from operator import add
from typing import Any, Literal
from typing_extensions import Annotated, TypedDict


Route = Literal[
    "conversation", "documents", "database", "mixed", "auto_investigate", "unknown"
]


class AgentState(TypedDict, total=False):
    question: str
    cleaned_question: str
    history: list[dict[str, Any]]
    # Les champs ci-dessous sont annotés avec `add` pour permettre à plusieurs
    # branches parallèles (doc_agent + db_agent) d'écrire en parallèle sans
    # InvalidUpdateError.
    groups: Annotated[list[str], add]
    tags: Annotated[list[str], add]
    references: dict[str, list[str]]
    route: Route
    selected_connectors: Annotated[list[str], add]
    selected_doc_filters: Annotated[list[str], add]
    doc_results: Annotated[list[dict[str, Any]], add]
    db_results: Annotated[list[dict[str, Any]], add]
    answer: str
    sources: Annotated[list[dict[str, Any]], add]
    conversation_id: str
    error: str | None
    refresh_attempted: bool