"""Node parse : nettoie la question et extrait les tags/groups/references."""
from __future__ import annotations

from rag_langchain.core.command_parser import parse_user_input


def parse_node(state: dict) -> dict:
    parsed = parse_user_input(state["question"])
    return {
        "cleaned_question": parsed.cleaned_question or parsed.raw,
        "groups": parsed.groups,
        "tags": parsed.tags,
        "references": parsed.references,
    }