"""Node planner : résout les #group / @tag en profils concrets.

Lève `AmbiguousTargetError` si un tag/groupe matche plusieurs profils
(comportement validé avec l'utilisateur).
"""
from __future__ import annotations

from ._helpers import get_meta_store


def planner_node(state: dict) -> dict:
    meta = get_meta_store()
    profiles = []
    seen: set[str] = set()

    for tag in state.get("tags", []):
        try:
            p = meta.resolve_tag(tag)
            if p.id not in seen:
                profiles.append(p)
                seen.add(p.id)
        except KeyError:
            # tag inconnu → on ignore silencieusement
            pass

    for grp in state.get("groups", []):
        try:
            p = meta.resolve_group(grp)
            if p.id not in seen:
                profiles.append(p)
                seen.add(p.id)
        except KeyError:
            pass

    # Mode auto_investigate : on interroge tout
    if state.get("route") == "auto_investigate":
        for p in meta.list_all():
            if p.id not in seen:
                profiles.append(p)
                seen.add(p.id)

    return {"selected_connectors": [p.id for p in profiles]}