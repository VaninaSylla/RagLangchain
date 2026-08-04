"""Node db_agent : exécute les requêtes sur les connecteurs sélectionnés.

Comportement validé :
- Pour chaque profil du planner : tente la requête via le connecteur.
- Si **aucun** résultat (rows vide + pas d'erreur) ET mode auto_investigate :
  on tente **une seule fois** un refresh du snapshot puis on retente.
- Fallback legacy : si le meta-store est vide (aucun profil enregistré), on
  retombe sur le SQLite historique (settings.sqlite_path) pour ne pas perdre
  la rétro-compat avec la V1.
- Le résultat est stocké dans `db_results` du state.
"""
from __future__ import annotations

from rag_langchain.connectors.sql_base import SQLConnector
from ._helpers import (
    get_meta_store, get_registry, get_snapshot_manager, serialise_query_result,
)


def _legacy_sqlite_connector():
    """Construit un SQLiteConnector sur la base historique (V1 fallback)."""
    from rag_langchain.config import settings as app_settings
    from rag_langchain.connectors.sqlite import SQLiteConnector
    return SQLiteConnector(db_path=str(app_settings.sqlite_path), alias="sqlite_legacy")


def _run_on_connector(connector, cleaned, references) -> dict:
    try:
        qr = connector.answer(
            question=cleaned,
            references=references,
            language="fr",
        )
        return serialise_query_result(qr)
    except Exception as e:
        return {
            "source_label": getattr(connector, "source_label", "?"),
            "native_query": "", "columns": [], "rows": [],
            "safe": False, "risk": "blocked", "error": str(e),
        }


def db_agent_node(state: dict) -> dict:
    meta = get_meta_store()
    registry = get_registry()
    snapshots = get_snapshot_manager()

    cleaned = state.get("cleaned_question") or state["question"]
    references = state.get("references", {})
    selected = list(state.get("selected_connectors", []))
    results: list[dict] = []
    refresh_done = False

    # Fallback legacy : meta-store vide → SQLite historique
    if not selected and state.get("route") in ("auto_investigate", "database"):
        try:
            connector = _legacy_sqlite_connector()
            results.append(_run_on_connector(connector, cleaned, references))
        except Exception as e:
            results.append({
                "source_label": "sqlite_legacy",
                "native_query": "", "columns": [], "rows": [],
                "safe": False, "risk": "blocked",
                "error": f"legacy fallback: {e}",
            })

    for pid in selected:
        profile = meta.get(pid)
        if profile is None:
            continue
        snap = snapshots.get_or_refresh(profile)
        connector = registry.build(profile)

        # On injecte le schéma depuis le snapshot pour économiser un appel BD
        try:
            if isinstance(connector, SQLConnector):
                connector.get_schema_text = lambda text=snap.schema_text: text  # type: ignore[assignment]
        except Exception:
            pass

        # 1ʳᵉ tentative
        result_dict = _run_on_connector(connector, cleaned, references)

        # Auto-investigate : refresh unique si pas de résultat
        empty_result = (
            result_dict.get("safe") and not result_dict.get("rows")
            and not result_dict.get("error")
        )
        if (
            empty_result
            and state.get("route") == "auto_investigate"
            and not refresh_done
        ):
            snap = snapshots.refresh(profile)
            try:
                if isinstance(connector, SQLConnector):
                    connector.get_schema_text = lambda text=snap.schema_text: text  # type: ignore[assignment]
            except Exception:
                pass
            result_dict = _run_on_connector(connector, cleaned, references)
            refresh_done = True

        results.append(result_dict)

    return {"db_results": results, "refresh_attempted": refresh_done}