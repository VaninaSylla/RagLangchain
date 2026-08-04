"""Helpers partagés entre les nodes du LangGraph."""
from __future__ import annotations

import json
from functools import lru_cache

from rag_langchain.config.settings import settings as app_settings
from rag_langchain.connectors.meta_store import ConnectorMetaStore
from rag_langchain.connectors.registry import ConnectorRegistry
from rag_langchain.connectors.snapshot import SnapshotManager


@lru_cache(maxsize=1)
def get_meta_store() -> ConnectorMetaStore:
    return ConnectorMetaStore(app_settings.connectors_db_path)


@lru_cache(maxsize=1)
def get_registry() -> ConnectorRegistry:
    return ConnectorRegistry(get_meta_store())


@lru_cache(maxsize=1)
def get_snapshot_manager() -> SnapshotManager:
    return SnapshotManager(get_registry(), app_settings.snapshots_dir)


def reset_caches() -> None:
    get_meta_store.cache_clear()
    get_registry.cache_clear()
    get_snapshot_manager.cache_clear()


def serialise_query_result(result) -> dict:
    """Convertit un QueryResult en dict (pour stockage dans AgentState)."""
    return {
        "source_label": result.source_label,
        "native_query": result.native_query,
        "columns": result.columns,
        "rows": result.rows,
        "safe": result.safe,
        "risk": result.risk,
        "error": result.error,
    }


def safe_json_dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)