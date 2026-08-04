"""Mémoire globale persistante et vectorisée via ChromaDB.

Chaque message est indexé avec ses métadonnées (conversation_id, role, route,
sources, timestamp). Permet :
- historique chronologique d'une conversation (`get_recent`)
- recherche sémantique dans toutes les conversations (`semantic_search`)
- listing de toutes les conversations (`list_conversations`)

La collection est partagée entre les agents LangGraph et les pages Streamlit.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config.settings import settings as app_settings
from ..core.ingestion import get_embeddings


@dataclass
class MemoryMessage:
    id: str
    conversation_id: str
    role: str                # "user" | "assistant" | "system"
    content: str
    route: str = ""          # "conversation" | "documents" | "database" | "mixed" | "auto_investigate"
    sources: list[dict] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass
class Conversation:
    id: str
    title: str
    started_at: float
    last_activity: float
    message_count: int = 0


class MemoryStore:
    """Couche mémoire globale, vectorisée via ChromaDB."""

    def __init__(
        self,
        chroma_dir: Path | str | None = None,
        collection_name: str | None = None,
        embedding_model: str | None = None,
    ):
        self.chroma_dir = Path(chroma_dir or app_settings.memory_chroma_dir)
        base_collection_name = collection_name or app_settings.memory_collection_name
        self.chroma_dir.mkdir(parents=True, exist_ok=True)

        from langchain_chroma import Chroma
        from langchain_core.documents import Document

        self._Chroma = Chroma
        self._Document = Document
        self._embeddings = get_embeddings()

        # Détection d'incompatibilité de dimension : si la collection existe déjà
        # avec une dimension différente (modèle d'embedding changé OU collection
        # initialisée avec DefaultEmbeddingFunction), on la supprime et on repart
        # propre — sinon upsert lèvera InvalidArgumentError (768 vs 384).
        expected_dim = self._probe_embedding_dim()
        actual_dim = self._probe_collection_dim(base_collection_name)
        if actual_dim is not None and actual_dim != expected_dim:
            self._safe_delete_collection(base_collection_name)
        self.collection_name = base_collection_name

        self._vs = Chroma(
            persist_directory=str(self.chroma_dir),
            embedding_function=self._embeddings,
            collection_name=self.collection_name,
        )

    def _probe_embedding_dim(self) -> int:
        """Calcule la dimension de l'embedding via un embedding test."""
        try:
            vec = self._embeddings.embed_query("dimension probe")
            return len(vec)
        except Exception:
            # Fallback conservateur pour nomic-embed-text
            return 768

    def _probe_collection_dim(self, name: str) -> int | None:
        """Retourne la dimension d'une collection existante, ou None.

        chromadb ≥0.4 n'expose plus `Collection.dimension` (supprimé) : on lit
        la dimension via un peek() sur les embeddings des documents existants.
        """
        try:
            client = self._Chroma(
                persist_directory=str(self.chroma_dir),
                embedding_function=self._embeddings,
                collection_name=name,
            )._client
            col = client.get_collection(name)
            peeked = col.peek(limit=1)
            embeddings = peeked.get("embeddings")
            if embeddings is None:
                return None
            import numpy as np
            arr = np.asarray(embeddings)
            return int(arr.shape[1]) if arr.ndim == 2 else None
        except Exception:
            return None

    def _safe_delete_collection(self, name: str) -> None:
        """Supprime une collection obsolète (best-effort)."""
        try:
            client = self._Chroma(
                persist_directory=str(self.chroma_dir),
                embedding_function=self._embeddings,
                collection_name=name,
            )._client
            client.delete_collection(name)
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Conversations                                                       #
    # ------------------------------------------------------------------ #
    def start_conversation(self, title: str = "Nouvelle conversation") -> str:
        cid = str(uuid.uuid4())
        # Stocker une "ancre" via un message système
        meta = {
            "conversation_id": cid,
            "title": title,
            "started_at": time.time(),
            "last_activity": time.time(),
            "kind": "conversation_anchor",
        }
        doc = self._Document(page_content=f"[conversation] {title}", metadata=meta)
        self._vs.add_documents([doc], ids=[f"anchor::{cid}"])
        return cid

    def list_conversations(self) -> list[Conversation]:
        """Liste toutes les conversations, triées par dernière activité décroissante."""
        # On interroge Chroma pour tous les documents avec kind='conversation_anchor'
        data = self._vs._collection.get(where={"kind": "conversation_anchor"})
        convs: list[Conversation] = []
        for i, _id in enumerate(data["ids"]):
            meta = data["metadatas"][i]
            cid = meta["conversation_id"]
            convs.append(Conversation(
                id=cid,
                title=meta.get("title", "(sans titre)"),
                started_at=float(meta.get("started_at", 0)),
                last_activity=float(meta.get("last_activity", 0)),
                message_count=self._count_messages(cid),
            ))
        convs.sort(key=lambda c: c.last_activity, reverse=True)
        return convs

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        for c in self.list_conversations():
            if c.id == conversation_id:
                return c
        return None

    # ------------------------------------------------------------------ #
    # Messages                                                            #
    # ------------------------------------------------------------------ #
    def append_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        *,
        route: str = "",
        sources: list[dict] | None = None,
    ) -> MemoryMessage:
        msg = MemoryMessage(
            id=str(uuid.uuid4()),
            conversation_id=conversation_id,
            role=role,
            content=content,
            route=route,
            sources=sources or [],
            timestamp=time.time(),
        )
        meta = {
            "conversation_id": conversation_id,
            "role": role,
            "route": route,
            "sources_json": json.dumps(msg.sources, ensure_ascii=False, default=str),
            "timestamp": msg.timestamp,
            "msg_id": msg.id,
            "kind": "message",
        }
        doc = self._Document(page_content=content, metadata=meta)
        self._vs.add_documents([doc], ids=[f"msg::{msg.id}"])
        # Met à jour last_activity de l'ancre
        self._update_anchor_activity(conversation_id)
        return msg

    def get_recent(self, conversation_id: str, limit: int = 20) -> list[MemoryMessage]:
        """Retourne les N derniers messages d'une conversation, en ordre chronologique."""
        data = self._vs._collection.get(
            where={"$and": [{"kind": "message"}, {"conversation_id": conversation_id}]}
        )
        msgs: list[MemoryMessage] = []
        for i, _id in enumerate(data["ids"]):
            meta = data["metadatas"][i]
            sources = json.loads(meta.get("sources_json", "[]"))
            msgs.append(MemoryMessage(
                id=meta.get("msg_id", _id),
                conversation_id=meta["conversation_id"],
                role=meta["role"],
                content=data["documents"][i],
                route=meta.get("route", ""),
                sources=sources,
                timestamp=float(meta.get("timestamp", 0)),
            ))
        msgs.sort(key=lambda m: m.timestamp)
        return msgs[-limit:]

    def semantic_search(
        self,
        query: str,
        *,
        k: int = 5,
        conversation_id: str | None = None,
    ) -> list[tuple[MemoryMessage, float]]:
        """Recherche sémantique dans l'historique (toutes conversations ou une seule)."""
        if conversation_id:
            where = {"$and": [{"kind": "message"}, {"conversation_id": conversation_id}]}
        else:
            where = {"kind": "message"}
        retriever = self._vs.as_retriever(search_kwargs={"k": k, "filter": where})
        docs = retriever.invoke(query)
        out: list[tuple[MemoryMessage, float]] = []
        for d in docs:
            meta = d.metadata
            sources = json.loads(meta.get("sources_json", "[]"))
            msg = MemoryMessage(
                id=meta.get("msg_id", ""),
                conversation_id=meta.get("conversation_id", ""),
                role=meta.get("role", ""),
                content=d.page_content,
                route=meta.get("route", ""),
                sources=sources,
                timestamp=float(meta.get("timestamp", 0)),
            )
            out.append((msg, 0.0))  # score non retourné par as_retriever.invoke
        return out

    # ------------------------------------------------------------------ #
    # Internals                                                           #
    # ------------------------------------------------------------------ #
    def _update_anchor_activity(self, conversation_id: str) -> None:
        # Met à jour last_activity et incrémente message_count via metadata
        anchor_id = f"anchor::{conversation_id}"
        existing = self._vs._collection.get(ids=[anchor_id])
        if not existing["ids"]:
            return
        meta = existing["metadatas"][0]
        meta["last_activity"] = time.time()
        # On doit fournir l'embedding explicitement : _collection.upsert lit
        # autrement l'embedding par défaut interne de chromadb (DefaultEmbeddingFunction,
        # 384 dim), ce qui provoque InvalidArgumentError si la collection attend 768.
        page_content = f"[conversation] {meta.get('title','')}"
        embedding = self._embeddings.embed_query(page_content)
        self._vs._collection.upsert(
            ids=[anchor_id],
            embeddings=[embedding],
            documents=[page_content],
            metadatas=[meta],
        )

    def _count_messages(self, conversation_id: str) -> int:
        data = self._vs._collection.get(
            where={"$and": [{"kind": "message"}, {"conversation_id": conversation_id}]}
        )
        return len(data["ids"])


__all__ = ["MemoryStore", "MemoryMessage", "Conversation"]