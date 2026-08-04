"""Page 0 — Gestion des documents (upload, indexation, suppression, aperçu).

Point d'entrée principal : on indexe d'abord, puis on chatte.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import streamlit as st

from rag_langchain.core.ingestion import (
    index_files,
    list_indexed_sources,
    get_vectorstore,
    purge_by_hash,
)
from rag_langchain.config.settings import settings as app_settings

st.set_page_config(page_title="Documents", page_icon="📄", layout="wide")

st.title("📄 Documents — Indexation & Gestion")

# --------------------------------------------------------------------- #
# Helpers UI
# --------------------------------------------------------------------- #
def _format_size(path: Path) -> str:
    try:
        size = path.stat().st_size
        for unit in ["o", "Ko", "Mo", "Go"]:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} To"
    except Exception:
        return "?"

def _file_type_emoji(suffix: str) -> str:
    return {
        ".pdf": "📕", ".docx": "📘", ".pptx": "📙", ".ppt": "📙",
        ".txt": "📝", ".md": "📝",
    }.get(suffix.lower(), "📄")

# --------------------------------------------------------------------- #
# Onglet Upload
# --------------------------------------------------------------------- #
tab_upload, tab_indexes, tab_status = st.tabs(
    ["📤 Upload & Indexer", "📚 Indexés", "⚙️ Statut"]
)

with tab_upload:
    st.subheader("Déposer des fichiers")
    st.caption("Formats : PDF, DOCX, PPTX, TXT, MD — max 50 Mo par fichier")

    uploaded = st.file_uploader(
        "Glissez-déposez ou cliquez",
        type=["pdf", "docx", "pptx", "ppt", "txt", "md"],
        accept_multiple_files=True,
        help="Plusieurs fichiers à la fois — l'indexation est séquentielle avec progression.",
    )

    if uploaded:
        # Sauvegarde temporaire dans data/documents/
        app_settings.documents_dir.mkdir(parents=True, exist_ok=True)
        saved_paths = []
        for up in uploaded:
            dest = app_settings.documents_dir / up.name
            dest.write_bytes(up.getbuffer())
            saved_paths.append(dest)

        st.info(f"{len(saved_paths)} fichier(s) prêt(s). Lancement de l'indexation…")

        # Zone de logs temps réel
        log_box = st.empty()
        logs = []

        def _progress_cb(msg: str):
            logs.append(msg)
            log_box.code("\n".join(logs[-30:]), language="text")

        with st.spinner("Indexation en cours…"):
            total = index_files(saved_paths, progress_callback=_progress_cb)

        st.success(f"✅ Terminé — {total} chunks indexés au total.")
        time.sleep(0.5)
        st.rerun()

# --------------------------------------------------------------------- #
# Onglet Documents Indexés
# --------------------------------------------------------------------- #
with tab_indexes:
    st.subheader("Documents présents dans Chroma")

    vs = get_vectorstore()
    sources = list_indexed_sources(vs)

    if not sources:
        st.info("Aucun document indexé. Utilisez l'onglet **Upload** pour commencer.")
    else:
        # Stats globales
        total_chunks = vs._collection.count()
        c1, c2 = st.columns(2)
        c1.metric("📄 Fichiers uniques", len(sources))
        c2.metric("🧩 Chunks totaux", total_chunks)

        st.divider()

        for src in sources:
            with st.expander(f"{_file_type_emoji(Path(src).suffix)} {src}", expanded=False):
                # Compter chunks pour ce fichier
                data = vs._collection.get(where={"source": src}, include=["metadatas", "documents"])
                n_chunks = len(data["ids"])
                st.caption(f"🧩 {n_chunks} chunks")

                # Aperçu du 1er chunk
                if data["documents"]:
                    preview = data["documents"][0][:500]
                    st.code(preview + ("…" if len(data["documents"][0]) > 500 else ""), language="text")

                cA, cB, cC = st.columns(3)
                # Réindexer
                if cA.button("🔄 Réindexer", key=f"reidx-{src}", use_container_width=True):
                    # Trouver le fichier source dans data/documents/
                    src_path = app_settings.documents_dir / src
                    if src_path.exists():
                        with st.spinner(f"Réindexation de {src}…"):
                            n = index_files([src_path])
                        st.success(f"✅ {src} réindexé ({n} chunks)")
                        st.rerun()
                    else:
                        st.error(f"Fichier source introuvable : {src_path}")

                # Supprimer
                if cB.button("🗑️ Supprimer", key=f"del-{src}", use_container_width=True):
                    # Supprimer de Chroma par source_hash
                    # On doit retrouver le hash via les métadonnées
                    meta = data["metadatas"][0] if data["metadatas"] else {}
                    file_hash = meta.get("source_hash")
                    if file_hash:
                        removed = purge_by_hash(vs, file_hash)
                        st.success(f"✅ {src} supprimé ({removed} chunks)")
                        # Supprimer aussi le fichier physique
                        src_path = app_settings.documents_dir / src
                        if src_path.exists():
                            src_path.unlink()
                        st.rerun()
                    else:
                        st.error("Impossible de retrouver l'empreinte du fichier.")

                # Télécharger (optionnel)
                src_path = app_settings.documents_dir / src
                if src_path.exists():
                    with open(src_path, "rb") as f:
                        cC.download_button(
                            "⬇️ Télécharger",
                            data=f.read(),
                            file_name=src,
                            mime="application/octet-stream",
                            use_container_width=True,
                        )

# --------------------------------------------------------------------- #
# Onglet Statut
# --------------------------------------------------------------------- #
with tab_status:
    st.subheader("État de la base vectorielle")

    vs = get_vectorstore()
    total = vs._collection.count()

    col1, col2, col3 = st.columns(3)
    col1.metric("Chunks indexés", total)
    col2.metric("Fichiers sources", len(list_indexed_sources(vs)))
    col3.metric("Répertoire Chroma", str(app_settings.chroma_dir))

    st.divider()
    st.markdown("**Configuration d'embedding**")
    st.json({
        "embedding_model": app_settings.embedding_model,
        "ollama_base_url": app_settings.ollama_base_url,
        "chunk_size": app_settings.chunk_size,
        "chunk_overlap": app_settings.chunk_overlap,
        "retrieve_k": app_settings.retrieve_k,
        "final_k": app_settings.final_k,
    })

    st.divider()
    st.markdown("**Répertoire documents (fichiers sources)**")
    docs_dir = app_settings.documents_dir
    if docs_dir.exists():
        files = list(docs_dir.iterdir())
        st.write(f"{len(files)} fichier(s) sur disque :")
        for f in files:
            st.write(f"  {_file_type_emoji(f.suffix)} {f.name} ({_format_size(f)})")
    else:
        st.warning(f"Dossier introuvable : {docs_dir}")

    st.divider()
    if st.button("🧹 Vider TOUT Chroma (irreversible)", type="secondary"):
        # Confirmation double-clic pattern
        if "confirm_purge" not in st.session_state:
            st.session_state.confirm_purge = True
            st.warning("⚠️ Cliquez une 2e fois pour confirmer la suppression totale.")
        else:
            try:
                vs._collection.delete(where={})
                # Supprimer aussi les fichiers sources
                for f in docs_dir.iterdir():
                    f.unlink()
                st.success("✅ Chroma vidé + fichiers sources supprimés.")
                st.session_state.confirm_purge = False
                time.sleep(0.5)
                st.rerun()
            except Exception as e:
                st.error(f"❌ {e}")