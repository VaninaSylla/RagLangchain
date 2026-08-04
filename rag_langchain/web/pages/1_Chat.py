"""Page 1 — Chat RAG multi-agents.

Pose une question, le LangGraph orchestrateur route entre docs / DB / mémoire.
"""
from __future__ import annotations

import streamlit as st

from rag_langchain.core.command_parser import parse_user_input
from rag_langchain.core.ingestion import get_vectorstore, list_indexed_sources
from rag_langchain.agents import run_question
from rag_langchain.memory import MemoryStore
from rag_langchain.agents.nodes._helpers import reset_caches


st.title("💬 Chat RAG multi-agents")

# Sidebar : métriques + filtres
with st.sidebar:
    st.subheader("📊 Contexte")
    vs = get_vectorstore()
    sources = list_indexed_sources(vs)
    st.metric("📄 Docs indexés", len(sources))
    st.metric("🧩 Chunks", vs._collection.count())
    
    source_filter = st.multiselect(
        "Filtrer par source",
        options=sources,
        default=[],
        help="Laissez vide pour chercher dans tous les documents."
    )
    
    st.divider()
    st.caption("Utilisez #groupe @tag pour cibler les bases.")

if "conv_id" not in st.session_state:
    st.session_state.conv_id = ""
if "messages" not in st.session_state:
    st.session_state.messages = []

memory = MemoryStore()

# Init / reprise de conversation
if not st.session_state.conv_id:
    convs = memory.list_conversations()
    if convs:
        options = ["— Nouvelle conversation —"] + [f"{c.title[:30]} ({c.id[:8]})" for c in convs]
        choice = st.selectbox("Conversation", options)
        if choice.startswith("— Nouvelle"):
            st.session_state.conv_id = memory.start_conversation("Nouvelle conversation")
        else:
            cid = convs[options.index(choice) - 1].id
            st.session_state.conv_id = cid
            # Recharger l'historique visuel
            for m in memory.get_recent(cid, limit=50):
                st.session_state.messages.append((m.role, m.content))
    else:
        st.session_state.conv_id = memory.start_conversation("Nouvelle conversation")

# Affichage
for role, text in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(text)

# Input
question = st.chat_input("Pose ta question… (utilise #groupe @tag)")
if question:
    parsed = parse_user_input(question)
    st.session_state.messages.append(("user", question))
    with st.chat_message("user"):
        st.markdown(question)

    memory.append_message(st.session_state.conv_id, "user", question,
                          route="user_input", sources=[])

    with st.chat_message("assistant"):
        with st.spinner("Orchestration LangGraph…"):
            try:
                reset_caches()
                final = run_question(
                    question, 
                    conversation_id=st.session_state.conv_id,
                    source_filter=source_filter if source_filter else None
                )
                answer = final.get("answer", "(aucune réponse)")
                route = final.get("route", "?")
                st.markdown(answer)
                with st.expander(f"🧭 Route: {route}"):
                    st.json({
                        "groups": final.get("groups", []),
                        "tags": final.get("tags", []),
                        "selected_connectors": final.get("selected_connectors", []),
                        "refresh_attempted": final.get("refresh_attempted", False),
                        "sources": final.get("sources", []),
                    })
            except Exception as e:
                answer = f"❌ Erreur : {e}"
                st.markdown(answer)

    st.session_state.messages.append(("assistant", answer))
    memory.append_message(st.session_state.conv_id, "assistant", answer,
                          route=final.get("route", "") if 'final' in dir() else "",
                          sources=final.get("sources", []) if 'final' in dir() else [])