import sys
from pathlib import Path

# Streamlit sets sys.path[0] to the script's directory (rag_langchain/web/),
# so the package 'rag_langchain' itself is not importable from there.
# Prepending the project root to sys.path makes `import rag_langchain` work
# whether the app is launched via `streamlit run …`, `python app_streamlit.py`,
# or any other entry point that doesn't change cwd.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from rag_langchain.core.ingestion import index_files, get_vectorstore, get_embeddings, list_indexed_sources
from rag_langchain.core.rag_chain import (
    condense_question, retrieve_and_rerank, generate_answer_stream,
    classify_question, answer_from_history_stream,
)
from rag_langchain.core.command_parser import parse_user_input
from rag_langchain.config import settings

UPLOAD_DIR = settings.documents_dir
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

language = "fr"

st.set_page_config(page_title="RAG interactif", layout="centered")
st.title(" RAG interactif")
st.caption("Upload de documents · Reformulation · Reranking · Multi-BD (SQLite, PG, Mongo)")

# ------------------------------------------------------------------
# Chargement du vectorstore (mis en cache)
# ------------------------------------------------------------------
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# @Function Description: Ouvre la connexion à la base vectorielle Chroma,
# mise en cache par Streamlit pour n'être recréée qu'une seule fois.
# ------------------------------------------------------------------------------
# @Parameter:
#                 - (aucun)
# @Returnvalue:
#                 - Chroma - Instance connectée à "chroma_db/".
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
@st.cache_resource
def load_vectorstore():
    embeddings = get_embeddings()
    return get_vectorstore(embeddings)

vectorstore = load_vectorstore()


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# @Function Description: Liste les fichiers du dossier d'upload en
# excluant les placeholders Git (.gitkeep) et les fichiers temporaires
# Office (~$document.docx). Utilisé uniquement en fallback quand la base
# Chroma est encore vide (avant tout indexage), pour proposer à
# l'utilisateur les fichiers qu'il pourrait indexer depuis le disque.
# ------------------------------------------------------------------------------
# @Parameter:
#                 - (aucun)
# @Returnvalue:
#                 - list[Path] - Chemins de fichiers éligibles, triés.
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def list_filesystem_documents():
    def _eligible(p: Path) -> bool:
        name = p.name
        if name == ".gitkeep":
            return False
        if name.startswith("~$"):
            return False
        return p.is_file()

    return sorted(p for p in UPLOAD_DIR.glob("*") if _eligible(p))


# ------------------------------------------------------------------
# Sidebar : upload et indexation
# ------------------------------------------------------------------
with st.sidebar:
    st.header("📁 Documents")
    uploaded_files = st.file_uploader(
        "Ajoute tes documents (PDF, TXT, DOCX, PPTX)",
        type=["pdf", "txt", "docx", "pptx", "ppt"],
        accept_multiple_files=True,
        key="doc_uploader",
    )

    if uploaded_files and st.button("Indexer les documents", key="index_btn"):
        saved_paths = []
        for uf in uploaded_files:
            dest = UPLOAD_DIR / uf.name
            with open(dest, "wb") as f:
                f.write(uf.getbuffer())
            saved_paths.append(dest)

        status = st.empty()
        log_lines = []

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # @Function Description: Callback de progression passé à index_files().
        # Met à jour l'affichage Streamlit en direct pendant l'indexation.
        # ------------------------------------------------------------------------------
        # @Parameter:
        #                 - msg: str - Message de progression.
        # @Returnvalue:
        #                 - None
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        def progress(msg):
            log_lines.append(msg)
            status.text("\n".join(log_lines))

        # On capture toute exception pour qu'elle s'affiche dans l'UI au
        # lieu de faire s'évanouir l'interface (Streamlit rerun et cache
        # le traceback derrière un écran vide si l'exception n'est pas
        # attrapée dans le callback bouton).
        try:
            with st.spinner("Indexation en cours..."):
                total = index_files(saved_paths, progress_callback=progress, vectorstore=vectorstore)
        except Exception as e:
            st.error(f"❌ Erreur d'indexation : {type(e).__name__} : {e}")
            import traceback
            with st.expander("Traceback complet"):
                st.code(traceback.format_exc())
        else:
            st.success(f"✅ {total} chunks indexés.")
            # Invalide le cache de Chroma pour que les prochains reads voient
            # les nouveaux chunks, SANS forcer un rerun de l'UI (le rerun
            # naturel qui suit la fin du bouton suffit).
            load_vectorstore.clear()

    st.divider()

    # Source de vérité : Chroma. On ne lit le filesystem qu'en fallback
    # (collection vide avant tout indexage). On la mémorise dans
    # session_state pour stabiliser le multiselect entre les reruns.
    indexed_in_chroma = list_indexed_sources(vectorstore)
    if indexed_in_chroma:
        doc_list = indexed_in_chroma
        list_source = "Chroma"
    else:
        fs_docs = list_filesystem_documents()
        doc_list = [p.name for p in fs_docs]
        list_source = "dossier"

    st.caption(f"📄 {len(doc_list)} document(s) indexé(s) — source : {list_source}")
    for name in doc_list:
        st.caption(f"• {name}")

    st.divider()
    st.header(" Mode de recherche")
    mode_options = {
        "Auto (RAG)": "auto",
        "📄 Documents (RAG)": "docs",
        "🗄️ Employés (SQLite)": "sqlite",
        "🛒 Achats (Postgres)": "postgres",
        "🛠️ Services (Mongo)": "mongo",
        "🔗 Jointure multi-BD (Federated)": "federated"
    }
    selected_mode = st.selectbox(
        "Choisis la source de données :",
        list(mode_options.keys()),
        index=0,
        key="mode_select",
    )

    st.divider()
    st.header("Filtrer la recherche")
    selected_docs = st.multiselect(
        "Chercher uniquement dans (vide = tous) :",
        options=doc_list,
        key="doc_filter",
    )
    # On définit 'source_filter' AVANT de l'utiliser dans le code
    source_filter = selected_docs if selected_docs else None

try:
    chunk_count = vectorstore._collection.count()
except Exception:
    chunk_count = "?"

with st.sidebar:
    if chunk_count == 0:
        st.warning("⚠️ Aucun chunk indexé pour l'instant.")
    else:
        st.info(f"✅ {chunk_count} chunk(s) actuellement indexé(s).")

# ------------------------------------------------------------------
# État de la conversation
# ------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "history_pairs" not in st.session_state:
    st.session_state.history_pairs = []

for role, text in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(text)

# --- Aide à la saisie : palette de commandes ---
with st.container():
    st.caption("💡 Préfixes : `/employees` `/purchases` `/services` `/docs` `/all` · "
               "Références : `@employé:DUPONT` `@service:RH` `@doc:rapport.pdf`")
    user_input = st.chat_input("Pose ta question… (ou tape / pour voir les commandes)")

if user_input:
    parsed = parse_user_input(user_input)
    question = parsed.cleaned_question
    if user_input:
     parsed = parse_user_input(user_input)
    
    # Si l'utilisateur n'a pas tapé de / mais a choisi un mode dans la sidebar
    if parsed.command == "auto" and selected_mode != "auto":
        parsed.command = mode_options[selected_mode]
        
    question = parsed.cleaned_question
    # ... la suite du code reste exactement la même
    st.session_state.messages.append(("user", question))
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        history = st.session_state.history_pairs
        full_response = "" # Initialisation pour éviter les NameError à la fin
        sources = []

        # ==========================================================
        # BRANCHE 1 : COMMANDE FEDERATED (/all)
        # ==========================================================
        if parsed.command == "federated":
            st.caption("🔗 Réponse multi-sources (fédération)")
            with st.spinner("Planification et exécution des requêtes…"):
                from rag_langchain.connectors.federator import Federator
                fed = Federator(
                    sqlite_path=str(settings.sqlite_path),
                    pg_dsn=settings.pg_dsn,
                    mongo_uri=settings.mongo_uri,
                    mongo_db=settings.mongo_db
                )
                full_response, db_sources, plan = fed.answer(question, parsed.references, language)

            st.markdown(full_response)
            with st.expander("🧭 Plan d'exécution"):
                st.json(plan)
            with st.expander("🗄️ Sources base de données"):
                for r in db_sources:
                    st.caption(f"[{r.source_label}]")
                    st.code(r.native_query, language="json" if "Mongo" in r.source_label else "sql")
                    if r.rows:
                        st.dataframe(r.rows)

                # ==========================================================
        # BRANCHE 2 : COMMANDE SPECIFIQUE (/sql, /sqlite, /postgres, /mongo)
        # ==========================================================
        elif parsed.command in ["sqlite", "postgres", "mongo", "database"]:
            mode_label = "SQL (SQLite)" if parsed.command == "database" else parsed.command
            st.caption(f"🗄️ Réponse générée via {mode_label}")
            with st.spinner("Génération et exécution de la requête..."):
                # Si l'utilisateur tape /sql, on le redirige par défaut vers SQLite
                if parsed.command in ["sqlite", "database"]:
                    from rag_langchain.connectors.sqlite import SQLiteConnector
                    conn = SQLiteConnector(db_path=str(settings.sqlite_path))
                elif parsed.command == "postgres":
                    from rag_langchain.connectors.postgres import PostgresConnector
                    conn = PostgresConnector(dsn=settings.pg_dsn)
                else:
                    from rag_langchain.connectors.mongo import MongoConnector
                    conn = MongoConnector(uri=settings.mongo_uri, db_name=settings.mongo_db)

                result = conn.answer(question, parsed.references, language)

        # ==========================================================
        # BRANCHE 3 : ROUTAGE AUTOMATIQUE (Conversation, SQL auto, Docs)
        # ==========================================================
        else:
            route = classify_question(question, history, language)

            if route == "conversation":
                st.caption("💬 Réponse basée sur l'historique")
                stream = answer_from_history_stream(question, history, language)
                placeholder = st.empty()
                for chunk in stream:
                    full_response += chunk
                    placeholder.markdown(full_response + "▌")
                placeholder.markdown(full_response)

            elif route == "database":
                st.caption("🗄️ Réponse générée via SQLite (auto)")
                with st.spinner("Génération et exécution de la requête SQL..."):
                    from rag_langchain.connectors.sqlite import SQLiteConnector
                    conn = SQLiteConnector(db_path=str(settings.sqlite_path))
                    result = conn.answer(question, parsed.references, language)

                if result.safe:
                    full_response = "Requête exécutée avec succès."
                    if result.rows:
                        st.dataframe(result.rows)
                else:
                    full_response = f"❌ Erreur : {result.error}"
                st.markdown(full_response)
                with st.expander("🗄️ Détails de la requête SQL"):
                    st.code(result.native_query, language="sql")

            else:
                # Étape 1 : reformulation
                with st.spinner("Reformulation de la question..."):
                    standalone_q = condense_question(question, history, language)

                if standalone_q != question:
                    st.caption(f"🔄 Question reformulée : *{standalone_q}*")

                # Étape 2 : récupération + reranking
                with st.spinner("Recherche des passages pertinents..."):
                    chunks = retrieve_and_rerank(standalone_q, vectorstore, source_filter=source_filter)

                if not chunks:
                    full_response = "Je n'ai trouvé aucun passage pertinent dans les documents indexés."
                    st.markdown(full_response)
                else:
                    # Étape 3 : génération avec streaming
                    stream, sources = generate_answer_stream(standalone_q, chunks, language)
                    placeholder = st.empty()
                    for chunk in stream:
                        full_response += chunk
                        placeholder.markdown(full_response + "▌")
                    placeholder.markdown(full_response)

                    if sources:
                     import re
                    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                    # @Author:        John MANGA | Digit-Tech-Innov Solutions and Services
                    # @Creation:      20.07.2026
                    # ------------------------------------------------------------------------------
                    # @Function Description: Analyse la réponse générée par le LLM pour 
                    # extraire les numéros de citations [n] effectivement utilisés, afin 
                    # de n'afficher que les sources pertinentes dans l'interface.
                    # ------------------------------------------------------------------------------
                    # @Parameter:
                    #                 - (aucun, utilise les variables locales)
                    # @Returnvalue:
                    #                 - None - Met à jour l'interface Streamlit.
                    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                    cited_nums_str = re.findall(r'\[(\d+)\]', full_response)
                    cited_nums = sorted(list(set(int(n) for n in cited_nums_str if n.isdigit())))
                    
                    # Si le LLM n'a cité aucun numéro, on affiche toutes les sources par défaut
                    if not cited_nums:
                        cited_nums = list(range(1, len(sources) + 1))

                    st.write("**📚 Sources citées**")
                    cols = st.columns(len(cited_nums))
                    for col, i in zip(cols, cited_nums):
                        if 0 < i <= len(sources):
                            with col:
                                with st.popover(f"[{i}]"):
                                    st.caption(sources[i - 1])
                                    st.write(chunks[i - 1].page_content)
    st.session_state.messages.append(("assistant", full_response))
    st.session_state.history_pairs.append((question, full_response))
   