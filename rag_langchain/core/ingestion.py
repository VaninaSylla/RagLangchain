
import os
import sys
import hashlib
from pathlib import Path
from typing import Iterable

from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from rag_langchain.config import settings

# Configuration pour le découpage par titres Markdown
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

HEADERS_TO_SPLIT_ON = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]


def load_single_file(file_path: Path):
    """
    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @Function Description: Charge un fichier unique (PDF, TXT, DOCX ou PPTX) et le
    convertit en une liste de Documents LangChain. Choisit automatiquement le
    loader adapté selon l'extension, et normalise les métadonnées (nom de
    fichier source, numéro de page) pour un affichage propre dans les
    citations générées plus tard par le RAG.
    --------------------------------------------------------------------------
    @Parameter:
        - file_path: Path - Chemin vers le fichier à charger (.pdf, .txt,
          .docx ou .pptx).

    @Returnvalue:
        - list[Document] - Un document LangChain par page (PDF) ou par
          fichier (TXT/DOCX/PPTX), chacun avec metadata["source"] (nom de fichier)
          et metadata["page"] (numéro de page, si applicable, indexé à partir
          de 1).
    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    suffix = file_path.suffix.lower()

    # Imports différés : évite de charger 'unstructured' (lourd, utilisé uniquement
    # pour les .docx/.pptx) au démarrage de l'application si on ne traite que des PDF/TXT.
    if suffix == ".pdf":
        from langchain_community.document_loaders import PyMuPDFLoader
        # PyMuPDFLoader (basé sur la librairie 'fitz') gère mieux la mise en
        # page que PyPDFLoader : les colonnes, les formules simples et les
        # symboles sont moins souvent cassés. Il conserve aussi le numéro
        # de page exact dans metadata["page"], indispensable pour les
        # citations précises demandées.
        loader = PyMuPDFLoader(str(file_path))
    elif suffix == ".txt":
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader(str(file_path), encoding="utf-8")
    elif suffix == ".docx":
        from langchain_community.document_loaders import UnstructuredWordDocumentLoader
        loader = UnstructuredWordDocumentLoader(str(file_path))
    elif suffix == ".pptx" or suffix == ".ppt":
        from langchain_community.document_loaders import UnstructuredPowerPointLoader
        loader = UnstructuredPowerPointLoader(str(file_path))
    else:
        raise ValueError(f"Format non supporté : {suffix}")

    docs = loader.load()

    # Uniformise le nom de la source (juste le nom de fichier, pas le chemin
    # complet) pour un affichage propre dans les citations.
    for d in docs:
        d.metadata["source"] = file_path.name
        # PyMuPDFLoader indexe les pages à partir de 0 -> on affiche depuis 1
        if "page" in d.metadata:
            d.metadata["page"] = d.metadata["page"] + 1

    return docs


def split_documents(raw_docs):
    """
    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @Function Description: Découpe une liste de Documents LangChain en chunks
    en utilisant une approche hybride. Tente d'abord un découpage sémantique
    par titres Markdown (si présents), puis applique un découpage récursif
    classique pour garantir que les chunks ne dépassent pas CHUNK_SIZE. Les
    métadonnées d'origine (source, page, titre) sont conservées.
    --------------------------------------------------------------------------
    @Parameter:
        - raw_docs: list[Document] - Documents bruts (sortie de
          load_single_file), un par page ou par fichier.

    @Returnvalue:
        - list[Document] - Chunks de CHUNK_SIZE caractères (1000), avec un
          chevauchement de CHUNK_OVERLAP caractères (150) entre chunks
          consécutifs. Les métadonnées d'origine (source, page) sont
          conservées sur chaque chunk.
    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=False  # On garde le titre dans le texte pour le contexte du LLM
    )
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    
    final_chunks = []
    
    for doc in raw_docs:
        # Étape Markdown : Si le texte n'a pas de #, ça renvoie juste le doc entier.
        try:
            md_chunks = md_splitter.split_text(doc.page_content)
        except Exception:
            md_chunks = [doc]
            
        # Étape Récursive : On découpe les sections pour respecter la taille max
        chunks = splitter.split_documents(md_chunks)
        
        # Préservation des métadonnées d'origine (source, page)
        for chunk in chunks:
            chunk.metadata.update(doc.metadata)
            final_chunks.append(chunk)
            
    return final_chunks


def get_embeddings():
    """
    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @Function Description: Instancie le modèle d'embedding utilisé pour
    vectoriser les chunks de texte et les questions posées, via Ollama en
    local (aucun appel cloud). On cible explicitement l'API publique
    d'Ollama (http://localhost:11434) plutôt que de laisser la lib deviner
    le port d'un llama-server interne qui change à chaque chargement de
    modèle — c'est la cause principale des erreurs
    "connectex: No connection could be made" (status 400) vues sur
    l'endpoint /tokenize pendant l'indexation d'un PDF.
    --------------------------------------------------------------------------
    @Parameter:
        - (aucun)

    @Returnvalue:
        - OllamaEmbeddings - Instance configurée sur le modèle
          EMBEDDING_MODEL ("nomic-embed-text") et l'URL Ollama explicite.
    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    return OllamaEmbeddings(
        model=settings.embedding_model,
        base_url=settings.ollama_base_url,
        client_kwargs={"timeout": 120.0},
    )


def get_vectorstore(embeddings=None):
    """
    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @Function Description: Ouvre la base vectorielle Chroma persistée sur
    disque (dossier PERSIST_DIR), en la créant si elle n'existe pas encore.
    Utilisée aussi bien pendant l'indexation (écriture) que pendant
    l'interrogation (lecture) : c'est le point de jonction entre ces deux
    phases du pipeline.
    --------------------------------------------------------------------------
    @Parameter:
        - embeddings: OllamaEmbeddings | None - Modèle d'embedding à utiliser.
          Si None, get_embeddings() est appelé automatiquement.

    @Returnvalue:
        - Chroma - Instance connectée à la base persistée dans "chroma_db/".
    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    if embeddings is None:
        embeddings = get_embeddings()
    return Chroma(persist_directory=str(settings.chroma_dir), embedding_function=embeddings)


def _file_sha256(file_path: Path) -> str:
    """
    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @Function Description: Calcule l'empreinte SHA-256 d'un fichier en
    lisant par chunks de 1 MB (évite de tout charger en RAM pour un gros
    PDF). Cette empreinte sert de clé d'identification unique pour
    détecter une ré-indexation et purger l'ancienne version dans Chroma.
    --------------------------------------------------------------------------
    @Parameter:
        - file_path: Path - Chemin vers le fichier à hacher.

    @Returnvalue:
        - str - Hex digest SHA-256 (64 caractères).
    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    h = hashlib.sha256()
    with open(file_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def purge_by_hash(vectorstore, file_hash: str):
    """
    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @Function Description: Supprime tous les chunks de Chroma associés à
    une empreinte SHA-256 donnée (i.e. tous les chunks issus d'une
    version précédente du même fichier). On utilise l'API de filtre Chroma
    via le store, qui route vers le moteur sous-jacent (duckdb+parquet) et
    fonctionne de la même façon que ChromaDB en mode HTTP.
    --------------------------------------------------------------------------
    @Parameter:
        - vectorstore: Chroma - Instance cible.
        - file_hash: str - Hex digest SHA-256 (metadata["source_hash"]).

    @Returnvalue:
        - int - Nombre approximatif de chunks supprimés (Chroma ne
          retourne pas toujours le compte exact sur toutes les versions,
          on tolère ce flou et on déduit depuis le delete si possible).
    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    try:
        # Tente de compter avant suppression (utile pour le progress log)
        try:
            existing = vectorstore._collection.get(where={"source_hash": file_hash})
            removed = len(existing.get("ids", []) or [])
        except Exception:
            removed = 0
        vectorstore.delete(where={"source_hash": file_hash})
        return removed
    except Exception:
        return 0


def list_indexed_sources(vectorstore):
    """
    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @Function Description: Retourne la liste triée et dédupliquée des noms
    de fichiers réellement indexés dans Chroma (d'après metadata["source"]).
    C'est la source de vérité pour la sidebar Streamlit : on n'affiche
    plus jamais un fichier du filesystem qui n'est pas dans la base,
    et inversement. Robuste face à un Chroma verrouille (lock SQLite WAL)
    juste après écriture : retombe sur un set vide plutôt que de lever.
    --------------------------------------------------------------------------
    @Parameter:
        - vectorstore: Chroma - Instance cible.

    @Returnvalue:
        - list[str] - Noms de fichiers uniques, triés alphabétiquement.
    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    try:
        try:
            data = vectorstore._collection.get(include=["metadatas"], limit=100000)
            metadatas = data.get("metadatas") or []
        except Exception:
            return []

        sources = set()
        for m in metadatas:
            if isinstance(m, dict):
                src = m.get("source")
                if isinstance(src, str) and src:
                    sources.add(src)

        return sorted(sources)
    except Exception:
        return []


def _safe_progress_callback(progress_callback):
    """
    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @Function Description: Wrap a progress callback so that non-UTF-8 consoles
    (Windows cp1252 PowerShell) don't crash on the ✅ / ❌ emojis. Streamlit's
    own `st.text` already accepts Unicode and bypasses this wrapper; the
    shim only affects callbacks whose output is a real terminal/pipe.
    # ------------------------------------------------------------------------------
    # @Parameter:
    #                 - progress_callback: Callable[[str], None] - Original callback.
    # @Returnvalue:
    #                 - Callable[[str], None] - Safe variant that drops/replaces
    #                   characters the current stdout codec can't encode.
    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    if progress_callback is None:
        return None

    def safe_cb(message: str) -> None:
        try:
            progress_callback(message)
        except UnicodeEncodeError:
            encoding = (getattr(sys.stdout, "encoding", None) or "ascii").lower()
            replacement = "?" if encoding.startswith("ascii") else ""
            sanitized = message.encode(encoding, errors="replace").decode(encoding, errors="replace")
            if replacement:
                sanitized = sanitized.replace("\u2705", "[OK]").replace("\u274c", "[ERR]")
            progress_callback(sanitized)

    return safe_cb


def index_files(file_paths, progress_callback=None, vectorstore=None):
    """
    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @Function Description: Fonction "chef d'orchestre" de l'indexation :
    pour chaque fichier fourni, charge son contenu, le découpe en chunks, et
    l'ajoute à la base Chroma. Appelée aussi bien par ingest.py (indexation
    en lot en ligne de commande) que par le bouton "Indexer" de
    app_streamlit.py (upload web) — la logique n'est écrite qu'une seule fois.
    --------------------------------------------------------------------------
    @Parameter:
        - file_paths: list[Path | str] - Chemins des fichiers à indexer.
        - progress_callback: Callable[[str], None] | None - Fonction optionnelle
          appelée avec un message texte à chaque étape (ex: print en CLI,
          affichage st.text en Streamlit), pour suivre la progression.
        - vectorstore: Chroma | None - Instance existante. Si None, en crée une.

    @Returnvalue:
        - int - Nombre total de chunks indexés, tous fichiers confondus. Les
          fichiers en erreur sont signalés via progress_callback mais
          n'interrompent pas le traitement des fichiers suivants.
    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    if vectorstore is None:
        embeddings = get_embeddings()
        vectorstore = get_vectorstore(embeddings)

    safe_progress = _safe_progress_callback(progress_callback)

    total_chunks = 0
    for file_path in file_paths:
        file_path = Path(file_path)
        try:
            if safe_progress:
                safe_progress(f"Chargement de {file_path.name}...")

            # Calcule l'empreinte du fichier pour la dédup. On le fait
            # avant le split : un fichier identique (même contenu, même
            # nom) sera détecté et remplacé, pas dupliqué.
            try:
                file_hash = _file_sha256(file_path)
            except Exception as e:
                if safe_progress:
                    safe_progress(f"❌ Erreur sur {file_path.name} : {e}")
                continue

            raw_docs = load_single_file(file_path)
            chunks = split_documents(raw_docs)

            # Tag chaque chunk avec le hash AVANT écriture, puis purge
            # toute ancienne version portant ce même hash. Conséquence :
            # réindexer un PDF ne crée aucun doublon, ça remplace.
            for chunk in chunks:
                chunk.metadata["source_hash"] = file_hash

            removed = purge_by_hash(vectorstore, file_hash)

            vectorstore.add_documents(chunks)
            total_chunks += len(chunks)
            extra = f" (remplace {removed} ancien(s) chunk(s))" if removed else ""
            if safe_progress:
                safe_progress(f"✅ {file_path.name} indexé ({len(chunks)} chunks){extra}")
        except Exception as e:
            if safe_progress:
                safe_progress(f"❌ Erreur sur {file_path.name} : {e}")

    return total_chunks
