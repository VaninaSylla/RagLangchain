
import os
from pathlib import Path

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
    local (aucun appel cloud).
    --------------------------------------------------------------------------
    @Parameter:
        - (aucun)

    @Returnvalue:
        - OllamaEmbeddings - Instance configurée sur le modèle
          EMBEDDING_MODEL ("nomic-embed-text").
    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    return OllamaEmbeddings(model=settings.embedding_model)


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

    total_chunks = 0
    for file_path in file_paths:
        file_path = Path(file_path)
        try:
            if progress_callback:
                progress_callback(f"Chargement de {file_path.name}...")
            raw_docs = load_single_file(file_path)
            chunks = split_documents(raw_docs)
            vectorstore.add_documents(chunks)
            total_chunks += len(chunks)
            if progress_callback:
                progress_callback(f"✅ {file_path.name} indexé ({len(chunks)} chunks)")
        except Exception as e:
            if progress_callback:
                progress_callback(f"❌ Erreur sur {file_path.name} : {e}")

    return total_chunks
