"""
ingest.py
---------
CLI entry point: indexes all documents from the configured data/documents/
folder into the Chroma vector store.
"""

from rag_langchain.core.ingestion import index_files
from rag_langchain.config import settings

SUPPORTED = {".pdf", ".txt", ".docx", ".pptx", ".ppt"}


def main():
    docs_dir = settings.documents_dir
    if not docs_dir.exists():
        print(f"Folder '{docs_dir}' does not exist. Create it and place your files there.")
        return

    files = [f for f in docs_dir.rglob("*") if f.is_file() and f.suffix.lower() in SUPPORTED]

    if not files:
        print("No PDF/TXT/DOCX/PPTX files found in 'data/documents/'.")
        return

    print(f"{len(files)} file(s) found. Indexing in progress...\n")
    total = index_files(files, progress_callback=print)
    print(f"\nDone. {total} chunks indexed in 'data/chroma_db/'.")


if __name__ == "__main__":
    main()