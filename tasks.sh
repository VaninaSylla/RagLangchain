#!/usr/bin/env bash
# tasks.sh — convenience runners for POSIX shells (Linux/macOS) and Git-Bash (Windows)
# Usage:  ./tasks.sh install   |   ./tasks.sh web   |   ./tasks.sh test ...

set -euo pipefail

TASK="${1:-help}"

run() {
    echo ">>> $*"
    "$@"
}

case "$TASK" in
    install)     run pip install -r requirements.txt ;;
    init-sqlite) run python -m rag_langchain.scripts.init_sqlite_employees ;;
    init-pg)     run python -m rag_langchain.scripts.init_postgres_purchases ;;
    init-mongo)  run python -m rag_langchain.scripts.init_mongo_services ;;
    index)       run python -m rag_langchain.cli.index ;;
    chat)        run python -m rag_langchain.cli.chat ;;
    web)         run python -m streamlit run app_streamlit.py ;;
    test)        run pytest -q ;;
    clean)
        find . -type d -name "__pycache__" -exec rm -rf {} +
        echo "Cleaned __pycache__ folders."
        ;;
    help)
        echo "Available tasks:"
        echo "  install      Install Python dependencies"
        echo "  init-sqlite  Seed the SQLite employees DB"
        echo "  init-pg      Seed the PostgreSQL purchases DB (server must be up)"
        echo "  init-mongo   Seed the MongoDB services DB (server must be up)"
        echo "  index        Index every file in data/documents/ into Chroma"
        echo "  chat         Launch the CLI chat"
        echo "  web          Launch the Streamlit web UI"
        echo "  test         Run pytest"
        echo "  clean        Remove __pycache__ folders"
        ;;
    *)
        echo "Unknown task: $TASK" >&2
        echo "Run ./tasks.sh help for the list of available tasks." >&2
        exit 1
        ;;
esac
