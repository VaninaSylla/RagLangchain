# Makefile — convenience runners for *nix shells / Git-Bash
.PHONY: help install init-sqlite init-pg init-mongo index chat web test clean

help:
	@echo "Available targets:"
	@echo "  install      Install Python dependencies"
	@echo "  init-sqlite  Seed the SQLite employees DB"
	@echo "  init-pg      Seed the PostgreSQL purchases DB (server must be up)"
	@echo "  init-mongo   Seed the MongoDB services DB (server must be up)"
	@echo "  index        Index every file in data/documents/ into Chroma"
	@echo "  chat         Launch the CLI chat"
	@echo "  web          Launch the Streamlit web UI"
	@echo "  test         Run pytest"
	@echo "  clean        Remove __pycache__ folders"

install:
	pip install -r requirements.txt

init-sqlite:
	python -m rag_langchain.scripts.init_sqlite_employees

init-pg:
	python -m rag_langchain.scripts.init_postgres_purchases

init-mongo:
	python -m rag_langchain.scripts.init_mongo_services

index:
	python -m rag_langchain.cli.index

chat:
	python -m rag_langchain.cli.chat

web:
	streamlit run app_streamlit.py

test:
	pytest -q

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
