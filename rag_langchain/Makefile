.PHONY: install dev-install ingest chat web init-sqlite init-postgres init-mongo clean

install:
	pip install -r requirements.txt

dev-install:
	pip install -e .

ingest:
	python -m rag_langchain.cli.index

chat:
	python -m rag_langchain.cli.chat

web:
	streamlit run rag_langchain/web/streamlit_app.py --server.port=8501

init-sqlite:
	python -m rag_langchain.scripts.init_sqlite_employees

init-postgres:
	python -m rag_langchain.scripts.init_postgres_purchases

init-mongo:
	python -m rag_langchain.scripts.init_mongo_services

clean:
	rm -rf __pycache__ */__pycache__ */*/__pycache__ .pytest_cache
	rm -rf data/chroma_db
