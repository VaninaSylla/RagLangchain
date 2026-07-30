from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parent.parent / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # App
    app_name: str = "RAG LangChain"
    app_env: str = "development"
    log_level: str = "INFO"

    # Paths — base_dir is the repository root (one level above the package),
    # so that data/documents, data/sqlite, and data/chroma_db live at the
    # project root by default.
    base_dir: Path = Path(__file__).resolve().parent.parent.parent
    data_dir: Path = Path(__file__).resolve().parent.parent.parent / "data"
    documents_dir: Path = Path(__file__).resolve().parent.parent.parent / "data" / "documents"
    sqlite_dir: Path = Path(__file__).resolve().parent.parent.parent / "data" / "sqlite"
    chroma_dir: Path = Path(__file__).resolve().parent.parent.parent / "data" / "chroma_db"

    # SQLite
    sqlite_db_name: str = "employees.db"
    @property
    def sqlite_path(self) -> Path:
        return self.sqlite_dir / self.sqlite_db_name

    # Postgres
    pg_dsn: str = "postgresql://postgres:Admin_Vanina@localhost:5432/purchases_db"

    # MongoDB
    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db: str = "services_db"

    # Ollama / LLM
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "qwen3-4b"
    embedding_model: str = "nomic-embed-text"

    # RAG parameters
    chunk_size: int = 1000
    chunk_overlap: int = 150
    retrieve_k: int = 20
    final_k: int = 4
    rerank_model: str = "BAAI/bge-reranker-base"

    # Streamlit
    streamlit_port: int = 8501
    streamlit_address: str = "0.0.0.0"

    # Chroma
    chroma_telemetry: bool = False


settings = Settings()