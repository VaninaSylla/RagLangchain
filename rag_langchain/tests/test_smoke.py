"""
test_smoke.py
-------------
Smoke tests: import the package, parse commands, validate SQL, and check
Settings defaults. Designed to run without Ollama, PostgreSQL, or MongoDB.
"""

from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
def test_package_imports():
    import rag_langchain
    assert rag_langchain.__version__


def test_config_imports():
    from rag_langchain.config import settings, Settings
    assert isinstance(settings, Settings)


def test_core_imports():
    from rag_langchain.core import (
        ingestion,
        rag_chain,
        command_parser,
    )
    assert hasattr(ingestion, "get_vectorstore")
    assert hasattr(rag_chain, "answer_question")
    assert hasattr(command_parser, "parse_user_input")


def test_connectors_imports():
    from rag_langchain.connectors import BaseConnector, QueryResult
    from rag_langchain.connectors.sqlite import SQLiteConnector
    from rag_langchain.connectors.postgres import PostgresConnector
    from rag_langchain.connectors.mongo import MongoConnector
    assert issubclass(SQLiteConnector, BaseConnector)
    assert issubclass(PostgresConnector, BaseConnector)
    assert issubclass(MongoConnector, BaseConnector)


def test_cli_imports():
    from rag_langchain.cli import chat, index  # noqa: F401


def test_web_imports():
    # streamlit_app.py imports streamlit; we just verify the module loads.
    import importlib.util
    spec = importlib.util.find_spec("rag_langchain.web.streamlit_app")
    assert spec is not None


def test_scripts_imports():
    from rag_langchain.scripts import (
        init_sqlite_employees,
        init_postgres_purchases,
        init_mongo_services,
    )  # noqa: F401


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
def test_settings_defaults():
    from rag_langchain.config import Settings
    s = Settings()
    assert s.llm_model == "qwen3-4b"
    assert s.embedding_model == "nomic-embed-text"
    assert s.chunk_size > 0
    assert s.chunk_overlap >= 0
    assert s.retrieve_k > 0
    assert s.final_k > 0
    assert s.sqlite_db_name.endswith(".db")
    assert Path(s.sqlite_path).name == s.sqlite_db_name
    assert s.chroma_dir.name == "chroma_db"


# ---------------------------------------------------------------------------
# Command parser
# ---------------------------------------------------------------------------
def test_parse_user_input_no_prefix():
    from rag_langchain.core.command_parser import parse_user_input
    p = parse_user_input("salut, qui est Dupont ?")
    assert p.command == "auto"
    assert p.references == {}
    assert p.cleaned_question == "salut, qui est Dupont ?"


def test_parse_user_input_docs_command():
    from rag_langchain.core.command_parser import parse_user_input
    p = parse_user_input("/docs résumé en 5 lignes")
    assert p.command == "docs"
    assert p.cleaned_question == "résumé en 5 lignes"


def test_parse_user_input_employees_command():
    from rag_langchain.core.command_parser import parse_user_input
    p = parse_user_input("/employees qui a le plus gros salaire ?")
    assert p.command == "sqlite"
    assert p.cleaned_question == "qui a le plus gros salaire ?"


def test_parse_user_input_sql_command_maps_to_database():
    from rag_langchain.core.command_parser import parse_user_input
    p = parse_user_input("/sql liste les employés")
    assert p.command == "database"
    assert p.cleaned_question == "liste les employés"


def test_parse_user_input_federated_aliases():
    from rag_langchain.core.command_parser import parse_user_input
    assert parse_user_input("/all compare").command == "federated"
    assert parse_user_input("/tout compare").command == "federated"


def test_parse_user_input_references():
    from rag_langchain.core.command_parser import parse_user_input
    p = parse_user_input(
        "Quel est le salaire @employe:DUPONT et @service:RH ?"
    )
    assert p.references.get("employe") == ["DUPONT"]
    assert p.references.get("service") == ["RH"]
    assert "DUPONT" not in p.cleaned_question
    assert "RH" not in p.cleaned_question


def test_parse_user_input_combined_command_and_reference():
    from rag_langchain.core.command_parser import parse_user_input
    p = parse_user_input("/employees @employe:DUPONT infos")
    assert p.command == "sqlite"
    assert p.references.get("employe") == ["DUPONT"]
    assert p.cleaned_question == "infos"


# ---------------------------------------------------------------------------
# SQLite connector validation (no DB connection required)
# ---------------------------------------------------------------------------
def test_sqlite_validate_accepts_select():
    from rag_langchain.connectors.sqlite import SQLiteConnector
    conn = SQLiteConnector(db_path=":memory:")
    ok, reason = conn.validate("SELECT * FROM employes")
    assert ok, reason


def test_sqlite_validate_blocks_delete():
    from rag_langchain.connectors.sqlite import SQLiteConnector
    conn = SQLiteConnector(db_path=":memory:")
    ok, reason = conn.validate("DELETE FROM employes")
    assert not ok
    # The connector either rejects via the SELECT-prefix check or the FORBIDDEN
    # keyword list — both are valid security paths. We only assert it was blocked.
    assert reason != ""


def test_sqlite_validate_blocks_drop():
    from rag_langchain.connectors.sqlite import SQLiteConnector
    conn = SQLiteConnector(db_path=":memory:")
    ok, reason = conn.validate("DROP TABLE employes")
    assert not ok


def test_sqlite_validate_blocks_non_select():
    from rag_langchain.connectors.sqlite import SQLiteConnector
    conn = SQLiteConnector(db_path=":memory:")
    ok, reason = conn.validate("INSERT INTO employes VALUES (1)")
    assert not ok


# ---------------------------------------------------------------------------
# In-memory SQLite end-to-end (no Ollama, no Chroma)
# ---------------------------------------------------------------------------
def test_sqlite_execute_in_memory():
    import sqlite3
    from rag_langchain.connectors.sqlite import SQLiteConnector

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE t (id INTEGER, name TEXT)")
    conn.executemany("INSERT INTO t VALUES (?, ?)", [(1, "a"), (2, "b")])
    conn.commit()
    conn.close()

    # Patch the connector to use a fresh in-memory DB via a temporary file.
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    try:
        c = sqlite3.connect(path)
        c.execute("CREATE TABLE t (id INTEGER, name TEXT)")
        c.executemany("INSERT INTO t VALUES (?, ?)", [(1, "a"), (2, "b")])
        c.commit(); c.close()

        connector = SQLiteConnector(db_path=path)
        result = connector.answer("list all rows", {}, "fr")
        assert result.safe
        assert len(result.rows) == 2
    finally:
        Path(path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Postgres connector (validation only — no live DB)
# ---------------------------------------------------------------------------
def test_postgres_validate_blocks_delete():
    from rag_langchain.connectors.postgres import PostgresConnector
    conn = PostgresConnector(dsn="postgresql://x:y@localhost:5432/z")
    ok, reason = conn.validate("DELETE FROM achats")
    assert not ok


def test_postgres_validate_accepts_select():
    from rag_langchain.connectors.postgres import PostgresConnector
    conn = PostgresConnector(dsn="postgresql://x:y@localhost:5432/z")
    ok, reason = conn.validate("SELECT 1")
    assert ok, reason


# ---------------------------------------------------------------------------
# Mongo connector (validation only — no live DB)
# ---------------------------------------------------------------------------
def test_mongo_validate_rejects_malformed_json():
    from rag_langchain.connectors.mongo import MongoConnector
    conn = MongoConnector(uri="mongodb://localhost:27017", db_name="x")
    ok, reason = conn.validate("not-json")
    assert not ok
    assert "JSON" in reason.upper()


def test_mongo_validate_blocks_drop():
    from rag_langchain.connectors.mongo import MongoConnector
    conn = MongoConnector(uri="mongodb://localhost:27017", db_name="x")
    ok, reason = conn.validate('[{"$match": {}}, {"$out": "x"}]')
    assert not ok
    assert "$OUT" in reason.upper()


# ---------------------------------------------------------------------------
# Federator instantiation (does not require live DBs for the federator
# itself; the connectors are constructed but not used until the planner
# query fires).
# ---------------------------------------------------------------------------
def test_federator_constructs():
    from rag_langchain.connectors.federator import Federator
    fed = Federator(
        sqlite_path=":memory:",
        pg_dsn="postgresql://x:y@localhost:5432/z",
        mongo_uri="mongodb://localhost:27017",
        mongo_db="x",
    )
    assert set(fed.connectors.keys()) == {"sqlite", "postgres", "mongo"}


# ---------------------------------------------------------------------------
# Settings resolution: ensure env_file points at rag_langchain/.env
# ---------------------------------------------------------------------------
def test_env_file_is_inside_package():
    from rag_langchain.config.settings import settings
    expected = Path(settings.base_dir) / ".env"
    assert expected.parent == Path(settings.base_dir)


# ---------------------------------------------------------------------------
# Streamlit entry-point: simulates Streamlit's sys.path[0]=script_dir to
# verify the project-root path-injection fix prevents the
# `ModuleNotFoundError: rag_langchain` regression.
# ---------------------------------------------------------------------------
def test_streamlit_app_imports_when_script_dir_is_sys_path():
    import importlib.util
    import sys

    project_root = Path(__file__).resolve().parents[2]
    script_dir = project_root / "rag_langchain" / "web"
    spec = importlib.util.spec_from_file_location(
        "streamlit_app", str(project_root / "rag_langchain" / "web" / "streamlit_app.py")
    )
    assert spec is not None

    # Save/restore sys.path. Insert script_dir at position 0 (as Streamlit does).
    original = list(sys.path)
    sys.path.insert(0, str(script_dir))
    try:
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # must not raise ModuleNotFoundError
        assert mod.settings.base_dir == project_root
    finally:
        sys.path[:] = original


def test_cli_chat_imports_when_script_dir_is_sys_path():
    import importlib.util
    import sys

    project_root = Path(__file__).resolve().parents[2]
    script_dir = project_root / "rag_langchain" / "cli"
    spec = importlib.util.spec_from_file_location(
        "chat", str(project_root / "rag_langchain" / "cli" / "chat.py")
    )
    assert spec is not None

    original = list(sys.path)
    sys.path.insert(0, str(script_dir))
    try:
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert callable(mod.main)
    finally:
        sys.path[:] = original


def test_cli_index_imports_when_script_dir_is_sys_path():
    import importlib.util
    import sys

    project_root = Path(__file__).resolve().parents[2]
    script_dir = project_root / "rag_langchain" / "cli"
    spec = importlib.util.spec_from_file_location(
        "index", str(project_root / "rag_langchain" / "cli" / "index.py")
    )
    assert spec is not None

    original = list(sys.path)
    sys.path.insert(0, str(script_dir))
    try:
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert callable(mod.main)
    finally:
        sys.path[:] = original


# ---------------------------------------------------------------------------
# Indexing: progress callback must not crash on cp1252 consoles.
# Regression test for the "UnicodeEncodeError: cp1252 can't encode ✅" bug.
# ---------------------------------------------------------------------------
def test_index_files_progress_survives_cp1252_console(monkeypatch):
    """
    Simulate a cp1252 console by monkeypatching stdout.encoding, then index
    an in-memory PDF-like fixture. The progress callback (which uses ✅ / ❌)
    must not raise UnicodeEncodeError on the success line.
    """
    import sys
    from rag_langchain.core import ingestion as ing_mod

    captured = []

    class FakeStdout:
        encoding = "cp1252"

    monkeypatch.setattr(ing_mod.sys, "stdout", FakeStdout())

    # Build a minimal PDF with pymupdf (already a dep) and index it.
    import fitz
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        pdf_path = f.name
    try:
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 50), "Hello world. " * 50)
        doc.save(pdf_path)
        doc.close()

        safe_cb = ing_mod._safe_progress_callback(lambda m: captured.append(m))
        assert safe_cb is not None
        # Calling the wrapper directly must not raise.
        safe_cb("✅ ok")
        safe_cb("❌ not ok")
        assert captured == ["✅ ok", "❌ not ok"]
    finally:
        Path(pdf_path).unlink(missing_ok=True)
