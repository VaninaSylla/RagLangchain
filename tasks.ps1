# tasks.ps1 — convenience runners for PowerShell
# Usage:  .\tasks.ps1 install   |   .\tasks.ps1 web   |   .\tasks.ps1 test ...

param(
    [Parameter(Position=0)]
    [ValidateSet("install","init-sqlite","init-pg","init-mongo","index","chat","web","test","clean")]
    [string]$Task = "help"
)

$ErrorActionPreference = "Stop"

function Run($cmd) {
    Write-Host ">>> $cmd" -ForegroundColor Cyan
    Invoke-Expression $cmd
}

switch ($Task) {
    "install"     { Run "pip install -r requirements.txt" }
    "init-sqlite" { Run "python -m rag_langchain.scripts.init_sqlite_employees" }
    "init-pg"     { Run "python -m rag_langchain.scripts.init_postgres_purchases" }
    "init-mongo"  { Run "python -m rag_langchain.scripts.init_mongo_services" }
    "index"       { Run "python -m rag_langchain.cli.index" }
    "chat"        { Run "python -m rag_langchain.cli.chat" }
    "web"         { Run "streamlit run rag_langchain/web/streamlit_app.py" }
    "test"        { Run "pytest -q" }
    "clean"       {
        Get-ChildItem -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
        Write-Host "Cleaned __pycache__ folders." -ForegroundColor Green
    }
    default {
        Write-Host @"
Available tasks:
  install      Install Python dependencies
  init-sqlite  Seed the SQLite employees DB
  init-pg      Seed the PostgreSQL purchases DB (server must be up)
  init-mongo   Seed the MongoDB services DB (server must be up)
  index        Index every file in data/documents/ into Chroma
  chat         Launch the CLI chat
  web          Launch the Streamlit web UI
  test         Run pytest
  clean        Remove __pycache__ folders
"@
    }
}
