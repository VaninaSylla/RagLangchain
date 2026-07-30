# RAG LangChain — Assistant interactif multi-sources

Assistant RAG (Retrieval-Augmented Generation) qui combine :

- **Mémoire conversationnelle** (reformulation automatique des questions)
- **Reranking cross-encoder** (résultats très précis)
- **Multi-bases** : SQLite (employés) · PostgreSQL (achats) · MongoDB (services)
- **Interface** : CLI Python ou application web Streamlit

Le tout en local, avec [Ollama](https://ollama.com) comme moteur LLM/embedding.

---

## Sommaire

1. [Architecture du projet](#architecture-du-projet)
2. [Flux de données](#flux-de-données)
3. [Pipeline RAG](#pipeline-rag)
4. [Palette de commandes](#palette-de-commandes)
5. [Bases de données](#bases-de-données)
6. [Installation](#installation)
7. [Configuration (.env)](#configuration-env)
8. [Utilisation](#utilisation)
9. [Tests](#tests)
10. [Limites connues](#limites-connues)

---

## Architecture du projet

```mermaid
flowchart TB
    subgraph UI["Interfaces utilisateur"]
        CLI["CLI<br/>python -m rag_langchain.cli.chat"]
        WEB["Streamlit<br/>streamlit run rag_langchain/web/streamlit_app.py"]
        SHIM1["ingest.py"]
        SHIM2["rag_app.py"]
        SHIM3["app_streamlit.py"]
    end

    subgraph PKG["rag_langchain/ (package)"]
        subgraph CLI_DIR["cli/"]
            CHAT["chat.py"]
            IDX["index.py"]
        end
        subgraph CORE["core/"]
            ING["ingestion.py<br/>(load, split, embed)"]
            CHAIN["rag_chain.py<br/>(route, reformulate, retrieve, rerank, generate)"]
            PARSER["command_parser.py"]
        end
        subgraph CONN["connectors/"]
            SQLITE["sqlite.py"]
            PG["postgres.py"]
            MONGO["mongo.py"]
            FED["federator.py"]
            BASE["base.py"]
        end
        subgraph CFG["config/"]
            SET["settings.py<br/>(pydantic-settings)"]
            LOG["logging.py"]
        end
        subgraph SCR["scripts/"]
            S1["init_sqlite_employees"]
            S2["init_postgres_purchases"]
            S3["init_mongo_services"]
        end
        subgraph TST["tests/"]
            T["test_smoke.py"]
        end
        subgraph WEB_DIR["web/"]
            APP["streamlit_app.py"]
        end
    end

    subgraph DATA["data/"]
        DOCS["documents/<br/>PDF · DOCX · PPTX · TXT"]
        CHROMA["chroma_db/<br/>(vecteurs)"]
        SQLDB["sqlite/employees.db"]
    end

    subgraph EXT["Services externes"]
        OLLAMA[("Ollama<br/>LLM + embeddings")]
        PGSQL[("PostgreSQL")]
        MGDB[("MongoDB")]
    end

    CLI --> CHAT
    WEB --> APP
    SHIM1 --> IDX
    SHIM2 --> CHAT
    SHIM3 --> APP

    CHAT --> CHAIN
    CHAT --> ING
    CHAT --> PARSER
    CHAT --> FED
    IDX --> ING
    APP --> CHAIN
    APP --> ING
    APP --> PARSER
    APP --> FED

    CHAIN --> SQLITE
    CHAIN --> RET["retrieve_and_rerank"]
    ING --> CHROMA
    RET --> CHROMA
    RET --> OLLAMA
    SQLITE --> SQLDB
    PG --> PGSQL
    MONGO --> MGDB
    FED --> SQLITE
    FED --> PG
    FED --> MONGO

    SET -. config .- CFG
    S1 --> SQLDB
    S2 --> PGSQL
    S3 --> MGDB
    DOCS --> ING
```

### Arborescence détaillée

```
RagLangchain/
├── data/                          # Données persistantes (git-ignored)
│   ├── documents/                 # Fichiers à indexer (PDF, DOCX, PPTX, TXT)
│   ├── sqlite/employees.db        # Base SQLite (employés)
│   └── chroma_db/                 # Base vectorielle Chroma
│
├── rag_langchain/                 # Package Python
│   ├── cli/
│   │   ├── index.py               # Indexation des documents
│   │   └── chat.py                # Chat interactif CLI
│   ├── core/
│   │   ├── ingestion.py           # Chargement, découpage, embeddings
│   │   ├── rag_chain.py           # Pipeline RAG principal
│   │   └── command_parser.py      # Décodage de /docs, @employe, etc.
│   ├── connectors/
│   │   ├── base.py                # Interface abstraite + QueryResult
│   │   ├── sqlite.py              # SQLite (employés)
│   │   ├── postgres.py            # PostgreSQL (achats)
│   │   ├── mongo.py               # MongoDB (services)
│   │   └── federator.py           # Fédération multi-bases
│   ├── web/
│   │   └── streamlit_app.py       # Interface Streamlit
│   ├── scripts/
│   │   ├── init_sqlite_employees.py
│   │   ├── init_postgres_purchases.py
│   │   └── init_mongo_services.py
│   ├── config/
│   │   ├── settings.py            # pydantic-settings
│   │   └── logging.py
│   ├── tests/test_smoke.py        # pytest
│   └── .env                       # Variables locales (git-ignored)
│
├── app_streamlit.py               # Shims (alias)
├── ingest.py
├── rag_app.py
├── tasks.ps1 / Makefile           # Raccourcis
├── requirements.txt
└── README.md
```

---

## Flux de données

```mermaid
flowchart LR
    A["📄 Documents<br/>(PDF, DOCX...)"] --> B["load_single_file<br/>(PyMuPDF / unstructured)"]
    B --> C["split_documents<br/>(Markdown + Recursive)"]
    C --> D["OllamaEmbeddings<br/>(nomic-embed-text)"]
    D --> E["Chroma<br/>(data/chroma_db)"]

    U["❓ Question utilisateur"] --> P["parse_user_input<br/>(palette /commande @ref)"]
    P --> R["classify_question<br/>(router)"]
    R -->|conversation| H["answer_from_history<br/>(LLM sur historique)"]
    R -->|database| DB["SQLite / Postgres / Mongo<br/>ou Federator"]
    R -->|documents| Q["condense_question<br/>(reformulation)"]
    Q --> RR["retrieve_and_rerank<br/>(Chroma → cross-encoder)"]
    E --> RR
    RR --> G["generate_answer<br/>(LLM avec citations [n])"]
    G --> ANS["💬 Réponse + sources"]
    H --> ANS
    DB --> ANS
```

---

## Pipeline RAG

Le pipeline complet s'exécute en 5 étapes à chaque question « documents » :

```mermaid
sequenceDiagram
    autonumber
    participant U as Utilisateur
    participant P as command_parser
    participant R as router (LLM)
    participant C as condense (LLM)
    participant V as Chroma (vectorstore)
    participant X as Cross-encoder<br/>(reranker)
    participant G as generate (LLM)

    U->>P: "/docs résumé du document"
    P->>R: question + historique
    R-->>P: "documents"
    P->>C: question (si historique)
    C-->>P: question autonome
    P->>V: similarity_search(k=20)
    V-->>P: 20 chunks candidats
    P->>X: predict(pairs)
    X-->>P: scores
    P->>P: top 4 chunks
    P->>G: context + question
    G-->>U: réponse + citations [1][2]...
```

### Étapes en détail

| # | Étape | Module / fonction | Rôle |
|---|-------|-------------------|------|
| 1 | **Routage** | `core.rag_chain.classify_question` | Décide : `conversation` / `documents` / `database` |
| 2 | **Reformulation** | `condense_question` | Réécrit la question si elle dépend de l'historique (sinon, renvoie telle quelle) |
| 3 | **Récupération large** | `retrieve_and_rerank` (étape 1) | Chroma renvoie les **20 chunks** (`RETRIEVE_K=20`) les plus proches |
| 4 | **Reranking** | `retrieve_and_rerank` (étape 2) | Cross-encoder `BAAI/bge-reranker-base` renote et garde les **4 meilleurs** (`FINAL_K=4`) |
| 5 | **Génération** | `generate_answer` | Le LLM (`qwen3-4b`) rédige la réponse en citant `[1]`, `[2]…` |

### Garde-fous & améliorations

- **Reranking offline** : `HF_HUB_OFFLINE=1` est défini dès la 1ʳᵉ utilisation pour éviter de re-vérifier en ligne.
- **Citations** : chaque chunk retenu a ses métadonnées (`source`, `page`) injectées dans le prompt → le LLM cite `[1]`, `[2]…` qui correspondent à la liste finale de sources.
- **Routeur conversation** : vérification par marqueurs explicites (`"ma dernière question"`, `"conversation"`, …) pour éviter le faux positif « de quoi parle ce doc » → « de quoi a-t-on parlé ».

---

## Palette de commandes

```mermaid
flowchart LR
    Q["Question brute"] --> P["parse_user_input"]
    P --> CMD["command<br/>(détecté via /xxx)"]
    P --> REF["references<br/>(détectées via @type:val)"]
    P --> CLEAN["cleaned_question<br/>(envoyée au LLM)"]

    CMD -->|"/docs"| D["branche documents"]
    CMD -->|"/employees<br/>/sql"| S["branche SQLite"]
    CMD -->|"/purchases"| P2["branche Postgres"]
    CMD -->|"/services"| M["branche Mongo"]
    CMD -->|"/all<br/>/tout"| F["branche Federator"]
    CMD -->|"rien"| AUTO["routage automatique<br/>(LLM)"]
```

### Commandes

| Commande | Cible | Exemple |
|----------|-------|---------|
| `/docs` | Documents (RAG) | `/docs résume le PDF` |
| `/employees` · `/sql` · `/employes` · `/emp` | SQLite (employés) | `/employees qui a le plus gros salaire ?` |
| `/purchases` · `/achats` | PostgreSQL (achats) | `/purchases quel est le montant total ?` |
| `/services` · `/service` | MongoDB (services) | `/services liste les services` |
| `/all` · `/tout` | Fédération multi-bases | `/all compare les budgets des services avec les achats` |
| *rien* | Routage automatique | Le système détecte tout seul la bonne branche |

### Références (`@`)

```
@employe:DUPONT      @service:RH
@fournisseur:Dell    @doc:rapport.pdf
```

Elles sont extraites par `command_parser` et injectées dans le pipeline de la branche cible.

---

## Bases de données

```mermaid
erDiagram
    EMPLOYES ||--o{ ACHATS : "employe_id"
    SERVICES ||--o{ EMPLOYES : "service_id"
    SERVICES ||--o{ ACHATS : "service_id"

    EMPLOYES {
        int id PK
        text nom
        text poste
        int service_id FK
        real salaire
        text date_embauche
    }
    ACHATS {
        int id PK
        text fournisseur
        real montant
        text date_achat
        int employe_id FK
        int service_id FK
    }
    SERVICES {
        int _id PK
        text nom
        text responsable
        real budget
        text localisation
    }
```

| Base | Type | Fichier / DSN | Contenu | Seed |
|------|------|---------------|---------|------|
| **employés** | SQLite | `data/sqlite/employees.db` | 5 employés, salaires, services | `init_sqlite_employees` |
| **achats** | PostgreSQL | `purchases_db` (localhost:5432) | 4 achats, fournisseurs, montant | `init_postgres_purchases` |
| **services** | MongoDB | `services_db` (localhost:27017) | 4 services, responsables, budgets | `init_mongo_services` |

Les `service_id` sont chaînés entre les trois bases — c'est ce qui permet les requêtes fédérées (`/all`).

---

## Installation

```mermaid
flowchart LR
    A["1. Cloner / télécharger"] --> B["2. Créer venv"]
    B --> C["3. pip install -r requirements.txt"]
    C --> D["4. Éditer rag_langchain/.env"]
    D --> E["5. Lancer Ollama<br/>+ pull des modèles"]
    E --> F["6. Seed des bases<br/>(optionnel)"]
    F --> G["7. Indexer les documents"]
    G --> H["8. Lancer chat CLI<br/>ou Streamlit"]
```

### Étape 1 — Préparer l'environnement

```powershell
cd C:\Users\ngono\Desktop\RagLangchain
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Étape 2 — Ollama (obligatoire)

```powershell
ollama serve
ollama pull nomic-embed-text
ollama pull qwen3-4b
```

### Étape 3 — Bases de données (optionnel)

```powershell
# SQLite (local, aucune installation requise)
python -m rag_langchain.scripts.init_sqlite_employees

# PostgreSQL (installer + démarrer le service avant)
python -m rag_langchain.scripts.init_postgres_purchases

# MongoDB (installer + démarrer mongod avant)
python -m rag_langchain.scripts.init_mongo_services
```

> Voir `RUNBOOK.md` pour le détail complet d'installation PostgreSQL/MongoDB.

### Étape 4 — Indexer des documents

Placer des fichiers dans `data/documents/` puis :

```powershell
python -m rag_langchain.cli.index
```

---

## Configuration (.env)

Toutes les valeurs sont surchargeables via `rag_langchain/.env`.

```env
# --- App ---
APP_NAME=RAG LangChain
APP_ENV=development
LOG_LEVEL=INFO

# --- Paths (par défaut : <projet>/data) ---
# DATA_DIR=C:/Users/.../RagLangchain/data
# DOCUMENTS_DIR=C:/Users/.../RagLangchain/data/documents
# SQLITE_DIR=C:/Users/.../RagLangchain/data/sqlite
# CHROMA_DIR=C:/Users/.../RagLangchain/data/chroma_db

# --- SQLite ---
SQLITE_DB_NAME=employees.db

# --- PostgreSQL ---
PG_DSN=postgresql://postgres:CHANGE_ME@localhost:5432/purchases_db

# --- MongoDB ---
MONGO_URI=mongodb://localhost:27017
MONGO_DB=services_db

# --- Ollama / LLM ---
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=qwen3-4b
EMBEDDING_MODEL=nomic-embed-text

# --- RAG ---
CHUNK_SIZE=1000
CHUNK_OVERLAP=150
RETRIEVE_K=20
FINAL_K=4
RERANK_MODEL=BAAI/bge-reranker-base

# --- Streamlit ---
STREAMLIT_PORT=8501
STREAMLIT_ADDRESS=0.0.0.0
```

---

## Utilisation

### Raccourcis (recommandé)

```powershell
.\tasks.ps1 install      # pip install -r requirements.txt
.\tasks.ps1 init-sqlite  # seed SQLite
.\tasks.ps1 init-pg      # seed PostgreSQL
.\tasks.ps1 init-mongo   # seed MongoDB
.\tasks.ps1 index        # indexation des documents
.\tasks.ps1 chat         # CLI chat
.\tasks.ps1 web          # Streamlit
.\tasks.ps1 test         # pytest
.\tasks.ps1 clean        # supprime __pycache__
```

### Commandes `python -m`

```powershell
# Indexation
python -m rag_langchain.cli.index

# Chat CLI
python -m rag_langchain.cli.chat

# Streamlit
streamlit run rag_langchain/web/streamlit_app.py
```

### Exemples de conversation

```text
Question > Bonjour
💬 (Réponse basée sur l'historique)

Question > /employees qui a le plus gros salaire ?
🗄️ Réponse générée via SQLite
   Marie Curie — 600000

Question > /services liste les services
🛠️ Réponse générée via Mongo
   1. Ressources Humaines (budget 5 000 000)
   2. Informatique (budget 15 000 000)
   …

Question > /all compare les budgets des services avec les achats
🔗 Réponse multi-sources (fédération)
   (plan d'exécution + citations)

Question > /docs @doc:chapter 4_ chemical bonding.pdf parle des liaisons ?
📄 Réponse générée via Documents
   [1] chapter 4_ chemical bonding.pdf, page 4 …
```

---

## Tests

```powershell
.\tasks.ps1 test
# ou
pytest -q
```

26 tests couvrent :

- import de chaque module
- parsing de la palette (`/docs`, `/employees`, `/all`, `@employe:…`)
- validation SQL (SELECT / DELETE / DROP / INSERT)
- validation Mongo (JSON, `$out`)
- construction du `Federator`
- defaults de `Settings` et résolution du `.env`

---

## Limites connues

- **PDF scientifiques** : les formules mathématiques ne sont pas reconnues (extraction texte brut).
- **CPU** : le reranking cross-encoder tourne sur CPU → latence de quelques secondes.
- **Mémoire** : la conversation n'est pas persistée entre les redémarrages.
- **Services externes** : PostgreSQL et MongoDB doivent être lancés manuellement avant utilisation.
- **Premier lancement** : `BAAI/bge-reranker-base` (~1.1 Go) est téléchargé depuis Hugging Face.
