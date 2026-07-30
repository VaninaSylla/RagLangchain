# RUNBOOK — Guide pas-à-pas pour lancer le projet

> **Public** : toute personne qui veut faire tourner `RagLangchain` sur sa machine
> Windows pour la première fois, sans se perdre dans les détails.
>
> **Temps estimé** : 30 à 60 minutes (dont la majorité pour installer PostgreSQL et MongoDB si besoin).

---

## Sommaire

1. [Vue d'ensemble](#vue-densemble)
2. [Prérequis](#prérequis)
3. [Étape 1 — Python & venv](#étape-1--python--venv)
4. [Étape 2 — Dépendances Python](#étape-2--dépendances-python)
5. [Étape 3 — Ollama (LLM local)](#étape-3--ollama-llm-local)
6. [Étape 4 — Configuration `.env`](#étape-4--configuration-env)
7. [Étape 5 — Bases de données](#étape-5--bases-de-données)
8. [Étape 6 — Premier lancement](#étape-6--premier-lancement)
9. [Étape 7 — Utilisation](#étape-7--utilisation)
10. [Diagnostic & erreurs](#diagnostic--erreurs)
11. [Maintenance](#maintenance)

---

## Vue d'ensemble

```mermaid
flowchart LR
    P[Python] --> V[venv]
    V --> I[pip install]
    I --> O[Ollama]
    O --> M[pull 2 modèles]
    I --> E[.env]
    O --> DB[Postgres / Mongo]
    DB --> S[seed scripts]
    S --> CH[CLI chat]
    S --> ST[Streamlit]
    CH --> OK[✅]
    ST --> OK
```

---

## Prérequis

| Outil | Version | Pourquoi | Vérification |
|-------|---------|----------|--------------|
| **Python** | ≥ 3.10 | Runtime | `python --version` |
| **PowerShell** | 5.1+ | Shell par défaut Windows | `$PSVersionTable.PSVersion` |
| **Ollama** | dernière | Moteur LLM + embeddings | `ollama --version` |
| **PostgreSQL** *(optionnel)* | 13+ | Base `purchases_db` | `psql --version` |
| **MongoDB** *(optionnel)* | 6+ | Base `services_db` | `mongod --version` |
| **Connexion Internet** | — | Téléchargement des modèles HF + Ollama | — |

> Les 3 premières étapes (Python, pip, Ollama) sont **obligatoires**. PostgreSQL et MongoDB ne sont nécessaires que si vous voulez utiliser les branches `/purchases`, `/services` ou `/all`.

---

## Étape 1 — Python & venv

### 1.1 Vérifier Python

```powershell
python --version
# Doit afficher Python 3.10 ou supérieur
```

Si absent, télécharger depuis [python.org](https://www.python.org/downloads/) (cocher **Add Python to PATH**).

### 1.2 Activer PowerShell pour les scripts

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

### 1.3 Créer l'environnement virtuel

```powershell
cd C:\Users\ngono\Desktop\RagLangchain
python -m venv venv
.\venv\Scripts\Activate.ps1
```

> Votre prompt doit maintenant commencer par `(venv)`.

### 1.4 Mettre pip à jour

```powershell
python -m pip install --upgrade pip
```

---

## Étape 2 — Dépendances Python

```powershell
pip install -r requirements.txt
```

La première installation prend 2 à 5 minutes (téléchargement de torch, transformers, etc.).

### Alternative — raccourci

```powershell
.\tasks.ps1 install
```

---

## Étape 3 — Ollama (LLM local)

### 3.1 Installer Ollama

Télécharger depuis [ollama.com/download](https://ollama.com/download) et lancer l'installateur.

### 3.2 Démarrer le serveur Ollama

Ollama démarre automatiquement avec l'installateur. Si ce n'est pas le cas :

```powershell
ollama serve
```

### 3.3 Télécharger les modèles

Dans **un autre terminal** :

```powershell
ollama pull nomic-embed-text    # ~274 Mo — embeddings
ollama pull qwen3-4b            # ~2.5 Go — génération
```

### 3.4 Vérifier

```powershell
ollama list
# Doit afficher nomic-embed-text et qwen3-4b
```

```mermaid
flowchart LR
    Q["question utilisateur"] --> O[Ollama]
    O --> E["nomic-embed-text<br/>(embedding)"]
    O --> L["qwen3-4b<br/>(generation)"]
    L --> R["réponse"]
```

---

## Étape 4 — Configuration `.env`

### 4.1 Ouvrir le fichier

```powershell
code rag_langchain\.env
# ou
notepad rag_langchain\.env
```

### 4.2 Adapter les valeurs critiques

```env
# Mot de passe PostgreSQL — ADAPTER !
PG_DSN=postgresql://postgres:VOTRE_MOT_DE_PASSE@localhost:5432/purchases_db

# MongoDB (par défaut OK si mongod tourne sur le port standard)
MONGO_URI=mongodb://localhost:27017
MONGO_DB=services_db

# LLM (par défaut OK)
LLM_MODEL=qwen3-4b
EMBEDDING_MODEL=nomic-embed-text
```

> Les autres paramètres ont des valeurs par défaut fonctionnelles.

### 4.3 Vérifier la résolution

```powershell
python -c "from rag_langchain.config import settings; print(settings.sqlite_path)"
# Doit afficher : C:\Users\ngono\Desktop\RagLangchain\data\sqlite\employees.db
```

---

## Étape 5 — Bases de données

Trois branches possibles. Choisir selon vos besoins :

```mermaid
flowchart TD
    Q["Vous voulez utiliser…"] --> A["Documents uniquement ?"]
    Q --> B["/employees ou /sql ?"]
    Q --> C["/purchases ?"]
    Q --> D["/services ?"]
    Q --> E["/all ?"]

    A -->|Oui| OK1["Rien à installer<br/>SQLite + Chroma suffisent"]
    B -->|Oui| OK1
    C -->|Oui| INST1["Installer PostgreSQL"]
    D -->|Oui| INST2["Installer MongoDB"]
    E -->|Oui| INST1
    E -->|Oui| INST2
```

### 5.1 SQLite — **(obligatoire, déjà inclus)**

```powershell
python -m rag_langchain.scripts.init_sqlite_employees
# Affiche : SQLite database created at '…\data\sqlite\employees.db' with 5 employees.
```

> Idempotent : si la base existe déjà, le script s'arrête proprement.

### 5.2 PostgreSQL — *(optionnel)*

#### Installation

1. Télécharger depuis [postgresql.org/download/windows](https://www.postgresql.org/download/windows/)
2. Pendant l'installation :
   - Mémoriser le **mot de passe** saisi pour l'utilisateur `postgres`
   - Port par défaut : `5432`
3. Démarrer le service : `services.msc` → service « postgresql-x64-XX » → Démarrer

#### Vérification

```powershell
# Vérifier que le port répond
Test-NetConnection -ComputerName localhost -Port 5432
```

#### Seed

```powershell
python -m rag_langchain.scripts.init_postgres_purchases
# Affiche : Database 'purchases_db' created / Table 'achats' created and filled ...
```

> ⚠️ Si le script affiche une erreur de connexion : vérifier que `PG_DSN` dans `.env` contient le bon mot de passe.

### 5.3 MongoDB — *(optionnel)*

#### Installation

1. Télécharger depuis [mongodb.com/try/download/community](https://www.mongodb.com/try/download/community)
2. Choisir une installation **complète** (inclut `mongod`)
3. Démarrer le service : `services.msc` → service « MongoDB Server » → Démarrer

#### Vérification

```powershell
Test-NetConnection -ComputerName localhost -Port 27017
```

#### Seed

```powershell
python -m rag_langchain.scripts.init_mongo_services
# Affiche : MongoDB database 'services_db' created with 4 services.
```

### 5.4 Raccourci

```powershell
.\tasks.ps1 init-sqlite
.\tasks.ps1 init-pg
.\tasks.ps1 init-mongo
```

---

## Étape 6 — Premier lancement

### 6.1 Placer des documents

Mettre des fichiers dans `data/documents/` (formats acceptés : `.pdf`, `.txt`, `.docx`, `.pptx`, `.ppt`).

```powershell
# Exemple : copier un PDF
Copy-Item "C:\Users\Moi\Downloads\rapport.pdf" "data\documents\"
```

### 6.2 Indexer

```powershell
python -m rag_langchain.cli.index
# Affiche : N file(s) found. Indexing in progress...
#          ✅ fichier.pdf indexé (X chunks)
#          Done. X chunks indexed in 'data/chroma_db/'.
```

> ⏱️ La première exécution télécharge le modèle d'embedding (`nomic-embed-text`) et le cross-encoder (`BAAI/bge-reranker-base`, ~1.1 Go) — prévoir 5-10 minutes.

### 6.3 Lancer le chat

```powershell
python -m rag_langchain.cli.chat
```

Vous devez voir :

```
Loading vector store...
Ready! Type your question (or 'exit' to quit).

Question > _
```

### 6.4 Lancer Streamlit (optionnel)

```powershell
streamlit run rag_langchain/web/streamlit_app.py
```

Le navigateur s'ouvre automatiquement sur `http://localhost:8501`.

---

## Étape 7 — Utilisation

### 7.1 Tester les bases

```text
Question > /employees qui a le plus gros salaire ?
```

→ Affiche la requête SQL exécutée + les résultats.

### 7.2 Tester les documents

```text
Question > /docs fais-moi un résumé de mes documents
```

→ Affiche la réponse + une liste de sources `[1] [2] …`.

### 7.3 Tester la mémoire conversationnelle

```text
Question > Bonjour, comment ça marche ?
Question > Et pour les bases de données ?
```

→ La 2ᵉ question doit être reformulée en tenant compte du contexte.

### 7.4 Tester la fédération

```text
Question > /all compare les budgets des services avec les achats
```

→ Affiche un plan d'exécution + les résultats croisés + la synthèse.

### 7.5 Quitter

```text
Question > exit
```

---

## Diagnostic & erreurs

### Problèmes courants

| Symptôme | Cause probable | Solution |
|----------|----------------|----------|
| `ModuleNotFoundError: rag_langchain` | Pas dans le bon dossier | `cd C:\Users\ngono\Desktop\RagLangchain` |
| `ollama: command not found` | Ollama non installé ou PATH non mis à jour | Réinstaller Ollama, redémarrer le terminal |
| `ConnectionRefused: 11434` | `ollama serve` pas lancé | Lancer Ollama manuellement |
| `psycopg2.OperationalError: could not connect` | PostgreSQL pas démarré ou mauvais mot de passe | Démarrer le service PG, vérifier `PG_DSN` |
| `ServerSelectionTimeoutError: localhost:27017` | MongoDB pas démarré | Démarrer le service MongoDB |
| `Please install langchain_text_splitters` | pip install incomplet | `pip install langchain-text-splitters` |
| Chroma reste vide après indexation | Mauvais chemin | Vérifier `CHROMA_DIR` dans `.env` |
| Réponse très lente (> 30 s) | Premier appel au reranker | Normal — attendez, les suivants sont plus rapides |

### Checklist rapide

```powershell
# 1. Tout est-il installé ?
python --version
ollama --version

# 2. Ollama tourne-t-il ?
Test-NetConnection -ComputerName localhost -Port 11434

# 3. Les modèles sont-ils là ?
ollama list

# 4. Le venv est-il activé ?
# (Le prompt doit commencer par (venv))

# 5. Les bases existent-elles ?
Test-Path "data\sqlite\employees.db"

# 6. Les documents sont-ils indexés ?
Test-Path "data\chroma_db\chroma.sqlite3"
```

### Réinitialiser complètement

```powershell
# 1. Supprimer la base vectorielle
Remove-Item -Recurse -Force data\chroma_db
New-Item -ItemType File -Path data\chroma_db\.gitkeep -Force

# 2. Supprimer la base SQLite
Remove-Item -Force data\sqlite\employees.db

# 3. Re-seeder
python -m rag_langchain.scripts.init_sqlite_employees
python -m rag_langchain.cli.index
```

### Logs utiles

```powershell
# Streamlit affiche tout en clair
streamlit run rag_langchain/web/streamlit_app.py

# CLI : activer le mode debug
$env:LOG_LEVEL = "DEBUG"
python -m rag_langchain.cli.chat
```

---

## Maintenance

### Mettre à jour les dépendances

```powershell
pip install --upgrade -r requirements.txt
```

### Ajouter un document

```powershell
# 1. Copier le fichier
Copy-Item "nouveau.pdf" "data\documents\"

# 2. Réindexer
python -m rag_langchain.cli.index
```

### Réindexer depuis zéro

```powershell
Remove-Item -Recurse -Force data\chroma_db
New-Item -ItemType File -Path data\chroma_db\.gitkeep -Force
python -m rag_langchain.cli.index
```

### Lancer les tests

```powershell
.\tasks.ps1 test
# ou
pytest -q
```

### Nettoyer les caches Python

```powershell
.\tasks.ps1 clean
```

### Mettre à jour Ollama

```powershell
# Windows : réinstaller depuis ollama.com
# Puis mettre à jour les modèles :
ollama pull nomic-embed-text
ollama pull qwen3-4b
```

---

## Résumé express (TL;DR)

```powershell
# Setup
cd C:\Users\ngono\Desktop\RagLangchain
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Ollama
ollama serve                    # dans un autre terminal
ollama pull nomic-embed-text
ollama pull qwen3-4b

# Bases
python -m rag_langchain.scripts.init_sqlite_employees

# Documents
Copy-Item "mon.pdf" "data\documents\"

# Lancer
python -m rag_langchain.cli.index
python -m rag_langchain.cli.chat
```

Bon run ! 🚀
