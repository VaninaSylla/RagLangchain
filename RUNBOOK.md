# RUNBOOK — Guide pas-à-pas pour lancer le projet

> **Public** : toute personne qui veut faire tourner `RagLangchain` sur sa machine
> pour la première fois, sans se perdre dans les détails. Les commandes sont données
> pour **Git-Bash (Windows)** et **Linux/macOS** — adaptez le chemin du venv
> (`venv/Scripts/` vs `venv/bin/`).
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
| **Shell** | — | Git-Bash (Windows) ou bash/zsh (Linux/macOS) | `bash --version` |
| **Ollama** | dernière | Moteur LLM + embeddings | `ollama --version` |
| **PostgreSQL** *(optionnel)* | 13+ | Base `purchases_db` | `psql --version` |
| **MongoDB** *(optionnel)* | 6+ | Base `services_db` | `mongod --version` |
| **Connexion Internet** | — | Téléchargement des modèles HF + Ollama | — |

> Les 3 premières étapes (Python, pip, Ollama) sont **obligatoires**. PostgreSQL et MongoDB ne sont nécessaires que si vous voulez utiliser les branches `/purchases`, `/services` ou `/all`.

---

## Étape 1 — Python & venv

### 1.1 Vérifier Python

```bash
python --version
# Doit afficher Python 3.10 ou supérieur
```

Si absent, télécharger depuis [python.org](https://www.python.org/downloads/) (cocher **Add Python to PATH** sous Windows).

### 1.2 Créer l'environnement virtuel

```bash
cd C:\Users\ngono\Desktop\RagLangchain
python -m venv venv

# Git-Bash (Windows)
source venv/Scripts/activate
# Linux / macOS
# source venv/bin/activate
```

> Votre prompt doit maintenant commencer par `(venv)`.

### 1.3 Mettre pip à jour

```bash
python -m pip install --upgrade pip
```

---

## Étape 2 — Dépendances Python

```bash
pip install -r requirements.txt
```

La première installation prend 2 à 5 minutes (téléchargement de torch, transformers, etc.).

### Alternative — raccourci

```bash
./tasks.sh install
```

---

## Étape 3 — Ollama (LLM local)

### 3.1 Installer Ollama

Télécharger depuis [ollama.com/download](https://ollama.com/download) et lancer l'installateur.

### 3.2 Démarrer le serveur Ollama

Ollama démarre automatiquement avec l'installateur. Si ce n'est pas le cas :

```bash
ollama serve
```

### 3.3 Télécharger les modèles

Dans **un autre terminal** :

```bash
ollama pull nomic-embed-text    # ~274 Mo — embeddings
ollama pull qwen3-4b            # ~2.5 Go — génération
```

### 3.4 Vérifier

```bash
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

```bash
code rag_langchain/.env
# ou
nano rag_langchain/.env          # Linux/macOS
# ou
notepad rag_langchain\.env       # Windows
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

```bash
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

```bash
python -m rag_langchain.scripts.init_sqlite_employees
# Affiche : SQLite database created at '…\data\sqlite\employees.db' with 5 employees.
```

> Idempotent : si la base existe déjà, le script s'arrête proprement.

### 5.2 PostgreSQL — *(optionnel)*

#### Installation

1. Télécharger depuis [postgresql.org/download/windows](https://www.postgresql.org/download/windows/) (Windows) ou via le gestionnaire de paquets (`sudo apt install postgresql` sur Ubuntu, `brew install postgresql` sur macOS)
2. Pendant l'installation :
   - Mémoriser le **mot de passe** saisi pour l'utilisateur `postgres`
   - Port par défaut : `5432`
3. Démarrer le service : `services.msc` → service « postgresql-x64-XX » → Démarrer (Windows)

#### Vérification

```bash
# Vérifier que le port répond
curl -s telnet://localhost:5432 || echo "port 5432 fermé"
```

#### Seed

```bash
python -m rag_langchain.scripts.init_postgres_purchases
# Affiche : Database 'purchases_db' created / Table 'achats' created and filled ...
```

> ⚠️ Si le script affiche une erreur de connexion : vérifier que `PG_DSN` dans `.env` contient le bon mot de passe.

### 5.3 MongoDB — *(optionnel)*

#### Installation

1. Télécharger depuis [mongodb.com/try/download/community](https://www.mongodb.com/try/download/community)
2. Choisir une installation **complète** (inclut `mongod`)
3. Démarrer le service : `services.msc` → service « MongoDB Server » → Démarrer (Windows)

#### Vérification

```bash
curl -s telnet://localhost:27017 || echo "port 27017 fermé"
```

#### Seed

```bash
python -m rag_langchain.scripts.init_mongo_services
# Affiche : MongoDB database 'services_db' created with 4 services.
```

### 5.4 Raccourci

```bash
./tasks.sh init-sqlite
./tasks.sh init-pg
./tasks.sh init-mongo
```

---

## Étape 6 — Premier lancement

### 6.1 Placer des documents

Mettre des fichiers dans `data/documents/` (formats acceptés : `.pdf`, `.txt`, `.docx`, `.pptx`, `.ppt`).

```bash
# Exemple : copier un PDF
cp "C:/Users/Moi/Downloads/rapport.pdf" data/documents/
```

### 6.2 Indexer

```bash
python -m rag_langchain.cli.index
# Affiche : N file(s) found. Indexing in progress...
#          ✅ fichier.pdf indexé (X chunks)
#          Done. X chunks indexed in 'data/chroma_db/'.
```

> ⏱️ La première exécution télécharge le modèle d'embedding (`nomic-embed-text`) et le cross-encoder (`BAAI/bge-reranker-base`, ~1.1 Go) — prévoir 5-10 minutes.

### 6.3 Lancer le chat

```bash
python -m rag_langchain.cli.chat
```

Vous devez voir :

```
Loading vector store...
Ready! Type your question (or 'exit' to quit).

Question > _
```

### 6.4 Lancer Streamlit (optionnel)

```bash
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

```bash
# 1. Tout est-il installé ?
python --version
ollama --version

# 2. Ollama tourne-t-il ?
curl -s telnet://localhost:11434 || echo "port 11434 fermé"

# 3. Les modèles sont-ils là ?
ollama list

# 4. Le venv est-il activé ?
# (Le prompt doit commencer par (venv))

# 5. Les bases existent-elles ?
test -f data/sqlite/employees.db

# 6. Les documents sont-ils indexés ?
test -f data/chroma_db/chroma.sqlite3
```

### Réinitialiser complètement

```bash
# 1. Supprimer la base vectorielle
rm -rf data/chroma_db
touch data/chroma_db/.gitkeep

# 2. Supprimer la base SQLite
rm -f data/sqlite/employees.db

# 3. Re-seeder
python -m rag_langchain.scripts.init_sqlite_employees
python -m rag_langchain.cli.index
```

### Logs utiles

```bash
# Streamlit affiche tout en clair
streamlit run rag_langchain/web/streamlit_app.py

# CLI : activer le mode debug
export LOG_LEVEL="DEBUG"
python -m rag_langchain.cli.chat
```

---

## Maintenance

### Mettre à jour les dépendances

```bash
pip install --upgrade -r requirements.txt
```

### Ajouter un document

```bash
# 1. Copier le fichier
cp "nouveau.pdf" data/documents/

# 2. Réindexer
python -m rag_langchain.cli.index
```

### Réindexer depuis zéro

```bash
rm -rf data/chroma_db
touch data/chroma_db/.gitkeep
python -m rag_langchain.cli.index
```

### Lancer les tests

```bash
./tasks.sh test
# ou
pytest -q
```

### Nettoyer les caches Python

```bash
./tasks.sh clean
```

### Mettre à jour Ollama

```bash
# Windows : réinstaller depuis ollama.com
# Linux/macOS : réinstaller depuis le site officiel
# Puis mettre à jour les modèles :
ollama pull nomic-embed-text
ollama pull qwen3-4b
```

---

## Résumé express (TL;DR)

```bash
# Setup
cd C:\Users\ngono\Desktop\RagLangchain
python -m venv venv
source venv/Scripts/activate      # Git-Bash (Windows) — sinon : venv/bin/activate (Linux/macOS)
pip install -r requirements.txt

# Ollama
ollama serve                    # dans un autre terminal
ollama pull nomic-embed-text
ollama pull qwen3-4b

# Bases
python -m rag_langchain.scripts.init_sqlite_employees

# Documents
cp "mon.pdf" data/documents/

# Lancer
python -m rag_langchain.cli.index
python -m rag_langchain.cli.chat
```

Bon run ! 🚀
