import re, sqlite3
from .base import BaseConnector, QueryResult
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

FORBIDDEN = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "ATTACH", "PRAGMA", "REPLACE"]
MAX_ROWS = 50

class SQLiteConnector(BaseConnector):
    source_label = "SQLite:employés"

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # @Function Description: Initialise le connecteur SQLite avec le chemin de la
    # base de données et le modèle LLM à utiliser pour la génération.
    # ------------------------------------------------------------------------------
    # @Parameter:
    #                 - db_path: str - Chemin vers le fichier de base SQLite.
    #                 - llm_model: str - Nom du modèle Ollama à utiliser.
    # @Returnvalue:
    #                 - (aucun)
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self, db_path="data/sqlite/employees.db", llm_model="qwen3-4b"):
        self.db_path = db_path
        self.llm = ChatOllama(model=llm_model, temperature=0.1)

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # @Function Description: Ouvre une connexion à la base SQLite locale.
    # ------------------------------------------------------------------------------
    # @Parameter:
    #                 - (aucun)
    # @Returnvalue:
    #                 - sqlite3.Connection - Connexion ouverte.
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _conn(self): 
        return sqlite3.connect(self.db_path)

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # @Function Description: Extrait le schéma de la base SQLite (tables et colonnes)
    # en interrogeant sqlite_master et PRAGMA table_info.
    # ------------------------------------------------------------------------------
    # @Parameter:
    #                 - (aucun)
    # @Returnvalue:
    #                 - str - Schéma textuel injecté au LLM.
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_schema_text(self) -> str:
        conn = self._conn()
        try:
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [r[0] for r in cur.fetchall()]
            lines = []
            for t in tables:
                cur.execute(f"PRAGMA table_info({t})")
                cols = [f"{c[1]} ({c[2]})" for c in cur.fetchall()]
                lines.append(f"Table {t} : " + ", ".join(cols))
            return "\n".join(lines)
        finally:
            conn.close()

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++-
    # @Function Description: Génère une requête SQL SQLite via le LLM en fonction de
    # la question et des filtres @employé.
    # ------------------------------------------------------------------------------
    # @Parameter:
    #                 - question: str - Question utilisateur.
    #                 - references: dict - Filtres (ex: {"employé": ["DUPONT"]}).
    #                 - language: str - Langue.
    # @Returnvalue:
    #                 - str - Requête SQL générée et nettoyée.
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def generate_query(self, question, references, language):
        schema = self.get_schema_text()
        ref_hint = ""
        if "employé" in references or "employe" in references:
            noms = references.get("employé", references.get("employe"))
            ref_hint = f"\nFiltre imposé : le nom de l'employé doit être dans {noms}."

        prompt = ChatPromptTemplate.from_template(
            """Schéma SQLite :\n{schema}\n\nQuestion : {question}{ref_hint}\n
            Écris UNIQUEMENT la requête SQL brute, terminée par ';'. Pas de Markdown."""
        )
        chain = prompt | self.llm | StrOutputParser()
        raw = chain.invoke({"schema": schema, "question": question, "ref_hint": ref_hint})
        return re.sub(r"```sql|```", "", raw).strip()

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # @Function Description: Valide la requête SQL pour s'assurer qu'elle commence
    # par SELECT et ne contient aucun mot-clé destructeur.
    # ------------------------------------------------------------------------------
    # @Parameter:
    #                 - sql: str - Requête SQL à valider.
    # @Returnvalue:
    #                 - tuple[bool, str] - (True si sûre, False si bloquée + raison).
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def validate(self, sql):
        n = sql.strip().upper()
        if not n.startswith("SELECT"):
            return False, "Non-SELECT bloqué (connecteur lecture seule)."
        for kw in FORBIDDEN:
            if kw in n:
                return False, f"Mot-clé interdit : {kw}."
        return True, ""

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # @Function Description: Exécute le SELECT validé sur SQLite. Ajoute un LIMIT
    # automatique si absent pour éviter de surcharger la mémoire.
    # ------------------------------------------------------------------------------
    # @Parameter:
    #                 - sql: str - Requête validée.
    # @Returnvalue:
    #                 - QueryResult - Résultat formaté.
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def execute(self, sql):
        if "LIMIT" not in sql.upper():
            sql = sql.rstrip(";") + f" LIMIT {MAX_ROWS};"
        conn = self._conn()
        try:
            cur = conn.cursor(); cur.execute(sql)
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
            return QueryResult(self.source_label, sql, cols, rows, safe=True)
        finally:
            conn.close()