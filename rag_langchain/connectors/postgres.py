import re
import psycopg2
from psycopg2.extras import RealDictCursor
from .base import BaseConnector, QueryResult
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

FORBIDDEN = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE", "GRANT", "REVOKE"]
MAX_ROWS = 50

class PostgresConnector(BaseConnector):
    source_label = "Postgres:achats"

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # @Function Description: Initialise le connecteur PostgreSQL avec le DSN et le
    # modèle LLM à utiliser.
    # ------------------------------------------------------------------------------
    # @Parameter:
    #                 - dsn: str - Chaîne de connexion PostgreSQL.
    #                 - llm_model: str - Nom du modèle Ollama.
    # @Returnvalue:
    #                 - (aucun)
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self, dsn, llm_model="qwen3-4b"):
        self.dsn = dsn
        self.llm = ChatOllama(model=llm_model, temperature=0.1)

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # @Function Description: Ouvre une connexion à la base PostgreSQL.
    # ------------------------------------------------------------------------------
    # @Parameter:
    #                 - (aucun)
    # @Returnvalue:
    #                 - psycopg2.connection - Connexion ouverte.
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _conn(self):
        return psycopg2.connect(self.dsn)

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # @Function Description: Extrait le schéma du schéma 'public' de PostgreSQL en
    # interrogeant information_schema.columns.
    # ------------------------------------------------------------------------------
    # @Parameter:
    #                 - (aucun)
    # @Returnvalue:
    #                 - str - Schéma textuel.
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_schema_text(self):
        conn = self._conn()
        try:
            cur = conn.cursor()
            cur.execute("""
                SELECT table_name, column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = 'public'
                ORDER BY table_name, ordinal_position;
            """)
            tables = {}
            for tbl, col, typ in cur.fetchall():
                tables.setdefault(tbl, []).append(f"{col} ({typ})")
            return "\n".join(f"Table {t} : " + ", ".join(c) for t, c in tables.items())
        finally:
            conn.close()

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # @Function Description: Génère une requête SQL PostgreSQL via le LLM avec filtres
    # optionnels (fournisseur, date).
    # ------------------------------------------------------------------------------
    # @Parameter:
    #                 - question: str - Question utilisateur.
    #                 - references: dict - Références extraites.
    #                 - language: str - Langue.
    # @Returnvalue:
    #                 - str - Requête SQL générée.
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def generate_query(self, question, references, language):
        schema = self.get_schema_text()
        ref_hint = ""
        if "fournisseur" in references:
            ref_hint = f"\nFiltre imposé : fournisseur dans {references['fournisseur']}."
        if "date" in references:
            ref_hint += f"\nFiltre date : {references['date']}."

        prompt = ChatPromptTemplate.from_template(
            """Schéma PostgreSQL :\n{schema}\n\nQuestion : {question}{ref_hint}\n
            Écris UNIQUEMENT la requête SQL PostgreSQL brute, terminée par ';'.
            Utilise UNIQUEMENT le schéma public. Pas de Markdown."""
        )
        chain = prompt | self.llm | StrOutputParser()
        raw = chain.invoke({"schema": schema, "question": question, "ref_hint": ref_hint})
        return re.sub(r"```sql|```", "", raw).strip()

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # @Function Description: Valide la requête PostgreSQL (SELECT et mots-clés interdits).
    # ------------------------------------------------------------------------------
    # @Parameter:
    #                 - sql: str - Requête à valider.
    # @Returnvalue:
    #                 - tuple[bool, str] - (True si valide, raison si bloquée).
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def validate(self, sql):
        n = sql.strip().upper()
        if not n.startswith("SELECT"):
            return False, "Non-SELECT bloqué."
        for kw in FORBIDDEN:
            if kw in n:
                return False, f"Mot-clé interdit : {kw}."
        return True, ""

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # @Function Description: Exécute la requête sur PostgreSQL avec un curseur dictionnaire.
    # ------------------------------------------------------------------------------
    # @Parameter:
    #                 - sql: str - Requête validée.
    # @Returnvalue:
    #                 - QueryResult - Résultat de la requête.
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def execute(self, sql):
        if "LIMIT" not in sql.upper():
            sql = sql.rstrip(";") + f" LIMIT {MAX_ROWS};"
        conn = self._conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql)
                rows = [dict(r) for r in cur.fetchall()]
                cols = list(rows[0].keys()) if rows else []
            return QueryResult(self.source_label, sql, cols, rows, safe=True)
        finally:
            conn.close()  