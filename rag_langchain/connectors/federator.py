import json, re
from typing import Optional
from .base import BaseConnector, QueryResult
from .sqlite import SQLiteConnector
from .postgres import PostgresConnector
from .mongo import MongoConnector
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

PLANNER_PROMPT = ChatPromptTemplate.from_template(
    """Tu es un planificateur de requêtes multi-bases. Voici les sources disponibles:

1. SQLite:employés — contient les employés (nom, poste, service_id, salaire)
2. Postgres:achats — contient les achats (fournisseur, montant, employe_id, service_id)
3. Mongo:services — contient les services (service_id, nom, responsable, budget)

Question de l'utilisateur : {question}

Décide quelles sources il faut interroger pour répondre. Formule une sous-question claire pour chaque source. Si une source dépend d'une autre pour obtenir un ID, indique-le clairement dans la sous-question (ex: "Trouve le responsable du service ayant l'ID trouvé dans SQLite").

Réponds en JSON STRICT :
{{
  "sources": [
    {{"source": "sqlite", "sub_question": "..."}},
    {{"source": "postgres", "sub_question": "..."}},
    {{"source": "mongo", "sub_question": "..."}}
  ]
}}

Ne mentionne QUE les sources pertinentes.
"""
)

SYNTHESIS_PROMPT = ChatPromptTemplate.from_template(
    """Tu es un analyste de données expert. Tu dois répondre à la question de l'utilisateur
en croisant les résultats provenant de différentes bases de données.

Question initiale : {question}

Résultats par source :
{results}

CONSIGNES STRICTES :
1. Analyse les tableaux de résultats ci-dessus.
2. Fais la jointure mentale entre les tableaux en utilisant les colonnes ID (ex: si SQLite donne service_id=2, cherche la ligne avec service_id=2 dans Mongo).
3. Rédige une réponse claire en {language} qui fusionne ces informations.
4. Cite la source entre crochets pour chaque fait, par exemple [SQLite:employés] ou [Mongo:services].

Réponse finale (avec citations) :"""
)

class Federator:
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # @Function Description: Initialise le fédérateur en instanciant les 3 connecteurs
    # spécifiques (SQLite, Postgres, Mongo) et le LLM de synthèse.
    # ------------------------------------------------------------------------------
    # @Parameter:
    #                 - sqlite_path: str - Chemin de la base SQLite.
    #                 - pg_dsn: str - DSN de connexion PostgreSQL.
    #                 - mongo_uri: str - URI de connexion MongoDB.
    #                 - mongo_db: str - Nom de la base MongoDB.
    #                 - llm_model: str - Nom du modèle Ollama.
    # @Returnvalue:
    #                 - (aucun)
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self, sqlite_path, pg_dsn, mongo_uri, mongo_db,
                 llm_model="qwen3-4b"):
        self.connectors: dict[str, BaseConnector] = {
            "sqlite":   SQLiteConnector(sqlite_path, llm_model),
            "postgres": PostgresConnector(pg_dsn, llm_model),
            "mongo":    MongoConnector(mongo_uri, mongo_db, llm_model),
        }
        self.llm = ChatOllama(model=llm_model, temperature=0.2)

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # @Function Description: Utilise le LLM pour analyser la question globale et
    # décider quelles bases de données interroger, en générant une sous-question
    # pour chacune.
    # ------------------------------------------------------------------------------
    # @Parameter:
    #                 - question: str - Question globale de l'utilisateur.
    #                 - language: str - Langue de l'interaction.
    # @Returnvalue:
    #                 - dict - Plan JSON contenant les sources et sous-questions.
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _plan(self, question, language):
        chain = PLANNER_PROMPT | self.llm | StrOutputParser()
        raw = chain.invoke({"question": question})
        cleaned = re.sub(r"```json|```", "", raw).strip()
        return json.loads(cleaned)

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # @Function Description: Orchestre la requête multi-sources : planifie, exécute
    # les sous-requêtes en parallèle sur les connecteurs, puis fusionne les résultats
    # via un LLM de synthèse avec citations.
    # ------------------------------------------------------------------------------
    # @Parameter:
    #                 - question: str - Question utilisateur.
    #                 - references: dict - Filtres de la palette de commandes.
    #                 - language: str - Langue.
    # @Returnvalue:
    #                 - tuple[str, list[QueryResult], dict] - (Réponse synthétisée,
    #                   liste des résultats bruts par BD, plan d'exécution).
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def answer(self, question, references, language="fr"):
        try:
            plan = self._plan(question, language)
        except Exception as e:
            return f"❌ Échec de planification : {e}", [], []

        results: list[QueryResult] = []
        for entry in plan["sources"]:
            src = entry["source"]
            sub_q = entry["sub_question"]
            if src not in self.connectors:
                continue
            res = self.connectors[src].answer(sub_q, references, language)
            results.append(res)

        results_text_parts = []
        citations_info = []
        for r in results:
            header = f"[{r.source_label}] Requête: {r.native_query}"
            if not r.safe:
                body = f"Bloquée : {r.error}"
            elif not r.rows:
                body = "(aucun résultat)"
            else:
                body = " | ".join(r.columns) + "\n" + \
                       "\n".join(" | ".join(str(v) for v in row.values())
                                 for row in r.rows[:20])
            results_text_parts.append(f"{header}\n{body}")
            citations_info.append(r)

        results_text = "\n\n---\n\n".join(results_text_parts)

        chain = SYNTHESIS_PROMPT | self.llm | StrOutputParser()
        answer = chain.invoke({
            "question": question, "results": results_text, "language": language,
        })
        return answer, citations_info, plan