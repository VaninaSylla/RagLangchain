import json
import re
from bson import json_util
from pymongo import MongoClient
from .base import BaseConnector, QueryResult
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

MAX_ROWS = 50

class MongoConnector(BaseConnector):
    source_label = "Mongo:services"

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++-
    # @Function Description: Initialise le connecteur MongoDB avec l'URI, le nom de
    # la base et le modèle LLM.
    # ------------------------------------------------------------------------------
    # @Parameter:
    #                 - uri: str - URI de connexion MongoDB.
    #                 - db_name: str - Nom de la base de données.
    #                 - llm_model: str - Nom du modèle Ollama.
    # @Returnvalue:
    #                 - (aucun)
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self, uri, db_name, llm_model="qwen3-4b"):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.llm = ChatOllama(model=llm_model, temperature=0.1)

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++-
    # @Function Description: Échantillonne les collections MongoDB pour déduire les
    # champs disponibles (schéma flexible) et génère une description textuelle.
    # ------------------------------------------------------------------------------
    # @Parameter:
    #                 - (aucun)
    # @Returnvalue:
    #                 - str - Schéma textuel des collections.
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_schema_text(self):
        lines = []
        for coll_name in self.db.list_collection_names():
            sample = list(self.db[coll_name].find().limit(5))
            if not sample:
                lines.append(f"Collection {coll_name} : (vide)")
                continue
            keys = set()
            for doc in sample:
                keys.update(doc.keys())
            lines.append(f"Collection {coll_name} : " + ", ".join(sorted(keys)))
        return "\n".join(lines)

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++-
    # @Function Description: Génère un pipeline d'agrégation MongoDB en JSON via le
    # LLM, en intégrant les filtres @service si présents.
    # ------------------------------------------------------------------------------
    # @Parameter:
    #                 - question: str - Question utilisateur.
    #                 - references: dict - Filtres.
    #                 - language: str - Langue.
    # @Returnvalue:
    #                 - str - Chaîne JSON représentant le pipeline d'agrégation.
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def generate_query(self, question, references, language):
        schema = self.get_schema_text()
        ref_hint = ""
        if "service" in references:
            ref_hint = f"\nFiltre imposé : nom du service dans {references['service']}."

        prompt = ChatPromptTemplate.from_template(
            """Schéma MongoDB :\n{schema}\n\nQuestion : {question}{ref_hint}\n
            Écris UNIQUEMENT un pipeline d'agrégation MongoDB en JSON valide,
            commençant par '[' et finissant par ']'. Pas de Markdown, pas de
            commentaires. Limite à {max_rows} documents avec $limit."""
        )
        chain = prompt | self.llm | StrOutputParser()
        raw = chain.invoke({
            "schema": schema, "question": question,
            "ref_hint": ref_hint, "max_rows": MAX_ROWS,
        })
        return re.sub(r"```json|```", "", raw).strip()

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++-
    # @Function Description: Valide le JSON du pipeline et bloque les opérateurs
    # destructeurs MongoDB ($drop, $out, etc.).
    # ------------------------------------------------------------------------------
    # @Parameter:
    #                 - pipeline_str: str - JSON du pipeline.
    # @Returnvalue:
    #                 - tuple[bool, str] - (True si valide, raison du blocage sinon).
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def validate(self, pipeline_str):
        try:
            forbidden_ops = ["$drop", "$out", "$merge", "$rename"]
            for op in forbidden_ops:
                if op in pipeline_str:
                    return False, f"Opérateur interdit : {op}."
            json.loads(pipeline_str)
            return True, ""
        except json.JSONDecodeError as e:
            return False, f"JSON invalide : {e}"

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++-
    # @Function Description: Exécute le pipeline d'agrégation sur la première collection
    # de la base, et formate les résultats pour les rendre serialisables.
    # ------------------------------------------------------------------------------
    # @Parameter:
    #                 - pipeline_str: str - Pipeline validé.
    # @Returnvalue:
    #                 - QueryResult - Résultats de l'agrégation.
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def execute(self, pipeline_str):
        pipeline = json.loads(pipeline_str)
        coll_name = self.db.list_collection_names()[0]
        rows = list(self.db[coll_name].aggregate(pipeline))
        rows = json.loads(json_util.dumps(rows))
        cols = list(rows[0].keys()) if rows else []
        return QueryResult(self.source_label, pipeline_str, cols, rows, safe=True)