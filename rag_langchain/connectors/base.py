from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

@dataclass
class QueryResult:
    """Résultat normalisé d'un connecteur, agnostique de la BD sous-jacente."""
    source_label: str          # ex: "SQLite:employés", "Postgres:achats", "Mongo:services"
    native_query: str          # SQL ou pipeline Mongo (pour affichage debug)
    columns: list[str]         # noms de colonnes ou champs clés
    rows: list[dict[str, Any]] # lignes sous forme de dicts
    safe: bool                 # True si exécutée, False si bloquée
    risk: str = "none"         # "none" | "needs_confirmation" | "blocked"
    error: str | None = None

class BaseConnector(ABC):
    """Interface commune à tous les connecteurs de BD."""
    source_label: str = "unknown"

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # @Function Description: Description textuelle du schéma de la base de données,
    # injectée dans le prompt du LLM pour générer la requête.
    # ------------------------------------------------------------------------------
    # @Parameter:
    #                 - (aucun)
    # @Returnvalue:
    #                 - str - Le schéma formaté en texte.
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @abstractmethod
    def get_schema_text(self) -> str:
        pass

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # @Function Description: Demande au LLM de générer la requête native (SQL ou
    # pipeline Mongo) en fonction de la question et des filtres.
    # ------------------------------------------------------------------------------
    # @Parameter:
    #                 - question: str - Question en langage naturel.
    #                 - references: dict - Filtres @type:valeur extraits.
    #                 - language: str - Langue de l'interaction.
    # @Returnvalue:
    #                 - str - La requête générée.
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @abstractmethod
    def generate_query(self, question: str, references: dict, language: str) -> str:
        pass

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # @Function Description: Garde-fou de sécurité validant la requête générée avant
    # exécution (ex: bloquer les requêtes non-SELECT).
    # ------------------------------------------------------------------------------
    # @Parameter:
    #                 - query: str - La requête à valider.
    # @Returnvalue:
    #                 - tuple[bool, str] - (est_valide, raison_du_blocage).
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @abstractmethod
    def validate(self, query: str) -> tuple[bool, str]:
        pass

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # @Function Description: Exécute la requête validée sur la base de données et
    # retourne les résultats sous forme de dictionnaires.
    # ------------------------------------------------------------------------------
    # @Parameter:
    #                 - query: str - La requête validée.
    # @Returnvalue:
    #                 - QueryResult - L'objet contenant les données extraites.
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @abstractmethod
    def execute(self, query: str) -> QueryResult:
        pass

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++--------------
    # @Function Description: Pipeline complet du connecteur : génération, validation,
    # exécution. Gère les erreurs et bloque les actions non sécurisées.
    # ------------------------------------------------------------------------------
    # @Parameter:
    #                 - question: str - Question de l'utilisateur.
    #                 - references: dict - Références extraites par la palette.
    #                 - language: str - Langue.
    # @Returnvalue:
    #                 - QueryResult - Résultat final contenant les données ou l'erreur.
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def answer(self, question: str, references: dict, language: str = "fr") -> QueryResult:
        try:
            q = self.generate_query(question, references, language)
            ok, reason = self.validate(q)
            if not ok:
                return QueryResult(
                    source_label=self.source_label, native_query=q,
                    columns=[], rows=[], safe=False, risk="blocked", error=reason,
                )
            return self.execute(q)
        except Exception as e:
            return QueryResult(
                source_label=self.source_label, native_query="",
                columns=[], rows=[], safe=False, risk="blocked", error=str(e),
            )