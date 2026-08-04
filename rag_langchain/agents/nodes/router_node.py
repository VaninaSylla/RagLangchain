"""Node router : décide si la question parle aux docs, à la BD, ou les deux.

Comportement spécial « auto_investigate » : si la question est posée **sans**
aucun tag/groupe explicite, le router bascule dans ce mode, qui force
l'orchestrateur à interroger **tous** les connecteurs et la collection docs.
Si rien ne remonte, le node `db_agent` tentera un refresh de snapshot avant
que `synthesis` ne renvoie un message « aucune information ».
"""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

from ..state import AgentState, Route


ROUTER_PROMPT = ChatPromptTemplate.from_template("""
Tu es un routeur intelligent pour un système RAG multi-sources.

L'utilisateur a posé la question : « {question} »

Sources disponibles :
- documents : documents indexés dans ChromaDB (PDF, DOCX, PPTX, TXT)
- databases : bases de données SQL/NoSQL connectées (résolues via #groupe / @tag)
- conversation : la conversation actuelle (mémoire)

Décide la route :
- "conversation" : si la question est purement méta (résumer, rappeler, etc.)
- "documents" : si la question cherche clairement du contenu dans des documents
- "database" : si la question cherche clairement des données structurées
- "mixed" : si les deux sont possibles et la combinaison aide
- "auto_investigate" : si la question est vague / sans contexte et il faut
  tenter toutes les sources pour trouver la bonne.

Réponds UNIQUEMENT par un JSON : {{"route": "<valeur>"}}
""".strip())


def _classify_with_llm(question: str) -> Route:
    llm = ChatOllama(model="qwen3-4b", temperature=0.0)
    chain = ROUTER_PROMPT | llm | StrOutputParser()
    import json, re
    raw = chain.invoke({"question": question})
    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0)).get("route", "unknown")  # type: ignore[return-value]
        except Exception:
            pass
    return "unknown"


def router_node(state: dict) -> dict:
    has_tags = bool(state.get("groups")) or bool(state.get("tags"))
    if not has_tags:
        # Pas de contexte explicite → mode investigation automatique
        return {"route": "auto_investigate"}

    try:
        route = _classify_with_llm(state.get("cleaned_question") or state["question"])
    except Exception:
        route = "mixed"
    return {"route": route}