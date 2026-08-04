"""Node synthesis : produit la réponse finale.

Garantit le comportement "no hallucination" :
- Si `doc_results` est vide ET `db_results` est vide (ou ne contient que des
  erreurs), renvoie un message explicite « aucune information trouvée ».
- Sinon, délègue au LLM pour synthétiser avec citations.
"""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama


NO_INFO_MSG = (
    "Je n'ai trouvé aucune information pertinente, ni dans les documents "
    "indexés, ni dans les bases de données auxquelles je suis connecté. "
    "Vérifie qu'un document a été indexé ou qu'une base est bien enregistrée "
    "avec les bons tags (#groupe / @tag)."
)


SYNTH_PROMPT = ChatPromptTemplate.from_template("""
Tu es un assistant RAG multi-sources. Tu dois synthétiser une réponse à la
question de l'utilisateur en utilisant UNIQUEMENT les éléments fournis.

Question : {question}

Contexte documentaire (extraits pertinents) :
{doc_context}

Résultats de bases de données (colonnes + lignes) :
{db_context}

RÈGLE ABSOLUE : si une base de données renvoie un chiffre ou un fait direct
(colonnes "count", "nombre", "total", etc.), tu DOIS utiliser cette donnée
comme réponse principale. N'invente JAMAIS une réponse chiffrée qui ne vient
pas des sources DB ci-dessus.

Rédige une réponse en français, concise, avec des citations :
- [Doc: <nom_fichier> p.<page>] pour chaque extrait utilisé
- [<source_label>] pour chaque source de données utilisée

Si le contexte est insuffisant, dis-le honnêtement.
""".strip())


def _has_meaningful_data(doc_results: list[dict], db_results: list[dict]) -> bool:
    if doc_results:
        return True
    for r in db_results:
        if r.get("rows") or r.get("error"):
            return True
    return False


def _format_doc_context(docs: list[dict]) -> str:
    if not docs:
        return "(aucun)"
    lines = []
    for i, d in enumerate(docs, 1):
        meta = d.get("metadata", {})
        src = meta.get("source", "?")
        page = meta.get("page", "?")
        lines.append(f"[Doc{i}] source={src} p.{page}\n{d.get('content','')[:800]}")
    return "\n\n".join(lines)


def _format_db_context(results: list[dict]) -> str:
    if not results:
        return "(aucune)"
    lines = []
    for r in results:
        label = r.get("source_label", "?")
        if r.get("error"):
            lines.append(f"[{label}] ERREUR: {r['error']}")
            continue
        rows = r.get("rows") or []
        cols = r.get("columns") or []
        if rows:
            header = " | ".join(cols)
            body = "\n".join(
                " | ".join(str(row.get(c, "")) for c in cols) for row in rows[:10]
            )
            lines.append(f"[{label}] (colonnes: {header})\n{body}")
        else:
            lines.append(f"[{label}] (aucune ligne)")
    return "\n\n".join(lines)


def synthesis_node(state) -> dict:
    """Produit la réponse finale — ne retourne QUE les champs modifiés."""
    question = state.get("cleaned_question") or state["question"]
    doc_results = state.get("doc_results") or []
    db_results = state.get("db_results") or []
    upstream_error = state.get("error")

    if not _has_meaningful_data(doc_results, db_results):
        if upstream_error:
            return {
                "answer": f"{NO_INFO_MSG}\n\n(Détail technique : {upstream_error})",
                "sources": [],
            }
        return {"answer": NO_INFO_MSG, "sources": []}

    try:
        llm = ChatOllama(model="qwen3-4b", temperature=0.2)
        chain = SYNTH_PROMPT | llm | StrOutputParser()
        answer = chain.invoke({
            "question": question,
            "doc_context": _format_doc_context(doc_results),
            "db_context": _format_db_context(db_results),
        })
        sources = []
        for d in doc_results:
            m = d.get("metadata", {})
            sources.append({"kind": "doc", "source": m.get("source"),
                            "page": m.get("page")})
        for r in db_results:
            sources.append({"kind": "db", "source_label": r.get("source_label"),
                            "query": r.get("native_query")})
        return {"answer": answer, "sources": sources}
    except Exception as e:
        return {"answer": f"Erreur de synthèse : {e}", "sources": []}