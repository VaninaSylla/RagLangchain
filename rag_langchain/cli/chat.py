"""
chat.py
-------
CLI interactive RAG chat: conversation memory, question reformulation,
reranking, and citations with source + page number.
"""

from rag_langchain.core.ingestion import get_vectorstore
from rag_langchain.core.rag_chain import answer_question
from rag_langchain.core.command_parser import parse_user_input


def main():
    print("Loading vector store...")
    vectorstore = get_vectorstore()
    language = "fr"

    print("Ready! Type your question (or 'exit' to quit).\n")
    print("Palette: /docs, /sql, /employees, /purchases, /services, /all "
          "| references @employe:DUPONT @service:RH\n")

    history_pairs = []

    while True:
        raw = input("Question > ").strip()
        if raw.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break
        if not raw:
            continue

        parsed = parse_user_input(raw)
        question = parsed.cleaned_question or raw

        if parsed.command == "federated":
            from rag_langchain.connectors.federator import Federator
            from rag_langchain.config import settings

            fed = Federator(
                sqlite_path=str(settings.sqlite_path),
                pg_dsn=settings.pg_dsn,
                mongo_uri=settings.mongo_uri,
                mongo_db=settings.mongo_db,
            )
            full_response, _db_sources, _plan = fed.answer(
                question, parsed.references, language
            )
            print(f"\n--- Answer (federated) ---\n{full_response}\n")
            history_pairs.append((question, full_response))
            continue

        answer, sources, standalone_q, _pending_sql = answer_question(
            question, history_pairs, vectorstore, language,
            references=parsed.references,
        )

        if standalone_q and standalone_q != question:
            print(f"\n(Reformulated question: {standalone_q})")

        print(f"\n--- Answer ---\n{answer}\n")

        if sources:
            print("Cited sources:")
            for i, s in enumerate(sources, start=1):
                print(f"  [{i}] {s}")
        print()

        history_pairs.append((question, answer))


if __name__ == "__main__":
    main()
