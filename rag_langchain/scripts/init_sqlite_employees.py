from pathlib import Path
import sqlite3

from rag_langchain.config import settings


def main():
    db_path = settings.sqlite_path
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if db_path.exists():
        print(f"Database '{db_path}' already exists. Delete it to recreate.")
        return

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE employes (
            id INTEGER PRIMARY KEY,
            nom TEXT NOT NULL,
            poste TEXT NOT NULL,
            service_id INTEGER NOT NULL,
            salaire REAL NOT NULL,
            date_embauche TEXT NOT NULL
        )
    """)

    employes = [
        (1, "Jean Dupont", "Developeur", 2, 450000, "2022-03-15"),
        (2, "Marie Curie", "Responsable RH", 1, 600000, "2021-01-10"),
        (3, "Paul Martin", "Magasinier", 3, 350000, "2023-06-01"),
        (4, "Sophie Bernard", "Cheffe de Projet IT", 2, 550000, "2020-11-20"),
        (5, "Lucas Lopez", "Acheteur", 4, 400000, "2024-02-05")
    ]
    cur.executemany("INSERT INTO employes VALUES (?, ?, ?, ?, ?, ?)", employes)

    conn.commit()
    conn.close()
    print(f"SQLite database created at '{db_path}' with 5 employees.")


if __name__ == "__main__":
    main()