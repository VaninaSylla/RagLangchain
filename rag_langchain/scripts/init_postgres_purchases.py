import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

from rag_langchain.config import settings


def main():
    dsn = settings.pg_dsn
    parts = dsn.replace("postgresql://", "").split("@")
    user_pass, host_port_db = parts[0], parts[1]
    pg_user, pg_password = user_pass.split(":")
    host_port, db_name = host_port_db.split("/")
    pg_host, pg_port = host_port.split(":") if ":" in host_port else (host_port, "5432")

    try:
        conn = psycopg2.connect(user=pg_user, password=pg_password, host=pg_host, port=pg_port, dbname="postgres")
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()

        cur.execute(f"SELECT 1 FROM pg_database WHERE datname='{db_name}'")
        exists = cur.fetchone()
        if exists:
            print(f"Database '{db_name}' already exists. Delete it first if you want to recreate.")
            return

        cur.execute(f"CREATE DATABASE {db_name};")
        print(f"Database '{db_name}' created.")
        cur.close()
        conn.close()

        conn = psycopg2.connect(user=pg_user, password=pg_password, host=pg_host, port=pg_port, dbname=db_name)
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE achats (
                id SERIAL PRIMARY KEY,
                fournisseur TEXT NOT NULL,
                montant REAL NOT NULL,
                date_achat TEXT NOT NULL,
                employe_id INTEGER NOT NULL,
                service_id INTEGER NOT NULL
            )
        """)

        achats = [
            ("Dell", 1500000, "2026-07-01", 1, 2),
            ("Office Depot", 50000, "2026-07-05", 2, 1),
            ("Boulanger", 300000, "2026-07-10", 5, 4),
            ("Castorama", 120000, "2026-07-12", 3, 3)
        ]
        cur.executemany("INSERT INTO achats (fournisseur, montant, date_achat, employe_id, service_id) VALUES (%s, %s, %s, %s, %s)", achats)

        conn.commit()
        print(f"Table 'achats' created and filled with 4 entries in {db_name}.")

    except Exception as e:
        print(f"PostgreSQL error: {e}")

    finally:
        if 'conn' in locals() and conn.closed == 0:
            conn.close()


if __name__ == "__main__":
    main()