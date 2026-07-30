from pymongo import MongoClient

from rag_langchain.config import settings


def main():
    try:
        client = MongoClient(settings.mongo_uri, serverSelectionTimeoutMS=3000)
        client.admin.command('ping')

        db = client[settings.mongo_db]
        services_col = db["services"]

        services_col.delete_many({})

        services = [
            {"_id": 1, "nom": "Ressources Humaines", "responsable": "Marie Curie", "budget": 5000000, "localisation": "Etage 1"},
            {"_id": 2, "nom": "Informatique", "responsable": "Sophie Bernard", "budget": 15000000, "localisation": "Etage 2"},
            {"_id": 3, "nom": "Logistique", "responsable": "Paul Martin", "budget": 8000000, "localisation": "Sous-sol"},
            {"_id": 4, "nom": "Achats", "responsable": "Lucas Lopez", "budget": 3000000, "localisation": "Etage 1"}
        ]

        services_col.insert_many(services)
        print(f"MongoDB database '{settings.mongo_db}' created with 4 services.")

    except Exception as e:
        print(f"MongoDB connection error: {e}")
        print("Make sure the MongoDB (mongod) server is running on your machine.")


if __name__ == "__main__":
    main()