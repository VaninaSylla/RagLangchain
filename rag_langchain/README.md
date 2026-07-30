# RAG interactif v2 — Upload, Chroma, Reranking, Citations

## Nouveautés par rapport à la v1

| Fonctionnalité | v1 | v2 |
|---|---|---|
| Ajout de documents | manuel dans `documents/` + script | **upload direct dans l'interface web** |
| Vectorstore | FAISS | **Chroma** (persistance auto, ajout incrémental) |
| Extraction PDF | pypdf | **PyMuPDF** (meilleure préservation de la mise en page) |
| Citations | nom de fichier seul | **nom de fichier + numéro de page** |
| Recherche | vectorielle simple (top-4) | **top-10 vectoriel puis reranking cross-encoder → top-4** |
| Conversation | question isolée à chaque fois | **mémoire + reformulation automatique de la question** |

## Installation

```powershell
pip uninstall faiss-cpu -y
pip install -r requirements.txt
```

⚠️ Le tout premier lancement téléchargera automatiquement le modèle de reranking
(`cross-encoder/ms-marco-MiniLM-L-6-v2`, ~80 Mo) depuis Hugging Face — il faut
une connexion internet la première fois, ensuite il est mis en cache localement.

## Utilisation

### Interface web (recommandée — upload inclus)
```powershell
streamlit run app_streamlit.py
```
1. Dans la barre latérale, dépose tes fichiers PDF/TXT/DOCX
2. Clique sur "📥 Indexer les documents"
3. Pose tes questions dans le chat — chaque réponse affiche ses sources avec numéro de page

### Ligne de commande (sans upload, lit `documents/`)
```powershell
python ingest.py       # indexe une fois
python rag_app.py       # puis pose tes questions
```

## Comment fonctionne le pipeline "interactif"

1. **Reformulation** : ta question est réécrite par le LLM en tenant compte des 3 derniers échanges, pour être compréhensible seule (`rag_chain_utils.condense_question`)
2. **Récupération large** : Chroma renvoie les 10 chunks les plus proches de la question reformulée
3. **Reranking** : un cross-encoder réévalue précisément ces 10 chunks et ne garde que les 4 meilleurs
4. **Génération avec citations** : le LLM répond en citant `[1]`, `[2]`... qui renvoient à la liste des sources (fichier + page) affichée sous la réponse

## Limites connues (à mentionner dans le rapport)

- L'extraction PyMuPDF améliore la structure générale du texte mais **ne reconnaît pas les formules LaTeX/mathématiques** comme des équations — elles sont extraites en texte brut, potentiellement désordonné. Une extraction fidèle des formules nécessiterait un OCR spécialisé (ex. Nougat), hors du périmètre de ce projet.
- Le reranking ajoute un léger temps de calcul supplémentaire à chaque question (le cross-encoder tourne sur CPU).
- La mémoire de conversation n'est pas persistée entre deux lancements de l'application (elle est réinitialisée à chaque redémarrage de Streamlit).
- Le choix de langue (français/anglais) ne s'applique qu'à la **génération de la réponse** — le modèle d'embedding (`nomic-embed-text`) reste le même quelle que soit la langue choisie. Si tes documents sont en anglais et que tu poses une question en français (ou l'inverse), la récupération reste généralement correcte car ce modèle gère raisonnablement le multilingue, mais la pertinence peut légèrement baisser par rapport à une question posée dans la langue d'origine des documents.