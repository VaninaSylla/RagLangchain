"""
rag_chain_utils.py
-------------------
Logique du pipeline RAG "interactif" :
  1. Reformulation de la question à partir de l'historique (condense question)
  2. Récupération large dans Chroma (top ~10)
  3. Reranking précis avec un cross-encoder (garde le top 4)
  4. Génération de la réponse avec citations [n] renvoyant à source + page
"""

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from rag_langchain.config import settings


_reranker = None


def get_reranker():
    """
    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # @Function Description: Charge (une seule fois, mis en cache dans la variable
    # globale _reranker) le modèle cross-encoder utilisé pour réévaluer précisément
    # les chunks candidats avant la génération. L'import de sentence-transformers
    # est différé jusqu'au premier appel pour ne pas ralentir le démarrage.
    # ------------------------------------------------------------------------------
    # @Parameter:
    #                 - (aucun)
    # @Returnvalue:
    #                 - CrossEncoder - Instance du modèle RERANK_MODEL, prête à
    #                   noter des paires (question, chunk). Si sentence-transformers
    #                   n'est pas installé, retourne None (fallback sans rerank).
    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    global _reranker
    if _reranker is None:
        # Import différé : sentence-transformers entraîne le chargement de torch/transformers,
        # plusieurs secondes. On ne paie ce coût qu'à la première question posée, pas au
        # démarrage de l'application (ce qui accélère nettement l'ouverture de Streamlit).
        try:
            import os
            # Mode hors-ligne : évite que Hugging Face revérifie en ligne s'il y a une mise à jour
            # à chaque lancement (peut traîner sur une connexion lente ou instable).
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            from sentence_transformers import CrossEncoder
            _reranker = CrossEncoder(settings.rerank_model)
        except Exception:
            # sentence-transformers non installé ou modèle indisponible → on continue
            # sans rerank plutôt que de faire planter le pipeline RAG.
            _reranker = None
    return _reranker


def get_llm():
    """
    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @Function Description: Instancie le modèle de génération (LLM) utilisé
    pour toutes les tâches de langage du pipeline : reformulation, routage,
    et génération de la réponse finale.
    --------------------------------------------------------------------------
    @Parameter:
        - (aucun)

    @Returnvalue:
        - ChatOllama - Instance configurée sur LLM_MODEL ("qwen3-4b"), avec
          une température de 0.1 (réponses factuelles, peu créatives).
    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    return ChatOllama(model=settings.llm_model, temperature=0.1)


# ----------------------------------------------------------------------
# 1. Reformulation de la question (rend le RAG "interactif")
# ----------------------------------------------------------------------
CONDENSE_PROMPTS = {
    "fr": ChatPromptTemplate.from_template(
        """Voici l'historique de la conversation :
{chat_history}

Nouveau message de l'utilisateur : {question}

Ta seule tâche : décider si ce message a besoin de l'historique pour être compris.

Ne reformule QUE si le message contient une référence implicite qui n'a de sens
qu'avec l'historique (un pronom comme "ça"/"cela", une comparaison du type
"et pourquoi pas X ?", une continuation du type "et sinon ?").

Dans TOUS les autres cas — et en particulier si le message est déjà une
instruction complète et autonome (par exemple : "fais un résumé du document
en 10 lignes", "traduis ce passage", "liste les points principaux", "explique
X") — renvoie ce message EXACTEMENT tel quel, sans le transformer en question,
sans changer son sujet, sans ajouter de détails venant de l'historique.

En cas de doute, ne reformule PAS. Réponds UNIQUEMENT avec le message final
(reformulé ou non), rien d'autre."""
    ),
    "en": ChatPromptTemplate.from_template(
        """Here is the conversation history:
{chat_history}

New user message: {question}

Your only task: decide whether this message needs the history to be understood.

Only rewrite it if it contains an implicit reference that only makes sense
with the history (a pronoun like "that", a comparison like "why not X
instead?", a continuation like "and otherwise?").

In ALL other cases — especially if the message is already a complete,
standalone instruction (e.g. "summarize the document in 10 lines",
"translate this passage", "list the main points", "explain X") — return this
message EXACTLY as is, without turning it into a question, without changing
its topic, without pulling in details from the history.

When in doubt, do NOT rewrite. Reply ONLY with the final message (rewritten
or not), nothing else."""
    ),
}


def condense_question(question: str, chat_history: list[tuple[str, str]], language: str = "fr") -> str:
    """
    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @Function Description: Reformule la question de l'utilisateur en une
    question autonome si (et seulement si) elle dépend de l'historique pour
    être comprise (ex: "et pourquoi pas X ?"). Les instructions déjà
    complètes (résumé, traduction, liste...) sont renvoyées inchangées, pour
    ne pas dénaturer l'intention de l'utilisateur.
    --------------------------------------------------------------------------
    @Parameter:
        - question: str - Message brut tapé par l'utilisateur.
        - chat_history: list[tuple[str, str]] - Liste de paires
          (question, réponse) des échanges précédents. Seuls les 3 derniers
          sont utilisés ici.
        - language: str - "fr" ou "en", langue du prompt de reformulation.

    @Returnvalue:
        - str - La question reformulée (autonome), ou la question d'origine
          si aucun historique n'existe, si aucune reformulation n'était
          nécessaire, ou si la sortie du LLM ressemble à une explication
          plutôt qu'à une question propre (garde-fou de robustesse).
    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    if not chat_history:
        return question

    history_text = "\n".join(f"Q: {q}\nR: {r}" for q, r in chat_history[-3:])  # 3 derniers échanges
    llm = get_llm()
    chain = CONDENSE_PROMPTS[language] | llm | StrOutputParser()
    raw = chain.invoke({"chat_history": history_text, "question": question}).strip()

    # Garde-fou : malgré la consigne, un petit modèle comme llama3.2 produit
    # parfois son raisonnement complet au lieu d'une simple question
    # reformulée (ex: "Le message X n'a pas besoin de l'historique car...").
    # Plutôt que d'utiliser ce texte pollué pour la recherche, on détecte ce
    # cas et on revient prudemment à la question d'origine.
    explanation_markers = [
        "car le message", "n'a pas besoin", "la réponse est",
        "il s'agit d'une instruction", "because the message", "the answer is",
    ]
    looks_like_explanation = (
        len(raw) > 3 * max(len(question), 20)
        or f'"{question}"' in raw
        or f"« {question} »" in raw
        or any(marker in raw.lower() for marker in explanation_markers)
    )
    if looks_like_explanation:
        return question

    return raw


# ----------------------------------------------------------------------
# 1bis. Routeur : la question porte-t-elle sur les documents, ou sur la
# conversation elle-même (ex: "quelle était ma dernière question ?") ?
# ----------------------------------------------------------------------
ROUTER_PROMPTS = {
    "fr": ChatPromptTemplate.from_template(
        """Une question va être posée à un assistant. Décide si cette
question porte :
- sur le CONTENU DES DOCUMENTS indexés (ex: "qu'est-ce que X ?", "résume Y",
  "de quoi parle [nom de document]", "de quoi parle ce document")
- sur LA CONVERSATION elle-même — c'est-à-dire les échanges DÉJÀ EUS entre
  toi et l'utilisateur (ex: "quelle était ma dernière question ?",
  "rappelle-moi mes 20 dernières questions", "de quoi a-t-on parlé avant
  DANS CETTE CONVERSATION ?")
- sur des DONNÉES STRUCTURÉES en base de données — que ce soit pour LIRE
  (ex: "combien de produits en stock ?", "liste les commandes du client X")
  OU pour MODIFIER ces données (ex: "ajoute un produit", "supprime la
  commande n°3", "mets à jour le prix de X")

ATTENTION, piège fréquent : "de quoi parle [un nom, un document, un sujet]"
est presque TOUJOURS une question DOCUMENTS (on demande le contenu d'un
document ou d'un sujet), MÊME SI la formulation ressemble à "de quoi a-t-on
parlé". Seule une question qui mentionne explicitement "notre conversation",
"nos échanges", "mes questions précédentes" ou équivalent est CONVERSATION.

Exemples :
- "de quoi parle ce document ?" -> DOCUMENTS
- "de quoi parle rapport.docx ?" -> DOCUMENTS
- "de quoi parle heroes ?" -> DOCUMENTS
- "résume le document" -> DOCUMENTS
- "quelle était ma dernière question ?" -> CONVERSATION
- "de quoi a-t-on parlé avant dans cette conversation ?" -> CONVERSATION
- "rappelle-moi mes questions précédentes" -> CONVERSATION
- "combien de produits en stock ?" -> DATABASE
- "ajoute un produit" -> DATABASE

Question à classer : {question}

Réponds UNIQUEMENT par un seul mot : CONVERSATION, DATABASE ou DOCUMENTS."""
    ),
    "en": ChatPromptTemplate.from_template(
        """A question is about to be asked to an assistant. Decide if this
question is about:
- the CONTENT OF THE INDEXED DOCUMENTS (e.g. "what is X?", "summarize Y",
  "what is [document name] about", "what is this document about")
- THE CONVERSATION itself — meaning exchanges that ALREADY HAPPENED between
  you and the user (e.g. "what was my last question?", "remind me of my
  last 20 questions", "what did we talk about earlier IN THIS
  CONVERSATION?")
- STRUCTURED DATA in a database — whether to READ it (e.g. "how many
  products in stock?", "list orders from client X") OR to MODIFY it (e.g.
  "add a product", "delete order #3", "update the price of X")

WATCH OUT for a common trap: "what is [a name, a document, a topic] about"
is ALMOST ALWAYS a DOCUMENTS question (asking for the content of a document
or topic), EVEN IF the phrasing resembles "what did we talk about". Only a
question that explicitly mentions "our conversation", "our exchanges", "my
previous questions" or similar is CONVERSATION.

Examples:
- "what is this document about?" -> DOCUMENTS
- "what is report.docx about?" -> DOCUMENTS
- "what is heroes about?" -> DOCUMENTS
- "summarize the document" -> DOCUMENTS
- "what was my last question?" -> CONVERSATION
- "what did we talk about earlier in this conversation?" -> CONVERSATION
- "remind me of my previous questions" -> CONVERSATION
- "how many products in stock?" -> DATABASE
- "add a product" -> DATABASE

Question to classify: {question}

Reply with ONLY one word: CONVERSATION, DATABASE or DOCUMENTS."""
    ),
}

MAX_HISTORY_FOR_RECALL = 20  # nombre maximum d'échanges rappelables sur la conversation


def classify_question(question: str, chat_history: list[tuple[str, str]], language: str = "fr") -> str:
    """
    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @Function Description: Détermine si la question porte sur le contenu des
    documents indexés, sur la conversation elle-même (ex: "quelle était ma
    dernière question ?"), ou sur des données structurées interrogeables en
    SQL (ex: "combien de produits en stock ?"). Permet d'orienter le pipeline
    vers la bonne branche de traitement.
    --------------------------------------------------------------------------
    @Parameter:
        - question: str - Question posée par l'utilisateur.
        - chat_history: list[tuple[str, str]] - Historique de la conversation.
          Sans historique, la question ne peut porter que sur les documents
          ou les données structurées.
        - language: str - "fr" ou "en", langue du prompt de classification.

    @Returnvalue:
        - str - "conversation", "database" ou "documents".
    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    llm = get_llm()
    chain = ROUTER_PROMPTS[language] | llm | StrOutputParser()
    result = chain.invoke({"question": question}).strip().upper()

    if "DATABASE" in result:
        return "database"

    if "CONVERSATION" in result:
        # Filet de sécurité : le routeur (un petit modèle) confond parfois
        # "de quoi parle X" (une question sur un document) avec "de quoi
        # a-t-on parlé" (une question sur la conversation), à cause de leur
        # ressemblance de surface. On n'accepte le verdict CONVERSATION que
        # si la question contient un marqueur explicite s'y référant.
        conversation_markers = {
            "fr": ["conversation", "échange", "questions précédentes", "historique",
                   "on a parlé", "avons discuté", "tu m'as dit", "je t'ai demandé",
                   "ma dernière question", "mes dernières questions",
                   "combien de questions", "cite les", "que j'ai posé", "que j ai pose"],
            "en": ["conversation", "exchange", "previous questions", "history",
                   "we talked", "you told me", "i asked you",
                   "my last question", "my last questions", "how many questions",
                   "list the questions", "that i asked"],
        }
        
        question_lower = question.lower()
        if any(marker in question_lower for marker in conversation_markers[language]):
            return "conversation"
        return "documents"

    return "documents"


META_ANSWER_PROMPTS = {
    "fr": ChatPromptTemplate.from_template(
        """Voici l'historique des derniers échanges de cette conversation
(question n°1 = la plus ancienne des échanges listés) :

{history}

Question de l'utilisateur À PROPOS de cette conversation : {question}

Réponds en te basant UNIQUEMENT sur cet historique, de façon précise et
concise. Si l'information demandée dépasse l'historique fourni ci-dessus,
dis-le clairement plutôt que d'inventer."""
    ),
    "en": ChatPromptTemplate.from_template(
        """Here is the history of the last exchanges in this conversation
(question #1 = the oldest of the listed exchanges):

{history}

User question ABOUT this conversation: {question}

Answer based ONLY on this history, precisely and concisely. If the requested
information goes beyond the history provided above, clearly say so rather
than making it up."""
    ),
}


def format_history_for_recall(chat_history: list[tuple[str, str]], max_pairs: int = MAX_HISTORY_FOR_RECALL) -> str:
    """
    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @Function Description: Met en forme les derniers échanges de la
    conversation en un texte numéroté, lisible par le LLM, pour répondre aux
    questions portant sur la conversation elle-même.
    --------------------------------------------------------------------------
    @Parameter:
        - chat_history: list[tuple[str, str]] - Historique complet
          (question, réponse) de la conversation.
        - max_pairs: int - Nombre maximum d'échanges à conserver, en partant
          des plus récents (par défaut MAX_HISTORY_FOR_RECALL = 20).

    @Returnvalue:
        - str - Texte numéroté ("1. Question : ... Réponse : ...", etc.),
          prêt à être injecté dans META_ANSWER_PROMPTS.
    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    recent = chat_history[-max_pairs:]
    lines = []
    for i, (q, a) in enumerate(recent, start=1):
        lines.append(f"{i}. Question : {q}\n   Réponse : {a}")
    return "\n".join(lines)


def answer_from_history(question: str, chat_history: list[tuple[str, str]], language: str = "fr") -> str:
    """
    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @Function Description: Répond directement à une question portant sur la
    conversation elle-même (ex: "quelle était ma dernière question ?"), à
    partir de l'historique uniquement — sans recherche documentaire ni
    citations, puisque la réponse ne provient pas des documents indexés.
    --------------------------------------------------------------------------
    @Parameter:
        - question: str - Question posée par l'utilisateur, à propos de la
          conversation.
        - chat_history: list[tuple[str, str]] - Historique (question, réponse)
          de la conversation.
        - language: str - "fr" ou "en".

    @Returnvalue:
        - str - Réponse générée à partir des derniers échanges (jusqu'à
          MAX_HISTORY_FOR_RECALL), en une seule fois (non streamée). Si aucun
          historique n'existe encore, renvoie directement un message le
          signalant, sans appeler le LLM.
    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    if not chat_history:
        no_history_msg = {
            "fr": "C'est notre tout premier échange dans cette conversation : il n'y a pas encore d'historique à te rappeler.",
            "en": "This is our very first exchange in this conversation: there's no history yet for me to recall.",
        }
        return no_history_msg[language]

    history_text = format_history_for_recall(chat_history)
    llm = get_llm()
    chain = META_ANSWER_PROMPTS[language] | llm | StrOutputParser()
    return chain.invoke({"history": history_text, "question": question})


def answer_from_history_stream(question: str, chat_history: list[tuple[str, str]], language: str = "fr"):
    """
    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @Function Description: Version en streaming de answer_from_history,
    utilisée par l'interface Streamlit pour afficher la réponse
    progressivement, mot par mot, plutôt que d'attendre la réponse complète.
    --------------------------------------------------------------------------
    @Parameter:
        - question: str - Question posée par l'utilisateur, à propos de la
          conversation.
        - chat_history: list[tuple[str, str]] - Historique (question, réponse)
          de la conversation.
        - language: str - "fr" ou "en".

    @Returnvalue:
        - Iterator[str] - Flux de fragments de texte à concaténer au fur et à
          mesure de leur réception (voir StrOutputParser + .stream()). Si
          aucun historique n'existe encore, renvoie un flux à un seul élément
          contenant le message le signalant, sans appeler le LLM.
    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    if not chat_history:
        no_history_msg = {
            "fr": "C'est notre tout premier échange dans cette conversation : il n'y a pas encore d'historique à te rappeler.",
            "en": "This is our very first exchange in this conversation: there's no history yet for me to recall.",
        }
        def _single_chunk():
            yield no_history_msg[language]
        return _single_chunk()

    history_text = format_history_for_recall(chat_history)
    llm = get_llm()
    chain = META_ANSWER_PROMPTS[language] | llm | StrOutputParser()
    return chain.stream({"history": history_text, "question": question})


# ----------------------------------------------------------------------
# 2 & 3. Récupération + reranking
# ----------------------------------------------------------------------
def retrieve_and_rerank(query: str, vectorstore, k_final: int = settings.final_k, source_filter: str | list[str] | None = None):
    """
    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # @Author:        John MANGA | Digit-Tech-Innov Solutions and Services
    # @Creation:      20.07.2026
    # ------------------------------------------------------------------------------
    # @Function Description: Récupère les chunks les plus pertinents pour une requête,
    # en deux temps : une recherche vectorielle large dans Chroma (rapide,
    # approximative), puis un reranking précis par cross-encoder qui ne garde que
    # les meilleurs résultats. Peut aussi restreindre la recherche à un ou plusieurs
    # documents via un filtre sur les métadonnées.
    # ------------------------------------------------------------------------------
    # @Parameter:
    #                 - query: str - Texte de la requête (question reformulée).
    #                 - vectorstore: Chroma - Base vectorielle à interroger.
    #                 - k_final: int - Nombre de chunks à conserver après reranking.
    #                 - source_filter: str | list[str] | None - Nom de fichier (ou
    #                   liste) pour restreindre la recherche (None = tous).
    # @Returnvalue:
    #                 - list[Document] - Les k_final chunks les plus pertinents, triés
    #                   du meilleur au moins bon score de reranking. Liste vide si
    #                   aucun candidat n'a été trouvé.
    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    
    search_kwargs = {"k": settings.retrieve_k}
    if source_filter:
            if isinstance(source_filter, list):
                search_kwargs["filter"] = {"source": {"$in": source_filter}}
            else:
                search_kwargs["filter"] = {"source": source_filter}

    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
    candidates = retriever.invoke(query)

    if not candidates:
            return []

    reranker = get_reranker()
    if reranker is None:
        # Pas de reranker dispo → on garde les top-k_final retournés par Chroma (déjà triés).
        return candidates[:k_final]

    pairs = [(query, doc.page_content) for doc in candidates]
    scores = reranker.predict(pairs)

    scored = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored[:k_final]]


# ----------------------------------------------------------------------
# 4. Génération avec citations [n] -> source + page
# ----------------------------------------------------------------------
ANSWER_PROMPTS = {
    "fr": ChatPromptTemplate.from_template(
        """Tu es un assistant qui répond UNIQUEMENT à partir des extraits fournis
ci-dessous, en FRANÇAIS. Si la réponse ne s'y trouve pas, dis clairement que
tu ne sais pas.

Pour CHAQUE affirmation, ajoute une référence entre crochets au numéro de
l'extrait correspondant, par exemple [1] ou [2][3]. N'invente jamais de
référence qui ne figure pas ci-dessous.

Extraits disponibles :
{context}

Question : {question}

Réponse (en français, avec références [n]) :"""
    ),
    "en": ChatPromptTemplate.from_template(
        """You are an assistant who answers ONLY based on the excerpts provided
below, in ENGLISH. If the answer isn't in them, clearly say you don't know.

For EVERY claim, add a bracketed reference to the matching excerpt number,
e.g. [1] or [2][3]. Never invent a reference that isn't listed below.

Available excerpts:
{context}

Question: {question}

Answer (in English, with [n] references):"""
    ),
}


def format_context_with_sources(chunks):
    """
    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @Function Description: Numérote les chunks retenus après reranking et
    construit en parallèle deux éléments : le texte de contexte à donner au
    LLM (avec ses numéros [1], [2]...) et la liste des sources correspondantes
    (nom de fichier + page), affichée à l'utilisateur après la réponse.
    --------------------------------------------------------------------------
    @Parameter:
        - chunks: list[Document] - Chunks sélectionnés par
          retrieve_and_rerank.

    @Returnvalue:
        - tuple[str, list[str]] - (texte de contexte numéroté, liste des
          libellés de sources "fichier, page X" dans le même ordre que la
          numérotation du contexte).
    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    context_parts = []
    sources = []
    for i, doc in enumerate(chunks, start=1):
        source = doc.metadata.get("source", "inconnue")
        page = doc.metadata.get("page")
        label = f"{source}" + (f", page {page}" if page else "")
        context_parts.append(f"[{i}] (source : {label})\n{doc.page_content}")
        sources.append(label)
    return "\n\n---\n\n".join(context_parts), sources


def generate_answer(question: str, chunks, language: str = "fr") -> tuple[str, list[str]]:
    """
    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @Function Description: Génère la réponse finale à partir des chunks
    retenus, en une seule fois (non streamée). Utilisée par la version CLI
    (rag_app.py).
    --------------------------------------------------------------------------
    @Parameter:
        - question: str - Question (déjà reformulée) à laquelle répondre.
        - chunks: list[Document] - Chunks de contexte sélectionnés.
        - language: str - "fr" ou "en".

    @Returnvalue:
        - tuple[str, list[str]] - (réponse générée avec citations [n], liste
          des sources correspondantes).
    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    context, sources = format_context_with_sources(chunks)
    llm = get_llm()
    chain = ANSWER_PROMPTS[language] | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    return answer, sources


def generate_answer_stream(question: str, chunks, language: str = "fr"):
    """
    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @Function Description: Version en streaming de generate_answer, utilisée
    par l'interface Streamlit pour afficher la réponse progressivement.
    --------------------------------------------------------------------------
    @Parameter:
        - question: str - Question (déjà reformulée) à laquelle répondre.
        - chunks: list[Document] - Chunks de contexte sélectionnés.
        - language: str - "fr" ou "en".

    @Returnvalue:
        - tuple[Iterator[str], list[str]] - (flux de fragments de texte à
          afficher au fur et à mesure, liste des sources correspondantes).
    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    context, sources = format_context_with_sources(chunks)
    llm = get_llm()
    chain = ANSWER_PROMPTS[language] | llm | StrOutputParser()
    return chain.stream({"context": context, "question": question}), sources


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# @Author:        John MANGA | Digit-Tech-Innov Solutions and Services
# @Creation:      20.07.2026
# ------------------------------------------------------------------------------
# @Function Description: Point d'entrée unique du pipeline RAG pour la
# version CLI (rag_app.py). Route d'abord la question (conversation vs
# documents vs database), puis exécute la branche correspondante.
# ------------------------------------------------------------------------------
# @Parameter:
#                 - question: str - Message brut tapé par l'utilisateur.
#                 - chat_history: list[tuple[str, str]] - Historique de la conversation.
#                 - vectorstore: Chroma - Base vectorielle à interroger.
#                 - language: str - "fr" ou "en".
# @Returnvalue:
#                 - tuple[str, list[str], str, str | None] - (réponse générée,
#                   liste des sources citées, question utilisée, requête SQL
#                   en attente le cas échéant).
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def answer_question(
    question: str,
    chat_history: list[tuple[str, str]],
    vectorstore,
    language: str = "fr",
    references: dict | None = None,
    source_filter: str | list[str] | None = None,
):
    """
    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # @Function Description: Point d'entrée unique du pipeline RAG. Route d'abord
    # la question (conversation vs documents vs database), puis exécute la
    # branche correspondante. Accepte des références @type:valeur (filtres
    # destinées aux connecteurs BD) et un filtre source pour les documents.
    # ------------------------------------------------------------------------------
    # @Parameter:
    #                 - question: str - Message brut tapé par l'utilisateur.
    #                 - chat_history: list[tuple[str, str]] - Historique de la conversation.
    #                 - vectorstore: Chroma - Base vectorielle à interroger.
    #                 - language: str - "fr" ou "en".
    #                 - references: dict | None - Filtres @type:valeur extraits par la palette.
    #                 - source_filter: str | list[str] | None - Restreint la recherche documents.
    # @Returnvalue:
    #                 - tuple[str, list[str], str, str | None] - (réponse générée,
    #                   liste des sources citées, question utilisée, requête SQL
    #                   en attente le cas échéant).
    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    if references is None:
        references = {}

    route = classify_question(question, chat_history, language)

    if route == "conversation":
        answer = answer_from_history(question, chat_history, language)
        return answer, [], question, None

    if route == "database":
        from rag_langchain.connectors.sqlite import SQLiteConnector
        conn = SQLiteConnector(db_path=str(settings.sqlite_path))
        result = conn.answer(question, references, language)

        if result.safe:
            answer = f"Requête SQL exécutée avec succès :\n{result.native_query}"
            if result.rows:
                import json
                answer += "\n\nRésultats :\n" + json.dumps(result.rows[:5], indent=2, ensure_ascii=False)
        else:
            answer = f"❌ Erreur ou requête bloquée : {result.error}"

        sql_source = [f"Requête SQL exécutée : {result.native_query}"] if result.safe else []
        return answer, sql_source, question, None

    # Branche Documents (RAG)
    standalone_question = condense_question(question, chat_history, language)
    chunks = retrieve_and_rerank(
        standalone_question, vectorstore, source_filter=source_filter
    )
    if not chunks:
        no_result_msg = {
            "fr": "Je n'ai trouvé aucun passage pertinent dans les documents indexés.",
            "en": "I couldn't find any relevant passage in the indexed documents.",
        }
        return no_result_msg[language], [], standalone_question, None

    answer, sources = generate_answer(standalone_question, chunks, language)
    return answer, sources, standalone_question, None