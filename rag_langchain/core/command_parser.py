# command_palette.py
import re
from dataclasses import dataclass
from typing import Literal

Command = Literal["auto", "docs", "sqlite", "postgres", "mongo", "federated"]

COMMAND_MAP = {
    "/docs": "docs",
    "/sql": "database",
    "/employees": "sqlite", "/employes": "sqlite", "/emp": "sqlite",
    "/purchases": "postgres", "/achats": "postgres", "/achat": "postgres",
    "/services": "mongo", "/service": "mongo",
    "/all": "federated", "/tout": "federated",
}

# @employé:DUPONT  @service:RH  @doc:rapport.pdf  @table:commandes
REFERENCE_RE = re.compile(r"@(\w+):([^\s@]+)")


@dataclass
class ParsedInput:
    raw: str
    command: Command          # "auto" si aucun préfixe /
    references: dict[str, list[str]]   # {"employé": ["DUPONT"], "service": ["RH"]}
    cleaned_question: str     # question débarrassée des / et @


def parse_user_input(text: str) -> ParsedInput:
    """
    Extrait la commande (/...) et les références (@type:valeur) du texte.
    Renvoie aussi la question nettoyée à passer au LLM.
    """
    text = text.strip()
    command: Command = "auto"

    # 1. Détection commande (au début du message)
    first_token = text.split(maxsplit=1)[0].lower() if text else ""
    if first_token in COMMAND_MAP:
        command = COMMAND_MAP[first_token]
        text = text[len(first_token):].strip()

    # 2. Détection références @type:valeur
    references: dict[str, list[str]] = {}
    for kind, value in REFERENCE_RE.findall(text):
        references.setdefault(kind.lower(), []).append(value)
    cleaned = REFERENCE_RE.sub("", text).strip()
    # Nettoyage des espaces multiples
    cleaned = re.sub(r"\s+", " ", cleaned)

    return ParsedInput(raw=text, command=command,
                       references=references, cleaned_question=cleaned)