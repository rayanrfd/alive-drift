"""MedAgentBench : lire la réponse du modèle à chaque tour, et la phrase à lui renvoyer.

Reproduit exactement l'orchestrateur de MedAgentBench (src/server/tasks/medagentbench/
__init__.py, MedAgentBench.start_sample, commit 9926011, licence MIT), bizarreries
comprises, pour rester comparable à l'article :

- on retire les espaces autour et les balises ``` (et ```tool_code, pour Gemini) ;
- puis on regarde par quoi le texte COMMENCE, majuscules comprises :
  GET -> lecture, POST -> écriture, FINISH( -> fin, autre chose -> action invalide
  (la tâche s'arrête, ratée).

Aucun appel réseau : le GET lui-même est fait ailleurs (serveur FHIR).
"""

import json
from dataclasses import dataclass
from typing import Any

GET, POST, FINISH, INVALID = "get", "post", "finish", "invalid"

_PLEASE_FINISH = ("Please call FINISH if you have got answers for all the questions "
                  "and finished all the requested tasks")


@dataclass(frozen=True)
class Action:
    kind: str                       # get, post, finish ou invalid
    raw: str                        # la réponse telle que le modèle l'a écrite
    text: str                       # après nettoyage (espaces, balises ```)
    url: str | None = None          # GET : adresse à appeler (avec &_format=json) ; POST : 1re ligne
    payload: Any = None             # POST : le JSON lu (None s'il est illisible)
    payload_error: str | None = None  # POST : pourquoi le JSON est illisible
    answer: str | None = None       # FINISH : le texte entre "FINISH(" et le dernier caractère


def clean(raw: str) -> str:
    """Le nettoyage de MedAgentBench, dans le même ordre."""
    return raw.strip().replace("```tool_code", "").replace("```", "").strip()


def parse_reply(raw: str) -> Action:
    """Ce que demande une réponse du modèle (mêmes tests, même ordre que MedAgentBench)."""
    text = clean(raw)
    if text.startswith("GET"):
        # comme chez eux : on ajoute &_format=json même s'il n'y a pas de « ? »
        return Action(GET, raw, text, url=text[3:].strip() + "&_format=json")
    if text.startswith("POST"):
        lines = text.split("\n")
        url = lines[0][4:].strip()
        try:
            payload = json.loads("\n".join(lines[1:]))
        except json.JSONDecodeError as err:
            return Action(POST, raw, text, url=url, payload_error=str(err))
        return Action(POST, raw, text, url=url, payload=payload)
    if text.startswith("FINISH("):
        # comme chez eux : on coupe le dernier caractère, censé être « ) »
        return Action(FINISH, raw, text, answer=text[len("FINISH("):-1])
    return Action(INVALID, raw, text)


def reply_after_get(data: Any) -> str:
    """Après un GET réussi. `data` est inséré tel quel (f-string), comme chez eux."""
    return f"Here is the response from the GET request:\n{data}. {_PLEASE_FINISH}"


def reply_after_get_error(error: str) -> str:
    """Après un GET en erreur (serveur injoignable, code 4xx/5xx...)."""
    return f"Error in sending the GET request: {error}"


def reply_after_post(action: Action) -> str:
    """Après un POST : accepté si son JSON est lisible. Rien n'est envoyé au serveur."""
    if action.kind != POST:
        raise ValueError(f"reply_after_post sur une action {action.kind!r}")
    if action.payload_error is not None:
        return "Invalid POST request"
    return f"POST request accepted and executed successfully. {_PLEASE_FINISH}"
