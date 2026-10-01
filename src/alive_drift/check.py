"""Décide si une réponse de run Alive est gardée ou rejetée.

Alive renvoie, pour chaque run, un objet JSON. Son bloc "model" dit quel
modèle a vraiment répondu. On rejette toute réponse servie autrement que
prévu : sinon on mesurerait la dérive d'un autre modèle sans le savoir.

Règle générale : un champ absent compte comme un échec. On ne suppose
jamais qu'un champ manquant veut dire "tout va bien".
"""

# Le raisonnement doit être coupé pour tous les modèles (générateur et juges).
EXPECTED_EFFORT = "none"


class RunRejected(Exception):
    """La réponse du run ne doit pas être utilisée."""


def check_run(run: dict, expected_model: str) -> None:
    """Lève RunRejected si la réponse ne doit pas être gardée.

    `run` est la réponse JSON d'Alive, déjà décodée en dict Python.
    Ne renvoie rien si tout est bon.
    """
    # --- 1. Le run a abouti et a produit du texte -------------------------
    if run.get("status") != "completed":
        raise RunRejected(f"status {run.get('status')!r} au lieu de 'completed'")
    content = run.get("content")
    if not isinstance(content, str) or not content.strip():
        raise RunRejected("réponse vide")

    # --- 2. Le bloc "model" existe ----------------------------------------
    block = run.get("model")
    if not isinstance(block, dict):
        raise RunRejected("pas de bloc 'model' dans la réponse")

    # --- 3. Le déploiement a demandé le bon modèle ------------------------
    if block.get("requested") != expected_model:
        raise RunRejected(
            f"modèle demandé {block.get('requested')!r}, attendu {expected_model!r}"
        )

    # --- 4. Alive n'a rien remplacé ----------------------------------------
    if block.get("fallback") is not False:
        raise RunRejected(f"fallback = {block.get('fallback')!r}")
    if block.get("substituted") is not False:
        raise RunRejected(f"substituted = {block.get('substituted')!r}")

    # --- 5. Un seul modèle a répondu, en un seul appel, et c'est le bon ---
    served = block.get("served")
    if not isinstance(served, list) or len(served) != 1:
        raise RunRejected(f"'served' devrait avoir 1 entrée : {served!r}")
    s = served[0]
    if s.get("model") != expected_model:
        raise RunRejected(f"servi par {s.get('model')!r}, attendu {expected_model!r}")
    if s.get("substituted") is not False:
        raise RunRejected(f"served[0].substituted = {s.get('substituted')!r}")
    if s.get("calls") != 1:
        raise RunRejected(f"{s.get('calls')!r} appels au lieu de 1")

    # --- 6. Le raisonnement est coupé ---------------------------------------
    if block.get("reasoning_effort") != EXPECTED_EFFORT:
        raise RunRejected(
            f"reasoning_effort {block.get('reasoning_effort')!r}, attendu {EXPECTED_EFFORT!r}"
        )
