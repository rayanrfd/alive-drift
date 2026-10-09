"""Décide si une réponse de run Alive est gardée ou rejetée.

Alive renvoie, pour chaque run, un objet JSON. Son bloc "model" dit quel
modèle a vraiment répondu. On rejette toute réponse servie autrement que
prévu : sinon on mesurerait la dérive d'un autre modèle sans le savoir.

Règle générale : un champ absent compte comme un échec. On ne suppose
jamais qu'un champ manquant veut dire "tout va bien".
"""

class RunRejected(Exception):
    """La réponse du run ne doit pas être utilisée."""


def check_run(run: dict, expected_model: str, expected_effort: str = "none",
              served_model: str | None = None) -> None:
    """Lève RunRejected si la réponse ne doit pas être gardée.

    `run` est la réponse JSON d'Alive, déjà décodée en Python.
    `expected_effort` : l'effort de raisonnement voulu ("none" = coupé).
    `served_model` : le nom du modèle qui doit répondre, si `expected_model` est
    un alias d'Alive (ex. "~anthropic/claude-haiku-latest" sert
    "anthropic/claude-haiku-5.5"). Par défaut, le même que `expected_model`.
    Ne renvoie rien si tout est bon.
    """
    # --- 0. La réponse est bien un objet JSON -------------------------------
    if not isinstance(run, dict):
        raise RunRejected(f"réponse mal formée : {type(run).__name__} au lieu d'un objet")

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
    if not isinstance(s, dict):
        raise RunRejected(f"served[0] mal formé : {s!r}")
    # Avec un alias, Alive répond avec le vrai nom du modèle : on exige ce nom-là.
    # Si l'alias est redirigé vers un autre modèle, le run est rejeté.
    expected_served = served_model or expected_model
    if s.get("model") != expected_served:
        raise RunRejected(f"servi par {s.get('model')!r}, attendu {expected_served!r}")
    if s.get("substituted") is not False:
        raise RunRejected(f"served[0].substituted = {s.get('substituted')!r}")
    if s.get("calls") != 1:
        raise RunRejected(f"{s.get('calls')!r} appels au lieu de 1")

    # --- 6. Le raisonnement est celui voulu ----------------------------------
    if block.get("reasoning_effort") != expected_effort:
        raise RunRejected(
            f"reasoning_effort {block.get('reasoning_effort')!r}, attendu {expected_effort!r}"
        )
    # "none" doit être vrai dans les faits : l'étiquette ne suffit pas, le
    # compteur de tokens de raisonnement doit valoir 0 (absent = échec).
    if expected_effort == "none":
        tokens = run.get("tokens")
        reasoning = tokens.get("reasoning") if isinstance(tokens, dict) else None
        if type(reasoning) is not int or reasoning != 0:
            raise RunRejected(f"{reasoning!r} tokens de raisonnement alors qu'il est coupé")

    # --- 7. De quoi retrouver le run et suivre les providers ---------------
    for label, value in (("execution_id", run.get("execution_id")),
                         ("provider", s.get("provider"))):
        if not isinstance(value, str) or not value:
            raise RunRejected(f"{label} absent")
