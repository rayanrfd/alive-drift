"""Lance UN run sur Alive et renvoie sa réponse JSON, vérifiée par check_run.

run_once : un appel, une réponse. Équivalent Python du curl de alive_quickref.md §4.
run_with_retries : run_once, réessayé si l'échec vient du serveur ou du réseau.
Pas encore de parallélisme.

Essai réel (un run, coût ~0,003 crédit) :

    export ALIVE_TOKEN=...
    uv run python -m alive_drift.client th-... dp-... mistralai/mistral-small-24b-instruct-2501 \
        "What is a normal resting heart rate?"
"""

import argparse
import json
import os
import sys
import time

import httpx

from alive_drift.check import RunRejected, check_run

BASE_URL = "https://brain-bzh.alive.dev/api/v1"

# Un run peut durer longtemps (Alive coupe à 1800 s). 300 s suffit pour une réponse.
TIMEOUT_S = 300.0

# Codes HTTP qui valent la peine d'un nouvel essai : serveur surchargé (429)
# ou en panne passagère (5xx). Un 502 d'Alive n'est pas facturé.
# Les autres erreurs (400, 401, 403, 404, 423...) viennent de NOTRE requête
# ou de notre jeton : réessayer ne changerait rien.
RETRYABLE_STATUS = {429, 500, 502, 503, 504}

# Attente avant le 2e puis le 3e essai. 2 délais = 3 essais au total.
RETRY_DELAYS_S = (2.0, 5.0)


def make_client(token: str, transport: httpx.BaseTransport | None = None) -> httpx.Client:
    """Client HTTP qui ajoute le jeton à chaque requête.

    `transport` sert seulement aux tests : on y branche un faux serveur.
    """
    return httpx.Client(
        base_url=BASE_URL,
        headers={"Authorization": f"Bearer {token}"},
        timeout=TIMEOUT_S,
        transport=transport,
    )


def run_once(client: httpx.Client, room_id: str, deployment_id: str,
             question: str, expected_model: str, expected_effort: str = "none") -> dict:
    """Envoie la question, renvoie le JSON d'Alive s'il passe check_run.

    Lève httpx.HTTPStatusError si Alive répond une erreur (401, 502...),
    RunRejected si la réponse ne doit pas être gardée.
    """
    url = f"/threads/{room_id}/deployments/{deployment_id}/runs"
    body = {
        "input": question,
        "parent_id": None,  # conversation neuve : aucune question précédente
        "stream": False,    # une seule réponse JSON, pas un flux
    }
    response = client.post(url, json=body)
    response.raise_for_status()  # code HTTP 4xx ou 5xx -> exception
    try:
        run = response.json()
    except ValueError:  # code 200 mais le corps n'est pas du JSON
        raise RunRejected("réponse 200 illisible (pas du JSON)") from None
    check_run(run, expected_model, expected_effort)
    return run


def is_retryable(err: Exception) -> bool:
    """Vrai si l'échec vient du serveur ou du réseau, pas de notre requête."""
    if isinstance(err, httpx.HTTPStatusError):
        return err.response.status_code in RETRYABLE_STATUS
    # TransportError : connexion impossible, coupée, ou délai dépassé (timeout).
    return isinstance(err, httpx.TransportError)


def run_with_retries(client: httpx.Client, room_id: str, deployment_id: str,
                     question: str, expected_model: str, expected_effort: str = "none",
                     sleep=time.sleep) -> dict:
    """run_once, avec jusqu'à 3 essais si l'échec est passager.

    RunRejected n'est JAMAIS réessayé : le run a abouti (donc il est facturé)
    mais a été servi autrement que demandé. On veut le voir, pas le masquer.

    `sleep` sert seulement aux tests, pour ne pas attendre pour de vrai.
    """
    attempts = len(RETRY_DELAYS_S) + 1
    for attempt in range(1, attempts + 1):
        try:
            return run_once(client, room_id, deployment_id, question,
                            expected_model, expected_effort)
        except (httpx.HTTPStatusError, httpx.TransportError) as err:
            if attempt == attempts or not is_retryable(err):
                raise  # dernier essai, ou erreur qui ne passera pas en réessayant
            delay = RETRY_DELAYS_S[attempt - 1]
            print(f"essai {attempt}/{attempts} échoué ({type(err).__name__}), "
                  f"nouvel essai dans {delay:g} s", file=sys.stderr)
            sleep(delay)
    raise AssertionError("inatteignable")  # la boucle rend ou lève toujours


def main(argv: list[str] | None = None, transport: httpx.BaseTransport | None = None,
         sleep=time.sleep) -> int:
    """Lance un run, affiche le JSON brut puis le verdict. Renvoie le code de sortie."""
    parser = argparse.ArgumentParser(description="Un run Alive, sortie brute + verdict.")
    parser.add_argument("room_id")
    parser.add_argument("deployment_id")
    parser.add_argument("expected_model")
    parser.add_argument("question")
    args = parser.parse_args(argv)

    token = os.environ.get("ALIVE_TOKEN")
    if not token:
        print("ALIVE_TOKEN absent : export ALIVE_TOKEN=... d'abord.", file=sys.stderr)
        return 2

    with make_client(token, transport) as client:
        try:
            run = run_with_retries(client, args.room_id, args.deployment_id,
                                   args.question, args.expected_model, sleep=sleep)
        except RunRejected as err:
            print(f"REJETÉ : {err}")
            return 1
    print(json.dumps(run, indent=2, ensure_ascii=False))
    print("ACCEPTÉ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
