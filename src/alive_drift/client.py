"""Lance UN run sur Alive et renvoie sa réponse JSON, vérifiée par check_run.

Pas encore de réessai ni de parallélisme : un appel, une réponse.
Équivalent Python du curl de alive_quickref.md §4.

Essai réel (un run, coût ~0,003 crédit) :

    export ALIVE_TOKEN=...
    uv run python -m alive_drift.client th-... dp-... mistralai/mistral-small-24b-instruct-2501 \
        "What is a normal resting heart rate?"
"""

import argparse
import json
import os
import sys

import httpx

from alive_drift.check import RunRejected, check_run

BASE_URL = "https://brain-bzh.alive.dev/api/v1"

# Un run peut durer longtemps (Alive coupe à 1800 s). 300 s suffit pour une réponse.
TIMEOUT_S = 300.0


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
             question: str, expected_model: str) -> dict:
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
    run = response.json()
    check_run(run, expected_model)
    return run


def main(argv: list[str] | None = None, transport: httpx.BaseTransport | None = None) -> int:
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
            run = run_once(client, args.room_id, args.deployment_id,
                           args.question, args.expected_model)
        except RunRejected as err:
            print(f"REJETÉ : {err}")
            return 1
    print(json.dumps(run, indent=2, ensure_ascii=False))
    print("ACCEPTÉ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
