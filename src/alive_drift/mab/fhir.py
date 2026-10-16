"""MedAgentBench : interroger le serveur FHIR (le faux dossier patient, dans Docker).

Reprend send_get_request et verify_fhir_server de MedAgentBench (src/server/tasks/
medagentbench/utils.py, commit 9926011, licence MIT), avec la même bibliothèque
(requests), donc les mêmes messages d'erreur envoyés au modèle :

- réponse 2xx : {"status_code", "data"} ; data est l'objet JSON si le serveur annonce
  exactement « application/json », sinon le texte brut (HAPI répond application/fhir+json :
  le modèle reçoit donc le JSON en texte) ;
- toute autre erreur (404, 500, adresse mal formée par le modèle...) : {"error": message},
  renvoyé au modèle comme chez eux.

Deux écarts volontaires, pour ne jamais mesurer une panne comme un échec de l'agent :
- si NOTRE serveur (celui de api_base) ne répond pas ou dépasse le délai, on lève
  ServerUnreachable : la campagne s'arrête au lieu de passer l'erreur au modèle ;
- un délai de 60 s (chez eux : aucun, une requête pouvait attendre indéfiniment).

    uv run python -m alive_drift.mab.fhir --check
    uv run python -m alive_drift.mab.fhir --get "http://localhost:8080/fhir/Patient?identifier=S6534835"
"""

import argparse
import sys
from typing import Any
from urllib.parse import urlsplit

import requests

from alive_drift.mab.tasks import FHIR_API_BASE

TIMEOUT_S = 60


class ServerUnreachable(RuntimeError):
    """Notre serveur FHIR ne répond pas : panne de notre côté, pas une erreur de l'agent."""


def _same_server(url: str, api_base: str) -> bool:
    a, b = urlsplit(url), urlsplit(api_base)
    return (a.scheme, a.hostname, a.port) == (b.scheme, b.hostname, b.port)


def fhir_get(url: str, api_base: str = FHIR_API_BASE, session: Any = None,
             timeout: float = TIMEOUT_S) -> dict:
    """Un GET, comme send_get_request de MedAgentBench.

    Lève ServerUnreachable si le serveur de api_base est injoignable ou trop lent.
    Une panne d'un AUTRE serveur (adresse inventée par le modèle) reste une erreur
    renvoyée au modèle, comme chez eux.
    """
    http = session if session is not None else requests
    try:
        response = http.get(url, timeout=timeout)
        response.raise_for_status()
        data = (response.json() if response.headers.get("Content-Type") == "application/json"
                else response.text)
    except (requests.ConnectionError, requests.Timeout) as err:
        if _same_server(url, api_base):
            raise ServerUnreachable(f"serveur FHIR {api_base} injoignable : {err}") from err
        return {"error": str(err)}
    except Exception as err:        # comme chez eux : toute autre erreur va au modèle
        return {"error": str(err)}
    return {"status_code": response.status_code, "data": data}


def check_server(api_base: str = FHIR_API_BASE, session: Any = None,
                 timeout: float = TIMEOUT_S) -> bool:
    """Vrai si {api_base}metadata répond 200, comme verify_fhir_server de MedAgentBench."""
    try:
        return fhir_get(f"{api_base}metadata", api_base, session, timeout).get("status_code") == 200
    except ServerUnreachable:
        return False


def main(argv: list[str] | None = None, session: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Interroger le serveur FHIR de MedAgentBench.")
    parser.add_argument("--api-base", default=FHIR_API_BASE, help="adresse du serveur FHIR")
    what = parser.add_mutually_exclusive_group(required=True)
    what.add_argument("--check", action="store_true", help="le serveur répond-il ?")
    what.add_argument("--get", metavar="URL", help="affiche la réponse d'un GET (texte envoyé au modèle)")
    args = parser.parse_args(argv)
    if args.check:
        if check_server(args.api_base, session):
            print(f"OK : {args.api_base} répond")
            return 0
        print(f"REFUS : {args.api_base} ne répond pas (Docker lancé ? port 8080 ?)", file=sys.stderr)
        return 2
    try:
        result = fhir_get(args.get, args.api_base, session)
    except ServerUnreachable as err:
        print(f"REFUS : {err}", file=sys.stderr)
        return 2
    if "error" in result:
        print(f"erreur renvoyée au modèle : {result['error']}")
        return 0
    text = str(result["data"])
    print(text)
    print(f"--- {len(text)} caractères (environ {len(text) // 4} tokens)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
