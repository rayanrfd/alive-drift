"""Tests de client.py, sans réseau.

Un faux serveur (httpx.MockTransport) remplace Alive : il enregistre la
requête reçue et renvoie une réponse choisie par le test. Par défaut, il
renvoie la vraie sortie d'Alive du 01/10 (REAL_RUN, définie dans test_check.py).
"""

import copy
import json

import httpx
import pytest

from alive_drift.check import RunRejected
from alive_drift.client import main, make_client, run_once
from test_check import MODEL, REAL_RUN  # même vraie sortie que les tests de check

TOKEN = "jeton-de-test"
ROOM = "th-Z7jWvllIYv1Eoa9z"
DEPLOYMENT = "dp-E8dfWyMHc4Q7oXnN"
QUESTION = "What is a normal resting heart rate?"


def fake_alive(status_code=200, body=None):
    """Faux serveur. Renvoie (transport, requests) ; `requests` liste ce qu'il a reçu."""
    requests = []
    payload = REAL_RUN if body is None else body

    def handler(request):
        requests.append(request)
        return httpx.Response(status_code, json=payload)

    return httpx.MockTransport(handler), requests


# --- La requête envoyée est exactement celle du curl ------------------------

def test_requete_envoyee():
    transport, requests = fake_alive()
    with make_client(TOKEN, transport) as client:
        run_once(client, ROOM, DEPLOYMENT, QUESTION, MODEL)

    assert len(requests) == 1
    req = requests[0]
    assert req.method == "POST"
    assert str(req.url) == (
        "https://brain-bzh.alive.dev/api/v1/threads/th-Z7jWvllIYv1Eoa9z"
        "/deployments/dp-E8dfWyMHc4Q7oXnN/runs"
    )
    assert req.headers["Authorization"] == "Bearer jeton-de-test"
    assert json.loads(req.content) == {"input": QUESTION, "parent_id": None, "stream": False}


# --- La réponse rendue est le JSON d'Alive, sans modification ---------------

def test_reponse_rendue_telle_quelle():
    transport, _ = fake_alive()
    with make_client(TOKEN, transport) as client:
        run = run_once(client, ROOM, DEPLOYMENT, QUESTION, MODEL)
    assert run == REAL_RUN


# --- check_run est bien appelé ----------------------------------------------

def test_reponse_avec_fallback_rejetee():
    bad = copy.deepcopy(REAL_RUN)
    bad["model"]["fallback"] = True
    transport, _ = fake_alive(body=bad)
    with make_client(TOKEN, transport) as client:
        with pytest.raises(RunRejected, match="fallback"):
            run_once(client, ROOM, DEPLOYMENT, QUESTION, MODEL)


# --- Erreurs HTTP : exception, un seul appel (pas de réessai à ce stade) -----

@pytest.mark.parametrize("code", [401, 502])
def test_erreur_http(code):
    transport, requests = fake_alive(status_code=code, body={"detail": "erreur"})
    with make_client(TOKEN, transport) as client:
        with pytest.raises(httpx.HTTPStatusError) as err:
            run_once(client, ROOM, DEPLOYMENT, QUESTION, MODEL)
    assert len(requests) == 1
    assert TOKEN not in str(err.value)  # le jeton n'apparaît jamais dans l'erreur


# --- Le script en ligne de commande -------------------------------------------

ARGS = [ROOM, DEPLOYMENT, MODEL, QUESTION]


def test_main_accepte(monkeypatch, capsys):
    monkeypatch.setenv("ALIVE_TOKEN", TOKEN)
    transport, _ = fake_alive()
    assert main(ARGS, transport) == 0
    out = capsys.readouterr().out
    assert out.strip().endswith("ACCEPTÉ")
    assert '"provider": "DeepInfra"' in out  # le JSON brut est bien affiché


def test_main_rejete(monkeypatch, capsys):
    monkeypatch.setenv("ALIVE_TOKEN", TOKEN)
    bad = copy.deepcopy(REAL_RUN)
    bad["model"]["served"][0]["model"] = "google/gemini-3.1-flash-lite"
    transport, _ = fake_alive(body=bad)
    assert main(ARGS, transport) == 1
    assert "REJETÉ : servi par" in capsys.readouterr().out


def test_main_sans_jeton(monkeypatch, capsys):
    monkeypatch.delenv("ALIVE_TOKEN", raising=False)
    transport, requests = fake_alive()
    assert main(ARGS, transport) == 2
    assert requests == []  # aucun appel sans jeton
