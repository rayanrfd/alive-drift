"""Tests des réessais (run_with_retries), sans réseau.

Le faux serveur joue un SCÉNARIO : une liste de réponses, une par appel.
Chaque élément est soit un code HTTP (200 renvoie la vraie sortie du 01/10),
soit une erreur réseau à lever.
"""

import copy

import httpx
import pytest

from alive_drift.check import RunRejected
from alive_drift.client import main, make_client, run_with_retries
from test_check import MODEL, REAL_RUN

TOKEN = "jeton-de-test"
ROOM = "th-Z7jWvllIYv1Eoa9z"
DEPLOYMENT = "dp-E8dfWyMHc4Q7oXnN"
QUESTION = "What is a normal resting heart rate?"


def scripted_alive(scenario, ok_body=None):
    """Faux serveur qui suit `scenario`. Renvoie (transport, calls) ; calls compte les appels."""
    steps = list(scenario)
    calls = []
    body_200 = REAL_RUN if ok_body is None else ok_body

    def handler(request):
        calls.append(request)
        step = steps.pop(0)
        if isinstance(step, Exception):
            raise step
        if step == 200:
            return httpx.Response(200, json=body_200)
        return httpx.Response(step, json={"detail": f"erreur {step}"})

    return httpx.MockTransport(handler), calls


def play(scenario, ok_body=None):
    """Lance run_with_retries sur le scénario. Renvoie (résultat ou exception, appels, attentes)."""
    transport, calls = scripted_alive(scenario, ok_body)
    sleeps = []
    with make_client(TOKEN, transport) as client:
        try:
            result = run_with_retries(client, ROOM, DEPLOYMENT, QUESTION, MODEL,
                                      sleep=sleeps.append)
        except Exception as err:  # le test regarde ensuite quelle exception
            result = err
    return result, len(calls), sleeps


# --- Échecs passagers puis succès ---------------------------------------------

def test_502_puis_succes():
    result, n_calls, sleeps = play([502, 200])
    assert result == REAL_RUN
    assert n_calls == 2
    assert sleeps == [2.0]


def test_deux_echecs_puis_succes():
    result, n_calls, sleeps = play([503, 502, 200])
    assert result == REAL_RUN
    assert n_calls == 3
    assert sleeps == [2.0, 5.0]


def test_429_puis_succes():
    result, n_calls, _ = play([429, 200])
    assert result == REAL_RUN
    assert n_calls == 2


def test_reseau_coupe_puis_succes():
    result, n_calls, _ = play([httpx.ConnectError("connexion refusée"), 200])
    assert result == REAL_RUN
    assert n_calls == 2


# --- Abandon après 3 essais ------------------------------------------------------

def test_trois_502_abandon():
    result, n_calls, sleeps = play([502, 502, 502])
    assert isinstance(result, httpx.HTTPStatusError)
    assert result.response.status_code == 502
    assert n_calls == 3
    assert sleeps == [2.0, 5.0]  # pas d'attente après le dernier essai


def test_trois_timeouts_abandon():
    result, n_calls, _ = play([httpx.ReadTimeout("trop long")] * 3)
    assert isinstance(result, httpx.ReadTimeout)
    assert n_calls == 3


# --- Erreurs qui ne passeront pas : un seul appel ------------------------------

@pytest.mark.parametrize("code", [400, 401, 403, 404, 423])
def test_erreur_de_notre_cote_pas_de_reessai(code):
    result, n_calls, sleeps = play([code, 200])
    assert isinstance(result, httpx.HTTPStatusError)
    assert result.response.status_code == code
    assert n_calls == 1
    assert sleeps == []


def test_run_rejete_pas_de_reessai():
    # Le run a abouti (facturé) mais servi par Gemini : on ne réessaie pas, on le signale.
    bad = copy.deepcopy(REAL_RUN)
    bad["model"]["served"][0]["model"] = "google/gemini-3.1-flash-lite"
    result, n_calls, sleeps = play([200, 200], ok_body=bad)
    assert isinstance(result, RunRejected)
    assert n_calls == 1
    assert sleeps == []


# --- Le script en ligne de commande utilise les réessais -----------------------

def test_main_reessaie(monkeypatch, capsys):
    monkeypatch.setenv("ALIVE_TOKEN", TOKEN)
    transport, calls = scripted_alive([502, 200])
    sleeps = []
    assert main([ROOM, DEPLOYMENT, MODEL, QUESTION], transport, sleep=sleeps.append) == 0
    assert len(calls) == 2
    assert sleeps == [2.0]
    assert capsys.readouterr().out.strip().endswith("ACCEPTÉ")
