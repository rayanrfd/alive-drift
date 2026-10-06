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


def play(scenario, ok_body=None, give_up=lambda: False):
    """Lance run_with_retries sur le scénario. Renvoie (résultat ou exception, appels, attentes)."""
    transport, calls = scripted_alive(scenario, ok_body)
    sleeps = []
    with make_client(TOKEN, transport) as client:
        try:
            result = run_with_retries(client, ROOM, DEPLOYMENT, QUESTION, MODEL,
                                      sleep=sleeps.append, give_up=give_up)
        except Exception as err:  # le test regarde ensuite quelle exception
            result = err
    return result, len(calls), sleeps


# --- Échecs passagers puis succès ---------------------------------------------

def test_502_puis_succes():
    result, n_calls, sleeps = play([502, 200])
    assert result == REAL_RUN
    assert n_calls == 2
    assert sleeps == [5.0]


def test_deux_echecs_puis_succes():
    result, n_calls, sleeps = play([503, 502, 200])
    assert result == REAL_RUN
    assert n_calls == 3
    assert sleeps == [5.0, 15.0]


def test_quatre_echecs_puis_succes_au_dernier_essai():
    # Le cas du 06/10 : le serveur d'Alive met du temps à revenir.
    result, n_calls, sleeps = play([502, 502, 502, 502, 200])
    assert result == REAL_RUN
    assert n_calls == 5
    assert sleeps == [5.0, 15.0, 30.0, 60.0]


def test_429_puis_succes():
    result, n_calls, _ = play([429, 200])
    assert result == REAL_RUN
    assert n_calls == 2


def test_reseau_coupe_puis_succes():
    result, n_calls, _ = play([httpx.ConnectError("connexion refusée"), 200])
    assert result == REAL_RUN
    assert n_calls == 2


# --- Abandon après 5 essais ------------------------------------------------------

def test_cinq_502_abandon():
    result, n_calls, sleeps = play([502] * 6)
    assert isinstance(result, httpx.HTTPStatusError)
    assert result.response.status_code == 502
    assert n_calls == 5  # pas un 6e essai
    assert sleeps == [5.0, 15.0, 30.0, 60.0]  # pas d'attente après le dernier essai


def test_cinq_timeouts_abandon():
    result, n_calls, _ = play([httpx.ReadTimeout("trop long")] * 6)
    assert isinstance(result, httpx.ReadTimeout)
    assert n_calls == 5


# --- give_up : on ne réessaie plus quand la campagne s'arrête -------------------

def test_give_up_deja_vrai_pas_de_reessai():
    result, n_calls, sleeps = play([502, 200], give_up=lambda: True)
    assert isinstance(result, httpx.HTTPStatusError)
    assert n_calls == 1
    assert sleeps == []  # on n'attend même pas


def test_give_up_devient_vrai_pendant_l_attente():
    answers = iter([False, True])  # faux avant l'attente, vrai après
    result, n_calls, sleeps = play([502, 200], give_up=lambda: next(answers))
    assert isinstance(result, httpx.HTTPStatusError)
    assert n_calls == 1  # le 2e essai n'est pas envoyé
    assert sleeps == [5.0]


def test_give_up_n_empeche_pas_un_succes():
    result, n_calls, _ = play([200], give_up=lambda: True)
    assert result == REAL_RUN


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
    assert sleeps == [5.0]
    assert capsys.readouterr().out.strip().endswith("ACCEPTÉ")
