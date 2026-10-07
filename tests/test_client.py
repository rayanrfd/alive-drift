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
from alive_drift.client import find_token, main, make_client, read_env_file, run_once
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


# --- Code 200 mais contenu inutilisable : rejet propre ------------------------

@pytest.mark.parametrize("content", [b"<html>erreur</html>", b"null", b"[]", b""])
def test_200_mal_forme_rejete(content):
    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=content))
    with make_client(TOKEN, transport) as client:
        with pytest.raises(RunRejected):
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


def no_token_in_terminal(monkeypatch):
    for name in ("ALIVE_TOKEN", "ALIVE_ACCESS_TOKEN"):
        monkeypatch.delenv(name, raising=False)


def test_main_sans_jeton(tmp_path, monkeypatch, capsys):
    no_token_in_terminal(monkeypatch)
    transport, requests = fake_alive()
    assert main(ARGS, transport, env_file=tmp_path / "absent.env") == 2
    assert requests == []  # aucun appel sans jeton
    assert "Jeton Alive introuvable" in capsys.readouterr().err


# --- Où le jeton est cherché ------------------------------------------------------

def test_jeton_lu_dans_le_fichier_env(tmp_path, monkeypatch):
    no_token_in_terminal(monkeypatch)
    env_file = tmp_path / ".env"
    env_file.write_text("ALIVE_ACCESS_TOKEN=jeton-du-fichier\n", encoding="utf-8")
    assert find_token(env_file) == "jeton-du-fichier"


def test_main_utilise_le_jeton_du_fichier_env(tmp_path, monkeypatch):
    no_token_in_terminal(monkeypatch)
    env_file = tmp_path / ".env"
    env_file.write_text("ALIVE_ACCESS_TOKEN=jeton-du-fichier\n", encoding="utf-8")
    transport, requests = fake_alive()
    assert main(ARGS, transport, env_file=env_file) == 0
    assert requests[0].headers["Authorization"] == "Bearer jeton-du-fichier"


@pytest.mark.parametrize("name", ["ALIVE_TOKEN", "ALIVE_ACCESS_TOKEN"])
def test_le_terminal_passe_avant_le_fichier(tmp_path, monkeypatch, name):
    no_token_in_terminal(monkeypatch)
    monkeypatch.setenv(name, "jeton-du-terminal")
    env_file = tmp_path / ".env"
    env_file.write_text("ALIVE_ACCESS_TOKEN=jeton-du-fichier\n", encoding="utf-8")
    assert find_token(env_file) == "jeton-du-terminal"


def test_pas_de_jeton_nulle_part(tmp_path, monkeypatch):
    no_token_in_terminal(monkeypatch)
    assert find_token(tmp_path / "absent.env") is None
    empty = tmp_path / ".env"
    empty.write_text("ALIVE_ACCESS_TOKEN=\nAUTRE=x\n", encoding="utf-8")
    assert find_token(empty) is None  # une valeur vide ne compte pas


def test_lecture_du_fichier_env(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# un commentaire\n"
        "\n"
        "SIMPLE=abc\n"
        'GUILLEMETS="avec espaces"\n'
        "export EXPORTE='xyz'\n"
        "AVEC_EGAL=a=b=c\n"
        "ligne sans signe egal\n",
        encoding="utf-8")
    assert read_env_file(env_file) == {
        "SIMPLE": "abc", "GUILLEMETS": "avec espaces", "EXPORTE": "xyz", "AVEC_EGAL": "a=b=c"}
    assert read_env_file(tmp_path / "absent.env") == {}
