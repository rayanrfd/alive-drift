"""Tests du parallélisme (generate avec workers > 1), sans réseau.

Mêmes outils que test_generate.py (faux serveur, run_generate). Deux ajouts :
- `slow` : le faux serveur attend un peu avant de répondre, pour que plusieurs
  questions soient réellement en cours en même temps ;
- le faux serveur "écho" répond « réponse à <question> », pour vérifier que
  chaque réponse est rangée avec SA question.
"""

import copy
import json
import threading
import time

import httpx
import pytest

from alive_drift.client import make_client
from alive_drift.generate import CampaignStopped, errors_path_for, generate, main
from test_check import REAL_RUN
from test_generate import (
    TOKEN, cli_args, fake_alive, make_config, make_questions, read_jsonl, run_generate,
)


def slow(seconds=0.02):
    """À passer en on_request : chaque appel au faux serveur dure `seconds`."""
    return lambda text: time.sleep(seconds)


def ids_in(path):
    return [r["question_id"] for r in read_jsonl(path)]


# --- Le risque principal : une réponse rangée avec la mauvaise question ----------

def test_chaque_reponse_reste_avec_sa_question(tmp_path):
    def handler(request):
        text = json.loads(request.content)["input"]
        time.sleep(0.03 if text.endswith(("0", "5")) else 0.001)  # arrivées dans le désordre
        body = copy.deepcopy(REAL_RUN)
        body["content"] = f"réponse à {text}"
        return httpx.Response(200, json=body)

    out = tmp_path / "A.jsonl"
    with make_client(TOKEN, httpx.MockTransport(handler)) as client:
        result = generate(client, make_questions(30), make_config(), out_path=out, workers=6)

    assert result == (30, 0)
    records = read_jsonl(out)
    assert sorted(r["question_id"] for r in records) == sorted(str(i) for i in range(30))
    for r in records:
        assert r["question"] == f"question {r['question_id']}"
        assert r["response"] == f"réponse à {r['question']}"
    # Les réponses sont bien arrivées dans le désordre : le test exerce le cas difficile.
    assert [r["question_id"] for r in records] != [str(i) for i in range(30)]


# --- C'est vraiment parallèle, et jamais plus que demandé -----------------------

def max_in_flight(tmp_path, workers, n_questions=12):
    """Lance generate et renvoie le plus grand nombre d'appels simultanés vus par le serveur."""
    lock = threading.Lock()
    state = {"now": 0, "max": 0}

    def handler(request):
        with lock:
            state["now"] += 1
            state["max"] = max(state["max"], state["now"])
        time.sleep(0.03)
        with lock:
            state["now"] -= 1
        return httpx.Response(200, json=REAL_RUN)

    with make_client(TOKEN, httpx.MockTransport(handler)) as client:
        generate(client, make_questions(n_questions), make_config(),
                 out_path=tmp_path / "A.jsonl", workers=workers)
    return state["max"]


def test_quatre_workers_quatre_appels_simultanes(tmp_path):
    assert max_in_flight(tmp_path, workers=4) == 4


def test_un_worker_un_seul_appel_a_la_fois(tmp_path):
    assert max_in_flight(tmp_path, workers=1) == 1


# --- Fichier de sortie lisible : pas de lignes mélangées --------------------------

def test_aucune_ligne_abimee_ni_en_double(tmp_path):
    out = tmp_path / "A.jsonl"
    result, inputs = run_generate(make_questions(200), out, workers=8)
    assert result == (200, 0)
    ids = ids_in(out)  # read_jsonl lèverait sur une ligne illisible
    assert len(ids) == 200 and len(set(ids)) == 200
    assert sorted(inputs) == sorted(f"question {i}" for i in range(200))  # un run par question


# --- Échecs ------------------------------------------------------------------------

def test_echecs_notes_en_parallele(tmp_path):
    out = tmp_path / "A.jsonl"
    plan = {"question 3": ["gemini"], "question 7": [502, 502, 502], "question 9": [502, 200]}
    result, _ = run_generate(make_questions(12), out, plan=plan, on_request=slow(), workers=4)

    assert result == (10, 2)
    assert sorted(ids_in(out), key=int) == ["0", "1", "2", "4", "5", "6", "8", "9", "10", "11"]
    errors = {e["question_id"]: e["error_type"] for e in read_jsonl(errors_path_for(out))}
    assert errors == {"3": "RunRejected", "7": "HTTPStatusError"}


# --- Arrêts : on n'envoie plus rien, mais ce qui est parti est rangé -----------

def test_401_arrete_en_parallele(tmp_path):
    out = tmp_path / "A.jsonl"
    result, inputs = run_generate(make_questions(60), out, plan={"question 0": [401]},
                                  on_request=slow(), workers=3)
    assert isinstance(result, httpx.HTTPStatusError)
    assert len(inputs) <= 12  # loin des 60 : les questions suivantes ne partent pas
    # Tout run parti et réussi est écrit (il est payé) : tous sauf la question 0.
    assert sorted(ids_in(out)) == sorted(t.split()[1] for t in inputs if t != "question 0")
    assert not errors_path_for(out).exists()  # un 401 n'est pas l'échec d'UNE question


def test_cinq_echecs_arretent_en_parallele(tmp_path):
    out = tmp_path / "A.jsonl"
    plan = {f"question {i}": ["gemini"] for i in range(60)}
    result, inputs = run_generate(make_questions(60), out, plan=plan,
                                  on_request=slow(), workers=4)
    assert isinstance(result, CampaignStopped)
    assert 5 <= len(inputs) <= 5 + 2 * 4  # 5, plus les runs déjà partis
    assert len(read_jsonl(errors_path_for(out))) == len(inputs)  # chacun est noté


# --- Reprise ---------------------------------------------------------------------

def test_reprise_en_parallele(tmp_path):
    out = tmp_path / "A.jsonl"
    run_generate(make_questions(20), out, limit=7, workers=3)
    result, inputs = run_generate(make_questions(20), out, workers=3)

    assert result == (13, 0)
    assert sorted(inputs) == sorted(f"question {i}" for i in range(7, 20))  # 0..6 pas renvoyées
    ids = ids_in(out)
    assert len(ids) == 20 and len(set(ids)) == 20


# --- workers invalide --------------------------------------------------------------

@pytest.mark.parametrize("workers", [0, -2])
def test_workers_invalide_refuse(tmp_path, workers):
    out = tmp_path / "A.jsonl"
    result, inputs = run_generate(make_questions(3), out, workers=workers)
    assert isinstance(result, ValueError)
    assert "workers doit valoir au moins 1" in str(result)
    assert inputs == []
    assert not out.exists()


# --- Ligne de commande ------------------------------------------------------------

def test_main_workers(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ALIVE_TOKEN", TOKEN)
    args, out = cli_args(tmp_path, 10, "--workers", "4")
    transport, inputs = fake_alive()
    assert main(args, transport) == 0
    assert len(set(ids_in(out))) == 10
    assert "10 réponses gardées, 0 échecs" in capsys.readouterr().out


def test_main_workers_zero_code_2(tmp_path, monkeypatch):
    monkeypatch.setenv("ALIVE_TOKEN", TOKEN)
    args, out = cli_args(tmp_path, 3, "--workers", "0")
    transport, inputs = fake_alive()
    assert main(args, transport) == 2
    assert inputs == []
