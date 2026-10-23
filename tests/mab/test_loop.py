"""Tests de mab/loop.py : la boucle de MedAgentBench sur Alive. Ni réseau, ni Docker.

Le faux Alive se comporte comme le vrai : un run AVEC input est placé sous un message
utilisateur neuf qu'Alive crée sous parent_id. L'agent est scripté : `replies` donne,
pour chaque tâche, ses réponses successives. Le faux serveur FHIR note les adresses reçues.
"""

import copy
import itertools
import json

import httpx
import pytest

from alive_drift.check import RunRejected
from alive_drift.client import make_client, run_once
from alive_drift.config import Config
from alive_drift.mab.actions import reply_after_get, reply_after_get_error
from alive_drift.mab.fhir import ServerUnreachable
from alive_drift.mab.loop import (COMPLETED, INVALID_ACTION, LIMIT_REACHED, main, play_task,
                                  run_campaign, select_tasks)
from alive_drift.mab.tasks import first_message, load_funcs, load_tasks, tasks_fingerprint
from alive_drift.recorder import CampaignStopped
from test_check import MODEL, REAL_RUN

TOKEN = "jeton-de-test"
CONFIG = Config(name="mab", room="th-MAB", deployment="dp-MAB", model=MODEL, reasoning_effort="none")
TASKS = load_tasks()
FUNCS = load_funcs()
FINGERPRINT = tasks_fingerprint()
GET_P = "GET http://localhost:8080/fhir/Patient?identifier=S1"
FINISH_OK = 'FINISH(["S1"])'
POST_OK = 'POST http://localhost:8080/fhir/Observation\n{"resourceType": "Observation"}'


class FakeAlive:
    """Rend, pour chaque run, la réponse suivante prévue pour la tâche (lue dans l'input
    du 1er tour). `bad` : {(tâche, tour): "gemini" | "same_parent"} pour casser un run.
    `start` : premier numéro des ids ; le changer pour une relance, sinon les execution_id
    (ex-1, ex-2...) reviennent et la campagne les refuse, comme elle doit le faire."""

    def __init__(self, replies, bad=None, start=1):
        self.replies = {k: list(v) for k, v in replies.items()}
        self.bad = bad or {}
        self.bodies = []                 # (input, parent_id) reçus, dans l'ordre
        self.task_of = {}                # message de l'agent -> (tâche, tour)
        self.n = itertools.count(start)

    def handler(self, request):
        body = json.loads(request.content)
        self.bodies.append((body["input"], body["parent_id"]))
        if body["parent_id"] is None:
            task_id = next(t.id for t in TASKS if f"Question: {t.instruction}" in body["input"]
                           and body["input"] == first_message(t, FUNCS))
            round_no = 1
        else:
            task_id, prev = self.task_of[body["parent_id"]]
            round_no = prev + 1
        n = next(self.n)
        run = copy.deepcopy(REAL_RUN)
        run["id"], run["parent_id"], run["execution_id"] = f"mg-a{n}", f"mg-u{n}", f"ex-{n}"
        run["content"] = self.replies[task_id].pop(0)
        kind = self.bad.get((task_id, round_no))
        if kind == "gemini":
            run["model"]["served"][0]["model"] = "google/gemini-3.1-flash-lite"
        elif kind == "same_parent":
            run["parent_id"] = body["parent_id"]
        self.task_of[run["id"]] = (task_id, round_no)
        return httpx.Response(200, json=run)


class FakeFhir:
    def __init__(self, result=None):
        self.result = result if result is not None else {"status_code": 200, "data": '{"total": 1}'}
        self.urls = []

    def __call__(self, url):
        self.urls.append(url)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def campaign(tmp_path, replies, tasks, *, bad=None, fhir=None, workers=1, max_rounds=8, start=1):
    alive = FakeAlive(replies, bad, start)
    fhir = fhir or FakeFhir()
    out = tmp_path / "mab.jsonl"
    with make_client(TOKEN, httpx.MockTransport(alive.handler)) as client:
        try:
            result = run_campaign(client, CONFIG, tasks, FUNCS, out_path=out, fingerprint=FINGERPRINT,
                                  max_rounds=max_rounds, workers=workers, fhir=fhir, sleep=lambda s: None)
        except Exception as err:
            result = err
    return result, alive, fhir, out


def lines(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


T1, T2 = TASKS[0], TASKS[1]


# --- Une tâche, tour par tour -----------------------------------------------------------------

def test_get_puis_finish(tmp_path):
    result, alive, fhir, out = campaign(tmp_path, {T1.id: [GET_P, FINISH_OK]}, [T1])
    assert result == (1, 0)
    # 1er tour : branche neuve avec le premier message ; 2e tour : sous la réponse de l'agent
    first = first_message(T1, FUNCS)
    reply = reply_after_get('{"total": 1}')
    assert alive.bodies == [(first, None), (reply, "mg-a1")]
    assert fhir.urls == ["http://localhost:8080/fhir/Patient?identifier=S1&_format=json"]
    [rec] = lines(out)
    assert rec["status"] == COMPLETED and rec["result"] == '["S1"]' and rec["n_rounds"] == 2
    assert [h["role"] for h in rec["history"]] == ["user", "agent", "user", "agent"]
    assert [h["content"] for h in rec["history"]] == [first, GET_P, reply, FINISH_OK]
    assert [r["execution_id"] for r in rec["runs"]] == ["ex-1", "ex-2"]
    assert rec["runs"][1]["parent_id"] == "mg-u2" and rec["fingerprint"] == FINGERPRINT


def test_get_en_erreur_renvoye_a_l_agent(tmp_path):
    fhir = FakeFhir({"error": "404 Client Error"})
    result, alive, _, out = campaign(tmp_path, {T1.id: [GET_P, FINISH_OK]}, [T1], fhir=fhir)
    assert result == (1, 0)
    assert alive.bodies[1][0] == reply_after_get_error("404 Client Error")


def test_post_accepte_sans_appeler_le_serveur(tmp_path):
    result, alive, fhir, out = campaign(tmp_path, {T1.id: [POST_OK, FINISH_OK]}, [T1])
    assert result == (1, 0) and fhir.urls == []
    assert alive.bodies[1][0].startswith("POST request accepted and executed successfully.")


def test_post_illisible_on_continue(tmp_path):
    result, alive, _, out = campaign(tmp_path, {T1.id: ["POST url\n{oops", FINISH_OK]}, [T1])
    assert result == (1, 0)
    assert alive.bodies[1][0] == "Invalid POST request"
    assert lines(out)[0]["status"] == COMPLETED


def test_action_invalide_arrete_la_tache(tmp_path):
    result, alive, _, out = campaign(tmp_path, {T1.id: ["I think the answer is S1."]}, [T1])
    assert result == (1, 0) and len(alive.bodies) == 1
    rec = lines(out)[0]
    assert rec["status"] == INVALID_ACTION and rec["result"] is None and rec["n_rounds"] == 1


def test_huit_tours_sans_finish(tmp_path):
    result, alive, fhir, out = campaign(tmp_path, {T1.id: [GET_P] * 9}, [T1])
    assert result == (1, 0) and len(alive.bodies) == 8 and len(fhir.urls) == 8
    rec = lines(out)[0]
    assert rec["status"] == LIMIT_REACHED and rec["n_rounds"] == 8 and len(rec["history"]) == 17
    # chaque tour part sous la réponse précédente de l'agent
    assert [p for _, p in alive.bodies] == [None] + [f"mg-a{i}" for i in range(1, 8)]


def test_max_rounds_reglable(tmp_path):
    result, alive, _, out = campaign(tmp_path, {T1.id: [GET_P] * 3}, [T1], max_rounds=2)
    assert len(alive.bodies) == 2 and lines(out)[0]["status"] == LIMIT_REACHED


def test_balises_retirees_comme_chez_eux(tmp_path):
    result, _, _, out = campaign(tmp_path, {T1.id: ["```tool_code\nFINISH([3])\n```"]}, [T1])
    assert lines(out)[0]["result"] == "[3]"


# --- Pannes : jamais comptées comme un échec de l'agent ---------------------------------------

def test_serveur_fhir_injoignable_arrete_tout(tmp_path):
    fhir = FakeFhir(ServerUnreachable("serveur FHIR injoignable"))
    result, alive, _, out = campaign(tmp_path, {T1.id: [GET_P], T2.id: [GET_P]}, [T1, T2], fhir=fhir)
    assert isinstance(result, ServerUnreachable)
    assert lines(out) == [] and len(alive.bodies) == 1       # la 2e tâche n'est pas lancée


def test_run_rejete_tache_en_erreur_puis_reprise(tmp_path):
    result, _, _, out = campaign(tmp_path, {T1.id: [GET_P, FINISH_OK]}, [T1], bad={(T1.id, 2): "gemini"})
    assert result == (0, 1) and lines(out) == []
    [err] = lines(tmp_path / "mab.errors.jsonl")
    assert err["task_id"] == T1.id and err["round"] == 2 and err["error_type"] == "RunRejected"
    # relancée : la tâche repart de zéro, dans une conversation neuve
    result, alive, _, out = campaign(tmp_path, {T1.id: [GET_P, FINISH_OK]}, [T1], start=100)
    assert result == (1, 0) and alive.bodies[0][1] is None


def test_run_place_sous_le_parent_lui_meme_rejete(tmp_path):
    # au 2e tour, la réponse doit être sous un message NEUF, pas directement sous l'agent
    result, _, _, out = campaign(tmp_path, {T1.id: [GET_P, FINISH_OK]}, [T1], bad={(T1.id, 2): "same_parent"})
    assert result == (0, 1) and lines(out) == []


def test_erreur_imprevue_arrete_proprement(tmp_path):
    # une réponse du serveur FHIR d'une forme inconnue (ni data ni error) : KeyError dans
    # play_task. Avant : plantage. Maintenant : journal, arrêt, rien compté contre l'agent.
    result, alive, _, out = campaign(tmp_path, {T1.id: [GET_P], T2.id: [GET_P]}, [T1, T2], fhir=FakeFhir({}))
    assert isinstance(result, CampaignStopped) and "KeyError" in str(result) and T1.id in str(result)
    assert lines(out) == [] and len(alive.bodies) == 1            # la 2e tâche n'est pas lancée
    [err] = lines(tmp_path / "mab.errors.jsonl")
    assert err["task_id"] == T1.id and "KeyError" in err["error"]


def test_commande_erreur_imprevue(tmp_path, capsys):
    configs, env = write_configs(tmp_path)
    code = main(["--config", "mab", "--configs", str(configs), "--out", str(tmp_path / "o.jsonl"),
                 "--task", "task1_1"], transport=httpx.MockTransport(FakeAlive({"task1_1": [GET_P]}).handler),
                fhir=FakeFhir({}), server_ok=lambda base: True, sleep=lambda s: None, env_file=env)
    assert code == 1 and "erreur imprévue" in capsys.readouterr().err


def test_cinq_taches_en_erreur_d_affilee_arretent(tmp_path):
    tasks = TASKS[:6]
    replies = {t.id: [FINISH_OK] for t in tasks}
    bad = {(t.id, 1): "gemini" for t in tasks}
    result, alive, _, out = campaign(tmp_path, replies, tasks, bad=bad)
    assert isinstance(result, CampaignStopped) and len(alive.bodies) == 5


# --- Reprise et choix des tâches -------------------------------------------------------------

def test_reprise_saute_les_taches_faites(tmp_path):
    campaign(tmp_path, {T1.id: [FINISH_OK]}, [T1])
    result, alive, _, out = campaign(tmp_path, {T2.id: [FINISH_OK]}, [T1, T2], start=100)
    assert result == (1, 0) and len(alive.bodies) == 1
    assert [r["task_id"] for r in lines(out)] == [T1.id, T2.id]


def test_reprise_refuse_un_fichier_d_une_autre_config(tmp_path):
    campaign(tmp_path, {T1.id: [FINISH_OK]}, [T1])
    out = tmp_path / "mab.jsonl"
    other = Config(name="autre", room="th-MAB", deployment="dp-MAB", model=MODEL, reasoning_effort="none")
    with make_client(TOKEN, httpx.MockTransport(FakeAlive({}).handler)) as client:
        with pytest.raises(ValueError, match="config"):
            run_campaign(client, other, [T1], FUNCS, out_path=out, fingerprint=FINGERPRINT,
                         fhir=FakeFhir(), sleep=lambda s: None)


def test_reprise_refuse_un_fichier_d_une_autre_room(tmp_path):
    campaign(tmp_path, {T1.id: [FINISH_OK]}, [T1])
    out = tmp_path / "mab.jsonl"
    moved = Config(name="mab", room="th-AUTRE", deployment="dp-MAB", model=MODEL, reasoning_effort="none")
    with make_client(TOKEN, httpx.MockTransport(FakeAlive({}).handler)) as client:
        with pytest.raises(ValueError, match="room_id"):
            run_campaign(client, moved, [T1], FUNCS, out_path=out, fingerprint=FINGERPRINT,
                         fhir=FakeFhir(), sleep=lambda s: None)


def test_plusieurs_taches_en_parallele(tmp_path):
    tasks = TASKS[:4]
    replies = {t.id: [GET_P, FINISH_OK] for t in tasks}
    result, alive, _, out = campaign(tmp_path, replies, tasks, workers=4)
    assert result == (4, 0) and sorted(r["task_id"] for r in lines(out)) == sorted(t.id for t in tasks)
    by_id = {t.id: t for t in tasks}
    for rec in lines(out):          # chaque conversation garde sa tâche et sa branche
        assert rec["history"][0]["content"] == first_message(by_id[rec["task_id"]], FUNCS)
        assert rec["status"] == COMPLETED and rec["n_rounds"] == 2
    parents = [p for _, p in alive.bodies if p is not None]
    assert len(parents) == 4 and len(set(parents)) == 4   # 4 branches distinctes


def test_execution_id_deja_vu_refuse(tmp_path):
    # une relance qui recevrait un run déjà enregistré : la tâche n'est pas enregistrée deux fois
    campaign(tmp_path, {T1.id: [FINISH_OK]}, [T1])
    result, _, _, out = campaign(tmp_path, {T2.id: [FINISH_OK]}, [T1, T2])
    assert result == (0, 1) and [r["task_id"] for r in lines(out)] == [T1.id]


def test_select_tasks():
    assert [t.id for t in select_tasks(TASKS, task_id="task5_1")] == ["task5_1"]
    assert len(select_tasks(TASKS, limit=3)) == 3
    per = select_tasks(TASKS, per_type=1)
    assert len(per) == 10 and len({t.type for t in per}) == 10
    with pytest.raises(ValueError):
        select_tasks(TASKS, task_id="task99_1")
    with pytest.raises(ValueError):
        select_tasks(TASKS, per_type=0)


def test_play_task_seule():
    calls = []

    def send(text, parent):
        calls.append(parent)
        run = copy.deepcopy(REAL_RUN)
        run["id"], run["content"] = f"mg-a{len(calls)}", [GET_P, FINISH_OK][len(calls) - 1]
        return run

    outcome = play_task(T1, FUNCS, send, FakeFhir())
    assert outcome["status"] == COMPLETED and calls == [None, "mg-a1"]


# --- client.run_once avec parent_via_input ----------------------------------------------------

def one_run(parent_of_run, parent_id="mg-a1", question="texte"):
    def handler(request):
        run = copy.deepcopy(REAL_RUN)
        run["parent_id"] = parent_of_run
        return httpx.Response(200, json=run)
    with make_client(TOKEN, httpx.MockTransport(handler)) as client:
        return run_once(client, "th", "dp", question, MODEL, parent_id=parent_id, parent_via_input=True)


def test_parent_via_input():
    assert one_run("mg-u9")["parent_id"] == "mg-u9"
    with pytest.raises(RunRejected, match="message neuf"):
        one_run("mg-a1")                 # directement sous l'agent : pas de message pour l'input
    with pytest.raises(RunRejected):
        one_run(None)
    with pytest.raises(ValueError):
        one_run("mg-u9", parent_id=None)


# --- Ligne de commande ------------------------------------------------------------------------

def write_configs(tmp_path):
    path = tmp_path / "configs.json"
    path.write_text(json.dumps({"mab": {"room": "th-MAB", "deployment": "dp-MAB", "model": MODEL,
                                        "reasoning_effort": "none"}}), encoding="utf-8")
    env = tmp_path / ".env"
    env.write_text("ALIVE_ACCESS_TOKEN=x\n", encoding="utf-8")
    return path, env


def test_commande_une_tache(tmp_path, capsys):
    configs, env = write_configs(tmp_path)
    alive = FakeAlive({"task1_1": [GET_P, FINISH_OK]})
    out = tmp_path / "out.jsonl"
    code = main(["--config", "mab", "--configs", str(configs), "--out", str(out), "--task", "task1_1"],
                transport=httpx.MockTransport(alive.handler), fhir=FakeFhir(),
                server_ok=lambda base: True, sleep=lambda s: None, env_file=env)
    assert code == 0 and lines(out)[0]["task_id"] == "task1_1"
    assert "1 tâches enregistrées" in capsys.readouterr().out


def test_commande_refuse_si_serveur_fhir_absent(tmp_path, capsys):
    configs, env = write_configs(tmp_path)
    alive = FakeAlive({})
    code = main(["--config", "mab", "--configs", str(configs), "--out", str(tmp_path / "o.jsonl"),
                 "--task", "task1_1"], transport=httpx.MockTransport(alive.handler),
                server_ok=lambda base: False, env_file=env)
    assert code == 2 and alive.bodies == []
    assert "serveur FHIR" in capsys.readouterr().err
