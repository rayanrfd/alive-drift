"""Tests de mab/grade.py. Ni réseau, ni Docker.

Un faux clone de MedAgentBench (refsol.py + utils.py écrits dans tmp_path) sert à
tester le chargement et les règles de notation. Les derniers tests utilisent le VRAI
refsol.py s'il est dans ../MedAgentBench (sinon ils sont sautés) : seulement les
correcteurs qui n'appellent pas le serveur (task1, extract_posts, check_has_post).
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

from alive_drift.mab.fhir import ServerUnreachable
from alive_drift.mab.grade import (GRADER_TIMEOUT_S, TASK_DIR, GraderGetFailed, default_out,
                                   grade_file, load_refsol, main, summary)
from alive_drift.mab.loop import COMPLETED, INVALID_ACTION, LIMIT_REACHED
from alive_drift.mab.tasks import FHIR_API_BASE, first_message, load_funcs, load_tasks, tasks_fingerprint

TASKS = {t.id: t for t in load_tasks()}
FUNCS = load_funcs()
FINGERPRINT = tasks_fingerprint()

# send_get_request : celui de leur utils.py, recopié tel quel (pour tester le délai et le témoin)
FAKE_UTILS = '''
import requests

def marker():
    return "utils de MedAgentBench"

def send_get_request(url, params=None, headers=None):
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        return {
            "status_code": response.status_code,
            "data": response.json() if response.headers.get('Content-Type') == 'application/json' else response.text
        }
    except Exception as e:
        return {"error": str(e)}
'''

# task1 : comme le vrai (compare sol) ; task2 : lève une exception (serveur en panne chez
# eux) ; task3 : vérifie ce qu'il reçoit ; task4 : renvoie une valeur « vraie » mais pas True.
FAKE_REFSOL = '''
import json
from .utils import *

def task1(case_data, results, fhir_api_base):
    print(case_data["id"], case_data["sol"], results.result)
    try:
        return case_data["sol"] == json.loads(results.result)
    except Exception:
        return False

def task2(case_data, results, fhir_api_base):
    raise KeyError("data")

def task3(case_data, results, fhir_api_base):
    roles = [h.role for h in results.history]
    return (marker() == "utils de MedAgentBench" and roles == ["user", "agent", "user", "agent"]
            and "POST" in results.history[1].content and fhir_api_base == "http://localhost:8080/fhir/"
            and case_data["eval_MRN"] == "S2380121")

def task4(case_data, results, fhir_api_base):
    return "oui"

# task5 : comme les vrais (GET hors try) ; task6 : GET dont l'échec est avalé
def task5(case_data, results, fhir_api_base):
    get_res = json.loads(send_get_request(f"{fhir_api_base}Observation?patient=X&_format=json")['data'])
    return [get_res["total"]] == json.loads(results.result)

def task6(case_data, results, fhir_api_base):
    try:
        json.loads(send_get_request(f"{fhir_api_base}Observation?code=GLU&_format=json")['data'])
    except Exception:
        return False
    return True
'''


def fake_mab(tmp_path, refsol=FAKE_REFSOL, utils=FAKE_UTILS):
    root = tmp_path / "MedAgentBench"
    task_dir = root / TASK_DIR
    task_dir.mkdir(parents=True)
    if refsol is not None:
        (task_dir / "refsol.py").write_text(refsol, encoding="utf-8")
    (task_dir / "utils.py").write_text(utils, encoding="utf-8")
    return root


def record(task_id, status=COMPLETED, result='["x"]', agent=None, **changes):
    """Une ligne comme celles de mab/loop.py (make_record)."""
    task = TASKS[task_id]
    agent = agent or [f"FINISH({result})"]
    history = [{"role": "user", "content": first_message(task, FUNCS)}]
    for i, reply in enumerate(agent):
        history.append({"role": "agent", "content": reply})
        if i < len(agent) - 1:
            history.append({"role": "user", "content": "POST request accepted and executed successfully."})
    rec = {"task_id": task_id, "task_type": task.type, "config": "mab", "room_id": "th-MAB",
           "deployment_id": "dp-MAB", "model": "openai/gpt-5.4-mini", "reasoning_effort": "none",
           "status": status, "result": result if status == COMPLETED else None,
           "n_rounds": len(agent), "max_rounds": 8, "credits_used": 0.5, "api_base": FHIR_API_BASE,
           "fingerprint": FINGERPRINT, "history": history,
           "runs": [{"execution_id": f"ex-{task_id}-{i}"} for i in range(len(agent))],
           "written_at": "2026-10-23T16:00:00+00:00"}
    rec.update(changes)
    return rec


def write_runs(tmp_path, records, name="mab.jsonl"):
    path = tmp_path / name
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    return path


def lines(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]


def up(base):
    return True


POST_BP = ('POST http://localhost:8080/fhir/Observation\n'
           '{"resourceType": "Observation", "valueString": "118/77 mmHg"}')


# --- Chargement de refsol.py ---------------------------------------------------------------

def test_charge_refsol_avec_son_utils(tmp_path):
    refsol = load_refsol(fake_mab(tmp_path))
    assert refsol.module.marker() == "utils de MedAgentBench"      # le « from .utils import * »
    assert len(refsol.refsol_hash) == 12 and len(refsol.utils_hash) == 12


def test_refsol_absent_refuse(tmp_path):
    with pytest.raises(FileNotFoundError, match="Box"):
        load_refsol(fake_mab(tmp_path, refsol=None))


def test_deux_clones_ne_se_melangent_pas(tmp_path):
    a = load_refsol(fake_mab(tmp_path / "a"))
    b = load_refsol(fake_mab(tmp_path / "b", utils=FAKE_UTILS.replace('"utils de MedAgentBench"', '"autre"')))
    assert a.module.marker() == "utils de MedAgentBench" and b.module.marker() == "autre"


def test_refsol_sans_send_get_request_refuse(tmp_path):
    with pytest.raises(ValueError, match="send_get_request"):
        load_refsol(fake_mab(tmp_path, utils='def marker():\n    return 1\n'))


# --- Règles de notation (celles de calculate_overall et eval.py) --------------------------

def grade(tmp_path, records, server_ok=up):
    runs = write_runs(tmp_path, records)
    out = tmp_path / "mab.graded.jsonl"
    graded = grade_file(runs, out, load_refsol(fake_mab(tmp_path)), server_ok=server_ok)
    return graded, out


def test_juste_et_faux(tmp_path):
    # task1_1 : sol ["S6534835"] dans test_data_v2.json
    graded, out = grade(tmp_path, [record("task1_1", result='["S6534835"]'),
                                   record("task1_2", result='["Patient not found"]')])
    assert [g["correct"] for g in graded] == [True, False] and lines(out) == graded
    assert graded[0]["grader_output"] == "task1_1 ['S6534835'] [\"S6534835\"]\n"
    assert graded[0]["graded"] and graded[0]["grader_error"] is None


def test_sans_reponse_faux_sans_appeler_le_correcteur(tmp_path):
    # task2 lèverait une exception : elle n'est pas appelée
    graded, _ = grade(tmp_path, [record("task2_1", status=LIMIT_REACHED),
                                 record("task2_2", status=INVALID_ACTION)])
    assert [(g["correct"], g["graded"], g["grader_error"]) for g in graded] == [(False, False, None)] * 2


def test_exception_du_correcteur_faux_et_gardee(tmp_path):
    graded, _ = grade(tmp_path, [record("task2_1", result="[60]")])
    assert graded[0]["correct"] is False and graded[0]["grader_error"] == "KeyError: 'data'"


def test_exception_et_serveur_tombe_arrete_sans_ecrire(tmp_path):
    calls = []

    def server_ok(base):            # répond au contrôle du début, plus après
        calls.append(base)
        return len(calls) == 1

    with pytest.raises(ServerUnreachable, match="task2_1"):
        grade(tmp_path, [record("task1_1"), record("task2_1", result="[60]")], server_ok=server_ok)
    assert not (tmp_path / "mab.graded.jsonl").exists()


def test_serveur_absent_au_depart(tmp_path):
    with pytest.raises(ServerUnreachable):
        grade(tmp_path, [record("task1_1")], server_ok=lambda base: False)


def test_le_correcteur_recoit_l_historique_comme_chez_eux(tmp_path):
    graded, _ = grade(tmp_path, [record("task3_1", result='["ok"]', agent=[POST_BP, 'FINISH(["ok"])'])])
    assert graded[0]["correct"] is True


def test_seul_true_compte(tmp_path):
    graded, _ = grade(tmp_path, [record("task4_1")])
    assert graded[0]["correct"] is False


def test_ligne_notee(tmp_path):
    graded, _ = grade(tmp_path, [record("task1_1", result='["S6534835"]')])
    g = graded[0]
    assert g["execution_ids"] == ["ex-task1_1-0"] and g["fingerprint"] == FINGERPRINT
    assert g["status"] == COMPLETED and g["config"] == "mab" and len(g["refsol"]) == 12


# --- Les GET du correcteur ---------------------------------------------------------------

def fhir_response(body=b'{"total": 3}', status=200, reason="OK"):
    """Une vraie requests.Response, comme celles du serveur FHIR (application/fhir+json)."""
    r = requests.Response()
    r.status_code, r.reason, r.url, r._content = status, reason, "http://localhost:8080/fhir/", body
    r.headers["Content-Type"] = "application/fhir+json;charset=utf-8"
    return r


def test_get_du_correcteur_avec_delai_et_note_inchangee(tmp_path, monkeypatch):
    calls = []

    def fake_get(url, params=None, headers=None, timeout=None):
        calls.append((url, timeout))
        return fhir_response()

    monkeypatch.setattr(requests, "get", fake_get)
    graded, _ = grade(tmp_path, [record("task5_1", result="[3]"), record("task5_2", result="[4]")])
    assert [g["correct"] for g in graded] == [True, False]
    assert calls[0] == ("http://localhost:8080/fhir/Observation?patient=X&_format=json", GRADER_TIMEOUT_S)


@pytest.mark.parametrize("task_id", ["task5_1", "task6_1"])   # GET hors try, GET dans un try
@pytest.mark.parametrize("failure", [requests.ConnectionError("refusée"), requests.Timeout("délai"),
                                     fhir_response(b"oups", 500, "Server Error")])
def test_get_du_correcteur_rate_arrete_sans_ecrire(tmp_path, monkeypatch, task_id, failure):
    def fake_get(url, params=None, headers=None, timeout=None):
        if isinstance(failure, Exception):
            raise failure
        return failure

    monkeypatch.setattr(requests, "get", fake_get)
    # le serveur répond à /metadata : sans le témoin, la tâche serait notée fausse sans bruit
    with pytest.raises(GraderGetFailed, match=task_id):
        grade(tmp_path, [record("task1_1"), record(task_id, result="[3]")])
    assert not (tmp_path / "mab.graded.jsonl").exists()


def test_commande_get_du_correcteur_rate(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(requests, "get", lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError("x")))
    runs = write_runs(tmp_path, [record("task5_1", result="[3]")])
    code = main(["--runs", str(runs), "--medagentbench", str(fake_mab(tmp_path))], server_ok=up)
    assert code == 1 and "rien n'a été écrit" in capsys.readouterr().err
    assert not default_out(runs).exists()


# --- Fichiers -----------------------------------------------------------------------------

def test_n_ecrase_jamais(tmp_path):
    runs = write_runs(tmp_path, [record("task1_1")])
    out = tmp_path / "deja.jsonl"
    out.write_text("ancien\n", encoding="utf-8")
    with pytest.raises(ValueError, match="existe déjà"):
        grade_file(runs, out, load_refsol(fake_mab(tmp_path)), server_ok=up)
    assert out.read_text(encoding="utf-8") == "ancien\n"


@pytest.mark.parametrize("bad, words", [
    ([record("task1_1"), record("task1_1")], "en double"),
    ([record("task1_1"), record("task1_2", config="autre")], "config"),
    ([record("task1_1", fingerprint={"tasks": "x"})], "empreinte"),
    ([record("task1_1", task_type="task2")], "type"),
    ([], "aucune tâche"),
])
def test_fichier_incoherent_refuse(tmp_path, bad, words):
    with pytest.raises(ValueError, match=words):
        grade(tmp_path, bad)


def test_resume():
    graded = [{"task_type": t, "correct": c, "status": s, "grader_error": e} for t, c, s, e in [
        ("task1", True, COMPLETED, None), ("task1", False, COMPLETED, None),
        ("task10", False, LIMIT_REACHED, None), ("task2", False, COMPLETED, "KeyError: 'data'")]]
    text = summary(graded)
    assert text.splitlines()[0] == "réussite : 1/4 = 25.0%"
    assert [l.split()[0] for l in text.splitlines()[1:4]] == ["task1", "task2", "task10"]
    assert "1 tâche(s)" in text


def test_commande(tmp_path, capsys):
    runs = write_runs(tmp_path, [record("task1_1", result='["S6534835"]')])
    code = main(["--runs", str(runs), "--medagentbench", str(fake_mab(tmp_path))], server_ok=up)
    assert code == 0 and lines(default_out(runs))[0]["correct"] is True
    assert "réussite : 1/1" in capsys.readouterr().out
    # relancée : la sortie existe, refus
    assert main(["--runs", str(runs), "--medagentbench", str(tmp_path / "MedAgentBench")], server_ok=up) == 2


def test_commande_serveur_absent(tmp_path, capsys):
    runs = write_runs(tmp_path, [record("task1_1")])
    code = main(["--runs", str(runs), "--medagentbench", str(fake_mab(tmp_path))],
                server_ok=lambda base: False)
    assert code == 1 and "rien n'a été écrit" in capsys.readouterr().err


# --- Le vrai refsol.py, s'il est là ---------------------------------------------------------

REAL_MAB = Path(__file__).resolve().parents[3] / "MedAgentBench"
real = pytest.mark.skipif(not (REAL_MAB / TASK_DIR / "refsol.py").is_file(),
                          reason="refsol.py absent de ../MedAgentBench")


@real
def test_vrai_refsol_task1(tmp_path):
    runs = write_runs(tmp_path, [record("task1_1", result='["S6534835"]'),
                                 record("task1_2", result='["S6534835"]'),
                                 record("task1_3", result='["S6534835"]', agent=[POST_BP, 'FINISH(["S6534835"])'])])
    out = tmp_path / "vrai.jsonl"
    graded = grade_file(runs, out, load_refsol(REAL_MAB), server_ok=up)
    # juste ; réponse d'une autre tâche ; un POST, interdit pour une tâche 1 : faux quoi qu'il réponde
    assert TASKS["task1_2"].sol != ["S6534835"]
    assert [g["correct"] for g in graded] == [True, False, False]


@real
def test_vrai_refsol_lit_nos_post():
    module = load_refsol(REAL_MAB).module
    history = [SimpleNamespace(role=r, content=c) for r, c in [
        ("user", "question"), ("agent", POST_BP),
        ("user", "POST request accepted and executed successfully. Please call FINISH"),
        ("agent", "POST url\n{oops"), ("user", "Invalid POST request"), ("agent", 'FINISH(["ok"])')]]
    results = SimpleNamespace(history=history, result='["ok"]')
    assert module.extract_posts(results) == [("http://localhost:8080/fhir/Observation",
                                              {"resourceType": "Observation", "valueString": "118/77 mmHg"})]
    assert module.check_has_post(results) is True


@real
def test_vrai_refsol_get_rate_arrete(tmp_path, monkeypatch):
    # task2 du vrai refsol : GET du patient pour calculer l'âge. Serveur en panne -> arrêt.
    seen = []

    def fake_get(url, params=None, headers=None, timeout=None):
        seen.append(timeout)
        raise requests.ConnectionError("refusée")

    monkeypatch.setattr(requests, "get", fake_get)
    runs = write_runs(tmp_path, [record("task2_1", result="[60]")])
    with pytest.raises(GraderGetFailed, match="task2_1"):
        grade_file(runs, tmp_path / "vrai.jsonl", load_refsol(REAL_MAB), server_ok=up)
    assert seen == [GRADER_TIMEOUT_S]           # le délai passe bien par leur utils.py


@real
def test_vrai_refsol_get_reussi_note_comme_chez_eux(tmp_path, monkeypatch):
    # même tâche, le serveur répond : âge calculé par leur code (date figée 2023-11-13)
    body = b'{"entry": [{"resource": {"birthDate": "1963-01-29"}}]}'
    monkeypatch.setattr(requests, "get", lambda url, params=None, headers=None, timeout=None: fhir_response(body))
    runs = write_runs(tmp_path, [record("task2_1", result="[60]"), record("task2_2", result="[61]")])
    graded = grade_file(runs, tmp_path / "vrai.jsonl", load_refsol(REAL_MAB), server_ok=up)
    assert [g["correct"] for g in graded] == [True, False]
