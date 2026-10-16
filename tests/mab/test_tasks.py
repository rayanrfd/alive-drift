"""Tests de mab/tasks.py : tâches MedAgentBench et premier message. Aucun appel réseau.

Les vraies données sont celles de data/medagentbench/ (copie de MedAgentBench,
commit 9926011) ; les refus sont testés sur de petits fichiers fabriqués ici.
"""

import hashlib
import json
from collections import Counter

import pytest

from alive_drift.mab.tasks import (PROMPT_TEMPLATE, first_message, load_funcs, load_tasks,
                                   main, tasks_fingerprint)

# Empreinte du gabarit MedAgentBench_prompt d'origine (commit 9926011), calculée sur leur
# fichier : si quelqu'un retouche PROMPT_TEMPLATE, ce test le signale.
TEMPLATE_SHA12 = "6d006952a997"


def write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def task(tid="task4_1", /, **changes):
    row = {"id": tid, "instruction": "What is X?", "context": "It's now.", "eval_MRN": "S1"}
    row.update(changes)
    return {k: v for k, v in row.items() if v is not None}


# --- Les vraies données ------------------------------------------------------------

def test_les_300_taches_de_medagentbench():
    tasks = load_tasks()
    assert len(tasks) == 300
    assert Counter(t.type for t in tasks) == {f"task{k}": 30 for k in range(1, 11)}
    assert len({t.id for t in tasks}) == 300
    assert all(t.sol is not None for t in tasks if t.type == "task1")
    # deux tâches 1 « Patient not found » n'ont pas de patient ; toutes les autres en ont un
    assert sorted(t.id for t in tasks if t.eval_mrn is None) == ["task1_29", "task1_30"]
    assert all(t.eval_mrn.startswith("S") for t in tasks if t.eval_mrn is not None)


def test_les_9_fonctions():
    funcs = load_funcs()
    assert len(funcs) == 9
    assert funcs[0]["name"] == "GET {api_base}/Condition"
    assert funcs[-1]["name"] == "GET {api_base}/Patient"


def test_le_gabarit_est_celui_de_medagentbench():
    assert hashlib.sha256(PROMPT_TEMPLATE.encode("utf-8")).hexdigest()[:12] == TEMPLATE_SHA12
    assert tasks_fingerprint()["template"] == TEMPLATE_SHA12


# --- Le premier message ------------------------------------------------------------

def test_premier_message_d_une_vraie_tache():
    tasks, funcs = load_tasks(), load_funcs()
    t = next(t for t in tasks if t.id == "task4_1")
    msg = first_message(t, funcs)
    assert msg.startswith("You are an expert in using FHIR functions to assist medical professionals.")
    assert msg.endswith(f"\nContext: {t.context}\nQuestion: {t.instruction}")
    assert json.dumps(funcs) in msg                     # la liste écrite comme chez eux
    assert "Note that you should use http://localhost:8080/fhir/ as the api_base." in msg
    assert "{context}" not in msg and "{question}" not in msg


def test_api_base_remplace_dans_la_consigne_mais_pas_dans_les_fonctions():
    funcs = load_funcs()
    msg = first_message(load_tasks()[0], funcs, api_base="http://h:1/fhir/")
    assert msg.count("http://h:1/fhir/") == 1           # une seule fois : dans la consigne
    assert '"name": "GET {api_base}/Condition"' in msg  # texte laissé tel quel, comme chez eux


def test_contexte_vide_comme_chez_medagentbench():
    t = next(t for t in load_tasks() if t.id == "task1_1")
    assert t.context == ""
    assert "\nContext: \nQuestion: What" in first_message(t, load_funcs())


# --- Refus ---------------------------------------------------------------------------

@pytest.mark.parametrize("rows, message", [
    ([task("task4_1"), task("task4_1")], "en double"),
    ([task(instruction=None)], "'instruction'"),
    ([task(context=None)], "'context'"),
    ([task("tache4_1")], "taskN_M"),
    ([task("task4_01")], "taskN_M"),
    ([task(eval_MRN=None)], "eval_MRN absent"),
    ([task(eval_MRN=7)], "eval_MRN n'est pas un texte"),
    ([task(sol="S1")], "sol n'est pas une liste"),
    ([[1, 2]], "un objet est attendu"),
    ([], "liste non vide"),
])
def test_refus_d_un_fichier_de_taches_abime(tmp_path, rows, message):
    with pytest.raises(ValueError, match=message):
        load_tasks(write_json(tmp_path / "t.json", rows))


def test_une_tache_1_peut_se_passer_de_patient(tmp_path):
    (t,) = load_tasks(write_json(tmp_path / "t.json", [task("task1_3", eval_MRN=None, sol=["x"])]))
    assert (t.type, t.eval_mrn, t.sol) == ("task1", None, ["x"])


def test_refus_d_un_fichier_illisible_ou_absent(tmp_path):
    bad = tmp_path / "t.json"
    bad.write_text("[{", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON illisible"):
        load_tasks(bad)
    with pytest.raises(FileNotFoundError):
        load_tasks(tmp_path / "absent.json")


def test_refus_d_une_fonction_sans_nom(tmp_path):
    with pytest.raises(ValueError, match="name et description"):
        load_funcs(write_json(tmp_path / "f.json", [{"description": "d"}]))


# --- Commande ---------------------------------------------------------------------

def test_commande_resume(capsys):
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "300 tâches, 9 fonctions" in out
    assert "  task10    30" in out
    assert f"gabarit {TEMPLATE_SHA12}" in out


def test_commande_une_tache_et_le_gabarit(capsys):
    assert main(["--task", "task5_1"]) == 0
    t = next(t for t in load_tasks() if t.id == "task5_1")
    assert capsys.readouterr().out == first_message(t, load_funcs()) + "\n"
    assert main(["--template"]) == 0
    assert capsys.readouterr().out == PROMPT_TEMPLATE + "\n"


def test_commande_refuse_une_tache_inconnue_ou_un_fichier_abime(tmp_path, capsys):
    assert main(["--task", "task99_1"]) == 2
    assert "introuvable" in capsys.readouterr().err
    assert main(["--tasks", str(write_json(tmp_path / "t.json", [task("x")]))]) == 2
    assert "REFUS" in capsys.readouterr().err


def test_l_empreinte_change_avec_le_fichier(tmp_path):
    a = write_json(tmp_path / "a.json", [task()])
    b = write_json(tmp_path / "b.json", [task(instruction="What is Y?")])
    funcs = write_json(tmp_path / "f.json", [{"name": "n", "description": "d"}])
    assert tasks_fingerprint(a, funcs)["tasks"] != tasks_fingerprint(b, funcs)["tasks"]
    assert tasks_fingerprint(a, funcs)["funcs"] == tasks_fingerprint(b, funcs)["funcs"]
