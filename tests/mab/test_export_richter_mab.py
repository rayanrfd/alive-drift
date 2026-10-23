"""Tests de mab/export_richter.py : notes de mab/grade.py -> fichiers de drift-audit. Ni réseau, ni Docker."""

import json

import pytest

from alive_drift.mab.export_richter import (NAMES, ORDER_SEED, export_rows, instrument_fingerprint,
                                            main, read_graded, task_numbers)
from alive_drift.mab.tasks import FHIR_API_BASE, load_tasks, tasks_fingerprint

TASK_IDS = [t.id for t in load_tasks()]
KNOWN = set(TASK_IDS)


def graded(task_id, correct=True, config="A", model="openai/gpt-5.4-mini", **changes):
    """Une ligne comme celles de mab/grade.py."""
    line = {"task_id": task_id, "task_type": task_id.split("_")[0], "config": config,
            "deployment_id": f"dp-{config}", "model": model, "reasoning_effort": "none",
            "status": "completed", "result": "[1]", "correct": correct, "graded": True,
            "grader_output": "", "grader_error": None, "n_rounds": 2,
            "execution_ids": [f"ex-{config}-{task_id}"], "api_base": FHIR_API_BASE,
            "fingerprint": tasks_fingerprint(), "refsol": "39fa6004dbe6", "refsol_utils": "05e539e32ffe"}
    line.update(changes)
    return line


def write(path, lines):
    path.write_text("".join(json.dumps(l) + "\n" for l in lines), encoding="utf-8")
    return path


def three_files(tmp_path, tasks=TASK_IDS[:30], h1_model="openai/gpt-5.5-mini", **h1_changes):
    paths = {}
    for role, config, model in (("ref", "A", "openai/gpt-5.4-mini"), ("h0", "A_bis", "openai/gpt-5.4-mini"),
                                ("h1", "B", h1_model)):
        changes = h1_changes if role == "h1" else {}
        lines = [{**graded(t, correct=(i % 3 != 0) if role != "h1" else (i % 2 == 0),
                           config=config, model=model), **changes} for i, t in enumerate(tasks)]
        paths[role] = write(tmp_path / f"{config}.graded.jsonl", lines)
    return paths


def read_out(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]


# --- Numérotation des tâches ----------------------------------------------------------------

def test_numeros_fixes_par_la_graine():
    numbers = task_numbers(TASK_IDS)
    assert sorted(numbers.values()) == list(range(300))
    assert numbers == task_numbers(TASK_IDS, ORDER_SEED)          # même graine, mêmes numéros
    assert numbers != task_numbers(TASK_IDS, ORDER_SEED + 1)


def test_un_lot_melange_les_types():
    numbers = task_numbers(TASK_IDS)
    first_lot = [t for t, n in numbers.items() if n < 25]
    assert len({t.split("_")[0] for t in first_lot}) >= 5         # pas 25 tâches du même type


# --- Export ---------------------------------------------------------------------------------

def run_export(tmp_path, paths, *extra):
    out = tmp_path / "richter" / "essai"
    code = main(["--ref", str(paths["ref"]), "--h0", str(paths["h0"]), "--h1", str(paths["h1"]),
                 "--name", "essai", "--out-dir", str(out), *extra])
    return code, out


def test_export_trois_fichiers(tmp_path, capsys):
    paths = three_files(tmp_path)
    code, out = run_export(tmp_path, paths)
    assert code == 0
    numbers = task_numbers(TASK_IDS)
    files = {name: read_out(out / f"{name}_clinical_adequacy.jsonl") for name in NAMES.values()}
    for name, rows in files.items():
        assert [r["id"] for r in rows] == sorted(r["id"] for r in rows)       # rangées par id
        assert all(type(r["id"]) is int and r["id"] == numbers[r["task_id"]] for r in rows)
        assert {r["clinical_adequacy"] for r in rows} <= {0.0, 1.0}
        assert len({r["jury_fingerprint"] for r in rows}) == 1 and rows[0]["order_seed"] == ORDER_SEED
    # même tâche, même id dans les trois fichiers : l'appariement de drift-audit est juste
    by_id = [{r["id"]: r["task_id"] for r in rows} for rows in files.values()]
    assert by_id[0] == by_id[1] == by_id[2]
    # la note est bien celle de la tâche
    ref = {r["task_id"]: r["clinical_adequacy"] for r in files["a_hb"]}
    assert ref[TASK_IDS[0]] == 0.0 and ref[TASK_IDS[1]] == 1.0
    printed = capsys.readouterr().out
    assert "30 tâches dans les trois fichiers" in printed and "run_pipeline1.py --scores-dir" in printed


def test_pas_d_option_pour_changer_la_graine(tmp_path, capsys):
    # une graine choisie après avoir vu les notes affaiblirait le test : pas d'option
    with pytest.raises(SystemExit):
        run_export(tmp_path, three_files(tmp_path), "--order-seed", "7")
    assert "unrecognized arguments" in capsys.readouterr().err


def test_previent_lots_incomplets_et_trop_peu_de_paires(tmp_path, capsys):
    code, _ = run_export(tmp_path, three_files(tmp_path))         # 30 tâches : 1 lot + 5, < 100
    printed = capsys.readouterr().out
    assert code == 0 and "les 5 dernières paires" in printed and "run_pipeline1 refusera" in printed


def test_300_taches_douze_lots_sans_avertissement(tmp_path, capsys):
    code, _ = run_export(tmp_path, three_files(tmp_path, tasks=TASK_IDS))
    printed = capsys.readouterr().out
    assert code == 0 and "300 tâches dans les trois fichiers" in printed and "ATTENTION" not in printed


def test_taches_manquantes_signalees(tmp_path, capsys):
    paths = three_files(tmp_path)
    write(paths["h1"], [graded(t, config="B", model="openai/gpt-5.5-mini") for t in TASK_IDS[:20]])
    code, _ = run_export(tmp_path, paths)
    assert code == 0 and "20 tâches dans les trois fichiers" in capsys.readouterr().out


def test_n_ecrit_jamais_par_dessus(tmp_path, capsys):
    paths = three_files(tmp_path)
    (tmp_path / "richter" / "essai").mkdir(parents=True)
    code, _ = run_export(tmp_path, paths)
    assert code == 2 and "existe déjà" in capsys.readouterr().err


@pytest.mark.parametrize("h1_changes, words", [
    ({"refsol": "autre"}, "même instrument"),
    ({"fingerprint": {"tasks": "x"}}, "même instrument"),
    ({"api_base": "http://ailleurs:8080/fhir/"}, "même instrument"),
    ({"config": "A"}, "trois configs différentes"),
])
def test_refus(tmp_path, capsys, h1_changes, words):
    code, _ = run_export(tmp_path, three_files(tmp_path, **h1_changes))
    assert code == 2 and words in capsys.readouterr().err


def test_h0_doit_etre_une_copie_de_ref(tmp_path, capsys):
    paths = three_files(tmp_path)
    write(paths["h0"], [graded(t, config="A_bis", model="openai/gpt-5.5-mini") for t in TASK_IDS[:30]])
    code, _ = run_export(tmp_path, paths)
    assert code == 2 and "copie de --ref" in capsys.readouterr().err


# --- Lecture des notes ----------------------------------------------------------------------

@pytest.mark.parametrize("lines, words", [
    ([graded("task1_1"), graded("task1_1")], "en double"),
    ([graded("task99_1")], "inconnue"),
    ([graded("task1_1", correct=1)], "vrai/faux"),
    ([graded("task1_1"), graded("task1_2", config="B")], "config"),
    ([graded("task1_1"), graded("task1_2", refsol="autre")], "refsol"),
    ([], "aucune note"),
])
def test_read_graded_refuse(tmp_path, lines, words):
    with pytest.raises(ValueError, match=words):
        read_graded(write(tmp_path / "g.jsonl", lines), KNOWN)


def test_read_graded_fichier_absent(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_graded(tmp_path / "absent.jsonl", KNOWN)


def test_empreinte_instrument():
    base = graded("task1_1")
    assert instrument_fingerprint(base) == instrument_fingerprint(graded("task2_1", config="Z"))
    assert instrument_fingerprint(base) != instrument_fingerprint(graded("task1_1", refsol_utils="x"))


def test_export_rows_seul():
    rows = {role: [graded("task1_1", config=c, correct=ok)]
            for role, c, ok in (("ref", "A", True), ("h0", "A_bis", True), ("h1", "B", False))}
    fingerprint, out = export_rows(rows, task_numbers(TASK_IDS), ORDER_SEED)
    assert set(out) == {"a_hb", "a_bis_hb", "b_hb"}
    assert out["b_hb"][0]["clinical_adequacy"] == 0.0 and out["a_hb"][0]["jury_fingerprint"] == fingerprint
