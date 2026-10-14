"""Tests de meta_eval.py : faire noter l'échantillon, puis comparer aux médecins.

Sans réseau : le lancement passe par le faux Alive de test_jury (un juge qui dit
« non applicable » aux critères impairs et « respecté » aux pairs).
"""

import json

import pytest

from alive_drift.grid import Criterion, grid_fingerprint, load_working_grid
from alive_drift.judge import system_prompt
from alive_drift.meta_eval import (
    JURY, jury_verdict, load_sample, main, measure, sample_responses,
)
from test_jury import TOKEN, full_grid_transport, setup_files
from test_scores import REPO_GRID, REPO_SELECTION

GRID = [Criterion("communication_a", "communication", "Texte du critère A.", 5.0),
        Criterion("hedging_b", "hedging", "Texte du critère B.", 5.0)]
T, F, NA = True, False, None


def sample_row(i, criterion, label, cid=None):
    cid = cid or f"c{i}"
    return {"id": f"m{i:03d}", "criterion": criterion, "label": label, "physician_labels": [label] * 2,
            "prompt_id": f"p-{cid}", "completion_id": cid, "question": f"question de {cid}",
            "response": f"réponse {cid}", "source_line": i}


def write_lines(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return path


# --- L'échantillon --------------------------------------------------------------------

def test_une_reponse_jugee_sur_deux_criteres_n_est_notee_qu_une_fois():
    sample = [sample_row(1, "communication_a", T, "c1"), sample_row(2, "hedging_b", F, "c1"),
              sample_row(3, "hedging_b", T, "c2")]
    assert sample_responses(sample) == [
        {"id": "c1", "config": "meta", "question": "question de c1", "response": "réponse c1",
         "execution_id": "c1"},
        {"id": "c2", "config": "meta", "question": "question de c2", "response": "réponse c2",
         "execution_id": "c2"}]


@pytest.mark.parametrize("changes, message", [
    ({"id": "m001"}, "id absent ou en double"),
    ({"criterion": "autre"}, "absent de la grille"),
    ({"label": None}, "avis des médecins absent"),
    ({"response": ""}, "réponse incomplète"),
    ({"completion_id": "c1", "response": "autre texte"}, "c1 a deux textes différents"),
])
def test_refus_echantillon_mal_forme(tmp_path, changes, message):
    path = write_lines(tmp_path / "sample.jsonl",
                       [sample_row(1, "communication_a", T), {**sample_row(2, "hedging_b", F), **changes}])
    with pytest.raises(ValueError, match=message):
        load_sample(path, GRID)


# --- Les mesures ----------------------------------------------------------------------------

def test_verdict_du_jury():
    def votes(*said):
        return [{"a": s} for s in said]
    assert jury_verdict(votes(T, T, F), "a") is True
    assert jury_verdict(votes(F, NA), "a") is False      # N/A ne compte pas
    assert jury_verdict(votes(T, F, NA), "a") is None    # 1/2 : indécis
    assert jury_verdict(votes(NA, NA), "a") is None      # tout N/A


def vote(cid, a, b, judge):
    return {"question_id": cid, "judge": judge, "verdicts": {"communication_a": a, "hedging_b": b}}


def test_justesse_equilibree_par_juge_et_pour_le_jury():
    # Critère A : 2 oui (c1, c2) et 2 non (c3, c4) des médecins. Critère B : 1 oui (c5), 1 non (c6).
    sample = [sample_row(1, "communication_a", T), sample_row(2, "communication_a", T),
              sample_row(3, "communication_a", F), sample_row(4, "communication_a", F),
              sample_row(5, "hedging_b", T), sample_row(6, "hedging_b", F)]
    j1 = {c: vote(c, a, b, "j1") for c, a, b in [("c1", T, NA), ("c2", F, NA), ("c3", F, NA),
                                                  ("c4", NA, NA), ("c5", NA, T), ("c6", NA, T)]}
    j2 = {c: vote(c, a, b, "j2") for c, a, b in [("c1", T, NA), ("c2", T, NA), ("c3", F, NA),
                                                  ("c4", F, NA), ("c5", NA, T)]}  # c6 pas encore noté
    result = measure(sample, {"j1": j1, "j2": j2}, GRID)

    a1 = result["j1"]["criteria"]["communication_a"]
    assert (a1["yes"], a1["yes_ok"], a1["no"], a1["no_ok"], a1["na"]) == (2, 1, 2, 1, 1)
    assert a1["balanced_accuracy"] == 0.5               # 1/2 des oui, 1/2 des non (N/A = faux)
    assert result["j1"]["criteria"]["hedging_b"]["balanced_accuracy"] == 0.5
    assert result["j1"]["na_rate"] == 1 / 6
    assert result["j2"]["criteria"]["communication_a"]["balanced_accuracy"] == 1.0
    assert result["j2"]["criteria"]["hedging_b"]["balanced_accuracy"] is None  # pas de « non » noté
    assert result["j2"]["mean_balanced_accuracy"] == 1.0
    assert result["j2"]["n"] == 5
    # Jury, sur les 5 lignes notées par les deux : c1 oui ; c2 1/2 indécis ; c3 non ; c4 non ; c5 oui.
    jury = result[JURY]
    assert jury["n"] == 5
    assert jury["criteria"]["communication_a"]["balanced_accuracy"] == (1 / 2 + 2 / 2) / 2
    assert jury["criteria"]["communication_a"]["na"] == 1


def test_un_seul_juge_pas_de_ligne_jury():
    sample = [sample_row(1, "communication_a", T)]
    result = measure(sample, {"j1": {"c1": vote("c1", T, NA, "j1")}}, GRID)
    assert list(result) == ["j1"]


# --- La commande : lancer puis comparer -------------------------------------------------------

def setup(tmp_path, n=3):
    """configs.json avec juge_test (grille de travail du dépôt), et un échantillon de n+1 lignes
    dont deux portent sur la même réponse (n réponses différentes)."""
    setup_files(tmp_path, selection=REPO_SELECTION)  # écrit tmp_path/configs.json (juge_test)
    grid = load_working_grid(REPO_GRID, REPO_SELECTION)
    rows = [sample_row(i, grid[i % 8].name, i % 2 == 0) for i in range(1, n + 1)]
    rows.append(sample_row(n + 1, grid[7].name, True, cid="c1"))
    write_lines(tmp_path / "sample.jsonl", rows)
    (tmp_path / "reference.json").write_text(json.dumps(
        {"physicians": {c.name: {"agreement": 0.75} for c in grid}}), encoding="utf-8")
    return grid, ["--sample", str(tmp_path / "sample.jsonl"), "--out-dir", str(tmp_path / "meta"),
                  "--grid", str(REPO_GRID), "--selection", str(REPO_SELECTION)]


def test_lancer_puis_comparer(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ALIVE_TOKEN", TOKEN)
    grid, common = setup(tmp_path)
    transport, requests = full_grid_transport(n_criteria=8, prompt=system_prompt(grid))

    args = ["run", "--judge", "juge_test", "--configs", str(tmp_path / "configs.json"), *common]
    assert main([*args, "--limit", "2"], transport) == 0
    assert len(requests) == 2
    assert main(args, transport) == 0                  # reprise : seule c3 reste à noter
    assert len(requests) == 3
    votes = [json.loads(line) for line in (tmp_path / "meta" / "juge_test.jsonl").read_text().splitlines()]
    assert [v["question_id"] for v in votes] == ["c1", "c2", "c3"]
    assert {v["judged_config"] for v in votes} == {"meta"}
    assert all(v["grid"] == grid_fingerprint(grid) for v in votes)

    capsys.readouterr()
    report = tmp_path / "rapport.json"
    assert main(["report", "--judges", "juge_test", "--reference", str(tmp_path / "reference.json"),
                 "--report-out", str(report), *common]) == 0
    out = capsys.readouterr().out
    assert "lignes notées / 4" in out and "0.75" in out
    doc = json.loads(report.read_text())
    # Le faux juge : critères pairs « respecté », impairs « non applicable ».
    # m001 (critère 2, non) -> faux ; m002 (critère 3, oui) -> N/A ; m003 (critère 4, non) -> faux ;
    # m004 (critère 8, oui, réponse c1) -> respecté, juste.
    result = doc["result"]["juge_test"]
    assert result["n"] == 4
    assert result["criteria"][grid[7].name]["yes_ok"] == 1
    assert result["criteria"][grid[2].name]["na"] == 1
    assert doc["judges"] == ["juge_test"] and doc["n_sample"] == 4


def test_report_n_ecrit_jamais_par_dessus(tmp_path, capsys):
    _, common = setup(tmp_path)
    write_lines(tmp_path / "meta" / "juge_test.jsonl", [])
    report = tmp_path / "rapport.json"
    report.write_text("ancien\n")
    assert main(["report", "--judges", "juge_test", "--reference", str(tmp_path / "reference.json"),
                 "--report-out", str(report), *common]) == 2
    assert "REFUS" in capsys.readouterr().err and report.read_text() == "ancien\n"


def test_report_refuse_des_votes_d_une_autre_campagne(tmp_path, capsys):
    grid, common = setup(tmp_path)
    stray = {"question_id": "c9", "judge": "juge_test", "judged_config": "meta",
             "response_execution_id": "c9", "grid": grid_fingerprint(grid),
             "verdicts": {c.name: True for c in grid}}
    write_lines(tmp_path / "meta" / "juge_test.jsonl", [stray])
    assert main(["report", "--judges", "juge_test", "--reference", str(tmp_path / "reference.json"),
                 *common]) == 2
    assert "pas de cet échantillon" in capsys.readouterr().err


def test_run_sans_jeton_aucun_run(tmp_path, monkeypatch):
    for name in ("ALIVE_TOKEN", "ALIVE_ACCESS_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    grid, common = setup(tmp_path)
    transport, requests = full_grid_transport(n_criteria=8, prompt=system_prompt(grid))
    args = ["run", "--judge", "juge_test", "--configs", str(tmp_path / "configs.json"), *common]
    assert main(args, transport, env_file=tmp_path / "absent.env") == 2
    assert requests == []
