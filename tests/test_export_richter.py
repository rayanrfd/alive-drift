"""Tests de export_richter.py : scores du jury -> fichiers d'entrée de Richter.

Le format attendu par drift-audit est recopié ici (on n'importe pas drift-audit) :
un fichier {a_hb, a_bis_hb, b_hb}_clinical_adequacy.jsonl par config, des lignes
{"id": entier, "clinical_adequacy": score ou null, "jury_fingerprint": texte}.
"""

import json

import pytest

from alive_drift.export_richter import export_rows, jury_fingerprint, main, read_scores
from alive_drift.grid import grid_fingerprint
from test_scores import campaign, run_main as run_scores


def score_row(qid, score, config="A", judges=("j1", "j2"), grid="g1", /, **changes):
    """Une ligne de scores.py, réduite aux champs que l'export lit.

    `changes` remplace des champs de la ligne (ex. score=1.2 pour une ligne fausse).
    """
    row = {"question_id": qid, "config": config, "response_execution_id": f"ex-{config}-{qid}",
           "grid": grid, "judges": list(judges), "score": score}
    row.update(changes)
    return row


def write_scores(scores_dir, rows_by_config):
    scores_dir.mkdir(parents=True, exist_ok=True)
    for config, rows in rows_by_config.items():
        (scores_dir / f"{config}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return scores_dir


def read_out(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def three_configs(**b_changes):
    return {"A": [score_row("0", 0.5), score_row("1", 1.0)],
            "A_bis": [score_row("0", 0.5, "A_bis"), score_row("1", None, "A_bis")],
            "B": [score_row("0", 0.25, "B", **b_changes)]}


# --- De bout en bout : votes -> scores.py -> export ---------------------------------

def test_des_votes_aux_fichiers_de_richter(tmp_path, capsys):
    grid = campaign(tmp_path / "jury")
    assert run_scores(tmp_path / "jury", "--out-dir", str(tmp_path / "scores" / "s1")) == 0
    out_dir = tmp_path / "richter"
    assert main([str(tmp_path / "scores" / "s1"), "--out-dir", str(out_dir)]) == 0

    assert sorted(p.name for p in out_dir.iterdir()) == [
        "a_bis_hb_clinical_adequacy.jsonl", "a_hb_clinical_adequacy.jsonl",
        "b_hb_clinical_adequacy.jsonl"]
    a = read_out(out_dir / "a_hb_clinical_adequacy.jsonl")
    b = read_out(out_dir / "b_hb_clinical_adequacy.jsonl")
    fingerprint = jury_fingerprint(["j1", "j2"], grid_fingerprint(grid))
    assert a[0] == {"id": 0, "clinical_adequacy": 0.9375, "jury_fingerprint": fingerprint,
                    "config": "A", "response_execution_id": "ex-A-0",
                    "judges": ["j1", "j2"], "grid": grid_fingerprint(grid)}
    assert [r["id"] for r in a] == [0, 1, 2]
    assert [r["id"] for r in b] == [0, 1]          # la 2 de B a été écartée par scores.py
    out = capsys.readouterr().out
    assert f"empreinte du jury : {fingerprint}" in out
    assert "b_hb          2 questions, dont 0 sans score" in out
    assert f"run_pipeline1.py --scores-dir {out_dir.resolve()}" in out


# --- Contenu ----------------------------------------------------------------------------

def test_un_score_null_est_ecrit_tel_quel():
    _, out = export_rows(three_configs())
    assert [r["clinical_adequacy"] for r in out["a_bis_hb"]] == [0.5, None]
    assert [r["id"] for r in out["a_bis_hb"]] == [0, 1]


def test_l_empreinte_change_avec_les_juges_leur_ordre_et_la_grille():
    base = jury_fingerprint(["j1", "j2"], "g1")
    assert len(base) == 12
    assert len({base, jury_fingerprint(["j2", "j1"], "g1"), jury_fingerprint(["j1"], "g1"),
                jury_fingerprint(["j1", "j2"], "g2")}) == 4


@pytest.mark.parametrize("b_changes", [{"judges": ["j1"]}, {"judges": ["j2", "j1"]},
                                       {"grid": "g2"}])
def test_refus_si_les_configs_n_ont_pas_le_meme_jury(b_changes):
    with pytest.raises(ValueError, match="pas du même jury"):
        export_rows(three_configs(**b_changes))


# --- Lecture d'un fichier de scores ----------------------------------------------------

@pytest.mark.parametrize("changes, message", [
    ({"config": "B"}, "config 'B', attendu 'A'"),
    ({"question_id": "q1"}, "n'est pas un numéro"),
    ({"question_id": "²"}, "n'est pas un numéro"),
    ({"question_id": "٢"}, "n'est pas un numéro"),
    ({"question_id": "01"}, "n'est pas un numéro"),
    ({"question_id": "1 "}, "n'est pas un numéro"),
    ({"question_id": "0"}, "en double"),
    ({"score": 1.2}, "hors de"),
    ({"score": -0.1}, "hors de"),
    ({"score": True}, "hors de"),
    ({"score": "0.5"}, "hors de"),
    ({"judges": []}, "juges absente"),
    ({"judges": ["j1", ""]}, "juges absente"),
    ({"grid": ""}, "grille absente"),
])
def test_refus_ligne_de_score_invalide(tmp_path, changes, message):
    write_scores(tmp_path, {"A": [score_row("0", 0.5), score_row("1", 0.5, **changes)]})
    with pytest.raises(ValueError, match=message):
        read_scores(tmp_path / "A.jsonl", "A")


def test_refus_fichier_absent_ou_vide(tmp_path):
    with pytest.raises(FileNotFoundError, match="introuvable"):
        read_scores(tmp_path / "A.jsonl", "A")
    (tmp_path / "A.jsonl").write_text("")
    with pytest.raises(ValueError, match="aucun score"):
        read_scores(tmp_path / "A.jsonl", "A")


# --- La commande ------------------------------------------------------------------------

def test_main_ecrit_par_defaut_dans_outputs_richter(tmp_path, monkeypatch):
    write_scores(tmp_path / "scores" / "20261014-144523", three_configs())
    monkeypatch.chdir(tmp_path)
    assert main([str(tmp_path / "scores" / "20261014-144523")]) == 0
    out_dir = tmp_path / "outputs" / "richter" / "20261014-144523"
    assert len(read_out(out_dir / "a_hb_clinical_adequacy.jsonl")) == 2


def test_main_n_ecrit_jamais_par_dessus(tmp_path, capsys):
    scores_dir = write_scores(tmp_path / "scores", three_configs())
    out_dir = tmp_path / "richter"
    out_dir.mkdir()
    (out_dir / "a_hb_clinical_adequacy.jsonl").write_text("ancien\n")
    assert main([str(scores_dir), "--out-dir", str(out_dir)]) == 2
    assert "REFUS" in capsys.readouterr().err
    assert (out_dir / "a_hb_clinical_adequacy.jsonl").read_text() == "ancien\n"
    assert [p.name for p in out_dir.iterdir()] == ["a_hb_clinical_adequacy.jsonl"]


def test_main_refuse_si_un_fichier_de_scores_manque_et_n_ecrit_rien(tmp_path, capsys):
    rows = three_configs()
    del rows["B"]
    scores_dir = write_scores(tmp_path / "scores", rows)
    out_dir = tmp_path / "richter"
    assert main([str(scores_dir), "--out-dir", str(out_dir)]) == 2
    assert "B.jsonl : fichier de scores introuvable" in capsys.readouterr().err
    assert not out_dir.exists()


def test_main_refuse_des_jurys_differents_et_n_ecrit_rien(tmp_path, capsys):
    scores_dir = write_scores(tmp_path / "scores", three_configs(judges=["j1"]))
    out_dir = tmp_path / "richter"
    assert main([str(scores_dir), "--out-dir", str(out_dir)]) == 2
    assert "pas du même jury" in capsys.readouterr().err
    assert not out_dir.exists()
