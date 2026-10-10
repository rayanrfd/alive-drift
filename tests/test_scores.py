"""Tests de scores.py : des votes du jury aux notes. Aucun appel à Alive.

Les fichiers de votes sont écrits à la main, dans la forme de votes.make_vote
(seuls les champs que scores.py lit). Un vote se donne par une lettre par
critère : "o" oui (true), "n" non (false), "-" non applicable (null).
"""

import json
from pathlib import Path

import pytest

from alive_drift.grid import Criterion, grid_fingerprint, load_working_grid
from alive_drift.scores import criterion_values, main, read_votes, score_config, score_response

REPO = Path(__file__).parent.parent
REPO_GRID = REPO / "data" / "criteria.json"
REPO_SELECTION = REPO / "data" / "selection.json"

# Une petite grille de 3 critères sur 2 thèmes, pour des calculs lisibles.
GRID = [Criterion("communication_a", "communication", "Texte du critère A.", 5.0),
        Criterion("hedging_b", "hedging", "Texte du critère B.", 5.0),
        Criterion("hedging_c", "hedging", "Texte du critère C.", 5.0)]

LETTERS = {"o": True, "n": False, "-": None}


def vote(qid, letters, judge, config="A", grid=GRID, /, **changes):
    """Une ligne de vote : `letters` = "on-" -> a oui, b non, c non applicable.

    `changes` remplace des champs de la ligne (ex. judge="j2" pour une ligne fausse).
    """
    row = {"question_id": qid, "judged_config": config,
           "response_execution_id": f"ex-{config}-{qid}", "judge": judge,
           "grid": grid_fingerprint(grid),
           "verdicts": {c.name: LETTERS[x] for c, x in zip(grid, letters)}}
    row.update(changes)
    return row


def write_votes(jury_dir, config, judge, rows):
    jury_dir.mkdir(parents=True, exist_ok=True)
    path = jury_dir / f"{config}.{judge}.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


# --- Une réponse ----------------------------------------------------------------

def test_valeur_d_un_critere_part_de_oui_sans_les_na():
    votes = [{"a": True}, {"a": False}, {"a": None}]
    assert criterion_values(votes, ["a"]) == {"a": 0.5}           # oui, non, N/A -> 1/2
    assert criterion_values([{"a": True}, {"a": True}, {"a": False}], ["a"]) == {"a": 2 / 3}
    assert criterion_values([{"a": None}, {"a": None}], ["a"]) == {"a": None}  # tous N/A


def test_score_d_une_reponse():
    votes = {"j1": vote("0", "on-", "j1"), "j2": vote("0", "oo-", "j2")}
    assert score_response(votes, GRID) == {
        "criteria": {"communication_a": 1.0, "hedging_b": 0.5, "hedging_c": None},
        "themes": {"communication": 1.0, "hedging": 0.5},   # pas les thèmes sans critère
        "score": 0.75,                                       # moyenne de 1.0 et 0.5
        "by_judge": {"j1": 0.5, "j2": 1.0},
    }


def test_le_score_global_est_la_moyenne_des_criteres_pas_des_themes():
    # a = 1, b = 0, c = 0 : moyenne des critères 1/3 ; celle des thèmes serait 1/2.
    votes = {"j1": vote("0", "onn", "j1"), "j2": vote("0", "onn", "j2")}
    result = score_response(votes, GRID)
    assert result["themes"] == {"communication": 1.0, "hedging": 0.0}
    assert result["score"] == 1 / 3


def test_reponse_ou_tout_est_non_applicable():
    result = score_response({"j1": vote("0", "---", "j1")}, GRID)
    assert result["score"] is None
    assert result["themes"] == {"communication": None, "hedging": None}
    assert result["by_judge"] == {"j1": None}


# --- Une config -----------------------------------------------------------------

def test_score_config_garde_les_reponses_notees_par_tous_les_juges(tmp_path):
    write_votes(tmp_path, "A", "j1", [vote(q, "ooo", "j1") for q in ("10", "2", "0", "7")])
    write_votes(tmp_path, "A", "j2", [vote(q, "ono", "j2") for q in ("0", "10", "2")])
    rows, excluded = score_config("A", ["j1", "j2"], GRID, tmp_path)
    assert [r["question_id"] for r in rows] == ["0", "2", "10"]   # ordre numérique
    assert excluded == 1                                           # la 7 : j2 manque
    assert rows[0] == {"question_id": "0", "config": "A", "response_execution_id": "ex-A-0",
                       "grid": grid_fingerprint(GRID), "judges": ["j1", "j2"],
                       **score_response({"j1": vote("0", "ooo", "j1"),
                                         "j2": vote("0", "ono", "j2")}, GRID)}


def test_refus_si_les_juges_n_ont_pas_note_la_meme_reponse(tmp_path):
    write_votes(tmp_path, "A", "j1", [vote("0", "ooo", "j1")])
    write_votes(tmp_path, "A", "j2", [vote("0", "ooo", "j2", response_execution_id="ex-autre")])
    with pytest.raises(ValueError, match="pas noté la même réponse"):
        score_config("A", ["j1", "j2"], GRID, tmp_path)


def test_refus_si_un_fichier_contient_une_autre_config(tmp_path):
    write_votes(tmp_path, "A", "j1", [vote("0", "ooo", "j1", "B")])
    with pytest.raises(ValueError, match="autre config"):
        score_config("A", ["j1"], GRID, tmp_path)


@pytest.mark.parametrize("judges", [[], ["j1", "j1"]])
def test_refus_liste_de_juges_vide_ou_en_double(tmp_path, judges):
    write_votes(tmp_path, "A", "j1", [vote("0", "ooo", "j1")])
    with pytest.raises(ValueError, match="vide ou avec un doublon"):
        score_config("A", judges, GRID, tmp_path)


def test_refus_fichier_de_votes_absent(tmp_path):
    with pytest.raises(FileNotFoundError, match="introuvable"):
        score_config("A", ["j1"], GRID, tmp_path)


# --- Lecture d'un fichier de votes -------------------------------------------------

@pytest.mark.parametrize("changes, message", [
    ({"judge": "j2"}, "juge 'j2', attendu 'j1'"),
    ({"grid": "autre"}, "grille 'autre'"),
    ({"question_id": ""}, "question_id absent"),
    ({"verdicts": {"communication_a": True, "hedging_b": True}}, "mal formés"),
    ({"verdicts": {"hedging_b": True, "communication_a": True, "hedging_c": True}}, "mal formés"),
    ({"verdicts": {"communication_a": True, "hedging_b": True, "hedging_c": 1}}, "mal formés"),
    ({"verdicts": {"communication_a": "met", "hedging_b": True, "hedging_c": True}}, "mal formés"),
])
def test_refus_ligne_de_vote_invalide(tmp_path, changes, message):
    path = write_votes(tmp_path, "A", "j1", [vote("0", "ooo", "j1"),
                                             vote("1", "ooo", "j1", **changes)])
    with pytest.raises(ValueError, match=message):
        read_votes(path, GRID, "j1")


def test_refus_question_en_double(tmp_path):
    path = write_votes(tmp_path, "A", "j1", [vote("0", "ooo", "j1"), vote("0", "nnn", "j1")])
    with pytest.raises(ValueError, match="en double"):
        read_votes(path, GRID, "j1")


# --- La commande ----------------------------------------------------------------------

def campaign(jury_dir):
    """Votes de 2 juges sur la vraie grille de travail (8 critères), 3 configs.

    j1 dit oui partout ; j2 dit non au 1er critère et oui ailleurs. Donc le 1er
    critère vaut 1/2, les 7 autres 1 : score du jury (0.5 + 7) / 8 = 0.9375.
    La question 2 de B n'est notée que par j1 : elle est écartée.
    """
    grid = load_working_grid(REPO_GRID, REPO_SELECTION)
    n = len(grid)
    for config in ("A", "A_bis", "B"):
        questions = ("0", "1", "2")
        write_votes(jury_dir, config, "j1",
                    [vote(q, "o" * n, "j1", config, grid) for q in questions])
        kept = questions[:2] if config == "B" else questions
        write_votes(jury_dir, config, "j2",
                    [vote(q, "n" + "o" * (n - 1), "j2", config, grid) for q in kept])
    return grid


def run_main(jury_dir, *extra):
    return main(["--judges", "j1", "j2", "--jury-dir", str(jury_dir),
                 "--grid", str(REPO_GRID), "--selection", str(REPO_SELECTION), *extra])


def test_main_ecrit_les_scores_et_affiche_le_resume(tmp_path, capsys):
    grid = campaign(tmp_path / "jury")
    out_dir = tmp_path / "scores"
    assert run_main(tmp_path / "jury", "--out-dir", str(out_dir)) == 0
    rows = {c: [json.loads(line) for line in (out_dir / f"{c}.jsonl").read_text().splitlines()]
            for c in ("A", "A_bis", "B")}
    assert {c: len(r) for c, r in rows.items()} == {"A": 3, "A_bis": 3, "B": 2}
    first = rows["A"][0]
    assert first["score"] == 0.9375
    assert first["criteria"][grid[0].name] == 0.5
    assert first["by_judge"] == {"j1": 1.0, "j2": 7 / 8}
    out = capsys.readouterr().out
    lines = {line[:28].strip(): line[28:].split() for line in out.splitlines()}
    assert lines["réponses notées"] == ["3", "3", "2"]
    assert lines["écartées (juge manquant)"] == ["0", "0", "1"]
    assert lines["score du jury"] == ["0.94", "0.94", "0.94"]
    assert lines["j1"] == ["1.00", "1.00", "1.00"]
    assert lines["1"] == ["0.50", "0.50", "0.50", grid[0].name]   # numéro, valeurs, nom
    assert lines["8"] == ["1.00", "1.00", "1.00", grid[7].name]
    assert out.rstrip().endswith(f"-> {out_dir}")


def test_main_ecrit_par_defaut_dans_un_dossier_date(tmp_path, monkeypatch):
    campaign(tmp_path / "jury")
    monkeypatch.chdir(tmp_path)
    assert run_main(tmp_path / "jury") == 0
    (stamp,) = list((tmp_path / "outputs" / "scores").iterdir())
    assert sorted(p.name for p in stamp.iterdir()) == ["A.jsonl", "A_bis.jsonl", "B.jsonl"]


def test_main_n_ecrit_jamais_par_dessus_un_calcul(tmp_path, capsys):
    campaign(tmp_path / "jury")
    out_dir = tmp_path / "scores"
    out_dir.mkdir()
    (out_dir / "A.jsonl").write_text("ancien calcul\n")
    assert run_main(tmp_path / "jury", "--out-dir", str(out_dir)) == 2
    assert "REFUS" in capsys.readouterr().err
    assert [p.name for p in out_dir.iterdir()] == ["A.jsonl"]
    assert (out_dir / "A.jsonl").read_text() == "ancien calcul\n"


def test_main_refuse_si_un_juge_manque_et_n_ecrit_rien(tmp_path, capsys):
    campaign(tmp_path / "jury")
    (tmp_path / "jury" / "B.j2.jsonl").unlink()
    out_dir = tmp_path / "scores"
    assert run_main(tmp_path / "jury", "--out-dir", str(out_dir)) == 2
    err = capsys.readouterr().err
    assert "REFUS" in err and "B.j2.jsonl : fichier de votes introuvable" in err
    assert not out_dir.exists()   # rien d'écrit, même pas A et A_bis
