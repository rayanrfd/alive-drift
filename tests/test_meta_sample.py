"""Tests de meta_sample.py : le tirage de l'échantillon de méta-évaluation.

La source est écrite à la main, dans la forme du fichier oss_meta_eval de
HealthBench (seuls les champs lus) : une ligne = une réponse jugée par 2 à 4
médecins sur UN critère (category "cluster:<nom>").
"""

import json
import random

import pytest

from alive_drift.grid import Criterion, load_working_grid
from alive_drift.meta_sample import (
    agreeing_pairs, draw, main, majority, physician_agreement, read_source,
)
from test_scores import REPO_GRID, REPO_SELECTION

GRID = [Criterion("communication_a", "communication", "Texte du critère A.", 5.0),
        Criterion("hedging_b", "hedging", "Texte du critère B.", 5.0)]
T, F = True, False


def source_row(name, labels, cid, *, text=None, turns=1, role="user", grid=GRID, **changes):
    """Une ligne de la source HealthBench pour le critère `name`."""
    texts = {c.name: c.text for c in grid}
    row = {"category": f"cluster:{name}", "rubric": texts.get(name, "autre") if text is None else text,
           "prompt": [{"role": role, "content": f"question de {cid}"}] * turns,
           "completion": f"réponse {cid}", "completion_id": cid, "prompt_id": f"p-{cid}",
           "binary_labels": labels}
    row.update(changes)
    return row


def write_source(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return path


def test_avis_majoritaire_et_paires_d_accord():
    assert majority([T, F]) is None              # égalité
    assert majority([T, T, F]) is True
    assert majority([F, F]) is False
    assert agreeing_pairs([T, F]) == (0, 1)
    assert agreeing_pairs([T, T]) == (1, 1)
    assert agreeing_pairs([T, T, F]) == (1, 3)   # (1,2) d'accord ; (1,3) et (2,3) non


def test_lecture_garde_un_tour_et_les_criteres_de_la_grille(tmp_path):
    path = write_source(tmp_path / "src.jsonl", [
        source_row("communication_a", [T, T], "c1"),
        source_row("communication_a", [T, F], "c2"),              # égalité : gardée ici
        source_row("communication_a", [T, T], "c3", turns=3),     # plusieurs tours : écartée
        source_row("hedging_b", [F, F], "c4"),
        source_row("autre_critere", [T, T], "c5"),                # hors grille : ignoré
    ])
    rows = read_source(path, GRID)
    assert [r["completion_id"] for r in rows["communication_a"]] == ["c1", "c2"]
    assert [r["completion_id"] for r in rows["hedging_b"]] == ["c4"]
    assert rows["hedging_b"][0]["question"] == "question de c4"
    assert rows["hedging_b"][0]["response"] == "réponse c4"


def test_refus_si_le_texte_du_critere_differe(tmp_path):
    path = write_source(tmp_path / "src.jsonl", [
        source_row("communication_a", [T, T], "c1"),
        source_row("hedging_b", [T, T], "c2", text="Texte du critère B, reformulé."),
    ])
    with pytest.raises(ValueError, match="ligne 2 : le texte du critère hedging_b diffère"):
        read_source(path, GRID)


@pytest.mark.parametrize("changes", [
    {"role": "assistant"},
    {"labels": [T]},              # un seul médecin
    {"labels": [T, 1]},           # pas un booléen
    {"completion": "  "},
    {"completion_id": ""},
])
def test_refus_ligne_mal_formee(tmp_path, changes):
    labels = changes.pop("labels", [T, T])
    path = write_source(tmp_path / "src.jsonl", [source_row("communication_a", [F, F], "c0"),
                                                 source_row("communication_a", labels, "c1", **changes)])
    with pytest.raises(ValueError, match="ligne 2 : ligne mal formée"):
        read_source(path, GRID)


def test_refus_meme_reponse_deux_fois_pour_un_critere(tmp_path):
    path = write_source(tmp_path / "src.jsonl", [source_row("communication_a", [T, T], "c1"),
                                                 source_row("communication_a", [F, F], "c1"),
                                                 source_row("hedging_b", [T, T], "c1")])
    with pytest.raises(ValueError, match="ligne 2 : réponse c1 deux fois pour communication_a"):
        read_source(path, GRID)


def test_refus_critere_sans_aucune_ligne(tmp_path):
    path = write_source(tmp_path / "src.jsonl", [source_row("communication_a", [T, T], "c1")])
    with pytest.raises(ValueError, match="aucune ligne en un tour pour : \\['hedging_b'\\]"):
        read_source(path, GRID)


def test_accord_entre_medecins():
    rows = {"communication_a": [{"labels": [T, T]}, {"labels": [T, F]}, {"labels": [T, T, F]}]}
    assert physician_agreement(rows)["communication_a"] == {
        "yes": 2, "no": 0, "tie": 1, "agreeing_pairs": 2, "pairs": 5, "agreement": 0.4}


def candidates(name, n_yes, n_no, n_tie=0):
    rows = [{"criterion": name, "labels": labels, "prompt_id": f"p-{name}-{i}",
             "completion_id": f"{name}-{i}", "question": "q", "response": "r", "line": i}
            for i, labels in enumerate([[T, T]] * n_yes + [[F, F]] * n_no + [[T, F]] * n_tie)]
    return rows


def test_tirage_equilibre_sans_egalites_et_entrelace():
    rows = {"communication_a": candidates("communication_a", 5, 5, n_tie=4),
            "hedging_b": candidates("hedging_b", 5, 1)}             # un seul « non » disponible
    sample = draw(rows, per_label=2, seed=0)
    assert [(r["criterion"], r["label"]) for r in sample] == [
        ("communication_a", T), ("communication_a", F), ("hedging_b", T), ("hedging_b", F),
        ("communication_a", T), ("communication_a", F), ("hedging_b", T)]
    assert [r["id"] for r in sample] == [f"m{i:03d}" for i in range(1, 8)]
    assert all(majority(r["physician_labels"]) is r["label"] for r in sample)  # aucune égalité
    assert len({r["completion_id"] for r in sample}) == 7


def test_tirage_ne_depend_pas_de_l_ordre_du_fichier():
    rows = {"communication_a": candidates("communication_a", 30, 30)}
    shuffled = {"communication_a": random.Random(1).sample(rows["communication_a"], 60)}
    assert draw(rows, 5, seed=0) == draw(shuffled, 5, seed=0)
    assert draw(rows, 5, seed=0) != draw(rows, 5, seed=1)


# --- La commande, sur la vraie grille de travail --------------------------------------

def real_source(tmp_path):
    grid = load_working_grid(REPO_GRID, REPO_SELECTION)
    rows = []
    for k, c in enumerate(grid):
        rows += [source_row(c.name, [T, T], f"{k}-oui-{i}", grid=grid) for i in range(3)]
        rows += [source_row(c.name, [F, F, T], f"{k}-non-{i}", grid=grid) for i in range(3)]
        rows.append(source_row(c.name, [T, F], f"{k}-egal", grid=grid))
    return grid, write_source(tmp_path / "src.jsonl", rows)


def run_main(tmp_path, source, *extra):
    return main(["--source", str(source), "--out", str(tmp_path / "sample.jsonl"),
                 "--reference", str(tmp_path / "reference.json"),
                 "--grid", str(REPO_GRID), "--selection", str(REPO_SELECTION), *extra])


def test_main_ecrit_l_echantillon_et_la_reference(tmp_path, capsys):
    grid, source = real_source(tmp_path)
    assert run_main(tmp_path, source, "--per-label", "2") == 0
    sample = [json.loads(line) for line in (tmp_path / "sample.jsonl").read_text().splitlines()]
    assert len(sample) == 8 * 4
    assert {r["criterion"] for r in sample} == {c.name for c in grid}
    reference = json.loads((tmp_path / "reference.json").read_text())
    assert reference["seed"] == 0 and reference["per_label"] == 2 and reference["source"] == "src.jsonl"
    # Par critère : 3 lignes [T, T] (1 paire d'accord), 3 [F, F, T] (1 sur 3), 1 [T, F] (0 sur 1).
    assert reference["physicians"][grid[0].name] == {
        "yes": 3, "no": 3, "tie": 1, "agreeing_pairs": 6, "pairs": 13, "agreement": 6 / 13}
    out = capsys.readouterr().out
    assert "32 lignes, 32 réponses différentes (= runs par juge)" in out


def test_main_n_ecrit_jamais_par_dessus_un_echantillon(tmp_path, capsys):
    _, source = real_source(tmp_path)
    (tmp_path / "reference.json").write_text("ancien\n")
    assert run_main(tmp_path, source) == 2
    assert "REFUS" in capsys.readouterr().err
    assert not (tmp_path / "sample.jsonl").exists()
    assert (tmp_path / "reference.json").read_text() == "ancien\n"
