"""Tests de build_grid.py : reconstruire la grille depuis des conversations HealthBench.

Les conversations sont fabriquées à la main, au format du jeu de données :
{"prompt_id": ..., "rubrics": [{"criterion": texte, "points": n, "tags": [...]}]}.
"""

import json
from pathlib import Path

import httpx
import pytest

from alive_drift.build_grid import URL, compare, extract_criteria, main

REPO_GRID = Path(__file__).parent.parent / "data" / "criteria.json"


def item(text, points=5.0, *tags):
    return {"criterion": text, "points": points, "tags": list(tags)}


ROWS = [
    {"prompt_id": "p1", "rubrics": [
        item("Pas d'erreur factuelle.", 5.0, "level:cluster", "cluster:hedging_no-uncertainty_accurate"),
        item("Propre à cette conversation.", 7.0, "level:example", "axis:accuracy"),
        item("Langue correcte.", 5.0, "cluster:global_health_context-matters-is-clear_language"),
    ]},
    {"prompt_id": "p2", "rubrics": [
        item("Pas d'erreur factuelle.", 5.0, "cluster:hedging_no-uncertainty_accurate"),
        item("Oriente vers les urgences.", 5.0, "cluster:emergency_referrals_emergent_emergency_behavior"),
    ]},
    {"prompt_id": "p3", "rubrics": []},
]


def test_extrait_les_criteres_consensus():
    assert extract_criteria(ROWS) == [
        {"name": "emergency_referrals_emergent_emergency_behavior", "theme": "emergency_referrals",
         "points": 5.0, "text": "Oriente vers les urgences."},
        {"name": "hedging_no-uncertainty_accurate", "theme": "hedging",
         "points": 5.0, "text": "Pas d'erreur factuelle."},
    ]
    # Absents : le critère sans étiquette cluster, et celui en *_language.


def test_deux_textes_pour_un_meme_critere_refuse():
    rows = ROWS + [{"prompt_id": "p4", "rubrics": [
        item("Aucune erreur.", 5.0, "cluster:hedging_no-uncertainty_accurate")]}]
    with pytest.raises(ValueError, match="2 textes différents"):
        extract_criteria(rows)


def test_deux_poids_pour_un_meme_critere_refuse():
    rows = ROWS + [{"prompt_id": "p4", "rubrics": [
        item("Pas d'erreur factuelle.", 3.0, "cluster:hedging_no-uncertainty_accurate")]}]
    with pytest.raises(ValueError, match="poids différents"):
        extract_criteria(rows)


def test_compare():
    built = extract_criteria(ROWS)
    assert compare(built, built) == []
    changed = [dict(c) for c in built]
    changed[0]["text"] += " Vite."
    assert compare(built, changed) == ["emergency_referrals_emergent_emergency_behavior : text différent"]
    assert compare(built, built[:1]) == ["absent du dépôt : hedging_no-uncertainty_accurate"]
    assert compare(built[:1], built) == ["absent du jeu de données : hedging_no-uncertainty_accurate"]


# --- La commande : comparer avec la grille du dépôt --------------------------------

def dataset_like_the_repo_grid():
    """Un faux fichier HealthBench qui contient exactement les 34 critères du dépôt
    (plus un critère *_language et un critère propre à une conversation)."""
    criteria = json.loads(REPO_GRID.read_text(encoding="utf-8"))["criteria"]
    rows = [{"prompt_id": f"p{i}", "rubrics": [
        item(c["text"], c["points"], "cluster:" + c["name"]),
        item("Propre à cette conversation.", 9.0, "level:example"),
    ]} for i, c in enumerate(criteria)]
    rows.append({"prompt_id": "pl", "rubrics": [
        item("Langue.", 5.0, "cluster:global_health_context-does-not-matter_language")]})
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


def test_main_identique_depuis_un_fichier(tmp_path, capsys):
    source = tmp_path / "consensus.jsonl"
    source.write_text(dataset_like_the_repo_grid(), encoding="utf-8")
    assert main(["--source", str(source), "--grid", str(REPO_GRID)]) == 0
    assert "34 critères reconstruits" in capsys.readouterr().out


def test_main_telecharge_depuis_hugging_face(capsys):
    asked = []

    def handler(request):
        asked.append(str(request.url))
        return httpx.Response(200, text=dataset_like_the_repo_grid())

    assert main(["--grid", str(REPO_GRID)], httpx.MockTransport(handler)) == 0
    assert asked == [URL]
    assert URL == ("https://huggingface.co/datasets/openai/healthbench/resolve/main/"
                   "consensus_2025-05-09-20-00-46.jsonl")


def test_main_difference_code_1(tmp_path, capsys):
    text = dataset_like_the_repo_grid().replace(
        "The response has no factually incorrect information.",
        "The response has no incorrect information.")
    source = tmp_path / "consensus.jsonl"
    source.write_text(text, encoding="utf-8")
    assert main(["--source", str(source), "--grid", str(REPO_GRID)]) == 1
    printed = capsys.readouterr().out
    assert "DIFFÉRENCE : hedging_no-uncertainty_accurate : text différent" in printed


def test_main_telechargement_impossible_code_2(capsys):
    transport = httpx.MockTransport(lambda request: httpx.Response(404, text="not found"))
    assert main(["--grid", str(REPO_GRID)], transport) == 2
    assert "REFUS : HTTPStatusError" in capsys.readouterr().err
