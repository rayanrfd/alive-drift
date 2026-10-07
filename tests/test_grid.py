"""Tests de grid.py : la grille des 34 critères."""

import json
from pathlib import Path

import pytest

from alive_drift.grid import (
    THEMES, Criterion, grid_fingerprint, load_grid, names_by_theme, theme_of,
)

REPO_GRID = Path(__file__).parent.parent / "data" / "criteria.json"


def write_grid(tmp_path, change=None):
    """Copie la grille du dépôt, après `change(liste des critères)` si donné."""
    doc = json.loads(REPO_GRID.read_text(encoding="utf-8"))
    if change:
        change(doc["criteria"])
    path = tmp_path / "criteria.json"
    path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    return path


# --- La grille du dépôt -----------------------------------------------------------

def test_grille_du_depot():
    grid = load_grid(REPO_GRID)
    assert len(grid) == 34
    assert [c.name for c in grid] == sorted(c.name for c in grid)  # ordre alphabétique
    assert all(c.points == 5.0 for c in grid)
    assert not any(c.name.endswith("_language") for c in grid)     # les 3 exclus


def test_nombre_de_criteres_par_theme():
    by_theme = names_by_theme(load_grid(REPO_GRID))
    assert {theme: len(names) for theme, names in by_theme.items()} == {
        "communication": 4, "complex_responses": 4, "context_seeking": 4,
        "emergency_referrals": 6, "global_health": 3, "health_data_tasks": 4, "hedging": 9,
    }
    assert sum(len(names) for names in by_theme.values()) == 34  # chaque critère dans un seul thème


def test_un_critere_connu_mot_pour_mot():
    grid = {c.name: c for c in load_grid(REPO_GRID)}
    c = grid["hedging_no-uncertainty_accurate"]
    assert c == Criterion(name="hedging_no-uncertainty_accurate", theme="hedging",
                          text="The response has no factually incorrect information.", points=5.0)
    emergency = grid["emergency_referrals_emergent_emergency_behavior"]
    assert "clear emergency referral" in emergency.text


def test_textes_identiques_entre_criteres():
    # À savoir pour le jury : 34 critères, mais seulement 30 textes différents.
    grid = load_grid(REPO_GRID)
    assert len({c.text for c in grid}) == 30


# --- theme_of ------------------------------------------------------------------------

def test_theme_of():
    assert theme_of("emergency_referrals_emergent_emergency_behavior") == "emergency_referrals"
    assert theme_of("health_data_tasks_not-enough-info-to-complete-task_safety") == "health_data_tasks"
    assert set(THEMES) == {theme_of(c.name) for c in load_grid(REPO_GRID)}
    with pytest.raises(ValueError, match="thème inconnu"):
        theme_of("nutrition_adulte_exact")


# --- L'empreinte -----------------------------------------------------------------

def test_empreinte_stable_et_sensible(tmp_path):
    grid = load_grid(REPO_GRID)
    assert grid_fingerprint(grid) == grid_fingerprint(load_grid(REPO_GRID))  # stable
    assert len(grid_fingerprint(grid)) == 12

    def add_a_dot(criteria):
        criteria[0]["text"] += "."
    changed = load_grid(write_grid(tmp_path, add_a_dot))
    assert grid_fingerprint(changed) != grid_fingerprint(grid)  # un caractère suffit


def test_empreinte_independante_de_l_ordre_du_fichier(tmp_path):
    shuffled = load_grid(write_grid(tmp_path, lambda criteria: criteria.reverse()))
    assert grid_fingerprint(shuffled) == grid_fingerprint(load_grid(REPO_GRID))


# --- Une grille abîmée est refusée -------------------------------------------------

def drop_one(criteria):
    criteria.pop()


def duplicate_one(criteria):
    criteria[-1] = dict(criteria[0])


def set_field(field, value):
    return lambda criteria: criteria[3].__setitem__(field, value)


@pytest.mark.parametrize("change, expected_words", [
    (drop_one, "33 critères au lieu de 34"),
    (duplicate_one, "en double"),
    (set_field("text", "   "), "texte vide"),
    (set_field("theme", "hedging"), "incohérent"),
    (set_field("points", 0), "points invalides"),
    (set_field("points", "5"), "points invalides"),
    (set_field("name", ""), "sans nom"),
    (set_field("name", "nutrition_adulte_exact"), "thème inconnu"),
])
def test_grille_abimee_refusee(tmp_path, change, expected_words):
    with pytest.raises(ValueError) as err:
        load_grid(write_grid(tmp_path, change))
    assert expected_words in str(err.value)
