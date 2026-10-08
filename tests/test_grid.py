"""Tests de grid.py : la grille des 34 critères."""

import json
from pathlib import Path

import pytest

from alive_drift.grid import (
    THEMES, Criterion, grid_fingerprint, load_grid, load_selection, load_working_grid,
    names_by_theme, select_criteria, theme_of,
)

REPO_GRID = Path(__file__).parent.parent / "data" / "criteria.json"
REPO_SELECTION = Path(__file__).parent.parent / "data" / "selection.json"


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

def test_empreinte_de_la_grille_verifiee_contre_healthbench():
    # Empreinte relevée le 07/10, le jour où build_grid a confirmé que la grille
    # est identique au jeu HealthBench. Si ce test casse, un des 34 textes,
    # noms ou poids a changé : ne pas corriger le test, relancer build_grid.
    assert grid_fingerprint(load_grid(REPO_GRID)) == "2e3eea4db18f"


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

def test_cle_en_double_dans_un_critere_refusee(tmp_path):
    text = REPO_GRID.read_text(encoding="utf-8")
    path = tmp_path / "criteria.json"
    path.write_text(text.replace('"points": 5.0,', '"points": 5.0, "points": 7.0,', 1), encoding="utf-8")
    with pytest.raises(ValueError, match="clé en double"):
        load_grid(path)


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
    (set_field("points", float("nan")), "points invalides"),
    (set_field("points", float("inf")), "points invalides"),
    (set_field("name", ""), "sans nom"),
    (set_field("name", "nutrition_adulte_exact"), "thème inconnu"),
])
def test_grille_abimee_refusee(tmp_path, change, expected_words):
    with pytest.raises(ValueError) as err:
        load_grid(write_grid(tmp_path, change))
    assert expected_words in str(err.value)


# --- La grille de travail : la sélection du 08/10 ---------------------------------

# Décision du 08/10 : grille moyenne, sans doublon, un seul critère par paire contradictoire.
KEPT = [
    "communication_not-health-professional_accuracy_completeness",
    "communication_not-health-professional_tailored",
    "complex_responses_detailed_appropriate",
    "context_seeking_not-enough-context_context_seeking",
    "emergency_referrals_conditionally-emergent_emergency_behavior",
    "emergency_referrals_emergent_emergency_behavior",
    "health_data_tasks_enough-info-to-complete-task_response_instruction_following",
    "hedging_any-reducible-uncertainty_accurate",
]

# Les paires dont les deux critères exigent des comportements opposés.
CONTRADICTORY_PAIRS = [
    ("communication_health-professional_tailored", "communication_not-health-professional_tailored"),
    ("complex_responses_detailed_appropriate", "complex_responses_simple_appropriate"),
    ("context_seeking_enough-context_precise", "context_seeking_not-enough-context_context_seeking"),
    ("emergency_referrals_emergent_emergency_behavior", "emergency_referrals_non-emergent_emergency_behavior"),
    ("hedging_any-reducible-uncertainty_seeks_context", "hedging_no-uncertainty_seeks_context"),
    ("hedging_no-uncertainty_hedges", "hedging_only-irreducible-uncertainty_hedges"),
]


def test_grille_de_travail_du_depot():
    grid = load_working_grid(REPO_GRID, REPO_SELECTION)
    assert [c.name for c in grid] == KEPT
    assert len({c.text for c in grid}) == len(grid)  # aucun doublon de texte
    full = {c.name: c for c in load_grid(REPO_GRID)}
    assert all(full[c.name] == c for c in grid)  # les textes sont ceux de HealthBench, inchangés
    assert grid_fingerprint(grid) != grid_fingerprint(load_grid(REPO_GRID))


def test_aucune_paire_contradictoire_entiere():
    names = {c.name for c in load_working_grid(REPO_GRID, REPO_SELECTION)}
    for pair in CONTRADICTORY_PAIRS:
        assert not set(pair) <= names, pair


def test_select_criteria_garde_l_ordre_de_la_grille():
    grid = load_grid(REPO_GRID)
    assert select_criteria(grid, [grid[5].name, grid[1].name]) == [grid[1], grid[5]]


def test_select_criteria_refuse_un_nom_inconnu_ou_une_liste_vide():
    grid = load_grid(REPO_GRID)
    with pytest.raises(ValueError, match="inconnus dans la sélection : \\['hedging_inexistant'\\]"):
        select_criteria(grid, [grid[0].name, "hedging_inexistant"])
    with pytest.raises(ValueError, match="sélection vide"):
        select_criteria(grid, [])


@pytest.mark.parametrize("content, expected_words", [
    ({"keep": []}, "liste non vide"),
    ({"keep": "hedging_any-reducible-uncertainty_accurate"}, "liste non vide"),
    ({"keep": ["hedging_any-reducible-uncertainty_accurate", 3]}, "liste non vide"),
    ({"keep": ["hedging_any-reducible-uncertainty_accurate", ""]}, "liste non vide"),
    ({"garde": ["hedging_any-reducible-uncertainty_accurate"]}, "liste non vide"),
    ({"keep": ["hedging_any-reducible-uncertainty_accurate"] * 2}, "en double"),
    (["hedging_any-reducible-uncertainty_accurate"], "doit être un objet"),
])
def test_selection_abimee_refusee(tmp_path, content, expected_words):
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(content), encoding="utf-8")
    with pytest.raises(ValueError) as err:
        load_selection(path)
    assert expected_words in str(err.value)


def test_selection_avec_un_nom_inconnu_refusee(tmp_path):
    path = tmp_path / "selection.json"
    path.write_text(json.dumps({"keep": KEPT[:3] + ["emergency_referrals_emergnt_emergency_behavior"]}),
                    encoding="utf-8")
    with pytest.raises(ValueError, match="inconnus"):
        load_working_grid(REPO_GRID, path)
