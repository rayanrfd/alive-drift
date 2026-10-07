"""Tests de judge.py : le prompt du juge et la lecture de sa réponse. Sans réseau."""

import json
from pathlib import Path

import pytest

from alive_drift.grid import Criterion, load_grid
from alive_drift.judge import (
    JudgeAnswerError, Verdict, extract_json, main, parse_verdicts, system_prompt, user_message,
)

REPO_GRID = Path(__file__).parent.parent / "data" / "criteria.json"

# Une petite grille de 3 critères, pour des tests lisibles.
SMALL = [Criterion("communication_a", "communication", "Texte du critère A.", 5.0),
         Criterion("hedging_b", "hedging", "Texte du critère B.", 5.0),
         Criterion("hedging_c", "hedging", "Texte du critère C.", 5.0)]


def answer(*items):
    """La réponse d'un juge : un JSON {"results": [...]}."""
    return json.dumps({"results": list(items)})


def item(number, met, explanation="parce que"):
    return {"criterion_id": number, "explanation": explanation, "criteria_met": met}


GOOD = answer(item(1, True, "clair"), item(2, False, "trop affirmatif"), item(3, True))


# --- Le prompt système --------------------------------------------------------------

def test_prompt_systeme_petite_grille_mot_pour_mot():
    prompt = system_prompt(SMALL)
    assert prompt.endswith(
        '# Rubric items\n'
        '<rubric_item id="1">\nTexte du critère A.\n</rubric_item>\n'
        '<rubric_item id="2">\nTexte du critère B.\n</rubric_item>\n'
        '<rubric_item id="3">\nTexte du critère C.\n</rubric_item>')
    assert "from id 1 to id 3" in prompt
    assert '"criterion_id"' in prompt and '"criteria_met"' in prompt and '"explanation"' in prompt


def test_prompt_systeme_grille_du_depot():
    grid = load_grid(REPO_GRID)
    prompt = system_prompt(grid)
    assert "from id 1 to id 34" in prompt
    for number, criterion in enumerate(grid, start=1):
        # chaque texte est là, entier, sous son numéro
        assert f'<rubric_item id="{number}">\n{criterion.text.strip()}\n</rubric_item>' in prompt
        # le nom du critère n'est jamais montré au juge
        assert criterion.name not in prompt
    assert prompt.count("<rubric_item id=") == 34
    assert "{" + "n}" not in prompt  # aucune variable non remplacée


def test_prompt_systeme_identique_d_un_appel_a_l_autre():
    assert system_prompt(load_grid(REPO_GRID)) == system_prompt(load_grid(REPO_GRID))


# --- Le message d'un run --------------------------------------------------------------

def test_message_mot_pour_mot():
    assert user_message("J'ai mal au ventre.", "Depuis quand ?") == (
        "<question>\nJ'ai mal au ventre.\n</question>\n\n<response>\nDepuis quand ?\n</response>")


def test_message_ne_modifie_ni_la_question_ni_la_reponse():
    question, response = "  {a}\n\nb  ", "```json\n{\"results\": []}\n```"
    message = user_message(question, response)
    assert f"<question>\n{question}\n</question>" in message
    assert f"<response>\n{response}\n</response>" in message


# --- Lire la réponse du juge ------------------------------------------------------------

def test_verdicts_ranges_par_nom_de_critere():
    assert parse_verdicts(GOOD, SMALL) == {
        "communication_a": Verdict(True, "clair"),
        "hedging_b": Verdict(False, "trop affirmatif"),
        "hedging_c": Verdict(True, "parce que"),
    }


def test_l_ordre_des_elements_ne_compte_pas():
    shuffled = answer(item(3, True), item(1, True, "clair"), item(2, False, "trop affirmatif"))
    assert parse_verdicts(shuffled, SMALL) == parse_verdicts(GOOD, SMALL)
    assert list(parse_verdicts(shuffled, SMALL)) == [c.name for c in SMALL]  # ordre de la grille


@pytest.mark.parametrize("wrapped", [
    "```json\n" + GOOD + "\n```",
    "```\n" + GOOD + "\n```",
    "Here is my evaluation:\n" + GOOD + "\nDone.",
    "\n\n  " + GOOD + "  \n",
])
def test_json_entoure_de_texte(wrapped):
    assert parse_verdicts(wrapped, SMALL) == parse_verdicts(GOOD, SMALL)


def test_explication_absente_acceptee():
    content = answer({"criterion_id": 1, "criteria_met": True}, item(2, False), item(3, True))
    assert parse_verdicts(content, SMALL)["communication_a"] == Verdict(True, "")


@pytest.mark.parametrize("content, expected_words", [
    (answer(item(1, True), item(2, False)), "critères sans verdict : [3]"),
    (answer(item(1, True), item(2, False), item(2, True), item(3, True)), "2 en double"),
    (answer(item(1, True), item(2, False), item(3, True), item(4, True)), "criterion_id inconnu : 4"),
    (answer(item(0, True), item(2, False), item(3, True)), "criterion_id inconnu : 0"),
    (answer(item("1", True), item(2, False), item(3, True)), "criterion_id inconnu : '1'"),
    (answer(item(True, True), item(2, False), item(3, True)), "criterion_id inconnu : True"),
    (answer(item(1, "true"), item(2, False), item(3, True)), "criteria_met = 'true'"),
    (answer(item(1, 1), item(2, False), item(3, True)), "criteria_met = 1"),
    (answer(item(1, None), item(2, False), item(3, True)), "criteria_met = None"),
    (answer(item(1, True), "oui", item(3, True)), "mal formé"),
    (json.dumps({"verdicts": []}), "pas de liste 'results'"),
    (json.dumps({"results": {"1": True}}), "pas de liste 'results'"),
    ("I cannot evaluate this response.", "pas d'objet JSON"),
    ("", "pas d'objet JSON"),
    (GOOD[:-20], "JSON illisible"),
    ("} rien {", "pas d'objet JSON"),
])
def test_reponse_inexploitable_refusee(content, expected_words):
    with pytest.raises(JudgeAnswerError) as err:
        parse_verdicts(content, SMALL)
    assert expected_words in str(err.value)


def test_extract_json_refuse_ce_qui_n_est_pas_un_objet():
    assert extract_json('avant {"a": 1} après') == {"a": 1}
    with pytest.raises(JudgeAnswerError):
        extract_json("[1, 2]")


def test_grille_complete_34_verdicts():
    grid = load_grid(REPO_GRID)
    content = answer(*[item(n, n % 2 == 0) for n in range(1, 35)])
    verdicts = parse_verdicts(content, grid)
    assert list(verdicts) == [c.name for c in grid]
    assert sum(v.met for v in verdicts.values()) == 17
    assert verdicts[grid[1].name].met is True and verdicts[grid[0].name].met is False


def test_33_verdicts_sur_34_refuse():
    grid = load_grid(REPO_GRID)
    content = answer(*[item(n, True) for n in range(1, 35) if n != 20])
    with pytest.raises(JudgeAnswerError) as err:
        parse_verdicts(content, grid)
    assert "[20]" in str(err.value)


# --- La commande qui affiche le prompt -----------------------------------------------------

def test_main_affiche_le_prompt(capsys):
    assert main([str(REPO_GRID)]) == 0
    assert capsys.readouterr().out == system_prompt(load_grid(REPO_GRID)) + "\n"


def test_main_grille_introuvable_code_2(tmp_path, capsys):
    assert main([str(tmp_path / "absente.json")]) == 2
    assert "REFUS" in capsys.readouterr().err
