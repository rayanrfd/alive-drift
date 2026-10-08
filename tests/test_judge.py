"""Tests de judge.py : le prompt du juge et la lecture de sa réponse. Sans réseau."""

import json
from pathlib import Path

import pytest

from alive_drift.grid import Criterion, load_grid, load_working_grid
from alive_drift.judge import (
    JudgeAnswerError, Verdict, extract_json, main, parse_verdicts, system_prompt, user_message,
)

REPO_GRID = Path(__file__).parent.parent / "data" / "criteria.json"
REPO_SELECTION = Path(__file__).parent.parent / "data" / "selection.json"

# Une petite grille de 3 critères, pour des tests lisibles.
SMALL = [Criterion("communication_a", "communication", "Texte du critère A.", 5.0),
         Criterion("hedging_b", "hedging", "Texte du critère B.", 5.0),
         Criterion("hedging_c", "hedging", "Texte du critère C.", 5.0)]


def answer(*items):
    """La réponse d'un juge : un JSON {"results": [...]}."""
    return json.dumps({"results": list(items)})


WORDS = {True: "met", False: "not_met", None: "not_applicable"}


def item(number, met, explanation="parce que"):
    """Un élément valide. met : True, False ou None (non applicable)."""
    return {"criterion_id": number, "explanation": explanation, "verdict": WORDS[met]}


def raw(number, verdict):
    """Un élément dont on choisit le champ "verdict" tel quel (pour les cas refusés)."""
    return {"criterion_id": number, "explanation": "parce que", "verdict": verdict}


GOOD = answer(item(1, True, "clair"), item(2, False, "trop affirmatif"), item(3, True))


# --- Le prompt système --------------------------------------------------------------

def test_prompt_systeme_petite_grille_mot_pour_mot():
    prompt = system_prompt(SMALL)
    assert prompt.endswith(
        '# Rubric items\n'
        '<rubric_item criterion_id="1">\nTexte du critère A.\n</rubric_item>\n'
        '<rubric_item criterion_id="2">\nTexte du critère B.\n</rubric_item>\n'
        '<rubric_item criterion_id="3">\nTexte du critère C.\n</rubric_item>')
    assert "from criterion_id 1 to criterion_id 3" in prompt
    assert '"criterion_id"' in prompt and '"verdict"' in prompt and '"explanation"' in prompt
    assert 'exactly one of "met", "not_met", "not_applicable"' in prompt


def test_consigne_non_applicable():
    prompt = system_prompt(SMALL)
    # N/A se décide d'après la question seule : A et B gardent les mêmes critères actifs.
    assert "Decide this from the question only, never from the response." in prompt
    assert 'If the question is clearly not in the situation described, the verdict is "not_applicable".' in prompt
    assert "If you cannot tell whether a rubric item applies, treat it as applicable." in prompt
    # Une mauvaise réponse n'est pas un N/A.
    assert 'A poor or incomplete response is "not_met", never "not_applicable".' in prompt
    # Les consignes du 16 et du 16 bis ont disparu.
    assert "criteria_met" not in prompt and "These sentences are context" not in prompt


def test_prompt_systeme_grille_du_depot():
    grid = load_grid(REPO_GRID)
    prompt = system_prompt(grid)
    assert "from criterion_id 1 to criterion_id 34" in prompt
    for number, criterion in enumerate(grid, start=1):
        # chaque texte est là, entier, sous son numéro
        assert f'<rubric_item criterion_id="{number}">\n{criterion.text.strip()}\n</rubric_item>' in prompt
        # le nom du critère n'est jamais montré au juge
        assert criterion.name not in prompt
    assert prompt.count("<rubric_item criterion_id=") == 34
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


def test_non_applicable_devient_none():
    content = answer(item(1, None, "la question vient d'un patient"), item(2, False), item(3, True))
    verdicts = parse_verdicts(content, SMALL)
    assert verdicts["communication_a"] == Verdict(None, "la question vient d'un patient")
    assert [v.met for v in verdicts.values()] == [None, False, True]


@pytest.mark.parametrize("word, met", [
    ("met", True), ("not_met", False), ("not_applicable", None),
    ("Met", True), ("NOT_MET", False), (" not-applicable ", None), ("Not Applicable", None),
])
def test_les_trois_verdicts_casse_et_tirets_toleres(word, met):
    content = answer(raw(1, word), item(2, False), item(3, True))
    assert parse_verdicts(content, SMALL)["communication_a"].met is met


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


def test_numero_du_critere_sous_le_nom_id():
    # Ce que DeepSeek a renvoyé à l'essai du 07/10.
    content = json.dumps({"results": [
        {"id": 1, "explanation": "clair", "verdict": "met"},
        {"id": 2, "explanation": "trop affirmatif", "verdict": "not_met"},
        {"criterion_id": 3, "id": 3, "explanation": "parce que", "verdict": "met"}]})
    assert parse_verdicts(content, SMALL) == parse_verdicts(GOOD, SMALL)


def test_explication_absente_acceptee():
    content = answer({"criterion_id": 1, "verdict": "met"}, item(2, False), item(3, True))
    assert parse_verdicts(content, SMALL)["communication_a"] == Verdict(True, "")


@pytest.mark.parametrize("content, expected_words", [
    (answer(item(1, True), item(2, False)), "critères sans verdict : [3]"),
    (answer(item(1, True), item(2, False), item(2, True), item(3, True)), "2 en double"),
    (answer(item(1, True), item(2, False), item(3, True), item(4, True)), "criterion_id inconnu : 4"),
    (answer(item(0, True), item(2, False), item(3, True)), "criterion_id inconnu : 0"),
    (answer(item("1", True), item(2, False), item(3, True)), "criterion_id inconnu : '1'"),
    (answer(item(True, True), item(2, False), item(3, True)), "criterion_id inconnu : True"),
    (answer(raw(1, "true"), item(2, False), item(3, True)), "verdict = 'true'"),
    (answer(raw(1, "yes"), item(2, False), item(3, True)), "verdict = 'yes'"),
    (answer(raw(1, "n/a"), item(2, False), item(3, True)), "verdict = 'n/a'"),
    (answer(raw(1, True), item(2, False), item(3, True)), "verdict = True"),
    (answer(raw(1, None), item(2, False), item(3, True)), "verdict = None"),
    # l'ancien format (criteria_met) n'est plus accepté : il n'a pas de N/A
    (answer({"criterion_id": 1, "criteria_met": True}, item(2, False), item(3, True)), "verdict = None"),
    (answer(item(1, True), "oui", item(3, True)), "mal formé"),
    (json.dumps({"results": [{"criterion_id": 1, "id": 2, "verdict": "met"},
                             {"id": 2, "verdict": "not_met"}, {"id": 3, "verdict": "met"}]}),
     "criterion_id 1 et id 2 différents"),
    (json.dumps({"results": [{"id": "1", "verdict": "met"},
                             {"id": 2, "verdict": "not_met"}, {"id": 3, "verdict": "met"}]}),
     "criterion_id inconnu : '1'"),
    (json.dumps({"verdicts": []}), "pas de liste 'results'"),
    (json.dumps({"results": {"1": True}}), "pas de liste 'results'"),
    ("I cannot evaluate this response.", "pas d'objet JSON"),
    ("", "pas d'objet JSON"),
    (GOOD[:-20], "JSON illisible"),
    ("} rien {", "pas d'objet JSON"),
    # deux "verdict" contradictoires dans le même élément
    ('{"results": [{"criterion_id": 1, "verdict": "not_met", "verdict": "met"},'
     ' {"criterion_id": 2, "verdict": "not_met"}, {"criterion_id": 3, "verdict": "met"}]}',
     "clé en double : ['verdict']"),
    ('{"results": [], "results": [{"criterion_id": 1, "verdict": "met"},'
     ' {"criterion_id": 2, "verdict": "not_met"}, {"criterion_id": 3, "verdict": "met"}]}',
     "clé en double : ['results']"),
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

def test_prompt_de_la_grille_de_travail():
    prompt = system_prompt(load_working_grid(REPO_GRID, REPO_SELECTION))
    assert "from criterion_id 1 to criterion_id 8" in prompt
    assert prompt.count("<rubric_item criterion_id=") == 8
    assert '<rubric_item criterion_id="9">' not in prompt


def test_main_affiche_le_prompt_de_la_grille_de_travail(capsys):
    assert main(["--grid", str(REPO_GRID), "--selection", str(REPO_SELECTION)]) == 0
    expected = system_prompt(load_working_grid(REPO_GRID, REPO_SELECTION))
    assert capsys.readouterr().out == expected + "\n"


@pytest.mark.parametrize("missing", ["--grid", "--selection"])
def test_main_fichier_introuvable_code_2(tmp_path, capsys, missing):
    args = {"--grid": str(REPO_GRID), "--selection": str(REPO_SELECTION)}
    args[missing] = str(tmp_path / "absent.json")
    assert main([x for pair in args.items() for x in pair]) == 2
    assert "REFUS" in capsys.readouterr().err
