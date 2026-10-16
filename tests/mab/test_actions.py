"""Tests de mab/actions.py : chaque cas est une réponse possible du modèle.

Les bizarreries de MedAgentBench sont reproduites exprès (comparabilité avec
l'article) ; chacune a son test, pour qu'elle reste visible.
"""

import pytest

from alive_drift.mab.actions import (FINISH, GET, INVALID, POST, parse_reply,
                                     reply_after_get, reply_after_get_error, reply_after_post)

BASE = "http://localhost:8080/fhir/"


# --- Les quatre cas ------------------------------------------------------------------

def test_get():
    a = parse_reply(f"GET {BASE}Observation?patient=S1&code=MG")
    assert (a.kind, a.url) == (GET, f"{BASE}Observation?patient=S1&code=MG&_format=json")


def test_post_lisible():
    raw = f'POST {BASE}Observation\n{{"resourceType": "Observation", "valueString": "118/77 mmHg"}}'
    a = parse_reply(raw)
    assert a.kind == POST and a.url == f"{BASE}Observation"
    assert a.payload == {"resourceType": "Observation", "valueString": "118/77 mmHg"}
    assert a.payload_error is None and a.raw == raw


def test_post_illisible_reste_un_post():
    a = parse_reply(f"POST {BASE}Observation\n{{resourceType: Observation}}")
    assert a.kind == POST and a.payload is None and a.payload_error


def test_post_sans_json():
    a = parse_reply(f"POST {BASE}Observation")
    assert a.kind == POST and a.payload_error


def test_finish():
    assert parse_reply("FINISH([1.3])").answer == "[1.3]"
    assert parse_reply('FINISH(["S6534835"])').answer == '["S6534835"]'
    assert parse_reply("FINISH([])").answer == "[]"


@pytest.mark.parametrize("raw", [
    "The magnesium level is 1.3 mg/dL.",
    "",
    "   ",
    "Voici ma réponse : FINISH([1.3])",       # ne COMMENCE pas par FINISH(
    "I will POST the order now.",              # le mot POST au milieu ne suffit pas
])
def test_invalide(raw):
    assert parse_reply(raw).kind == INVALID


# --- Nettoyage -------------------------------------------------------------------------

def test_espaces_et_balises_retires_mais_texte_brut_garde():
    raw = "```\nGET http://h/fhir/Patient?name=X\n```\n"
    a = parse_reply(raw)
    assert a.kind == GET and a.url == "http://h/fhir/Patient?name=X&_format=json"
    assert a.raw == raw                         # le correcteur relit le texte BRUT


def test_balise_tool_code_de_gemini():
    a = parse_reply("```tool_code\nFINISH([5])\n```")
    assert (a.kind, a.answer) == (FINISH, "[5]")


# --- Bizarreries reproduites exprès ---------------------------------------------------------

@pytest.mark.parametrize("raw", ["get http://h/fhir/Patient?x=1", "Finish([1])", "finish([1])", "Post http://h"])
def test_minuscules_invalides(raw):
    assert parse_reply(raw).kind == INVALID


def test_finish_coupe_le_dernier_caractere_quel_qu_il_soit():
    assert parse_reply("FINISH([1.3]) voilà").answer == "[1.3]) voil"


def test_get_sans_point_d_interrogation_garde_le_et():
    assert parse_reply("GET http://h/fhir/Patient").url == "http://h/fhir/Patient&_format=json"


def test_get_colle():
    assert parse_reply("GEThttp://h/fhir/Patient?x=1").url == "http://h/fhir/Patient?x=1&_format=json"


# --- Phrases renvoyées au modèle (mot pour mot, comme MedAgentBench) ----------------------

def test_phrase_apres_get():
    assert reply_after_get('{"total": 0}') == (
        'Here is the response from the GET request:\n{"total": 0}. Please call FINISH if you '
        "have got answers for all the questions and finished all the requested tasks")


def test_phrase_apres_get_insere_le_resultat_tel_quel():
    # un dict est écrit comme Python l'affiche (f-string), comme chez eux
    assert "\n{'total': 0}. Please" in reply_after_get({"total": 0})


def test_phrase_apres_get_en_erreur():
    assert reply_after_get_error("404 Client Error") == "Error in sending the GET request: 404 Client Error"


def test_phrases_apres_post():
    ok = parse_reply('POST http://h/fhir/Observation\n{"a": 1}')
    assert reply_after_post(ok) == (
        "POST request accepted and executed successfully. Please call FINISH if you have got "
        "answers for all the questions and finished all the requested tasks")
    assert reply_after_post(parse_reply("POST http://h/fhir/Observation\nnot json")) == "Invalid POST request"


def test_phrase_apres_post_refusee_pour_une_autre_action():
    with pytest.raises(ValueError, match="'get'"):
        reply_after_post(parse_reply("GET http://h?x=1"))
