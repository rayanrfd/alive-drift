"""Tests de check_run.

Point de départ : la VRAIE sortie d'Alive du 01/10 (curl de Rayan), copiée
telle quelle (seul le texte de la réponse est raccourci). Chaque test casse
UN champ et vérifie que la réponse est rejetée pour la bonne raison.
"""

import copy

import pytest

from alive_drift.check import RunRejected, check_run

MODEL = "mistralai/mistral-small-24b-instruct-2501"

REAL_RUN = {
    "id": "mg-sBZ3LMug33mOUX7O",
    "parent_id": "mg-zc03JpgBY8wQAHBx",
    "execution_id": "ex-WxDjC6ClZzye4T0a",
    "role": "agent",
    "status": "completed",
    "content": "A normal resting heart rate can vary slightly depending on ...",
    "credits_used": 0.002968,
    "started_at": 1790876500954,
    "completed_at": 1790876506130,
    "duration_ms": 5176,
    "tokens": {"prompt": 33, "completion": 181, "reasoning": 0},
    "model": {
        "requested": "mistralai/mistral-small-24b-instruct-2501",
        "served": [
            {
                "model": "mistralai/mistral-small-24b-instruct-2501",
                "provider": "DeepInfra",
                "calls": 1,
                "substituted": False,
            }
        ],
        "substituted": False,
        "fallback": False,
        "reasoning_effort": "none",
    },
}


def run_copy():
    """Copie profonde : un test qui modifie sa copie ne touche pas REAL_RUN."""
    return copy.deepcopy(REAL_RUN)


def assert_rejected(run, expected_words):
    """Vérifie le rejet ET que le message cite la bonne raison."""
    with pytest.raises(RunRejected) as err:
        check_run(run, MODEL)
    assert expected_words in str(err.value)


# --- La vraie réponse passe ------------------------------------------------

def test_vraie_reponse_acceptee():
    check_run(run_copy(), MODEL)  # ne doit rien lever


# --- 1. status et contenu ----------------------------------------------------

def test_status_failed():
    run = run_copy()
    run["status"] = "failed"
    assert_rejected(run, "status")


def test_contenu_vide():
    run = run_copy()
    run["content"] = "   "
    assert_rejected(run, "réponse vide")


def test_contenu_absent():
    run = run_copy()
    del run["content"]
    assert_rejected(run, "réponse vide")


# --- 2. bloc model -----------------------------------------------------------

def test_bloc_model_absent():
    run = run_copy()
    del run["model"]
    assert_rejected(run, "pas de bloc 'model'")


# --- 3. modèle demandé -------------------------------------------------------

def test_mauvais_modele_attendu():
    # La réponse est correcte, mais on attendait GPT-4.1 mini : rejet.
    with pytest.raises(RunRejected, match="modèle demandé"):
        check_run(run_copy(), "openai/gpt-4.1-mini")


# --- 4. fallback et substitution --------------------------------------------

def test_fallback_vrai():
    run = run_copy()
    run["model"]["fallback"] = True
    assert_rejected(run, "fallback")


def test_fallback_absent():
    run = run_copy()
    del run["model"]["fallback"]
    assert_rejected(run, "fallback")


def test_substitution():
    run = run_copy()
    run["model"]["substituted"] = True
    assert_rejected(run, "substituted")


# --- 5. served -----------------------------------------------------------------

def test_servi_par_gemini():
    # Le cas vu le 30/09 : mistral demandé, Gemini a répondu.
    run = run_copy()
    run["model"]["served"][0]["model"] = "google/gemini-3.1-flash-lite"
    assert_rejected(run, "servi par")


def test_deux_modeles_servis():
    run = run_copy()
    second = {"model": "google/gemini-3.1-flash-lite", "provider": "Google",
              "calls": 1, "substituted": True}
    run["model"]["served"].append(second)
    assert_rejected(run, "1 entrée")


def test_aucun_modele_servi():
    run = run_copy()
    run["model"]["served"] = []
    assert_rejected(run, "1 entrée")


def test_served_substitue():
    run = run_copy()
    run["model"]["served"][0]["substituted"] = True
    assert_rejected(run, "served[0].substituted")


def test_deux_appels():
    run = run_copy()
    run["model"]["served"][0]["calls"] = 2
    assert_rejected(run, "appels")


# --- 6. raisonnement ----------------------------------------------------------

def test_raisonnement_high():
    run = run_copy()
    run["model"]["reasoning_effort"] = "high"
    assert_rejected(run, "reasoning_effort")


def test_tokens_de_raisonnement_alors_que_coupe():
    # L'étiquette dit "none" mais le compteur dit que le modèle a raisonné.
    run = run_copy()
    run["tokens"]["reasoning"] = 500
    assert_rejected(run, "500 tokens de raisonnement")


def test_compteur_de_raisonnement_absent():
    run = run_copy()
    del run["tokens"]
    assert_rejected(run, "tokens de raisonnement")


def test_effort_high_voulu_et_servi():
    # Si la config demande "high", l'étiquette doit dire "high" et le compteur est libre.
    run = run_copy()
    run["model"]["reasoning_effort"] = "high"
    run["tokens"]["reasoning"] = 500
    check_run(run, MODEL, "high")


def test_effort_high_voulu_mais_none_servi():
    with pytest.raises(RunRejected, match="reasoning_effort 'none', attendu 'high'"):
        check_run(run_copy(), MODEL, "high")


# --- 7. execution_id et provider présents ------------------------------------

def test_execution_id_absent():
    run = run_copy()
    del run["execution_id"]
    assert_rejected(run, "execution_id absent")


def test_provider_absent():
    run = run_copy()
    del run["model"]["served"][0]["provider"]
    assert_rejected(run, "provider absent")


# --- 0. Réponse qui n'a pas la forme attendue : rejet, pas de plantage ---------

@pytest.mark.parametrize("bad", [None, [], "du texte", 42])
def test_reponse_pas_un_objet(bad):
    assert_rejected(bad, "réponse mal formée")


def test_served_pas_un_objet():
    run = run_copy()
    run["model"]["served"] = ["mistral"]
    assert_rejected(run, "served[0] mal formé")


# --- Le provider n'est PAS un motif de rejet ---------------------------------

def test_provider_venice_accepte():
    # Venice et DeepInfra alternent : on enregistre le provider, on ne rejette pas.
    run = run_copy()
    run["model"]["served"][0]["provider"] = "Venice"
    check_run(run, MODEL)
