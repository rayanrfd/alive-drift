"""Tests de generate.py, sans réseau.

Le faux serveur répond selon la QUESTION reçue : `plan` associe un texte de
question à la liste de ses réponses successives (une par appel). Une question
absente du plan reçoit la vraie sortie du 01/10 (code 200).
Étapes possibles : 200, "gemini" (200 mais servi par Gemini), "html" (200 mais
le contenu n'est pas du JSON), "same" (200 avec toujours le même execution_id),
ou un code d'erreur.

Comme le vrai Alive, le faux serveur donne un execution_id différent à chaque
run (unique_run), sauf pour l'étape "same".
"""

import copy
import itertools
import json
import re
import time
from pathlib import Path

import httpx
import pytest

from alive_drift.client import make_client
from alive_drift.config import Config
from alive_drift.files import errors_path_for, load_questions
from alive_drift.generate import generate, main
from alive_drift.recorder import CampaignStopped
from test_check import MODEL, REAL_RUN

TOKEN = "jeton-de-test"
ROOM = "th-Z7jWvllIYv1Eoa9z"
DEPLOYMENT = "dp-E8dfWyMHc4Q7oXnN"

GEMINI_RUN = copy.deepcopy(REAL_RUN)
GEMINI_RUN["model"]["served"][0]["model"] = "google/gemini-3.1-flash-lite"


def make_config(name="A", deployment=DEPLOYMENT, model=MODEL, effort="none", room=ROOM):
    return Config(name=name, room=room, deployment=deployment, model=model,
                  reasoning_effort=effort)


_run_numbers = itertools.count(1)


def unique_run(body=REAL_RUN):
    """Copie de `body` avec un execution_id neuf : ex-test-1, ex-test-2..."""
    run = copy.deepcopy(body)
    run["execution_id"] = f"ex-test-{next(_run_numbers)}"
    return run


def make_questions(n):
    return [{"id": str(i), "text": f"question {i}"} for i in range(n)]


def fake_alive(plan=None, on_request=None):
    """Renvoie (transport, inputs) ; `inputs` liste les questions reçues, dans l'ordre."""
    plan = {text: list(steps) for text, steps in (plan or {}).items()}
    inputs = []

    def handler(request):
        text = json.loads(request.content)["input"]
        inputs.append(text)
        if on_request:
            on_request(text)
        steps = plan.get(text)
        step = steps.pop(0) if steps else 200
        if step == 200:
            return httpx.Response(200, json=unique_run())
        if step == "gemini":
            return httpx.Response(200, json=unique_run(GEMINI_RUN))
        if step == "same":
            return httpx.Response(200, json=REAL_RUN)  # execution_id fixe
        if step == "html":
            return httpx.Response(200, content=b"<html>page d'erreur</html>")
        return httpx.Response(step, json={"detail": f"erreur {step}"})

    return httpx.MockTransport(handler), inputs


def run_generate(questions, out_path, plan=None, on_request=None, *,
                 config=None, limit=None, workers=1, sleep=lambda s: None):
    """Lance generate (config A par défaut). Renvoie (résultat ou exception, questions reçues).

    Par défaut les attentes entre essais sont supprimées. `sleep=None` garde la
    vraie attente de generate (celle qui s'interrompt à l'arrêt).
    """
    transport, inputs = fake_alive(plan, on_request)
    with make_client(TOKEN, transport) as client:
        try:
            result = generate(client, questions, config or make_config(),
                              out_path=out_path, limit=limit, workers=workers,
                              sleep=sleep)
        except Exception as err:
            result = err
    return result, inputs


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# --- load_questions ---------------------------------------------------------------

def write(path, lines):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_load_questions(tmp_path):
    path = write(tmp_path / "q.jsonl", ['{"id": "0", "text": "première"}', "",
                                        '{"id": "1", "text": "deuxième"}'])
    assert load_questions(path) == [{"id": "0", "text": "première"},
                                    {"id": "1", "text": "deuxième"}]


def test_load_questions_id_entier_devient_texte(tmp_path):
    # Format réel de data/healthbench.jsonl : {"id": 0, "text": "..."}
    path = write(tmp_path / "q.jsonl", ['{"id": 0, "text": "a"}', '{"id": 12, "text": "b"}'])
    assert [q["id"] for q in load_questions(path)] == ["0", "12"]


@pytest.mark.parametrize("lines, expected_words", [
    (['{"id": 0, "text": "a"}', '{"id": "0", "text": "b"}'], "en double"),
    (['{"id": true, "text": "a"}'], "'id' manquant"),
    (['{"id": "", "text": "a"}'], "'id' manquant"),

    (['{"id": "0", "text": "a"}', '{"id": "0", "text": "b"}'], "en double"),
    (['{"id": "0", "text": "  "}'], "'text' vide"),
    (['{"id": "0"}'], "'text' vide"),
    (['{"text": "a"}'], "'id' manquant"),
    (['[1, 2]'], "doit être un objet"),
    ([""], "aucune question"),
])
def test_load_questions_refuse(tmp_path, lines, expected_words):
    path = write(tmp_path / "q.jsonl", lines)
    with pytest.raises(ValueError) as err:
        load_questions(path)
    assert expected_words in str(err.value)


def test_fichier_healthbench_du_depot():
    # Le fichier livré avec le dépôt : 1000 questions, ids uniques (sinon load lève).
    path = Path(__file__).parent.parent / "data" / "healthbench.jsonl"
    questions = load_questions(path)
    assert len(questions) == 1000
    assert questions[0]["id"] == "0"
    assert questions[-1]["id"] == "999"
    assert questions[0]["text"].startswith("which antibiotic is safe")


# --- Cas normal : toutes les réponses sont gardées -------------------------------

def test_trois_questions(tmp_path):
    out = tmp_path / "A.jsonl"
    result, inputs = run_generate(make_questions(3), out)

    assert result == (3, 0)
    assert inputs == ["question 0", "question 1", "question 2"]  # une par run, dans l'ordre
    records = read_jsonl(out)
    assert [r["question_id"] for r in records] == ["0", "1", "2"]
    record = records[1]
    assert record.pop("execution_id").startswith("ex-test-")  # celui du faux serveur
    # L'heure d'écriture, en UTC : 2026-10-06T09:24:31+00:00
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\+00:00", record.pop("written_at"))
    assert record == {
        "question_id": "1",
        "config": "A",
        "room_id": "th-Z7jWvllIYv1Eoa9z",
        "deployment_id": "dp-E8dfWyMHc4Q7oXnN",
        "reasoning_effort": "none",
        "question": "question 1",
        "response": REAL_RUN["content"],
        "model": "mistralai/mistral-small-24b-instruct-2501",
        "provider": "DeepInfra",
        "prompt_tokens": 33,
        "completion_tokens": 181,
        "reasoning_tokens": 0,
        "credits_used": 0.002968,
        "duration_ms": 5176,
    }
    assert not errors_path_for(out).exists()  # aucun échec : pas de fichier d'erreurs


def test_reponse_ecrite_avant_la_question_suivante(tmp_path):
    # Quand le serveur reçoit la question i, les i réponses précédentes sont déjà sur disque.
    out = tmp_path / "A.jsonl"
    lines_seen = []
    run_generate(make_questions(3), out, on_request=lambda text: lines_seen.append(len(read_jsonl(out))))
    assert lines_seen == [0, 1, 2]


def test_accents_non_echappes(tmp_path):
    out = tmp_path / "A.jsonl"
    run_generate([{"id": "0", "text": "fièvre élevée"}], out)
    assert "fièvre élevée" in out.read_text(encoding="utf-8")


# --- Échecs : notés à part, la campagne continue --------------------------------

def test_run_rejete_note_et_on_continue(tmp_path):
    out = tmp_path / "A.jsonl"
    result, inputs = run_generate(make_questions(3), out, plan={"question 1": ["gemini"]})

    assert result == (2, 1)
    assert inputs == ["question 0", "question 1", "question 2"]  # pas de réessai du rejet
    assert [r["question_id"] for r in read_jsonl(out)] == ["0", "2"]
    errors = read_jsonl(errors_path_for(out))
    assert len(errors) == 1
    assert errors[0]["question_id"] == "1"
    assert errors[0]["config"] == "A"
    assert errors[0]["deployment_id"] == DEPLOYMENT
    assert errors[0]["error_type"] == "RunRejected"
    assert "servi par 'google/gemini-3.1-flash-lite'" in errors[0]["error"]
    assert errors[0]["detail"] is None  # un rejet n'est pas une erreur HTTP : pas de message d'Alive


def test_cinq_502_notes_et_on_continue(tmp_path):
    out = tmp_path / "A.jsonl"
    result, inputs = run_generate(make_questions(3), out, plan={"question 1": [502] * 5})

    assert result == (2, 1)
    assert inputs.count("question 1") == 5  # les 5 essais ont eu lieu
    assert [r["question_id"] for r in read_jsonl(out)] == ["0", "2"]
    errors = read_jsonl(errors_path_for(out))
    assert errors[0]["question_id"] == "1"
    assert errors[0]["error_type"] == "HTTPStatusError"
    assert TOKEN not in errors[0]["error"]
    assert errors[0]["detail"] == "erreur 502"  # le message renvoyé par (le faux) Alive
    assert errors[0]["written_at"].endswith("+00:00")


def test_502_puis_succes_reponse_gardee(tmp_path):
    out = tmp_path / "A.jsonl"
    result, _ = run_generate(make_questions(2), out, plan={"question 0": [502, 200]})
    assert result == (2, 0)
    assert not errors_path_for(out).exists()


def test_200_mal_forme_note_et_on_continue(tmp_path):
    # Une réponse cassée ne doit pas arrêter une campagne de 1000 questions.
    out = tmp_path / "A.jsonl"
    result, _ = run_generate(make_questions(3), out, plan={"question 1": ["html"]})
    assert result == (2, 1)
    assert [r["question_id"] for r in read_jsonl(out)] == ["0", "2"]
    assert read_jsonl(errors_path_for(out))[0]["error_type"] == "RunRejected"


# --- Un execution_id ne sert qu'une fois ----------------------------------------

def test_execution_id_deja_vu_compte_comme_echec(tmp_path):
    # Alive rend deux fois le même run pour deux questions différentes.
    out = tmp_path / "A.jsonl"
    plan = {"question 0": ["same"], "question 1": ["same"]}
    result, _ = run_generate(make_questions(3), out, plan=plan)

    assert result == (2, 1)
    assert [r["question_id"] for r in read_jsonl(out)] == ["0", "2"]
    errors = read_jsonl(errors_path_for(out))
    assert errors[0]["question_id"] == "1"
    assert "déjà enregistré" in errors[0]["error"]


def test_execution_id_deja_dans_le_fichier_refuse_a_la_reprise(tmp_path):
    out = tmp_path / "A.jsonl"
    run_generate(make_questions(2), out, plan={"question 0": ["same"]}, limit=1)
    result, _ = run_generate(make_questions(2), out, plan={"question 1": ["same"]})

    assert result == (0, 1)
    assert [r["question_id"] for r in read_jsonl(out)] == ["0"]


# --- La config décide où partent les runs et ce qu'on attend -------------------

def test_les_runs_partent_vers_le_deploiement_de_la_config(tmp_path):
    urls = []

    def handler(request):
        urls.append(str(request.url))
        return httpx.Response(200, json=REAL_RUN)

    config = Config(name="A", room="th-ROOM", deployment="dp-DEPLOY", model=MODEL,
                    reasoning_effort="none")
    with make_client(TOKEN, httpx.MockTransport(handler)) as client:
        generate(client, make_questions(1), config, out_path=tmp_path / "A.jsonl")
    assert urls == ["https://brain-bzh.alive.dev/api/v1/threads/th-ROOM/deployments/dp-DEPLOY/runs"]
    record = read_jsonl(tmp_path / "A.jsonl")[0]
    assert record["deployment_id"] == "dp-DEPLOY"
    assert record["room_id"] == "th-ROOM"


def test_tokens_mal_formes_avec_effort_high_ne_plante_pas(tmp_path):
    # Avec un effort autre que "none", le bloc tokens n'est pas vérifié : s'il est
    # cassé, on enregistre des compteurs vides au lieu de planter.
    body = copy.deepcopy(REAL_RUN)
    body["model"]["reasoning_effort"] = "high"
    body["tokens"] = "pas un objet"
    out = tmp_path / "A.jsonl"
    with make_client(TOKEN, httpx.MockTransport(lambda r: httpx.Response(200, json=body))) as client:
        result = generate(client, make_questions(1), make_config(effort="high"), out_path=out)
    assert result == (1, 0)
    assert read_jsonl(out)[0]["completion_tokens"] is None


def test_effort_de_la_config_verifie(tmp_path):
    # La config veut "high" ; le faux serveur répond avec "none" : tout est rejeté.
    out = tmp_path / "A.jsonl"
    result, _ = run_generate(make_questions(2), out, config=make_config(effort="high"))
    assert result == (0, 2)
    assert "attendu 'high'" in read_jsonl(errors_path_for(out))[0]["error"]


# --- limit -----------------------------------------------------------------------

@pytest.mark.parametrize("limit", [0, -1])
def test_limit_invalide_refuse(tmp_path, limit):
    out = tmp_path / "A.jsonl"
    result, inputs = run_generate(make_questions(3), out, limit=limit)
    assert isinstance(result, ValueError)
    assert inputs == []  # -1 aurait lancé toutes les questions sauf la dernière
    assert not out.exists()


@pytest.mark.parametrize("body, expected", [
    (b'{"detail": "rate limit exceeded"}', "rate limit exceeded"),
    (b'{"autre": 1}', "{'autre': 1}"),
    (b"<html>Bad Gateway</html>", "<html>Bad Gateway</html>"),
    (b"", None),
])
def test_message_d_alive_note_avec_l_echec(tmp_path, body, expected):
    out = tmp_path / "A.jsonl"
    transport = httpx.MockTransport(lambda request: httpx.Response(502, content=body))
    with make_client(TOKEN, transport) as client:
        result = generate(client, make_questions(1), make_config(), out_path=out,
                          sleep=lambda s: None)
    assert result == (0, 1)
    assert read_jsonl(errors_path_for(out))[0]["detail"] == expected


# --- Arrêts ----------------------------------------------------------------------

def test_a_l_arret_les_reessais_en_attente_sont_abandonnes(tmp_path):
    # Question 1 : un 502, donc elle attend 5 s avant son 2e essai.
    # Question 0 : un 401 un peu plus tard, qui arrête la campagne.
    # La question 1 doit se réveiller tout de suite et NE PAS renvoyer de run.
    out = tmp_path / "A.jsonl"
    plan = {"question 0": [401], "question 1": [502, 200]}
    slow_401 = lambda text: time.sleep(0.2) if text == "question 0" else None
    start = time.monotonic()
    result, inputs = run_generate(make_questions(2), out, plan=plan, on_request=slow_401,
                                  workers=2, sleep=None)  # sleep=None : la vraie attente

    assert isinstance(result, httpx.HTTPStatusError)
    assert result.response.status_code == 401
    assert time.monotonic() - start < 2.0          # pas les 5 s d'attente
    assert inputs.count("question 1") == 1         # pas de 2e essai après l'arrêt
    assert read_jsonl(out) == []


def test_401_arrete_tout(tmp_path):
    out = tmp_path / "A.jsonl"
    result, inputs = run_generate(make_questions(3), out, plan={"question 0": [401]})
    assert isinstance(result, httpx.HTTPStatusError)
    assert inputs == ["question 0"]  # les questions suivantes ne sont pas envoyées
    assert read_jsonl(out) == []


def test_cinq_echecs_de_suite_arretent_tout(tmp_path):
    out = tmp_path / "A.jsonl"
    plan = {f"question {i}": ["gemini"] for i in range(8)}
    result, inputs = run_generate(make_questions(8), out, plan=plan)
    assert isinstance(result, CampaignStopped)
    assert len(inputs) == 5  # 5 runs facturés, pas 8
    assert len(read_jsonl(errors_path_for(out))) == 5


def test_un_succes_remet_le_compteur_a_zero(tmp_path):
    # 4 échecs, 1 succès, 4 échecs : jamais 5 d'affilée, donc pas d'arrêt.
    out = tmp_path / "A.jsonl"
    plan = {f"question {i}": ["gemini"] for i in range(9) if i != 4}
    result, _ = run_generate(make_questions(9), out, plan=plan)
    assert result == (1, 8)


# --- Reprise : on saute ce qui est fait, on retente le reste --------------------

def ids_in(path):
    return [r["question_id"] for r in read_jsonl(path)]


def test_reprise_saute_les_questions_faites(tmp_path):
    out = tmp_path / "A.jsonl"
    run_generate(make_questions(5), out, limit=2)          # 1er lancement : questions 0 et 1
    result, inputs = run_generate(make_questions(5), out)   # 2e lancement : tout

    assert inputs == ["question 2", "question 3", "question 4"]  # 0 et 1 pas renvoyées
    assert result == (3, 0)  # le compte de CE lancement
    assert ids_in(out) == ["0", "1", "2", "3", "4"]


def test_reprise_apres_un_arret_401(tmp_path):
    out = tmp_path / "A.jsonl"
    result, _ = run_generate(make_questions(4), out, plan={"question 2": [401]})
    assert isinstance(result, httpx.HTTPStatusError)
    assert ids_in(out) == ["0", "1"]

    result, inputs = run_generate(make_questions(4), out)  # jeton réparé, on relance
    assert inputs == ["question 2", "question 3"]
    assert ids_in(out) == ["0", "1", "2", "3"]


def test_reprise_retente_les_echecs(tmp_path):
    out = tmp_path / "A.jsonl"
    run_generate(make_questions(3), out, plan={"question 1": ["gemini"]})
    assert ids_in(out) == ["0", "2"]

    result, inputs = run_generate(make_questions(3), out)
    assert inputs == ["question 1"]  # seule la question sans réponse est renvoyée
    assert result == (1, 0)
    assert ids_in(out) == ["0", "2", "1"]  # ajoutée à la fin
    # Le journal d'erreurs garde la trace de la tentative ratée.
    assert [e["question_id"] for e in read_jsonl(errors_path_for(out))] == ["1"]


def test_relance_quand_tout_est_fait(tmp_path):
    out = tmp_path / "A.jsonl"
    run_generate(make_questions(3), out)
    before = out.read_text(encoding="utf-8")

    result, inputs = run_generate(make_questions(3), out)
    assert inputs == []  # aucun run, donc aucun coût
    assert result == (0, 0)
    assert out.read_text(encoding="utf-8") == before


def test_les_lignes_existantes_ne_sont_jamais_modifiees(tmp_path):
    out = tmp_path / "A.jsonl"
    run_generate(make_questions(5), out, limit=2)
    before = out.read_text(encoding="utf-8")
    run_generate(make_questions(5), out)
    assert out.read_text(encoding="utf-8").startswith(before)


def test_limit_plus_petit_que_le_fichier(tmp_path):
    # 5 réponses déjà là, relance avec limit=3 : rien à faire, et pas de refus.
    out = tmp_path / "A.jsonl"
    run_generate(make_questions(5), out)
    result, inputs = run_generate(make_questions(5), out, limit=3)
    assert result == (0, 0)
    assert inputs == []


def test_journal_d_erreurs_existant_complete(tmp_path):
    out = tmp_path / "A.jsonl"
    errors = errors_path_for(out)
    errors.write_text('{"question_id": "9", "ancien": true}\n', encoding="utf-8")
    run_generate(make_questions(2), out, plan={"question 0": ["gemini"]})
    lines = read_jsonl(errors)
    assert lines[0] == {"question_id": "9", "ancien": True}  # l'ancienne ligne est intacte
    assert lines[1]["question_id"] == "0"


# --- Reprise refusée : le fichier n'est pas celui de cette campagne -------------

def refused(tmp_path, tamper=None, **relaunch):
    """Crée A.jsonl (3 questions, config A), l'abîme avec `tamper`, puis relance.

    Vérifie qu'aucun run n'est lancé et que le fichier n'est pas touché.
    Renvoie le message de l'erreur.
    """
    out = tmp_path / "A.jsonl"
    run_generate(make_questions(3), out)
    if tamper:
        out.write_text(tamper(out.read_text(encoding="utf-8")), encoding="utf-8")
    before = out.read_text(encoding="utf-8")

    questions = relaunch.pop("questions", make_questions(5))
    result, inputs = run_generate(questions, out, **relaunch)
    assert isinstance(result, ValueError)
    assert inputs == []
    assert out.read_text(encoding="utf-8") == before
    return str(result)


def test_refus_autre_config(tmp_path):
    assert "config 'A', attendu 'B'" in refused(tmp_path, config=make_config(name="B"))


def test_refus_autre_deploiement(tmp_path):
    # Même nom "A", mais configs.json pointe maintenant vers un autre déploiement.
    msg = refused(tmp_path, config=make_config(deployment="dp-AUTRE"))
    assert "déploiement 'dp-E8dfWyMHc4Q7oXnN', attendu 'dp-AUTRE'" in msg


def test_refus_autre_modele(tmp_path):
    msg = refused(tmp_path, config=make_config(model="openai/gpt-4.1-mini"))
    assert "attendu 'openai/gpt-4.1-mini'" in msg


def test_refus_autre_effort_de_raisonnement(tmp_path):
    # configs.json est passé de "none" à "high" entre deux lancements.
    msg = refused(tmp_path, config=make_config(effort="high"))
    assert "effort de raisonnement 'none', attendu 'high'" in msg


def test_refus_autre_room(tmp_path):
    assert "room 'th-Z7jWvllIYv1Eoa9z', attendu 'th-AUTRE'" in refused(
        tmp_path, config=make_config(room="th-AUTRE"))


def edit_line(index, **changes):
    """Fabrique un `tamper` qui modifie des champs de la ligne `index` (0 = première)."""
    def tamper(text):
        lines = text.splitlines()
        record = json.loads(lines[index])
        record.update(changes)
        lines[index] = json.dumps(record, ensure_ascii=False)
        return "\n".join(lines) + "\n"
    return tamper


def test_refus_ligne_avec_tokens_de_raisonnement(tmp_path):
    # Une ancienne ligne où le modèle a raisonné alors que la config dit "none".
    msg = refused(tmp_path, tamper=edit_line(1, reasoning_tokens=42))
    assert "ligne 2 : 42 tokens de raisonnement" in msg


def test_refus_execution_id_en_double_dans_le_fichier(tmp_path):
    # Deux questions qui portent le même run : l'une des deux réponses n'est pas la sienne.
    def same_id(text):
        first = json.loads(text.splitlines()[0])["execution_id"]
        return edit_line(2, execution_id=first)(text)
    msg = refused(tmp_path, tamper=same_id)
    assert "ligne 3 : execution_id" in msg and "en double" in msg


def test_refus_execution_id_absent_dans_le_fichier(tmp_path):
    assert "ligne 1 : execution_id absent" in refused(tmp_path, tamper=edit_line(0, execution_id=""))


def without_response(text):
    """Retire le champ response de la 2e ligne du fichier."""
    lines = text.splitlines()
    record = json.loads(lines[1])
    del record["response"]
    lines[1] = json.dumps(record, ensure_ascii=False)
    return "\n".join(lines) + "\n"


def test_refus_ligne_sans_reponse(tmp_path):
    # Sinon la question 1 serait comptée comme faite et jamais relancée.
    assert "ligne 2 : question '1' sans réponse" in refused(tmp_path, tamper=without_response)


def test_refus_question_inconnue(tmp_path):
    other = [{"id": "100", "text": "autre jeu de questions"}]
    assert "question '0' absente" in refused(tmp_path, questions=other)


def test_refus_texte_de_question_change(tmp_path):
    changed = make_questions(5)
    changed[1]["text"] = "question 1 reformulée"
    assert "texte de la question '1' a changé" in refused(tmp_path, questions=changed)


def test_refus_ligne_en_double(tmp_path):
    msg = refused(tmp_path, tamper=lambda text: text + text.splitlines()[0] + "\n")
    assert "ligne 4" in msg and "question '0' en double" in msg


def test_refus_ligne_illisible(tmp_path):
    msg = refused(tmp_path, tamper=lambda text: text + "ceci n'est pas du JSON\n")
    assert "ligne 4 : ligne illisible" in msg


def test_refus_ligne_vide(tmp_path):
    assert "ligne 4 : ligne illisible" in refused(tmp_path, tamper=lambda text: text + "\n")


def test_refus_derniere_ligne_coupee(tmp_path):
    # Écriture interrompue au milieu de la 3e ligne.
    assert "dernière ligne incomplète" in refused(tmp_path, tamper=lambda text: text[:-40])


def test_refus_derniere_ligne_sans_retour(tmp_path):
    # JSON complet mais sans "\n" final : la ligne suivante s'y collerait.
    assert "dernière ligne incomplète" in refused(tmp_path, tamper=lambda text: text[:-1])


def test_cree_le_dossier_de_sortie(tmp_path):
    out = tmp_path / "outputs" / "campagne" / "A.jsonl"
    result, _ = run_generate(make_questions(1), out)
    assert result == (1, 0)
    assert out.exists()


# --- Ligne de commande ----------------------------------------------------------

def cli_args(tmp_path, n_questions, *extra, config="A"):
    """Écrit un fichier de questions et un configs.json (config "A"), renvoie (args, sortie)."""
    questions = write(tmp_path / "q.jsonl",
                      [json.dumps(q) for q in make_questions(n_questions)])
    configs = tmp_path / "configs.json"
    configs.write_text(json.dumps({"A": {"room": ROOM, "deployment": DEPLOYMENT,
                                         "model": MODEL, "reasoning_effort": "none"}}),
                       encoding="utf-8")
    out = tmp_path / "A.jsonl"
    args = ["--config", config, "--configs", str(configs), "--questions", str(questions),
            "--out", str(out), *extra]
    return args, out


def test_main_limit(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ALIVE_TOKEN", TOKEN)
    args, out = cli_args(tmp_path, 10, "--limit", "3")
    transport, inputs = fake_alive()
    assert main(args, transport) == 0
    assert inputs == ["question 0", "question 1", "question 2"]
    assert len(read_jsonl(out)) == 3
    assert "3 réponses gardées, 0 échecs" in capsys.readouterr().out


def test_main_utilise_le_jeton_du_fichier_env(tmp_path, monkeypatch):
    for name in ("ALIVE_TOKEN", "ALIVE_ACCESS_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("ALIVE_ACCESS_TOKEN=jeton-du-fichier\n", encoding="utf-8")
    args, out = cli_args(tmp_path, 2)
    seen = []

    def handler(request):
        seen.append(request.headers["Authorization"])
        return httpx.Response(200, json=unique_run())

    assert main(args, httpx.MockTransport(handler), env_file=env_file) == 0
    assert seen == ["Bearer jeton-du-fichier"] * 2


def test_main_avec_echec_code_1(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ALIVE_TOKEN", TOKEN)
    args, out = cli_args(tmp_path, 2)
    transport, _ = fake_alive({"question 1": ["gemini"]})
    assert main(args, transport) == 1
    assert "1 réponses gardées, 1 échecs" in capsys.readouterr().out


def test_main_401_code_1(tmp_path, monkeypatch):
    monkeypatch.setenv("ALIVE_TOKEN", TOKEN)
    args, out = cli_args(tmp_path, 2)
    transport, inputs = fake_alive({"question 0": [401]})
    assert main(args, transport) == 1
    assert len(inputs) == 1


@pytest.mark.parametrize("config", ["B", "a", "A "])
def test_main_config_inconnue_code_2(tmp_path, monkeypatch, config):
    monkeypatch.setenv("ALIVE_TOKEN", TOKEN)
    args, out = cli_args(tmp_path, 2, config=config)
    transport, inputs = fake_alive()
    assert main(args, transport) == 2
    assert inputs == []
    assert not out.exists()


def test_main_configs_json_introuvable_code_2(tmp_path, monkeypatch):
    monkeypatch.setenv("ALIVE_TOKEN", TOKEN)
    args, out = cli_args(tmp_path, 2)
    (tmp_path / "configs.json").rename(tmp_path / "ailleurs.json")
    transport, inputs = fake_alive()
    assert main(args, transport) == 2
    assert inputs == []


def test_main_fichier_de_questions_vide_code_2(tmp_path, monkeypatch):
    # Avant : "0 réponses gardées, 0 échecs" et code 0, comme si tout allait bien.
    monkeypatch.setenv("ALIVE_TOKEN", TOKEN)
    args, out = cli_args(tmp_path, 0)
    transport, inputs = fake_alive()
    assert main(args, transport) == 2
    assert not out.exists()


def test_main_limit_negatif_code_2(tmp_path, monkeypatch):
    monkeypatch.setenv("ALIVE_TOKEN", TOKEN)
    args, out = cli_args(tmp_path, 3, "--limit", "-1")
    transport, inputs = fake_alive()
    assert main(args, transport) == 2
    assert inputs == []


def test_main_sans_jeton(tmp_path, monkeypatch):
    for name in ("ALIVE_TOKEN", "ALIVE_ACCESS_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    args, out = cli_args(tmp_path, 2)
    transport, inputs = fake_alive()
    assert main(args, transport, env_file=tmp_path / "absent.env") == 2
    assert inputs == []
    assert not out.exists()


def test_main_relance(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ALIVE_TOKEN", TOKEN)
    args, out = cli_args(tmp_path, 4, "--limit", "2")
    transport, _ = fake_alive()
    assert main(args, transport) == 0

    args_full = [a for a in args if a not in ("--limit", "2")]  # même commande, sans limite
    transport, inputs = fake_alive()
    assert main(args_full, transport) == 0
    assert inputs == ["question 2", "question 3"]
    assert len(read_jsonl(out)) == 4
    assert "2 réponses gardées, 0 échecs à ce lancement" in capsys.readouterr().out


def test_main_fichier_d_une_autre_campagne_code_2(tmp_path, monkeypatch):
    monkeypatch.setenv("ALIVE_TOKEN", TOKEN)
    args, out = cli_args(tmp_path, 2)
    out.write_text("déjà là\n", encoding="utf-8")
    transport, inputs = fake_alive()
    assert main(args, transport) == 2
    assert inputs == []
    assert out.read_text(encoding="utf-8") == "déjà là\n"  # intact


# --- Une config sur un alias d'Alive -------------------------------------------------

def test_generation_sur_un_alias(tmp_path):
    # Le déploiement demande l'alias ; Alive répond avec le vrai nom, qu'on exige et qu'on note.
    alias, real = "~anthropic/claude-haiku-latest", "anthropic/claude-haiku-5.5"
    config = Config(name="H", room=ROOM, deployment=DEPLOYMENT, model=alias,
                    reasoning_effort="none", served_model=real)

    def handler(request):
        run = unique_run()
        run["model"]["requested"] = alias
        run["model"]["served"][0]["model"] = real
        return httpx.Response(200, json=run)

    out = tmp_path / "H.jsonl"
    with make_client(TOKEN, httpx.MockTransport(handler)) as client:
        assert generate(client, make_questions(2), config, out_path=out, sleep=lambda s: None) == (2, 0)
        assert [r["model"] for r in read_jsonl(out)] == [real, real]
        # La reprise relit les lignes avec le vrai nom, sans erreur.
        assert generate(client, make_questions(3), config, out_path=out, sleep=lambda s: None) == (1, 0)
