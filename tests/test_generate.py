"""Tests de generate.py, sans réseau.

Le faux serveur répond selon la QUESTION reçue : `plan` associe un texte de
question à la liste de ses réponses successives (une par appel). Une question
absente du plan reçoit la vraie sortie du 01/10 (code 200).
Étapes possibles : 200, "gemini" (200 mais servi par Gemini), ou un code d'erreur.
"""

import copy
import json
from pathlib import Path

import httpx
import pytest

from alive_drift.client import make_client
from alive_drift.generate import (
    CampaignStopped, errors_path_for, generate, load_questions, main,
)
from test_check import MODEL, REAL_RUN

TOKEN = "jeton-de-test"
ROOM = "th-Z7jWvllIYv1Eoa9z"
DEPLOYMENT = "dp-E8dfWyMHc4Q7oXnN"

GEMINI_RUN = copy.deepcopy(REAL_RUN)
GEMINI_RUN["model"]["served"][0]["model"] = "google/gemini-3.1-flash-lite"


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
            return httpx.Response(200, json=REAL_RUN)
        if step == "gemini":
            return httpx.Response(200, json=GEMINI_RUN)
        return httpx.Response(step, json={"detail": f"erreur {step}"})

    return httpx.MockTransport(handler), inputs


def run_generate(questions, out_path, plan=None, on_request=None):
    """Lance generate. Renvoie (résultat ou exception, questions reçues par le serveur)."""
    transport, inputs = fake_alive(plan, on_request)
    with make_client(TOKEN, transport) as client:
        try:
            result = generate(client, questions, room_id=ROOM, deployment_id=DEPLOYMENT,
                              expected_model=MODEL, config="A", out_path=out_path,
                              sleep=lambda s: None)
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
    assert records[1] == {
        "question_id": "1",
        "config": "A",
        "question": "question 1",
        "response": REAL_RUN["content"],
        "model": "mistralai/mistral-small-24b-instruct-2501",
        "provider": "DeepInfra",
        "execution_id": "ex-WxDjC6ClZzye4T0a",
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
    assert errors[0]["error_type"] == "RunRejected"
    assert "servi par 'google/gemini-3.1-flash-lite'" in errors[0]["error"]


def test_trois_502_notes_et_on_continue(tmp_path):
    out = tmp_path / "A.jsonl"
    result, inputs = run_generate(make_questions(3), out, plan={"question 1": [502, 502, 502]})

    assert result == (2, 1)
    assert inputs.count("question 1") == 3  # les 3 essais ont eu lieu
    assert [r["question_id"] for r in read_jsonl(out)] == ["0", "2"]
    errors = read_jsonl(errors_path_for(out))
    assert errors[0]["question_id"] == "1"
    assert errors[0]["error_type"] == "HTTPStatusError"
    assert TOKEN not in errors[0]["error"]


def test_502_puis_succes_reponse_gardee(tmp_path):
    out = tmp_path / "A.jsonl"
    result, _ = run_generate(make_questions(2), out, plan={"question 0": [502, 200]})
    assert result == (2, 0)
    assert not errors_path_for(out).exists()


# --- Arrêts ----------------------------------------------------------------------

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


# --- On n'écrase jamais ---------------------------------------------------------

@pytest.mark.parametrize("existing", ["A.jsonl", "A.errors.jsonl"])
def test_refuse_si_sortie_existe(tmp_path, existing):
    (tmp_path / existing).write_text("déjà là\n", encoding="utf-8")
    result, inputs = run_generate(make_questions(2), tmp_path / "A.jsonl")
    assert isinstance(result, FileExistsError)
    assert inputs == []  # aucun run lancé
    assert (tmp_path / existing).read_text(encoding="utf-8") == "déjà là\n"  # intact


def test_cree_le_dossier_de_sortie(tmp_path):
    out = tmp_path / "outputs" / "campagne" / "A.jsonl"
    result, _ = run_generate(make_questions(1), out)
    assert result == (1, 0)
    assert out.exists()


# --- Ligne de commande ----------------------------------------------------------

def cli_args(tmp_path, n_questions, *extra):
    questions = write(tmp_path / "q.jsonl",
                      [json.dumps(q) for q in make_questions(n_questions)])
    out = tmp_path / "A.jsonl"
    args = ["--questions", str(questions), "--room", ROOM, "--deployment", DEPLOYMENT,
            "--model", MODEL, "--config", "A", "--out", str(out), *extra]
    return args, out


def test_main_limit(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ALIVE_TOKEN", TOKEN)
    args, out = cli_args(tmp_path, 10, "--limit", "3")
    transport, inputs = fake_alive()
    assert main(args, transport) == 0
    assert inputs == ["question 0", "question 1", "question 2"]
    assert len(read_jsonl(out)) == 3
    assert "3 réponses gardées, 0 échecs" in capsys.readouterr().out


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


def test_main_sans_jeton(tmp_path, monkeypatch):
    monkeypatch.delenv("ALIVE_TOKEN", raising=False)
    args, out = cli_args(tmp_path, 2)
    transport, inputs = fake_alive()
    assert main(args, transport) == 2
    assert inputs == []
    assert not out.exists()


def test_main_sortie_existante_code_2(tmp_path, monkeypatch):
    monkeypatch.setenv("ALIVE_TOKEN", TOKEN)
    args, out = cli_args(tmp_path, 2)
    out.write_text("déjà là\n", encoding="utf-8")
    transport, inputs = fake_alive()
    assert main(args, transport) == 2
    assert inputs == []
