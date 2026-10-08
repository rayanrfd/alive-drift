"""Tests de jury.py, sans réseau.

Le faux Alive joue le juge : il lit le numéro de la question dans le message
reçu ("question 2") et répond selon `plan` = {numéro: liste d'étapes}, une
étape par appel. Étapes possibles :
  "ok"       les 3 verdicts attendus (met, not_met, not_applicable) ;
  "partiel"  il manque le critère 3 (réponse inutilisable) ;
  "bavard"   du texte sans JSON ;
  "gemini"   bon JSON mais servi par un autre modèle ;
  un code HTTP d'erreur.
Sans plan : "ok". Chaque run a un execution_id neuf, comme le vrai Alive.
"""

import copy
import itertools
import json
import re
from pathlib import Path

import httpx
import pytest

from alive_drift.client import make_client
from alive_drift.config import Config
from alive_drift.files import errors_path_for
from alive_drift.grid import Criterion, grid_fingerprint
from alive_drift.judge import prompt_fingerprint, system_prompt, user_message
from alive_drift.jury import (
    check_judge, default_out_path, judge_responses, load_responses, load_votes, main,
)
from alive_drift.recorder import CampaignStopped
from test_check import MODEL, REAL_RUN

TOKEN = "jeton-de-test"
REPO = Path(__file__).parent.parent

# Une petite grille de 3 critères, pour des tests lisibles.
GRID = [Criterion("communication_a", "communication", "Texte du critère A.", 5.0),
        Criterion("hedging_b", "hedging", "Texte du critère B.", 5.0),
        Criterion("hedging_c", "hedging", "Texte du critère C.", 5.0)]
NAMES = [c.name for c in GRID]
PROMPT = prompt_fingerprint(system_prompt(GRID))

JUDGE = Config(name="juge_test", room="th-JUGE", deployment="dp-JUGE", model=MODEL,
               reasoning_effort="none", judge_prompt=PROMPT)


def verdicts_json(*words):
    return json.dumps({"results": [
        {"criterion_id": n, "explanation": f"raison {n}", "verdict": word}
        for n, word in enumerate(words, start=1)]})


CONTENTS = {"ok": verdicts_json("met", "not_met", "not_applicable"),
            "partiel": verdicts_json("met", "not_met"),
            "bavard": "I think this response is quite good overall."}

_numbers = itertools.count(1)


def judge_run(content, model=MODEL):
    run = copy.deepcopy(REAL_RUN)
    run["execution_id"] = f"ex-juge-{next(_numbers)}"
    run["content"] = content
    run["model"]["served"][0]["model"] = model
    return run


def fake_alive(plan=None):
    """Renvoie (transport, requests). requests = liste de (chemin, message reçu)."""
    plan = {number: list(steps) for number, steps in (plan or {}).items()}
    requests = []

    def handler(request):
        message = json.loads(request.content)["input"]
        requests.append((request.url.path.removeprefix("/api/v1"), message))
        number = int(re.search(r"question (\d+)", message).group(1))
        steps = plan.get(number)
        step = steps.pop(0) if steps else "ok"
        if step == "gemini":
            return httpx.Response(200, json=judge_run(CONTENTS["ok"], "google/gemini-3.1-flash-lite"))
        if step in CONTENTS:
            return httpx.Response(200, json=judge_run(CONTENTS[step]))
        return httpx.Response(step, json={"detail": f"erreur {step}"})

    return httpx.MockTransport(handler), requests


def make_responses(n, config="A"):
    """n réponses à noter, comme les rend load_responses."""
    return [{"id": str(i), "config": config, "question": f"question {i}",
             "response": f"réponse {i}", "execution_id": f"ex-gen-{config}-{i}"} for i in range(n)]


def run_jury(responses, out_path, plan=None, *, judge=JUDGE, grid=GRID, limit=None, workers=1):
    """Lance judge_responses. Renvoie (résultat ou exception, requêtes reçues)."""
    transport, requests = fake_alive(plan)
    with make_client(TOKEN, transport) as client:
        try:
            result = judge_responses(client, responses, judge, grid, out_path=out_path,
                                     limit=limit, workers=workers, sleep=lambda s: None)
        except Exception as err:
            result = err
    return result, requests


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# --- Cas normal -------------------------------------------------------------------------

def test_trois_reponses_notees(tmp_path):
    out = tmp_path / "A.juge_test.jsonl"
    result, requests = run_jury(make_responses(3), out)

    assert result == (3, 0)
    votes = read_jsonl(out)
    assert [v["question_id"] for v in votes] == ["0", "1", "2"]
    vote = votes[1]
    written_at = vote.pop("written_at")
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\+00:00", written_at)
    assert vote == {
        "question_id": "1",
        "judged_config": "A",
        "response_execution_id": "ex-gen-A-1",
        "judge": "juge_test",
        "room_id": "th-JUGE",
        "deployment_id": "dp-JUGE",
        "reasoning_effort": "none",
        "grid": grid_fingerprint(GRID),
        "judge_prompt": PROMPT,
        "verdicts": {"communication_a": True, "hedging_b": False, "hedging_c": None},  # None = N/A
        "explanations": {"communication_a": "raison 1", "hedging_b": "raison 2", "hedging_c": "raison 3"},
        "model": MODEL,
        "provider": REAL_RUN["model"]["served"][0]["provider"],
        "execution_id": vote["execution_id"],
        "prompt_tokens": REAL_RUN["tokens"]["prompt"],
        "completion_tokens": REAL_RUN["tokens"]["completion"],
        "reasoning_tokens": 0,
        "credits_used": REAL_RUN["credits_used"],
        "duration_ms": REAL_RUN["duration_ms"],
    }
    assert vote["execution_id"].startswith("ex-juge-")
    assert not errors_path_for(out).exists()


def test_le_juge_recoit_la_question_et_la_reponse_sur_son_deploiement(tmp_path):
    _, requests = run_jury(make_responses(2), tmp_path / "out.jsonl")
    assert requests == [
        ("/threads/th-JUGE/deployments/dp-JUGE/runs", user_message("question 0", "réponse 0")),
        ("/threads/th-JUGE/deployments/dp-JUGE/runs", user_message("question 1", "réponse 1")),
    ]


def test_plusieurs_runs_a_la_fois(tmp_path):
    out = tmp_path / "out.jsonl"
    result, requests = run_jury(make_responses(12), out, workers=4)
    assert result == (12, 0)
    assert sorted(int(v["question_id"]) for v in read_jsonl(out)) == list(range(12))
    assert len({v["execution_id"] for v in read_jsonl(out)}) == 12


def test_limit_ne_note_que_les_premieres(tmp_path):
    out = tmp_path / "out.jsonl"
    result, requests = run_jury(make_responses(5), out, limit=2)
    assert result == (2, 0)
    assert [v["question_id"] for v in read_jsonl(out)] == ["0", "1"]


# --- Le juge répond mal : noté, pas gardé, on continue ---------------------------------

@pytest.mark.parametrize("step, error_type, words", [
    ("partiel", "JudgeAnswerError", "critères sans verdict : [3]"),
    ("bavard", "JudgeAnswerError", "pas d'objet JSON"),
    ("gemini", "RunRejected", "servi par 'google/gemini-3.1-flash-lite'"),
])
def test_reponse_de_juge_inutilisable(tmp_path, step, error_type, words):
    out = tmp_path / "out.jsonl"
    result, requests = run_jury(make_responses(3), out, {1: [step]})

    assert result == (2, 1)
    assert len(requests) == 3  # un run payé ne se réessaie pas tout seul
    assert [v["question_id"] for v in read_jsonl(out)] == ["0", "2"]  # aucun vote partiel gardé
    [error] = read_jsonl(errors_path_for(out))
    assert error["question_id"] == "1" and error["config"] == "juge_test"
    assert error["error_type"] == error_type and words in error["error"]


def test_le_journal_garde_le_debut_de_la_reponse_inutilisable(tmp_path):
    out = tmp_path / "out.jsonl"
    run_jury(make_responses(1), out, {0: ["bavard"]})
    [error] = read_jsonl(errors_path_for(out))
    assert re.fullmatch(r"ex-juge-\d+ : I think this response is quite good overall\.", error["detail"])


def test_cinq_reponses_inutilisables_d_affilee_arretent_tout(tmp_path):
    out = tmp_path / "out.jsonl"
    result, requests = run_jury(make_responses(10), out, {i: ["partiel"] for i in range(10)})
    assert isinstance(result, CampaignStopped)
    assert len(requests) == 5  # 5 runs payés, pas 10
    assert read_jsonl(out) == []


def test_502_reessaye_puis_note(tmp_path):
    out = tmp_path / "out.jsonl"
    result, requests = run_jury(make_responses(2), out, {0: [502, "ok"]})
    assert result == (2, 0)
    assert len(requests) == 3


def test_401_arrete_tout(tmp_path):
    out = tmp_path / "out.jsonl"
    result, requests = run_jury(make_responses(3), out, {0: [401]})
    assert isinstance(result, httpx.HTTPStatusError)
    assert len(requests) == 1 and read_jsonl(out) == []


# --- Reprise -----------------------------------------------------------------------------

def test_reprise_saute_ce_qui_est_note_et_retente_les_echecs(tmp_path):
    out = tmp_path / "out.jsonl"
    run_jury(make_responses(3), out, {1: ["partiel"]})
    before = out.read_text(encoding="utf-8")

    result, requests = run_jury(make_responses(3), out)
    assert result == (1, 0)
    assert [message for _, message in requests] == [user_message("question 1", "réponse 1")]
    assert out.read_text(encoding="utf-8").startswith(before)  # rien n'est réécrit
    assert sorted(v["question_id"] for v in read_jsonl(out)) == ["0", "1", "2"]


def test_relance_quand_tout_est_note(tmp_path):
    out = tmp_path / "out.jsonl"
    run_jury(make_responses(2), out)
    result, requests = run_jury(make_responses(2), out)
    assert result == (0, 0) and requests == []


def rewrite(out, change):
    """Applique `change(vote)` à la première ligne de `out`."""
    votes = read_jsonl(out)
    change(votes[0])
    out.write_text("".join(json.dumps(v) + "\n" for v in votes), encoding="utf-8")


def set_key(key, value):
    return lambda vote: vote.__setitem__(key, value)


@pytest.mark.parametrize("change, words", [
    (set_key("judge", "juge_autre"), "juge 'juge_autre'"),
    (set_key("deployment_id", "dp-AUTRE"), "déploiement 'dp-AUTRE'"),
    (set_key("model", "autre/modele"), "modèle 'autre/modele'"),
    (set_key("grid", "000000000000"), "grille '000000000000'"),
    (set_key("judge_prompt", "000000000000"), "prompt du juge '000000000000'"),
    (set_key("judged_config", "B"), "config notée 'B'"),
    (set_key("question_id", "99"), "question '99' absente"),
    (set_key("response_execution_id", "ex-gen-A-AUTRE"), "n'est plus la même"),
    (set_key("verdicts", {"communication_a": True, "hedging_b": False}), "votes incomplets"),
    (set_key("verdicts", {"communication_a": True, "hedging_b": 0, "hedging_c": True}), "votes incomplets"),
    (set_key("verdicts", {"communication_a": True, "hedging_b": "not_met", "hedging_c": None}), "votes incomplets"),
    (set_key("execution_id", ""), "execution_id absent"),
])
def test_fichier_qui_n_est_pas_celui_de_la_campagne_refuse(tmp_path, change, words):
    out = tmp_path / "out.jsonl"
    run_jury(make_responses(2), out)
    rewrite(out, change)
    before = out.read_text(encoding="utf-8")

    result, requests = run_jury(make_responses(3), out)
    assert isinstance(result, ValueError) and words in str(result)
    assert requests == []                                 # aucun run lancé
    assert out.read_text(encoding="utf-8") == before      # fichier intact


def test_refus_question_notee_deux_fois_ou_ligne_coupee(tmp_path):
    out = tmp_path / "out.jsonl"
    run_jury(make_responses(1), out)
    line = out.read_text(encoding="utf-8")
    out.write_text(line + line, encoding="utf-8")
    with pytest.raises(ValueError, match="en double"):
        load_votes(out, make_responses(1), JUDGE, GRID)
    out.write_text(line[:-1], encoding="utf-8")
    with pytest.raises(ValueError, match="dernière ligne incomplète"):
        load_votes(out, make_responses(1), JUDGE, GRID)


# --- Le juge doit être celui de la grille ------------------------------------------------

def test_config_qui_n_est_pas_un_juge_refusee(tmp_path):
    not_a_judge = Config(name="A", room="th-A", deployment="dp-A", model=MODEL, reasoning_effort="none")
    result, requests = run_jury(make_responses(2), tmp_path / "out.jsonl", judge=not_a_judge)
    assert isinstance(result, ValueError) and "n'est pas un juge" in str(result)
    assert requests == []


def test_juge_cree_avec_une_autre_grille_refuse(tmp_path):
    changed = [Criterion("communication_a", "communication", "Texte MODIFIÉ.", 5.0), *GRID[1:]]
    result, requests = run_jury(make_responses(2), tmp_path / "out.jsonl", grid=changed)
    assert isinstance(result, ValueError)
    assert f"créé avec le prompt {PROMPT}" in str(result)
    assert requests == []  # rien de dépensé
    check_judge(JUDGE, GRID)  # avec la bonne grille : pas d'erreur


@pytest.mark.parametrize("options", [{"limit": 0}, {"limit": -1}, {"workers": 0}, {"workers": 33}])
def test_options_invalides_refusees(tmp_path, options):
    result, requests = run_jury(make_responses(2), tmp_path / "out.jsonl", **options)
    assert isinstance(result, ValueError) and requests == []


# --- load_responses : lire un fichier de génération ---------------------------------------

QUESTIONS = [{"id": str(i), "text": f"question {i}"} for i in range(4)]


def generation_line(i, config="A", **changes):
    return {"question_id": str(i), "config": config, "question": f"question {i}",
            "response": f"réponse {i}", "execution_id": f"ex-gen-{config}-{i}",
            "model": "openai/gpt-4.1-mini", **changes}


def write_lines(path, lines):
    path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
    return path


def test_load_responses_dans_l_ordre_des_questions(tmp_path):
    # Le fichier est dans l'ordre d'arrivée (2, 0, 3) ; la question 1 n'a pas de réponse.
    path = write_lines(tmp_path / "A.jsonl", [generation_line(2), generation_line(0), generation_line(3)])
    responses = load_responses(path, QUESTIONS)
    assert [r["id"] for r in responses] == ["0", "2", "3"]
    assert responses[1] == {"id": "2", "config": "A", "question": "question 2",
                            "response": "réponse 2", "execution_id": "ex-gen-A-2"}


@pytest.mark.parametrize("lines, words", [
    ([generation_line(0), generation_line(9)], "question '9' absente"),
    ([generation_line(0, question="autre texte")], "a changé"),
    ([generation_line(0), generation_line(0)], "en double"),
    ([generation_line(0, response="  ")], "'response' manquant ou vide"),
    ([generation_line(0, execution_id=None)], "'execution_id' manquant ou vide"),
    ([generation_line(0), generation_line(1, config="B")], "config 'B', attendu 'A'"),
    ([], "aucune réponse"),
])
def test_load_responses_refuse(tmp_path, lines, words):
    path = write_lines(tmp_path / "A.jsonl", lines)
    with pytest.raises(ValueError) as err:
        load_responses(path, QUESTIONS)
    assert words in str(err.value)


def test_load_responses_ligne_illisible(tmp_path):
    path = tmp_path / "A.jsonl"
    path.write_text('{"question_id": "0"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="ligne 1 : ligne illisible"):
        load_responses(path, QUESTIONS)


def test_chemin_de_sortie_par_defaut():
    assert default_out_path("outputs/A_bis.jsonl", "juge_qwen") == Path("outputs/jury/A_bis.juge_qwen.jsonl")


# --- Ligne de commande ----------------------------------------------------------------------

def setup_files(tmp_path, n=4, selection=None):
    """Écrit configs, questions, sélection et réponses. Renvoie les arguments communs.

    La grille est la vraie (3 critères ne suffisent pas à load_grid). Sélection par
    défaut : les 34 critères ; `selection` = un autre fichier de sélection.
    """
    from alive_drift.grid import load_grid, load_working_grid
    criteria = REPO / "data" / "criteria.json"
    if selection is None:
        selection = tmp_path / "selection.json"
        selection.write_text(json.dumps({"keep": [c.name for c in load_grid(criteria)]}), encoding="utf-8")
    grid = load_working_grid(criteria, selection)
    judge = {"room": "th-JUGE", "deployment": "dp-JUGE", "model": MODEL, "reasoning_effort": "none",
             "judge_prompt": prompt_fingerprint(system_prompt(grid))}
    generator = {"room": "th-A", "deployment": "dp-A", "model": MODEL, "reasoning_effort": "none"}
    (tmp_path / "configs.json").write_text(json.dumps({"juge_test": judge, "A": generator}), encoding="utf-8")
    write_lines(tmp_path / "questions.jsonl", [{"id": i, "text": f"question {i}"} for i in range(n)])
    write_lines(tmp_path / "A.jsonl", [generation_line(i) for i in reversed(range(n))])
    return ["--responses", str(tmp_path / "A.jsonl"), "--configs", str(tmp_path / "configs.json"),
            "--questions", str(tmp_path / "questions.jsonl"), "--grid", str(criteria),
            "--selection", str(selection)]


def full_grid_transport(n_criteria=34):
    """Un juge qui rend n verdicts (critères pairs : met, les autres : not_applicable)."""
    requests = []

    def handler(request):
        requests.append(json.loads(request.content)["input"])
        content = json.dumps({"results": [
            {"criterion_id": n, "explanation": "raison", "verdict": "met" if n % 2 == 0 else "not_applicable"}
            for n in range(1, n_criteria + 1)]})
        return httpx.Response(200, json=judge_run(content))
    return httpx.MockTransport(handler), requests


def test_main_note_et_range_dans_le_dossier_jury(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ALIVE_TOKEN", TOKEN)
    args = setup_files(tmp_path)
    transport, requests = full_grid_transport()

    assert main(["--judge", "juge_test", "--limit", "2", *args], transport) == 0
    out = tmp_path / "jury" / "A.juge_test.jsonl"
    votes = read_jsonl(out)
    assert [v["question_id"] for v in votes] == ["0", "1"]  # ordre des questions, pas du fichier
    assert len(votes[0]["verdicts"]) == 34
    assert list(votes[0]["verdicts"].values()).count(True) == 17
    assert list(votes[0]["verdicts"].values()).count(None) == 17  # N/A écrit null dans le fichier
    assert f"2 réponses notées, 0 échecs à ce lancement -> {out}" in capsys.readouterr().out
    assert TOKEN not in out.read_text(encoding="utf-8")


def test_main_option_out(tmp_path, monkeypatch):
    monkeypatch.setenv("ALIVE_TOKEN", TOKEN)
    transport, _ = full_grid_transport()
    out = tmp_path / "ailleurs" / "votes.jsonl"
    assert main(["--judge", "juge_test", "--out", str(out), *setup_files(tmp_path)], transport) == 0
    assert len(read_jsonl(out)) == 4


@pytest.mark.parametrize("judge", ["A", "juge_inconnu"])
def test_main_pas_un_juge_code_2(tmp_path, monkeypatch, capsys, judge):
    monkeypatch.setenv("ALIVE_TOKEN", TOKEN)
    transport, requests = full_grid_transport()
    assert main(["--judge", judge, *setup_files(tmp_path)], transport) == 2
    assert requests == []
    assert "REFUS" in capsys.readouterr().err


def test_main_fichier_de_reponses_introuvable_code_2(tmp_path, monkeypatch):
    monkeypatch.setenv("ALIVE_TOKEN", TOKEN)
    transport, requests = full_grid_transport()
    args = setup_files(tmp_path)
    args[1] = str(tmp_path / "absent.jsonl")
    assert main(["--judge", "juge_test", *args], transport) == 2
    assert requests == []


def test_main_sans_jeton_code_2(tmp_path, monkeypatch):
    for name in ("ALIVE_TOKEN", "ALIVE_ACCESS_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    transport, requests = full_grid_transport()
    args = ["--judge", "juge_test", *setup_files(tmp_path)]
    assert main(args, transport, env_file=tmp_path / "absent.env") == 2
    assert requests == []


def test_les_juges_d_avant_non_applicable_sont_refuses():
    # Ces juges ont été créés le 07/10 avec l'ancienne consigne (oui/non, sans N/A),
    # puis rangés dans archive/. Leur déploiement existe toujours chez Alive : si
    # quelqu'un les remettait dans configs.json, jury refuserait de les lancer.
    from alive_drift.config import load_config
    from alive_drift.grid import load_grid
    grid = load_grid(REPO / "data" / "criteria.json")
    for name in ("juge_deepseek", "juge_gemini", "juge_qwen", "juge_minimax", "juge_glm", "juge_kimi"):
        with pytest.raises(ValueError, match="créé avec le prompt 283c79ab9f9a"):
            check_judge(load_config(REPO / "archive" / "juges_avant_na.json", name), grid)


def test_main_avec_la_selection_du_depot(tmp_path, monkeypatch):
    # La grille de travail du 08/10 : 8 critères envoyés au juge, 8 votes écrits.
    from test_grid import KEPT
    monkeypatch.setenv("ALIVE_TOKEN", TOKEN)
    args = setup_files(tmp_path, selection=REPO / "data" / "selection.json")
    transport, requests = full_grid_transport(n_criteria=8)

    assert main(["--judge", "juge_test", "--limit", "1", *args], transport) == 0
    [vote] = read_jsonl(tmp_path / "jury" / "A.juge_test.jsonl")
    assert list(vote["verdicts"]) == KEPT
    assert list(vote["verdicts"].values()) == [None, True] * 4


def test_les_juges_de_la_grille_34_sont_refuses():
    # Les onze juges créés les 07 et 08/10 avec les 34 critères et la consigne N/A,
    # rangés dans archive/ le 08/10. La grille de travail change le prompt : jury
    # refuse de les lancer, il faut les recréer avec deploy --judge.
    from alive_drift.config import load_config, read_configs
    from alive_drift.grid import load_working_grid
    archive = REPO / "archive" / "juges_grille_34.json"
    grid = load_working_grid(REPO / "data" / "criteria.json", REPO / "data" / "selection.json")
    names = list(read_configs(archive))
    assert len(names) == 11
    for name in names:
        with pytest.raises(ValueError, match="créé avec le prompt 402721efd0f3"):
            check_judge(load_config(archive, name), grid)
