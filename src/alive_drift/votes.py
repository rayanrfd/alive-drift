"""Lire et écrire les fichiers du jury. Aucun appel à Alive ici.

C'est l'équivalent de files.py pour la génération :
- load_responses, default_out_path : les réponses à noter, et où écrire ;
- make_vote : une ligne de vote ;
- load_votes : ce qui est déjà noté, pour la reprise.
"""

import json
from pathlib import Path

from alive_drift.config import Config
from alive_drift.files import now_utc, read_records
from alive_drift.grid import Criterion, grid_fingerprint


# --- Les réponses à noter -----------------------------------------------------------

def load_responses(path, questions: list[dict]) -> list[dict]:
    """Les réponses d'un fichier de génération, dans l'ordre du fichier de QUESTIONS.

    Renvoie une liste de {"id", "config", "question", "response", "execution_id"}, plus
    "message_id" et "room_id" si la ligne a un message_id (réponses générées depuis le
    morceau 37 : le juge « dans le fil » s'en sert comme parent).
    L'ordre des questions (et non celui du fichier, qui dépend de l'arrivée des
    réponses) garantit que --limit 3 note les 3 mêmes questions pour A, A_bis et B.

    Lève ValueError si une ligne est illisible, vient d'une autre config que la
    première, porte une question inconnue ou modifiée, n'a pas de réponse, ou si
    une question apparaît deux fois.
    """
    text_by_id = {q["id"]: q["text"] for q in questions}
    by_id = {}
    config = None
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, start=1):
            where = f"{path} ligne {n}"
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                row = None
            if not isinstance(row, dict):
                raise ValueError(f"{where} : ligne illisible")
            qid = row.get("question_id")
            if qid not in text_by_id:
                raise ValueError(f"{where} : question {qid!r} absente du fichier de questions")
            if row.get("question") != text_by_id[qid]:
                raise ValueError(f"{where} : le texte de la question {qid!r} a changé")
            if qid in by_id:
                raise ValueError(f"{where} : question {qid!r} en double")
            for field in ("config", "response", "execution_id"):
                if not isinstance(row.get(field), str) or not row[field].strip():
                    raise ValueError(f"{where} : {field!r} manquant ou vide")
            config = config or row["config"]
            if row["config"] != config:
                raise ValueError(f"{where} : config {row['config']!r}, attendu {config!r}")
            by_id[qid] = {"id": qid, "config": row["config"], "question": row["question"],
                          "response": row["response"], "execution_id": row["execution_id"]}
            if row.get("message_id") is not None:
                by_id[qid].update(message_id=row["message_id"], room_id=row.get("room_id"))
    if not by_id:
        raise ValueError(f"{path} : aucune réponse")
    return [by_id[q["id"]] for q in questions if q["id"] in by_id]


def default_out_path(responses_path, judge_name: str) -> Path:
    """outputs/A.jsonl + juge_qwen -> outputs/jury/A.juge_qwen.jsonl"""
    responses_path = Path(responses_path)
    return responses_path.parent / "jury" / f"{responses_path.stem}.{judge_name}.jsonl"


# --- Une ligne de vote ------------------------------------------------------------------

def make_vote(response: dict, run: dict, judge: Config, grid: list[Criterion]) -> dict:
    """La ligne enregistrée pour une réponse notée par un juge.

    `run` a passé check_run et porte en plus "verdicts" (voir attempt).
    Les votes sont rangés par NOM de critère : les scores (global, par thème,
    par critère) se recalculent ensuite sans rappeler le juge.
    Un vote vaut true, false, ou null quand le juge a répondu "non applicable".
    """
    served = run["model"]["served"][0]
    tokens = run.get("tokens")
    if not isinstance(tokens, dict):
        tokens = {}
    verdicts = run["verdicts"]
    return {
        "question_id": response["id"],
        "judged_config": response["config"],                 # A, A_bis ou B
        "response_execution_id": response["execution_id"],  # LA réponse notée
        "judge": judge.name,
        "room_id": judge.room,
        "deployment_id": judge.deployment,
        "reasoning_effort": judge.reasoning_effort,
        "grid": grid_fingerprint(grid),
        "judge_prompt": judge.judge_prompt,
        "verdicts": {name: verdict.met for name, verdict in verdicts.items()},
        "explanations": {name: verdict.explanation for name, verdict in verdicts.items()},
        "model": served["model"],
        "provider": served["provider"],
        "execution_id": run["execution_id"],
        "prompt_tokens": tokens.get("prompt"),
        "completion_tokens": tokens.get("completion"),
        "reasoning_tokens": tokens.get("reasoning"),
        "credits_used": run.get("credits_used"),
        "duration_ms": run.get("duration_ms"),
        "written_at": now_utc(),
    }


# --- Reprise ------------------------------------------------------------------------------

def load_votes(out_path: Path, responses: list[dict], judge: Config,
               grid: list[Criterion]) -> tuple[set[str], set[str]]:
    """Ce qui est déjà noté dans out_path : (ids des questions, execution_id des runs de juge).

    Deux ensembles vides si le fichier n'existe pas.
    Lève ValueError à la première ligne qui n'est pas de CETTE campagne : autre
    juge, autre déploiement, autre grille, autre prompt, autre config notée,
    réponse qui n'est plus celle du fichier de réponses, votes incomplets.
    """
    response_by_id = {r["id"]: r for r in responses}
    names = [criterion.name for criterion in grid]
    expected = (
        ("juge", "judge", judge.name),
        ("room", "room_id", judge.room),
        ("déploiement", "deployment_id", judge.deployment),
        ("modèle", "model", judge.served()),
        ("effort de raisonnement", "reasoning_effort", judge.reasoning_effort),
        ("grille", "grid", grid_fingerprint(grid)),
        ("prompt du juge", "judge_prompt", judge.judge_prompt),
        ("config notée", "judged_config", responses[0]["config"]),
    )
    done, executions = set(), set()
    for where, vote in read_records(out_path):
        for label, key, value in expected:
            if vote.get(key) != value:
                raise ValueError(f"{where} : {label} {vote.get(key)!r}, attendu {value!r}")
        qid = vote.get("question_id")
        if qid not in response_by_id:
            raise ValueError(f"{where} : question {qid!r} absente du fichier de réponses")
        if vote.get("response_execution_id") != response_by_id[qid]["execution_id"]:
            raise ValueError(f"{where} : la réponse à la question {qid!r} n'est plus la même")
        if qid in done:
            raise ValueError(f"{where} : question {qid!r} en double")
        verdicts = vote.get("verdicts")
        if (not isinstance(verdicts, dict) or list(verdicts) != names
                or any(type(met) not in (bool, type(None)) for met in verdicts.values())):
            raise ValueError(f"{where} : votes incomplets ou mal formés")
        execution = vote.get("execution_id")
        if not isinstance(execution, str) or not execution:
            raise ValueError(f"{where} : execution_id absent")
        if execution in executions:
            raise ValueError(f"{where} : execution_id {execution!r} en double")
        done.add(qid)
        executions.add(execution)
    return done, executions
