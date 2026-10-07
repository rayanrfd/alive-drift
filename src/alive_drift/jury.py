"""Fait noter les réponses d'UNE config par UN juge.

Pour chaque réponse : un run Alive sur le déploiement du juge (le message est
la question et la réponse, voir judge.py), puis une ligne écrite tout de suite
dans le fichier de sortie, avec les 34 oui/non du juge.

C'est generate.py avec une autre entrée et une autre ligne de sortie : même
parallélisme (pool.py), mêmes réessais (client.py), même arrêt après 5 échecs
d'affilée (recorder.py), même reprise en relançant la commande.

Un fichier par (config notée, juge) :

    outputs/jury/A.juge_qwen.jsonl          les votes gardés
    outputs/jury/A.juge_qwen.errors.jsonl   les runs ratés ou inutilisables

Le jury complet = 3 configs x 3 juges = 9 commandes.

Avant le premier run, on vérifie sans rien dépenser que le juge porte bien le
prompt fabriqué depuis la grille d'aujourd'hui (empreinte dans configs.json).

Exemple sur 3 réponses (3 runs payants) :

    uv run python -m alive_drift.jury --judge juge_qwen --responses outputs/A.jsonl --limit 3
"""

import argparse
import json
import sys
import threading
from pathlib import Path

import httpx

from alive_drift.check import RunRejected
from alive_drift.client import TOKEN_HELP, find_token, make_client, run_with_retries
from alive_drift.config import Config, load_config
from alive_drift.files import load_questions, now_utc
from alive_drift.grid import Criterion, grid_fingerprint, load_grid
from alive_drift.judge import (
    JudgeAnswerError, parse_verdicts, prompt_fingerprint, system_prompt, user_message,
)
from alive_drift.pool import MAX_WORKERS, run_tasks
from alive_drift.recorder import CampaignStopped, Recorder


# --- Les réponses à noter -----------------------------------------------------------

def load_responses(path, questions: list[dict]) -> list[dict]:
    """Les réponses d'un fichier de génération, dans l'ordre du fichier de QUESTIONS.

    Renvoie une liste de {"id", "config", "question", "response", "execution_id"}.
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
    if not out_path.exists():
        return set(), set()
    content = out_path.read_text(encoding="utf-8")
    if content and not content.endswith("\n"):
        raise ValueError(f"{out_path} : dernière ligne incomplète (écriture interrompue ?)")

    response_by_id = {r["id"]: r for r in responses}
    names = [criterion.name for criterion in grid]
    expected = (
        ("juge", "judge", judge.name),
        ("room", "room_id", judge.room),
        ("déploiement", "deployment_id", judge.deployment),
        ("modèle", "model", judge.model),
        ("effort de raisonnement", "reasoning_effort", judge.reasoning_effort),
        ("grille", "grid", grid_fingerprint(grid)),
        ("prompt du juge", "judge_prompt", judge.judge_prompt),
        ("config notée", "judged_config", responses[0]["config"]),
    )
    done, executions = set(), set()
    for n, line in enumerate(content.splitlines(), start=1):
        where = f"{out_path} ligne {n}"
        try:
            vote = json.loads(line)
        except json.JSONDecodeError:
            vote = None
        if not isinstance(vote, dict):
            raise ValueError(f"{where} : ligne illisible")
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
                or any(type(met) is not bool for met in verdicts.values())):
            raise ValueError(f"{where} : votes incomplets ou mal formés")
        execution = vote.get("execution_id")
        if not isinstance(execution, str) or not execution:
            raise ValueError(f"{where} : execution_id absent")
        if execution in executions:
            raise ValueError(f"{where} : execution_id {execution!r} en double")
        done.add(qid)
        executions.add(execution)
    return done, executions


# --- La boucle ------------------------------------------------------------------------------

def check_judge(judge: Config, grid: list[Criterion]) -> None:
    """Lève ValueError si `judge` n'est pas un juge, ou pas celui de cette grille.

    Le prompt du juge est figé chez Alive depuis sa création. Si la grille ou
    les consignes ont changé depuis, ses numéros de critères ne correspondent
    plus aux nôtres : on refuse avant de dépenser quoi que ce soit.
    """
    if judge.judge_prompt is None:
        raise ValueError(f"la config {judge.name!r} n'est pas un juge (pas de 'judge_prompt')")
    current = prompt_fingerprint(system_prompt(grid))
    if judge.judge_prompt != current:
        raise ValueError(
            f"le juge {judge.name!r} a été créé avec le prompt {judge.judge_prompt}, "
            f"la grille et les consignes d'aujourd'hui donnent {current} : "
            "recréer le juge sous un autre nom (deploy --judge)")


def judge_responses(client: httpx.Client, responses: list[dict], judge: Config,
                    grid: list[Criterion], *, out_path, limit: int | None = None,
                    workers: int = 1, sleep=None) -> tuple[int, int]:
    """Un run de juge par réponse à noter. Renvoie (votes gardés, échecs) DE CE LANCEMENT.

    `responses` : la sortie de load_responses (une seule config).
    `limit` : ne considérer que les N premières réponses (au moins 1).
    `workers` : nombre de runs envoyés à Alive en même temps (1 à 32).
    `sleep` : fonction d'attente entre deux essais ; à ne passer que dans les tests.

    Lève ValueError, sans lancer de run, si une option est invalide, si le juge
    n'est pas celui de la grille, ou si out_path n'est pas le fichier de cette campagne.
    Lève l'erreur qui a arrêté la campagne (401, CampaignStopped,
    KeyboardInterrupt), une fois les runs en cours rangés.
    """
    # --- 1. Vérifier, sans rien dépenser -----------------------------------------
    if limit is not None and limit < 1:
        raise ValueError(f"limit doit valoir au moins 1 (reçu {limit})")
    if not 1 <= workers <= MAX_WORKERS:
        raise ValueError(f"workers doit être entre 1 et {MAX_WORKERS} (reçu {workers})")
    check_judge(judge, grid)

    # --- 2. Ce qui reste à noter (reprise) ----------------------------------------
    out_path = Path(out_path)
    done, executions = load_votes(out_path, responses, judge, grid)
    todo = [r for r in responses[:limit] if r["id"] not in done]
    if done:
        print(f"reprise : {len(done)} réponses déjà notées dans {out_path}, "
              f"{len(todo)} à faire", file=sys.stderr)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- 3. Une tentative = une réponse envoyée au juge, puis sa réponse lue ------
    # Tourne dans un fil secondaire : n'écrit rien, renvoie (réponse, run, erreur).
    stopping = threading.Event()
    if sleep is None:
        sleep = stopping.wait

    def attempt(response: dict):
        try:
            run = run_with_retries(client, judge.room, judge.deployment,
                                   user_message(response["question"], response["response"]),
                                   judge.model, judge.reasoning_effort, sleep=sleep,
                                   give_up=stopping.is_set)
            try:
                verdicts = parse_verdicts(run["content"], grid)
            except JudgeAnswerError as err:
                # Pour comprendre après coup : le run (payé) et le début de sa réponse.
                err.detail = f"{run['execution_id']} : {run['content'][:1000]}"
                raise
        except (RunRejected, httpx.HTTPStatusError, httpx.TransportError) as err:
            return response, None, err
        return response, {**run, "verdicts": verdicts}, None

    # --- 4. Lancer : pool distribue, recorder range ---------------------------------
    recorder = Recorder(judge, out_path, executions, total=len(todo),
                        make_record=lambda response, run, config: make_vote(response, run, config, grid))
    run_tasks(todo, attempt, recorder.record, workers=workers, stopping=stopping)

    if recorder.stop is not None:
        raise recorder.stop
    return recorder.n_ok, recorder.n_failed


# --- Ligne de commande -----------------------------------------------------------------------

def main(argv: list[str] | None = None, transport: httpx.BaseTransport | None = None,
         sleep=None, env_file=".env") -> int:
    parser = argparse.ArgumentParser(description="Fait noter les réponses d'une config par un juge.")
    parser.add_argument("--judge", required=True, help="nom d'un juge de configs.json, ex. juge_qwen")
    parser.add_argument("--responses", required=True, help="réponses à noter, ex. outputs/A.jsonl")
    parser.add_argument("--out", default=None,
                        help="défaut : <dossier des réponses>/jury/<réponses>.<juge>.jsonl")
    parser.add_argument("--configs", default="configs.json", help="fichier des configs")
    parser.add_argument("--questions", default="data/healthbench.jsonl", help="JSONL {id, text}")
    parser.add_argument("--grid", default="data/criteria.json", help="la grille des critères")
    parser.add_argument("--limit", type=int, default=None, help="ne noter que les N premières réponses")
    parser.add_argument("--workers", type=int, default=1, help="runs envoyés en même temps (1 à 32, défaut 1)")
    args = parser.parse_args(argv)
    out_path = Path(args.out) if args.out else default_out_path(args.responses, args.judge)

    token = find_token(env_file)
    if not token:
        print(TOKEN_HELP, file=sys.stderr)
        return 2

    with make_client(token, transport) as client:
        try:
            judge = load_config(args.configs, args.judge)
            grid = load_grid(args.grid)
            responses = load_responses(args.responses, load_questions(args.questions))
            n_ok, n_failed = judge_responses(client, responses, judge, grid, out_path=out_path,
                                             limit=args.limit, workers=args.workers, sleep=sleep)
        except (ValueError, OSError, KeyError) as err:
            print(f"REFUS : {err}", file=sys.stderr)
            return 2
        except (CampaignStopped, httpx.HTTPStatusError) as err:
            print(f"ARRÊT : {err}", file=sys.stderr)
            return 1
        except KeyboardInterrupt:
            print("INTERROMPU : relancer la même commande pour reprendre.", file=sys.stderr)
            return 130
    print(f"{n_ok} réponses notées, {n_failed} échecs à ce lancement -> {out_path}")
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
