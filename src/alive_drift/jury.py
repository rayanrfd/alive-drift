"""Fait noter les réponses d'UNE config par UN juge.

Pour chaque réponse : un run Alive sur le déploiement du juge (le message est
la question et la réponse, voir judge.py), puis une ligne écrite tout de suite
dans le fichier de sortie, avec un verdict par critère (true, false, ou null
pour "non applicable").

C'est generate.py avec une autre entrée et une autre ligne de sortie : même
parallélisme (pool.py), mêmes réessais (client.py), même arrêt après 5 échecs
d'affilée (recorder.py), même reprise en relançant la commande. Les fichiers
(réponses à noter, lignes de vote, reprise) sont lus et écrits par votes.py.

Un fichier par (config notée, juge) :

    outputs/jury/A.juge_qwen.jsonl          les votes gardés
    outputs/jury/A.juge_qwen.errors.jsonl   les runs ratés ou inutilisables

Le jury complet = 3 configs x 3 juges = 9 commandes.

Avant le premier run, on vérifie sans rien dépenser que le juge porte bien le
prompt fabriqué depuis la grille de travail d'aujourd'hui (data/criteria.json
réduit par data/selection.json ; empreinte dans configs.json).

Exemple sur 3 réponses (3 runs payants) :

    uv run python -m alive_drift.jury --judge juge_qwen --responses outputs/A.jsonl --limit 3
"""

import argparse
import sys
import threading
from pathlib import Path

import httpx

from alive_drift.check import RunRejected
from alive_drift.client import TOKEN_HELP, find_token, make_client, run_with_retries
from alive_drift.config import Config, load_config
from alive_drift.files import load_questions
from alive_drift.grid import Criterion, load_working_grid
from alive_drift.judge import (
    JudgeAnswerError, parse_verdicts, prompt_fingerprint, system_prompt, user_message,
)
from alive_drift.pool import MAX_WORKERS, run_tasks
from alive_drift.recorder import CampaignStopped, Recorder
from alive_drift.votes import default_out_path, load_responses, load_votes, make_vote


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
    parser.add_argument("--grid", default="data/criteria.json", help="les 34 critères")
    parser.add_argument("--selection", default="data/selection.json", help="les critères envoyés au juge")
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
            grid = load_working_grid(args.grid, args.selection)
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
