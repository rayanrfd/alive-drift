"""MedAgentBench sur Alive : faire passer les tâches à un agent, tour par tour (morceau 31).

Reproduit la boucle de MedAgentBench (src/server/tasks/medagentbench/__init__.py,
MedAgentBench.start_sample, commit 9926011, licence MIT) :

1. premier message : consigne, 9 fonctions FHIR, contexte et question (tasks.py) ;
2. au plus 8 tours (configs/tasks/medagentbench.yaml : max_round: 8). À chaque tour,
   l'agent répond et on lit sa réponse (actions.py) :
   - GET    -> on interroge le serveur FHIR (fhir.py) et on lui renvoie le résultat ;
   - POST   -> rien n'est envoyé au serveur : « POST request accepted... » (ou
               « Invalid POST request » si son JSON est illisible), et on continue ;
   - FINISH -> fin, statut « completed », on garde sa réponse ;
   - autre  -> fin, statut « agent invalid action » ;
3. après 8 tours sans FINISH : statut « task limit reached ».

Sur Alive, une tâche = une conversation dans la room de la config. Le 1er tour démarre
une branche neuve ; chaque tour suivant envoie notre phrase (résultat du GET...) comme
input, sous le dernier message de l'agent. Alive crée un message utilisateur pour cet
input, puis la réponse de l'agent dessous (parent_via_input). En conversation
« messages », l'agent voit toute la branche, ses réponses comme les siennes : c'est la
conversation de MedAgentBench.

Écarts volontaires (on ne compte jamais une panne de notre côté comme un échec de l'agent) :
- serveur FHIR injoignable (ServerUnreachable) : toute la campagne s'arrête ;
- run rejeté (mauvais modèle, fallback...) ou erreur d'Alive après réessais : la tâche
  n'est PAS enregistrée comme résultat ; elle va dans le journal d'erreurs et sera
  reprise de zéro au prochain lancement (chez eux : statut « task error »).
  Après 5 tâches en erreur d'affilée, on arrête.
- erreur imprévue dans une tâche (un bug, une réponse d'Alive d'une forme inconnue) :
  journal d'erreurs, puis arrêt propre (les tâches en cours vont au bout, ce qui est
  déjà écrit reste), au lieu d'un plantage qui perd les tâches en cours.

Pas de correction ici (morceau 32 : refsol.py). Chaque ligne garde tout l'historique,
avec les rôles « user » / « agent » de MedAgentBench, que lit refsol.py.

Fichiers (comme generate) :
- <out>               : une ligne par tâche terminée ; reprise : les tâches déjà là sont sautées ;
- <out>.errors.jsonl  : les tâches interrompues, avec la raison.

    uv run python -m alive_drift.mab.loop --config mab_g54mini --out outputs/mab/mab_g54mini.jsonl --task task1_1
    uv run python -m alive_drift.mab.loop --config mab_g54mini --out outputs/mab/mab_g54mini.jsonl --per-type 1
"""

import argparse
import sys
import threading
from pathlib import Path
from typing import Any, Callable

import httpx

from alive_drift.check import RunRejected
from alive_drift.client import TOKEN_HELP, find_token, is_retryable, make_client, run_with_retries
from alive_drift.config import Config, load_config
from alive_drift.files import (alive_detail, append_line, errors_path_for, message_id_of,
                               now_utc, read_records)
from alive_drift.mab.actions import (FINISH, GET, POST, parse_reply, reply_after_get,
                                     reply_after_get_error, reply_after_post)
from alive_drift.mab.fhir import ServerUnreachable, check_server, fhir_get
from alive_drift.mab.tasks import (FHIR_API_BASE, FUNCS_PATH, TASKS_PATH, Task, first_message,
                                   load_funcs, load_tasks, tasks_fingerprint)
from alive_drift.pool import MAX_WORKERS, run_tasks
from alive_drift.recorder import MAX_CONSECUTIVE_FAILURES, CampaignStopped

MAX_ROUNDS = 8

# Les statuts de MedAgentBench (src/typings/status.py, SampleStatus), mêmes textes.
COMPLETED = "completed"
INVALID_ACTION = "agent invalid action"
LIMIT_REACHED = "task limit reached"


class TaskInterrupted(Exception):
    """Une tâche n'a pas pu aller au bout à cause d'Alive (run rejeté, erreur HTTP...)."""

    def __init__(self, round_no: int, cause: Exception):
        super().__init__(f"tour {round_no} : {type(cause).__name__} : {cause}")
        self.round_no, self.cause = round_no, cause


def run_summary(round_no: int, run: dict) -> dict:
    """Ce qu'on garde d'un run (le texte de la réponse est dans l'historique)."""
    served = run["model"]["served"][0]
    tokens = run.get("tokens") if isinstance(run.get("tokens"), dict) else {}
    return {"round": round_no, "execution_id": run.get("execution_id"),
            "message_id": message_id_of(run), "parent_id": run.get("parent_id"),
            "model": served["model"], "provider": served["provider"],
            "prompt_tokens": tokens.get("prompt"), "completion_tokens": tokens.get("completion"),
            "reasoning_tokens": tokens.get("reasoning"), "credits_used": run.get("credits_used"),
            "duration_ms": run.get("duration_ms")}


def play_task(task: Task, funcs: list[dict], send: Callable[[str, str | None], dict],
              fhir: Callable[[str], dict], api_base: str = FHIR_API_BASE,
              max_rounds: int = MAX_ROUNDS) -> dict:
    """Une tâche, du premier message à la fin. Renvoie {status, result, history, runs}.

    send(texte, parent) : un run Alive vérifié (parent None au 1er tour, sinon le message
    de la dernière réponse de l'agent). fhir(url) : un GET, comme fhir.fhir_get.
    Lève TaskInterrupted si un run échoue, ServerUnreachable si le serveur FHIR tombe.
    """
    text = first_message(task, funcs, api_base)
    history = [{"role": "user", "content": text}]
    runs, parent = [], None
    for round_no in range(1, max_rounds + 1):
        try:
            run = send(text, parent)
        except (RunRejected, httpx.HTTPStatusError, httpx.TransportError) as err:
            raise TaskInterrupted(round_no, err) from err
        runs.append(run_summary(round_no, run))
        history.append({"role": "agent", "content": run["content"]})
        action = parse_reply(run["content"])
        if action.kind == FINISH:
            return {"status": COMPLETED, "result": action.answer, "history": history, "runs": runs}
        if action.kind == GET:
            got = fhir(action.url)          # ServerUnreachable remonte : on arrête tout
            text = reply_after_get(got["data"]) if "data" in got else reply_after_get_error(got["error"])
        elif action.kind == POST:
            text = reply_after_post(action)
        else:
            return {"status": INVALID_ACTION, "result": None, "history": history, "runs": runs}
        history.append({"role": "user", "content": text})
        parent = message_id_of(run)
        if parent is None:
            raise TaskInterrupted(round_no, RunRejected(f"réponse sans id de message : {run.get('id')!r}"))
    return {"status": LIMIT_REACHED, "result": None, "history": history, "runs": runs}


def make_record(task: Task, outcome: dict, config: Config, api_base: str, fingerprint: dict,
                max_rounds: int) -> dict:
    credits = [r["credits_used"] for r in outcome["runs"] if isinstance(r["credits_used"], (int, float))]
    return {"task_id": task.id, "task_type": task.type, "config": config.name,
            "room_id": config.room, "deployment_id": config.deployment, "model": config.model,
            "reasoning_effort": config.reasoning_effort, "status": outcome["status"],
            "result": outcome["result"], "n_rounds": len(outcome["runs"]), "max_rounds": max_rounds,
            "credits_used": round(sum(credits), 6), "api_base": api_base,
            "fingerprint": fingerprint, "history": outcome["history"], "runs": outcome["runs"],
            "written_at": now_utc()}


def load_done(out_path: Path, config: Config, fingerprint: dict, task_ids: set[str]) -> tuple[set[str], set[str]]:
    """(tâches déjà faites, execution_id déjà vus). Lève ValueError si le fichier n'est
    pas celui de cette campagne (autre config, room, déploiement, modèle, effort ou tâches)."""
    done, executions = set(), set()
    for where, rec in read_records(out_path):
        for field, expected in (("config", config.name), ("room_id", config.room),
                                ("deployment_id", config.deployment), ("model", config.model), ("reasoning_effort", config.reasoning_effort),
                                ("fingerprint", fingerprint)):
            if rec.get(field) != expected:
                raise ValueError(f"{where} : {field} {rec.get(field)!r}, attendu {expected!r}")
        tid = rec.get("task_id")
        if tid not in task_ids:
            raise ValueError(f"{where} : tâche {tid!r} inconnue")
        if tid in done:
            raise ValueError(f"{where} : tâche {tid!r} en double")
        done.add(tid)
        executions.update(r.get("execution_id") for r in rec.get("runs", []))
    return done, executions


def select_tasks(tasks: list[Task], task_id: str | None = None, limit: int | None = None,
                 per_type: int | None = None) -> list[Task]:
    """Les tâches de ce lancement : une seule, les N premières, ou les K premières de chaque type."""
    if task_id is not None:
        chosen = [t for t in tasks if t.id == task_id]
        if not chosen:
            raise ValueError(f"tâche {task_id!r} introuvable")
        return chosen
    if limit is not None and limit < 1:
        raise ValueError(f"limit doit valoir au moins 1 (reçu {limit})")
    if per_type is not None:
        if per_type < 1:
            raise ValueError(f"per-type doit valoir au moins 1 (reçu {per_type})")
        seen: dict[str, int] = {}
        chosen = []
        for t in tasks:
            if seen.get(t.type, 0) < per_type:
                seen[t.type] = seen.get(t.type, 0) + 1
                chosen.append(t)
        return chosen
    return tasks[:limit]


def run_campaign(client: httpx.Client, config: Config, tasks: list[Task], funcs: list[dict], *,
                 out_path, fingerprint: dict, api_base: str = FHIR_API_BASE,
                 max_rounds: int = MAX_ROUNDS, workers: int = 1, fhir: Callable[[str], dict] | None = None,
                 sleep=None) -> tuple[int, int]:
    """Fait passer les tâches pas encore faites. Renvoie (tâches enregistrées, tâches en erreur).

    Lève ValueError sans lancer de run si une option est invalide ou si out_path n'est pas
    le fichier de cette campagne ; lève ServerUnreachable, CampaignStopped ou l'erreur
    d'Alive qui a arrêté la campagne, une fois les tâches en cours rangées.
    """
    if not 1 <= workers <= MAX_WORKERS:
        raise ValueError(f"workers doit être entre 1 et {MAX_WORKERS} (reçu {workers})")
    if max_rounds < 1:
        raise ValueError(f"max_rounds doit valoir au moins 1 (reçu {max_rounds})")
    out_path = Path(out_path)
    done, executions = load_done(out_path, config, fingerprint, {t.id for t in tasks})
    todo = [t for t in tasks if t.id not in done]
    if done:
        print(f"reprise : {len(done)} tâches déjà dans {out_path}, {len(todo)} à faire", file=sys.stderr)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    errors_path = errors_path_for(out_path)
    fhir = fhir or (lambda url: fhir_get(url, api_base))

    stopping = threading.Event()
    if sleep is None:
        sleep = stopping.wait

    def send(text: str, parent: str | None) -> dict:
        return run_with_retries(client, config.room, config.deployment, text, config.model,
                                config.reasoning_effort, sleep=sleep, give_up=stopping.is_set,
                                served_model=config.served_model, parent_id=parent,
                                parent_via_input=parent is not None)

    def attempt(task: Task):
        try:
            return task, play_task(task, funcs, send, fhir, api_base, max_rounds), None
        except (TaskInterrupted, ServerUnreachable) as err:
            return task, None, err
        except Exception as err:            # un bug ou une réponse imprévue : on arrête proprement
            stop = CampaignStopped(f"erreur imprévue dans {task.id} : {type(err).__name__}: {err}")
            return task, None, TaskInterrupted(0, stop)

    state = {"ok": 0, "failed": 0, "consecutive": 0, "stop": None}

    def handle(result) -> bool:
        task, outcome, err = result
        if err is None:
            ids = {r["execution_id"] for r in outcome["runs"]}
            if ids & executions:
                err = TaskInterrupted(0, RunRejected(f"execution_id déjà enregistré : {sorted(ids & executions)}"))
            else:
                append_line(out_path, make_record(task, outcome, config, api_base, fingerprint, max_rounds))
                executions.update(ids)
                state["ok"] += 1
                state["consecutive"] = 0
                print(f"[{state['ok'] + state['failed']}/{len(todo)}] {task.id} : {outcome['status']} "
                      f"en {len(outcome['runs'])} tour(s)", file=sys.stderr)
                return False
        if isinstance(err, ServerUnreachable):
            state["stop"] = state["stop"] or err
            return True
        cause = err.cause
        if not (isinstance(cause, RunRejected) or is_retryable(cause)):
            state["stop"] = state["stop"] or cause      # 401, 404... : chaque tâche échouerait pareil
        append_line(errors_path, {"task_id": task.id, "config": config.name,
                                  "deployment_id": config.deployment, "round": err.round_no,
                                  "error_type": type(cause).__name__, "error": str(cause),
                                  "detail": alive_detail(cause), "written_at": now_utc()})
        state["failed"] += 1
        state["consecutive"] += 1
        print(f"[{state['ok'] + state['failed']}/{len(todo)}] {task.id} : ERREUR {err}", file=sys.stderr)
        if state["consecutive"] >= MAX_CONSECUTIVE_FAILURES and state["stop"] is None:
            state["stop"] = CampaignStopped(f"{state['consecutive']} tâches en erreur d'affilée")
        return state["stop"] is not None

    run_tasks(todo, attempt, handle, workers=workers, stopping=stopping)
    if state["stop"] is not None:
        raise state["stop"]
    return state["ok"], state["failed"]


def main(argv: list[str] | None = None, transport: httpx.BaseTransport | None = None,
         fhir: Callable[[str], dict] | None = None, server_ok: Callable[[str], bool] | None = None,
         sleep=None, env_file=".env") -> int:
    parser = argparse.ArgumentParser(description="Faire passer les tâches MedAgentBench à un agent sur Alive.")
    parser.add_argument("--config", required=True, help="config de l'agent dans configs.json")
    parser.add_argument("--configs", default="configs.json")
    parser.add_argument("--out", required=True, help="ex. outputs/mab/mab_g54mini.jsonl")
    parser.add_argument("--tasks", default=TASKS_PATH)
    parser.add_argument("--funcs", default=FUNCS_PATH)
    parser.add_argument("--api-base", default=FHIR_API_BASE, help="adresse du serveur FHIR")
    which = parser.add_mutually_exclusive_group()
    which.add_argument("--task", default=None, help="une seule tâche, ex. task1_1")
    which.add_argument("--limit", type=int, default=None, help="les N premières tâches du fichier")
    which.add_argument("--per-type", type=int, default=None, help="les K premières tâches de chaque type")
    parser.add_argument("--max-rounds", type=int, default=MAX_ROUNDS, help="tours au plus (défaut 8)")
    parser.add_argument("--workers", type=int, default=1, help="tâches en même temps (1 à 32)")
    args = parser.parse_args(argv)

    token = find_token(env_file)
    if not token:
        print(TOKEN_HELP, file=sys.stderr)
        return 2
    try:
        config = load_config(args.configs, args.config)
        tasks = select_tasks(load_tasks(args.tasks), args.task, args.limit, args.per_type)
        funcs = load_funcs(args.funcs)
        fingerprint = tasks_fingerprint(args.tasks, args.funcs)
    except (ValueError, OSError) as err:
        print(f"REFUS : {err}", file=sys.stderr)
        return 2
    if not (server_ok or check_server)(args.api_base):
        print(f"REFUS : le serveur FHIR {args.api_base} ne répond pas (Docker lancé ?)", file=sys.stderr)
        return 2
    with make_client(token, transport) as client:
        try:
            n_ok, n_failed = run_campaign(client, config, tasks, funcs, out_path=args.out,
                                          fingerprint=fingerprint, api_base=args.api_base,
                                          max_rounds=args.max_rounds, workers=args.workers,
                                          fhir=fhir, sleep=sleep)
        except (ValueError, OSError) as err:
            print(f"REFUS : {err}", file=sys.stderr)
            return 2
        except (ServerUnreachable, CampaignStopped, httpx.HTTPStatusError) as err:
            print(f"ARRÊT : {err}", file=sys.stderr)
            return 1
        except KeyboardInterrupt:
            print("INTERROMPU : relancer la même commande pour reprendre.", file=sys.stderr)
            return 130
    print(f"{n_ok} tâches enregistrées, {n_failed} en erreur à ce lancement -> {args.out}")
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
