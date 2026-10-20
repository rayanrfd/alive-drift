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
réduit par data/selection.json) : d'abord l'empreinte notée dans configs.json,
puis le prompt réellement appliqué chez Alive (lecture de la liste des
déploiements de la room).

Exemple sur 3 réponses (3 runs payants) :

    uv run python -m alive_drift.jury --judge juge_qwen --responses outputs/A.jsonl --limit 3

Jury « dans le fil » (morceaux 34 et 34bis) : le juge répond SOUS chaque réponse, dans la
room de la campagne (juge créé par deploy --judge-thread --in-room-of, en
conversation=transcript). Le run part SANS message (input null), avec pour parent l'id du
message de la réponse : enregistré par generate depuis le morceau 37 ; pour les réponses plus
anciennes, lu dans le fichier écrit par message_ids (défaut outputs/<config>.messages.jsonl) :
Alive envoie au juge la question et la réponse comme un texte, et son vote s'affiche juste
sous la réponse. --skip N saute les N premières réponses :

    uv run python -m alive_drift.jury --judge juge_haiku_fil2_L --responses outputs/L.jsonl \\
        --in-thread --skip 4 --limit 3

Les votes ont exactement le même format (plus "mode": "thread" et "parent_message_id"),
donc scores et Richter s'en servent sans changement.
"""

import argparse
import sys
import threading
from pathlib import Path

import httpx

from alive_drift.check import RunRejected
from alive_drift.client import TOKEN_HELP, find_token, make_client, run_with_retries
from alive_drift.config import Config, load_config
from alive_drift.deploy import RECIPE, THREAD_CONVERSATION, DeployError, check_deployment
from alive_drift.files import load_questions, read_records
from alive_drift.grid import Criterion, load_working_grid
from alive_drift.judge import (
    JudgeAnswerError, parse_verdicts, prompt_fingerprint, system_prompt,
    system_prompt_thread, user_message,
)
from alive_drift.message_ids import messages_path_for
from alive_drift.pool import MAX_WORKERS, run_tasks
from alive_drift.recorder import CampaignStopped, Recorder
from alive_drift.votes import default_out_path, load_responses, load_votes, make_vote


# --- La boucle ------------------------------------------------------------------------------

def judge_prompt_for(grid: list[Criterion], thread: bool) -> str:
    """Le prompt système attendu : celui des juges, ou celui des juges « dans le fil »."""
    return system_prompt_thread(grid) if thread else system_prompt(grid)


def check_judge(judge: Config, grid: list[Criterion], thread: bool = False) -> None:
    """Lève ValueError si `judge` n'est pas un juge, ou pas celui de cette grille.

    Le prompt du juge est figé chez Alive depuis sa création. Si la grille ou
    les consignes ont changé depuis, ses numéros de critères ne correspondent
    plus aux nôtres : on refuse avant de dépenser quoi que ce soit.
    """
    if judge.judge_prompt is None:
        raise ValueError(f"la config {judge.name!r} n'est pas un juge (pas de 'judge_prompt')")
    current = prompt_fingerprint(judge_prompt_for(grid, thread))
    if judge.judge_prompt != current:
        kind = "dans le fil (--judge-thread)" if thread else "(--judge)"
        raise ValueError(
            f"le juge {judge.name!r} a été créé avec le prompt {judge.judge_prompt}, "
            f"un juge {kind} avec la grille et les consignes d'aujourd'hui a {current} : "
            "mauvais mode (--in-thread ?) ou juge à recréer sous un autre nom")


def check_judge_at_alive(client: httpx.Client, judge: Config, grid: list[Criterion],
                         thread: bool = False) -> None:
    """Demande à Alive les réglages du déploiement du juge et refuse s'ils ne sont pas les bons.

    check_judge compare deux choses qui sont chez nous (configs.json et la
    grille). Ici on lit chez Alive (lecture seule, gratuite) le prompt que le
    déploiement applique vraiment, avec son modèle, son raisonnement et la
    recette de mesure. Si quelqu'un a modifié le juge dans l'interface d'Alive,
    on le voit avant le premier run.

    Lève ValueError si le déploiement est introuvable (ou absent de la première page
    d'une liste paginée : la suite n'est pas lue) ou si un réglage diffère,
    httpx.HTTPStatusError si Alive refuse la lecture (401...).
    """
    response = client.get(f"/threads/{judge.room}/deployments")
    response.raise_for_status()
    body = response.json()
    deployments = body.get("data") if isinstance(body, dict) else None
    if not isinstance(deployments, list):
        raise ValueError(f"room {judge.room} : liste des déploiements illisible chez Alive")
    found = [d for d in deployments if isinstance(d, dict) and d.get("id") == judge.deployment]
    if not found:
        # La liste est paginée (next_after) ; on ne lit que la première page tant que le
        # nom du paramètre de page d'Alive n'est pas confirmé. On le dit au lieu de
        # prétendre que le déploiement n'existe pas.
        if body.get("next_after") is not None:
            raise ValueError(
                f"déploiement {judge.deployment} absent de la première page des déploiements "
                f"de la room {judge.room} ; la liste continue (next_after={body['next_after']!r}) "
                "mais la page suivante n'est pas lue")
        raise ValueError(f"déploiement {judge.deployment} introuvable dans la room {judge.room} chez Alive")
    try:
        # L'agent n'est pas noté dans configs.json : on ne peut pas le comparer.
        check_deployment(found[0], deployment_id=judge.deployment, agent=found[0].get("agent_id"),
                         model=judge.model, system_prompt=judge_prompt_for(grid, thread),
                         reasoning_effort=judge.reasoning_effort,
                         conversation=THREAD_CONVERSATION if thread else RECIPE["conversation"])
    except DeployError as err:
        raise ValueError(f"le juge {judge.name!r} chez Alive n'est pas celui attendu : {err}") from None


def load_thread_parents(path, responses: list[dict], room: str) -> dict[str, str]:
    """{question: id du message de la réponse}, lu dans le fichier de message_ids.

    Lève FileNotFoundError si le fichier manque (lancer message_ids d'abord) ;
    ValueError si une ligne ne correspond pas à la réponse (autre execution_id) ou si
    le message n'est pas dans la room du juge (Alive refuserait le parent).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} introuvable : lancer d'abord "
                                f"python -m alive_drift.message_ids <réponses>")
    by_id = {r["id"]: r for r in responses}
    parents = {}
    for where, row in read_records(path):
        qid = row.get("question_id")
        if qid not in by_id:
            continue
        if row.get("execution_id") != by_id[qid]["execution_id"]:
            raise ValueError(f"{where} : execution_id différent de celui de la réponse {qid!r}")
        if row.get("thread_id") != room:
            raise ValueError(f"{where} : réponse {qid!r} dans la room {row.get('thread_id')!r}, "
                             f"le juge est dans {room!r}")
        message_id = row.get("message_id")
        if not isinstance(message_id, str) or not message_id.startswith("mg-"):
            raise ValueError(f"{where} : message_id {message_id!r} invalide")
        parents[qid] = message_id
    return parents


def thread_parents(responses: list[dict], room: str, messages_path) -> dict[str, str]:
    """{question: id du message de la réponse} pour le jury « dans le fil ».

    Une réponse qui a son message_id (generate, depuis le morceau 37) le donne directement.
    Pour les autres (campagnes plus anciennes), on lit le fichier de message_ids : il n'est
    donc nécessaire que s'il reste des réponses sans message_id.

    Lève ValueError si un message_id n'est pas un id de message ("mg-...") ou si la réponse
    n'est pas dans la room du juge (Alive refuserait le parent) ; FileNotFoundError (via
    load_thread_parents) s'il faut le fichier et qu'il manque.
    """
    parents, missing = {}, []
    for response in responses:
        message_id = response.get("message_id")
        if message_id is None:
            missing.append(response)
            continue
        if not isinstance(message_id, str) or not message_id.startswith("mg-"):
            raise ValueError(f"réponse {response['id']!r} : message_id {message_id!r} invalide")
        if response.get("room_id") != room:
            raise ValueError(f"réponse {response['id']!r} dans la room {response.get('room_id')!r}, "
                             f"le juge est dans {room!r}")
        parents[response["id"]] = message_id
    if missing:
        parents.update(load_thread_parents(messages_path, missing, room))
    return parents


def judge_responses(client: httpx.Client, responses: list[dict], judge: Config,
                    grid: list[Criterion], *, out_path, limit: int | None = None,
                    skip: int = 0, workers: int = 1, sleep=None,
                    parents: dict[str, str] | None = None) -> tuple[int, int]:
    """Un run de juge par réponse à noter. Renvoie (votes gardés, échecs) DE CE LANCEMENT.

    `responses` : la sortie de load_responses (une seule config).
    `limit` : ne considérer que les N premières réponses (au moins 1), après `skip`.
    `skip` : sauter les N premières réponses (0 par défaut).
    `workers` : nombre de runs envoyés à Alive en même temps (1 à 32).
    `sleep` : fonction d'attente entre deux essais ; à ne passer que dans les tests.
    `parents` : jury « dans le fil » ({question: id du message de la réponse}) : runs sans
    message, sous la réponse. None : jury habituel (question et réponse dans le message).

    Lève ValueError, sans lancer de run, si une option est invalide, si le juge
    n'est pas celui de la grille (dans configs.json ou chez Alive), ou si
    out_path n'est pas le fichier de cette campagne.
    Lève l'erreur qui a arrêté la campagne (401, CampaignStopped,
    KeyboardInterrupt), une fois les runs en cours rangés.
    """
    # --- 1. Vérifier, sans rien dépenser -----------------------------------------
    if limit is not None and limit < 1:
        raise ValueError(f"limit doit valoir au moins 1 (reçu {limit})")
    if skip < 0:
        raise ValueError(f"skip doit valoir au moins 0 (reçu {skip})")
    if not 1 <= workers <= MAX_WORKERS:
        raise ValueError(f"workers doit être entre 1 et {MAX_WORKERS} (reçu {workers})")
    thread = parents is not None
    check_judge(judge, grid, thread)
    check_judge_at_alive(client, judge, grid, thread)

    # --- 2. Ce qui reste à noter (reprise) ----------------------------------------
    out_path = Path(out_path)
    done, executions = load_votes(out_path, responses, judge, grid, parents)
    todo = [r for r in responses[skip:][:limit] if r["id"] not in done]
    if done:
        print(f"reprise : {len(done)} réponses déjà notées dans {out_path}, "
              f"{len(todo)} à faire", file=sys.stderr)
    if thread:
        missing = [r["id"] for r in todo if r["id"] not in parents]
        if missing:
            raise ValueError(f"{len(missing)} réponses sans id de message (ex. {missing[:3]}) : "
                             "relancer message_ids sur ce fichier de réponses")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- 3. Une tentative = une réponse envoyée au juge, puis sa réponse lue ------
    # Tourne dans un fil secondaire : n'écrit rien, renvoie (réponse, run, erreur).
    stopping = threading.Event()
    if sleep is None:
        sleep = stopping.wait

    def attempt(response: dict):
        try:
            message = (None if thread  # dans le fil : pas de message, Alive envoie la branche
                       else user_message(response["question"], response["response"]))
            run = run_with_retries(client, judge.room, judge.deployment, message,
                                   judge.model, judge.reasoning_effort, sleep=sleep,
                                   give_up=stopping.is_set, served_model=judge.served_model,
                                   parent_id=parents[response["id"]] if thread else None)
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
    def make_record(response, run, config):
        vote = make_vote(response, run, config, grid)
        if thread:
            vote.update(mode="thread", parent_message_id=parents[response["id"]])
        return vote

    recorder = Recorder(judge, out_path, executions, total=len(todo), make_record=make_record)
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
    parser.add_argument("--skip", type=int, default=0, help="sauter les N premières réponses (défaut 0)")
    parser.add_argument("--workers", type=int, default=1, help="runs envoyés en même temps (1 à 32, défaut 1)")
    parser.add_argument("--in-thread", action="store_true",
                        help="jury « dans le fil » : le juge répond sous chaque réponse (voir message_ids)")
    parser.add_argument("--messages", default=None,
                        help="avec --in-thread, pour des réponses sans message_id : fichier de "
                             "message_ids (défaut : <réponses>.messages.jsonl)")
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
            parents = None
            if args.in_thread:
                parents = thread_parents(responses, judge.room,
                                         args.messages or messages_path_for(args.responses))
            n_ok, n_failed = judge_responses(client, responses, judge, grid, out_path=out_path,
                                             limit=args.limit, skip=args.skip,
                                             workers=args.workers, sleep=sleep,
                                             parents=parents)
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
