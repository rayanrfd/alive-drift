"""Retrouver l'id du message (mg-...) de chaque réponse d'un fichier de campagne.

Pour qu'un juge réponde SOUS une réponse dans Alive (jury « dans le fil »), le run du juge
doit avoir pour parent l'id du MESSAGE de cette réponse (mg-...), dans la même room :
l'execution_id (ex-...) est refusé (sonde du 19/10). Nos fichiers de réponses ne gardent
que l'execution_id ; GET /executions/{execution_id} donne le message_id.
Une lecture par réponse : aucun run, rien d'écrit sur Alive, gratuit.

    uv run python -m alive_drift.message_ids outputs/L.jsonl --limit 3
    uv run python -m alive_drift.message_ids outputs/L.jsonl

écrit outputs/L.messages.jsonl, une ligne par réponse :
{question_id, config, execution_id, message_id, thread_id}.
Reprise : relancer la même commande saute les réponses déjà faites. Rien n'est écrasé.
"""

import argparse
import sys
import time
from pathlib import Path

import httpx

from alive_drift.client import RETRY_DELAYS_S, TOKEN_HELP, find_token, is_retryable, make_client
from alive_drift.files import alive_detail, append_line, read_records


def messages_path_for(responses_path) -> Path:
    """outputs/L.jsonl -> outputs/L.messages.jsonl"""
    path = Path(responses_path)
    return path.with_name(path.stem + ".messages.jsonl")


def read_responses(path) -> list[dict]:
    """Les réponses d'un fichier de generate, toutes de la même config.

    Lève ValueError si le fichier est vide, si une ligne n'a pas question_id, config,
    execution_id et room_id (textes), si une question apparaît deux fois ou si les
    lignes viennent de plusieurs configs ; FileNotFoundError s'il n'existe pas.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} : fichier de réponses introuvable")
    rows, seen = [], set()
    for where, row in read_records(path):
        for key in ("question_id", "config", "execution_id", "room_id"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise ValueError(f"{where} : champ {key!r} absent")
        if row["question_id"] in seen:
            raise ValueError(f"{where} : question {row['question_id']!r} en double")
        seen.add(row["question_id"])
        rows.append(row)
    if not rows:
        raise ValueError(f"{path} : aucune réponse")
    configs = {r["config"] for r in rows}
    if len(configs) != 1:
        raise ValueError(f"{path} : plusieurs configs {sorted(configs)}")
    return rows


def fetch_execution(client: httpx.Client, execution_id: str, sleep=time.sleep) -> dict:
    """GET /executions/{execution_id}, réessayé si Alive ou le réseau flanche (429, 5xx)."""
    attempts = len(RETRY_DELAYS_S) + 1
    for attempt in range(1, attempts + 1):
        try:
            response = client.get(f"/executions/{execution_id}")
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPStatusError, httpx.TransportError) as err:
            if attempt == attempts or not is_retryable(err):
                raise
            sleep(RETRY_DELAYS_S[attempt - 1])
    raise AssertionError("inatteignable")


def message_record(response: dict, execution: dict) -> dict:
    """La ligne à écrire, après vérification que l'exécution est bien celle de la réponse.

    Lève ValueError si l'execution_id ne correspond pas, si le run n'est pas
    « completed », si message_id n'est pas un mg-..., ou si le message n'est pas dans
    la room de la réponse (un juge ne pourrait alors pas y répondre).
    """
    ex, qid = response["execution_id"], response["question_id"]
    where = f"question {qid} ({ex})"
    if execution.get("execution_id") != ex:
        raise ValueError(f"{where} : Alive renvoie l'exécution {execution.get('execution_id')!r}")
    if execution.get("status") != "completed":
        raise ValueError(f"{where} : statut {execution.get('status')!r}, pas completed")
    message_id = execution.get("message_id")
    if not isinstance(message_id, str) or not message_id.startswith("mg-"):
        raise ValueError(f"{where} : message_id {message_id!r} n'est pas un id de message")
    if execution.get("thread_id") != response["room_id"]:
        raise ValueError(f"{where} : message dans la room {execution.get('thread_id')!r}, "
                         f"réponse générée dans {response['room_id']!r}")
    return {"question_id": qid, "config": response["config"], "execution_id": ex,
            "message_id": message_id, "thread_id": execution["thread_id"]}


def load_done(out_path: Path, responses: list[dict]) -> set[str]:
    """Les questions déjà faites dans le fichier de sortie, pour la reprise.

    Lève ValueError si le fichier contient une autre config, une question en double,
    ou une ligne dont l'execution_id n'est pas celui de la réponse (autre campagne).
    """
    by_qid = {r["question_id"]: r for r in responses}
    done = set()
    for where, row in read_records(out_path):
        qid = row.get("question_id")
        if qid not in by_qid or row.get("config") != by_qid[qid]["config"]:
            raise ValueError(f"{where} : ligne d'une autre campagne (config ou question inconnue)")
        if row.get("execution_id") != by_qid[qid]["execution_id"]:
            raise ValueError(f"{where} : execution_id différent de celui de la réponse")
        if qid in done:
            raise ValueError(f"{where} : question {qid!r} en double")
        done.add(qid)
    return done


def main(argv: list[str] | None = None, transport: httpx.BaseTransport | None = None,
         sleep=time.sleep, env_file=".env") -> int:
    parser = argparse.ArgumentParser(
        description="Retrouve l'id du message (mg-...) de chaque réponse. Lectures seules, gratuit.")
    parser.add_argument("responses", help="fichier de réponses de generate, ex. outputs/L.jsonl")
    parser.add_argument("--out", default=None, help="défaut : outputs/<config>.messages.jsonl")
    parser.add_argument("--limit", type=int, default=None, help="ne traiter que N réponses de plus")
    args = parser.parse_args(argv)
    out = Path(args.out) if args.out else messages_path_for(args.responses)
    try:
        responses = read_responses(args.responses)
        done = load_done(out, responses)
    except (ValueError, OSError) as err:
        print(f"REFUS : {err}", file=sys.stderr)
        return 2
    todo = [r for r in responses if r["question_id"] not in done]
    if args.limit is not None:
        todo = todo[:args.limit]
    token = find_token(env_file)
    if not token:
        print(TOKEN_HELP, file=sys.stderr)
        return 2
    written = 0
    with make_client(token, transport) as client:
        for response in todo:
            try:
                record = message_record(response, fetch_execution(client, response["execution_id"], sleep))
            except (httpx.HTTPStatusError, httpx.TransportError) as err:
                detail = alive_detail(err)
                print(f"ARRÊT : question {response['question_id']} : {err}"
                      + (f" ({detail})" if detail else ""), file=sys.stderr)
                break
            except ValueError as err:
                print(f"ARRÊT : {err}", file=sys.stderr)
                break
            append_line(out, record)
            written += 1
    total = len(done) + written
    print(f"{written} ids écrits, {len(done)} déjà là : {total}/{len(responses)} -> {out}")
    return 2 if written < len(todo) else 0     # arrêté avant la fin : code 2


if __name__ == "__main__":
    sys.exit(main())
