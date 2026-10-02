"""Génère les réponses d'UNE config (A, A_bis ou B) pour une liste de questions.

Pour chaque question : un run Alive (run_with_retries), puis une ligne écrite
tout de suite dans le fichier de sortie. Une question après l'autre, pas encore
de reprise ni de parallélisme.

Deux fichiers JSONL (une ligne = un objet JSON) :
- <out>            : les réponses gardées ;
- <out>.errors     : les questions sans réponse, avec la raison.

Exemple sur 3 questions (3 runs payants, quelques millièmes de crédit) :

    export ALIVE_TOKEN=...
    uv run python -m alive_drift.generate --questions data/healthbench.jsonl \
        --room th-... --deployment dp-... --model openai/gpt-4.1-mini \
        --config A --out outputs/A.jsonl --limit 3
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx

from alive_drift.check import RunRejected
from alive_drift.client import is_retryable, make_client, run_with_retries

# Après 5 questions ratées d'affilée, on arrête tout : le problème est général
# (modèle en panne, déploiement mal réglé). Un run rejeté est facturé : sans
# cette limite, un mauvais réglage coûterait 1000 runs inutiles.
MAX_CONSECUTIVE_FAILURES = 5


class CampaignStopped(Exception):
    """Trop d'échecs d'affilée : la campagne est arrêtée."""


# --- Lecture des questions ---------------------------------------------------

def load_questions(path) -> list[dict]:
    """Lit un JSONL de questions {"id": ..., "text": "..."}.

    Renvoie une liste de {"id": texte, "text": texte}, dans l'ordre du fichier.
    Lève ValueError si une ligne est mal formée ou si un id est en double.
    """
    questions = []
    seen = set()
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, start=1):
            if not line.strip():
                continue  # ligne vide tolérée
            row = json.loads(line)
            qid, text = row.get("id"), row.get("text")
            # Le fichier HealthBench a des ids entiers (0, 1, 2...) : on en fait du texte.
            # `type(...) is int` et non isinstance : True/False sont aussi des int en Python.
            if type(qid) is int:
                qid = str(qid)
            if not isinstance(qid, str) or not qid:
                raise ValueError(f"{path} ligne {n} : 'id' manquant")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{path} ligne {n} : 'text' vide")
            if qid in seen:
                raise ValueError(f"{path} ligne {n} : id {qid!r} en double")
            seen.add(qid)
            questions.append({"id": qid, "text": text})
    return questions


# --- Écriture ----------------------------------------------------------------

def make_record(question: dict, run: dict, config: str) -> dict:
    """La ligne enregistrée pour une réponse gardée.

    `run` a passé check_run : le bloc model et served[0] existent forcément.
    """
    served = run["model"]["served"][0]
    tokens = run.get("tokens") or {}
    return {
        "question_id": question["id"],
        "config": config,
        "question": question["text"],
        "response": run["content"],
        "model": served["model"],
        "provider": served.get("provider"),
        "execution_id": run.get("execution_id"),
        "prompt_tokens": tokens.get("prompt"),
        "completion_tokens": tokens.get("completion"),
        "reasoning_tokens": tokens.get("reasoning"),
        "credits_used": run.get("credits_used"),
        "duration_ms": run.get("duration_ms"),
    }


def append_line(path: Path, record: dict) -> None:
    """Ajoute une ligne JSON à la fin du fichier, puis le ferme (donc sur disque)."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def errors_path_for(out_path: Path) -> Path:
    """outputs/A.jsonl -> outputs/A.errors.jsonl"""
    return out_path.with_name(out_path.stem + ".errors.jsonl")


# --- La boucle -----------------------------------------------------------------

def generate(client: httpx.Client, questions: list[dict], *, room_id: str,
             deployment_id: str, expected_model: str, config: str,
             out_path, sleep=time.sleep) -> tuple[int, int]:
    """Un run par question. Renvoie (nombre de réponses gardées, nombre d'échecs).

    - Réponse gardée  -> une ligne dans out_path, écrite immédiatement.
    - Run rejeté, ou échec passager après 3 essais -> une ligne dans le fichier
      d'erreurs, et on passe à la question suivante.
    - Erreur de notre côté (401, 404...) -> on arrête tout : chaque question
      échouerait de la même façon.
    - 5 échecs d'affilée -> CampaignStopped.

    Refuse de démarrer si un des deux fichiers existe : on n'écrase jamais.
    """
    out_path = Path(out_path)
    errors_path = errors_path_for(out_path)
    for path in (out_path, errors_path):
        if path.exists():
            raise FileExistsError(f"{path} existe déjà : on n'écrase pas une sortie")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_ok = n_failed = consecutive = 0
    for i, question in enumerate(questions, start=1):
        try:
            run = run_with_retries(client, room_id, deployment_id,
                                   question["text"], expected_model, sleep=sleep)
        except (RunRejected, httpx.HTTPStatusError, httpx.TransportError) as err:
            if not isinstance(err, RunRejected) and not is_retryable(err):
                raise  # erreur de notre côté : inutile de continuer
            append_line(errors_path, {
                "question_id": question["id"],
                "config": config,
                "error_type": type(err).__name__,
                "error": str(err),
            })
            n_failed += 1
            consecutive += 1
            print(f"[{i}/{len(questions)}] question {question['id']} : ÉCHEC "
                  f"({type(err).__name__})", file=sys.stderr)
            if consecutive >= MAX_CONSECUTIVE_FAILURES:
                raise CampaignStopped(
                    f"{consecutive} échecs d'affilée, arrêt à la question {question['id']}"
                ) from err
            continue

        append_line(out_path, make_record(question, run, config))
        n_ok += 1
        consecutive = 0
        print(f"[{i}/{len(questions)}] question {question['id']} : ok", file=sys.stderr)
    return n_ok, n_failed


# --- Ligne de commande -----------------------------------------------------------

def main(argv: list[str] | None = None, transport: httpx.BaseTransport | None = None,
         sleep=time.sleep) -> int:
    parser = argparse.ArgumentParser(description="Génère les réponses d'une config sur Alive.")
    parser.add_argument("--questions", required=True, help="JSONL {id, text}")
    parser.add_argument("--room", required=True, help="th-...")
    parser.add_argument("--deployment", required=True, help="dp-...")
    parser.add_argument("--model", required=True, help="modèle attendu, ex. openai/gpt-4.1-mini")
    parser.add_argument("--config", required=True, help="nom de la config : A, A_bis ou B")
    parser.add_argument("--out", required=True, help="fichier de sortie, ex. outputs/A.jsonl")
    parser.add_argument("--limit", type=int, default=None, help="ne prendre que les N premières questions")
    args = parser.parse_args(argv)

    token = os.environ.get("ALIVE_TOKEN")
    if not token:
        print("ALIVE_TOKEN absent : export ALIVE_TOKEN=... d'abord.", file=sys.stderr)
        return 2

    questions = load_questions(args.questions)
    if args.limit is not None:
        questions = questions[:args.limit]

    with make_client(token, transport) as client:
        try:
            n_ok, n_failed = generate(
                client, questions, room_id=args.room, deployment_id=args.deployment,
                expected_model=args.model, config=args.config, out_path=args.out,
                sleep=sleep,
            )
        except FileExistsError as err:
            print(err, file=sys.stderr)
            return 2
        except (CampaignStopped, httpx.HTTPStatusError) as err:
            print(f"ARRÊT : {err}", file=sys.stderr)
            return 1
    print(f"{n_ok} réponses gardées, {n_failed} échecs -> {args.out}")
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
