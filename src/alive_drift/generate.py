"""Génère les réponses d'UNE config (A, A_bis ou B) pour une liste de questions.

Pour chaque question : un run Alive (run_with_retries), puis une ligne écrite
tout de suite dans le fichier de sortie. Une question après l'autre, pas encore
de parallélisme.

Deux fichiers JSONL (une ligne = un objet JSON) :
- <out>            : les réponses gardées. C'est LA référence : une question est
                     "faite" si et seulement si elle a une ligne ici ;
- <out>.errors     : journal des tentatives ratées, avec la raison.

Reprise : relancer la même commande saute les questions déjà dans <out> et
retente les autres (y compris celles qui avaient échoué).

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


# --- Reprise -------------------------------------------------------------------

def load_done_ids(out_path: Path, questions: list[dict], *, config: str,
                  expected_model: str) -> set[str]:
    """Ids des questions qui ont déjà une réponse dans out_path.

    Ensemble vide si le fichier n'existe pas (premier lancement).

    Avant de compléter un fichier, on vérifie qu'il est bien celui de CETTE
    campagne. Lève ValueError à la première ligne qui ne colle pas : on ne
    mélange jamais deux configs, deux modèles ou deux jeux de questions, et on
    ne répare pas le fichier tout seul.
    """
    if not out_path.exists():
        return set()
    content = out_path.read_text(encoding="utf-8")
    # Chaque ligne est écrite avec son retour à la ligne. S'il manque à la fin,
    # la dernière écriture a été coupée : la prochaine ligne s'y collerait.
    if content and not content.endswith("\n"):
        raise ValueError(f"{out_path} : dernière ligne incomplète (écriture interrompue ?)")

    text_by_id = {q["id"]: q["text"] for q in questions}
    done = set()
    for n, line in enumerate(content.splitlines(), start=1):
        where = f"{out_path} ligne {n}"
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            record = None
        if not isinstance(record, dict):
            raise ValueError(f"{where} : ligne illisible")
        qid = record.get("question_id")
        if record.get("config") != config:
            raise ValueError(f"{where} : config {record.get('config')!r}, attendu {config!r}")
        if record.get("model") != expected_model:
            raise ValueError(f"{where} : modèle {record.get('model')!r}, attendu {expected_model!r}")
        if qid not in text_by_id:
            raise ValueError(f"{where} : question {qid!r} absente du fichier de questions")
        if record.get("question") != text_by_id[qid]:
            raise ValueError(f"{where} : le texte de la question {qid!r} a changé")
        if qid in done:
            raise ValueError(f"{where} : question {qid!r} en double")
        done.add(qid)
    return done


# --- La boucle -----------------------------------------------------------------

def generate(client: httpx.Client, questions: list[dict], *, room_id: str,
             deployment_id: str, expected_model: str, config: str,
             out_path, limit: int | None = None, sleep=time.sleep) -> tuple[int, int]:
    """Un run par question à faire. Renvoie (réponses gardées, échecs) DE CE LANCEMENT.

    `limit` : ne considérer que les N premières questions.
    Les questions qui ont déjà une ligne dans out_path sont sautées (reprise).

    - Réponse gardée  -> une ligne dans out_path, écrite immédiatement.
    - Run rejeté, ou échec passager après 3 essais -> une ligne dans le fichier
      d'erreurs, et on passe à la question suivante.
    - Erreur de notre côté (401, 404...) -> on arrête tout : chaque question
      échouerait de la même façon.
    - 5 échecs d'affilée -> CampaignStopped.

    On n'écrase jamais : les deux fichiers sont seulement complétés par la fin.
    Si out_path existe mais n'est pas celui de cette campagne -> ValueError,
    aucun run lancé.
    """
    out_path = Path(out_path)
    errors_path = errors_path_for(out_path)
    # Vérifié contre TOUTES les questions, pas seulement les `limit` premières :
    # un fichier de 10 réponses reste valide si on relance avec --limit 3.
    done = load_done_ids(out_path, questions, config=config, expected_model=expected_model)
    todo = [q for q in questions[:limit] if q["id"] not in done]
    if done:
        print(f"reprise : {len(done)} réponses déjà dans {out_path}, "
              f"{len(todo)} questions à faire", file=sys.stderr)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_ok = n_failed = consecutive = 0
    for i, question in enumerate(todo, start=1):
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
            print(f"[{i}/{len(todo)}] question {question['id']} : ÉCHEC "
                  f"({type(err).__name__})", file=sys.stderr)
            if consecutive >= MAX_CONSECUTIVE_FAILURES:
                raise CampaignStopped(
                    f"{consecutive} échecs d'affilée, arrêt à la question {question['id']}"
                ) from err
            continue

        append_line(out_path, make_record(question, run, config))
        n_ok += 1
        consecutive = 0
        print(f"[{i}/{len(todo)}] question {question['id']} : ok", file=sys.stderr)
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

    with make_client(token, transport) as client:
        try:
            n_ok, n_failed = generate(
                client, questions, room_id=args.room, deployment_id=args.deployment,
                expected_model=args.model, config=args.config, out_path=args.out,
                limit=args.limit, sleep=sleep,
            )
        except ValueError as err:  # le fichier de sortie n'est pas celui de cette campagne
            print(f"REFUS : {err}", file=sys.stderr)
            return 2
        except (CampaignStopped, httpx.HTTPStatusError) as err:
            print(f"ARRÊT : {err}", file=sys.stderr)
            return 1
    print(f"{n_ok} réponses gardées, {n_failed} échecs à ce lancement -> {args.out}")
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
