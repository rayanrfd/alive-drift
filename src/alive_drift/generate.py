"""Génère les réponses d'UNE config (A, A_bis ou B) pour une liste de questions.

Pour chaque question : un run Alive (run_with_retries), puis une ligne écrite
tout de suite dans le fichier de sortie. Une question après l'autre par
défaut ; plusieurs en même temps avec --workers N.

Deux fichiers JSONL (une ligne = un objet JSON) :
- <out>            : les réponses gardées. C'est LA référence : une question est
                     "faite" si et seulement si elle a une ligne ici ;
- <out>.errors     : journal des tentatives ratées, avec la raison.

Reprise : relancer la même commande saute les questions déjà dans <out> et
retente les autres (y compris celles qui avaient échoué).

Ctrl-C : plus aucune question n'est envoyée, les runs en cours vont au bout et
sont rangés, puis le programme s'arrête. Un second Ctrl-C abandonne l'attente
(les réponses en cours sont alors perdues).

La config (déploiement, modèle, effort de raisonnement) est lue dans
configs.json par son nom : voir config.py.

Exemple sur 3 questions (3 runs payants, quelques millièmes de crédit) :

    export ALIVE_TOKEN=...
    uv run python -m alive_drift.generate --config sonde --out outputs/sonde.jsonl --limit 3
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import httpx

from alive_drift.check import RunRejected
from alive_drift.client import is_retryable, make_client, run_with_retries
from alive_drift.config import Config, load_config

# Après 5 questions ratées d'affilée, on arrête tout : le problème est général
# (modèle en panne, déploiement mal réglé). Un run rejeté est facturé : sans
# cette limite, un mauvais réglage coûterait 1000 runs inutiles.
MAX_CONSECUTIVE_FAILURES = 5

# Plafond du nombre de questions envoyées en même temps. À l'arrêt, les runs
# en cours vont au bout : ce plafond borne donc aussi ce qu'un arrêt peut coûter.
MAX_WORKERS = 32


class CampaignStopped(Exception):
    """Trop d'échecs d'affilée : la campagne est arrêtée."""


# --- Lecture des questions ---------------------------------------------------

def load_questions(path) -> list[dict]:
    """Lit un JSONL de questions {"id": ..., "text": "..."}.

    Renvoie une liste de {"id": texte, "text": texte}, dans l'ordre du fichier.
    Lève ValueError si une ligne est mal formée, si un id est en double, ou si
    le fichier ne contient aucune question.
    """
    questions = []
    seen = set()
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, start=1):
            if not line.strip():
                continue  # ligne vide tolérée
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path} ligne {n} : doit être un objet {{id, text}}")
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
    if not questions:
        raise ValueError(f"{path} : aucune question")
    return questions


# --- Écriture ----------------------------------------------------------------

def make_record(question: dict, run: dict, config: Config) -> dict:
    """La ligne enregistrée pour une réponse gardée.

    `run` a passé check_run : le bloc model et served[0] existent forcément.
    """
    served = run["model"]["served"][0]
    tokens = run.get("tokens")
    if not isinstance(tokens, dict):
        tokens = {}
    return {
        "question_id": question["id"],
        "config": config.name,
        "room_id": config.room,
        "deployment_id": config.deployment,
        "reasoning_effort": config.reasoning_effort,
        "question": question["text"],
        "response": run["content"],
        "model": served["model"],
        "provider": served["provider"],
        "execution_id": run["execution_id"],
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

def load_done(out_path: Path, questions: list[dict], config: Config) -> tuple[set[str], set[str]]:
    """Ce qui est déjà dans out_path : (ids des questions faites, execution_id utilisés).

    Deux ensembles vides si le fichier n'existe pas (premier lancement).

    Avant de compléter un fichier, on vérifie qu'il est bien celui de CETTE
    campagne. Lève ValueError à la première ligne qui ne colle pas : on ne
    mélange jamais deux configs, deux déploiements, deux modèles, deux efforts
    de raisonnement ou deux jeux de questions, et on ne répare pas le fichier
    tout seul.
    """
    if not out_path.exists():
        return set(), set()
    content = out_path.read_text(encoding="utf-8")
    # Chaque ligne est écrite avec son retour à la ligne. S'il manque à la fin,
    # la dernière écriture a été coupée : la prochaine ligne s'y collerait.
    if content and not content.endswith("\n"):
        raise ValueError(f"{out_path} : dernière ligne incomplète (écriture interrompue ?)")

    text_by_id = {q["id"]: q["text"] for q in questions}
    done = set()
    executions = set()
    for n, line in enumerate(content.splitlines(), start=1):
        where = f"{out_path} ligne {n}"
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            record = None
        if not isinstance(record, dict):
            raise ValueError(f"{where} : ligne illisible")
        qid = record.get("question_id")
        # Chaque ligne doit venir de la même config, champ par champ.
        for label, key, expected in (
            ("config", "config", config.name),
            ("room", "room_id", config.room),
            ("déploiement", "deployment_id", config.deployment),
            ("modèle", "model", config.model),
            ("effort de raisonnement", "reasoning_effort", config.reasoning_effort),
        ):
            if record.get(key) != expected:
                raise ValueError(f"{where} : {label} {record.get(key)!r}, attendu {expected!r}")
        # Même règle que check_run : "none" veut dire 0 token de raisonnement.
        reasoning = record.get("reasoning_tokens")
        if config.reasoning_effort == "none" and (type(reasoning) is not int or reasoning != 0):
            raise ValueError(f"{where} : {reasoning!r} tokens de raisonnement alors qu'il est coupé")
        if qid not in text_by_id:
            raise ValueError(f"{where} : question {qid!r} absente du fichier de questions")
        if record.get("question") != text_by_id[qid]:
            raise ValueError(f"{where} : le texte de la question {qid!r} a changé")
        if qid in done:
            raise ValueError(f"{where} : question {qid!r} en double")
        response = record.get("response")
        if not isinstance(response, str) or not response.strip():
            raise ValueError(f"{where} : question {qid!r} sans réponse")
        execution = record.get("execution_id")
        if not isinstance(execution, str) or not execution:
            raise ValueError(f"{where} : execution_id absent")
        if execution in executions:
            raise ValueError(f"{where} : execution_id {execution!r} en double")
        done.add(qid)
        executions.add(execution)
    return done, executions


# --- La boucle -----------------------------------------------------------------

def generate(client: httpx.Client, questions: list[dict], config: Config, *,
             out_path, limit: int | None = None, workers: int = 1,
             sleep=time.sleep) -> tuple[int, int]:
    """Un run par question à faire. Renvoie (réponses gardées, échecs) DE CE LANCEMENT.

    `config` : où envoyer les runs et ce qu'on attend d'eux (voir config.py).
    `limit` : ne considérer que les N premières questions (au moins 1).
    `workers` : nombre de questions envoyées à Alive en même temps (1 à 32).
    Les questions qui ont déjà une ligne dans out_path sont sautées (reprise).

    - Réponse gardée  -> une ligne dans out_path, écrite immédiatement.
    - Run rejeté, ou échec passager après 3 essais -> une ligne dans le fichier
      d'erreurs, et on passe à la question suivante.
    - Erreur de notre côté (401, 404...) -> on arrête tout : chaque question
      échouerait de la même façon.
    - 5 échecs d'affilée -> CampaignStopped.
    - Une réponse dont l'execution_id a déjà servi -> comptée comme un échec.
    - Ctrl-C -> KeyboardInterrupt, levée une fois les runs en cours rangés.

    Une question n'est envoyée que s'il y a une place libre parmi les `workers`
    ET qu'aucun arrêt n'est décidé. À l'arrêt il y a donc au plus `workers`
    runs en cours ; ils vont au bout et leur résultat est enregistré.
    Avec workers > 1, les lignes sont dans l'ordre d'arrivée des réponses.

    On n'écrase jamais : les deux fichiers sont seulement complétés par la fin.
    Si out_path existe mais n'est pas celui de cette campagne -> ValueError,
    aucun run lancé.
    """
    # questions[:-1] voudrait dire "toutes sauf la dernière" : on refuse.
    if limit is not None and limit < 1:
        raise ValueError(f"limit doit valoir au moins 1 (reçu {limit})")
    if not 1 <= workers <= MAX_WORKERS:
        raise ValueError(f"workers doit être entre 1 et {MAX_WORKERS} (reçu {workers})")
    out_path = Path(out_path)
    errors_path = errors_path_for(out_path)
    # Vérifié contre TOUTES les questions, pas seulement les `limit` premières :
    # un fichier de 10 réponses reste valide si on relance avec --limit 3.
    done, executions = load_done(out_path, questions, config)
    todo = [q for q in questions[:limit] if q["id"] not in done]
    if done:
        print(f"reprise : {len(done)} réponses déjà dans {out_path}, "
              f"{len(todo)} questions à faire", file=sys.stderr)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Une tentative = une question envoyée à Alive ---------------------------
    # Tourne dans un fil secondaire : elle n'écrit RIEN, elle renvoie seulement
    # (question, run, erreur). Un seul fil écrit les fichiers : le principal.
    def attempt(question: dict):
        try:
            run = run_with_retries(client, config.room, config.deployment,
                                   question["text"], config.model,
                                   config.reasoning_effort, sleep=sleep)
        except (RunRejected, httpx.HTTPStatusError, httpx.TransportError) as err:
            return question, None, err
        return question, run, None

    pool = ThreadPoolExecutor(max_workers=workers)
    remaining = iter(todo)   # les questions pas encore envoyées
    in_flight = set()        # les envois en cours (au plus `workers`)
    n_ok = n_failed = consecutive = 0
    stop = None              # l'exception qui arrête la campagne, levée une fois tout rangé
    interrupted = False      # Ctrl-C reçu une fois
    forced = False           # Ctrl-C reçu deux fois : on n'attend plus
    try:
        while True:
            # --- Envoyer : seulement s'il y a de la place et pas d'arrêt décidé ---
            while stop is None and len(in_flight) < workers:
                question = next(remaining, None)
                if question is None:
                    break
                in_flight.add(pool.submit(attempt, question))
            if not in_flight:
                break  # plus rien en cours, plus rien à envoyer

            # --- Attendre qu'au moins un envoi se termine ---------------------------
            try:
                finished, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
            except KeyboardInterrupt:
                if interrupted:
                    forced = True
                    raise
                interrupted = True
                stop = stop or KeyboardInterrupt()
                print(f"Ctrl-C : plus aucun envoi, j'attends les {len(in_flight)} runs en cours "
                      "(Ctrl-C à nouveau pour abandonner)", file=sys.stderr)
                continue

            # --- Ranger chaque résultat, toujours dans le fil principal -----------
            for future in finished:
                question, run, err = future.result()
                i = n_ok + n_failed + 1

                if err is None and run["execution_id"] in executions:
                    # Alive a rendu la réponse d'un run déjà enregistré.
                    err = RunRejected(f"execution_id {run['execution_id']!r} déjà enregistré")

                if err is None:
                    append_line(out_path, make_record(question, run, config))
                    executions.add(run["execution_id"])
                    n_ok += 1
                    consecutive = 0
                    print(f"[{i}/{len(todo)}] question {question['id']} : ok", file=sys.stderr)
                    continue

                if not isinstance(err, RunRejected) and not is_retryable(err):
                    # Erreur de notre côté (401, 404...) : inutile de continuer.
                    stop = stop or err
                    continue

                append_line(errors_path, {
                    "question_id": question["id"],
                    "config": config.name,
                    "deployment_id": config.deployment,
                    "error_type": type(err).__name__,
                    "error": str(err),
                })
                n_failed += 1
                consecutive += 1
                print(f"[{i}/{len(todo)}] question {question['id']} : ÉCHEC "
                      f"({type(err).__name__})", file=sys.stderr)
                if consecutive >= MAX_CONSECUTIVE_FAILURES and stop is None:
                    stop = CampaignStopped(
                        f"{consecutive} échecs d'affilée, arrêt à la question {question['id']}")
    finally:
        # Normalement plus rien n'est en cours ici. Sur une erreur imprévue, on
        # attend les runs partis ; après un second Ctrl-C, on n'attend pas.
        pool.shutdown(wait=not forced, cancel_futures=True)

    if stop is not None:
        raise stop
    return n_ok, n_failed


# --- Ligne de commande -----------------------------------------------------------

def main(argv: list[str] | None = None, transport: httpx.BaseTransport | None = None,
         sleep=time.sleep) -> int:
    parser = argparse.ArgumentParser(description="Génère les réponses d'une config sur Alive.")
    parser.add_argument("--config", required=True, help="nom d'une config de configs.json")
    parser.add_argument("--configs", default="configs.json", help="fichier des configs")
    parser.add_argument("--questions", default="data/healthbench.jsonl", help="JSONL {id, text}")
    parser.add_argument("--out", required=True, help="fichier de sortie, ex. outputs/A.jsonl")
    parser.add_argument("--limit", type=int, default=None, help="ne prendre que les N premières questions")
    parser.add_argument("--workers", type=int, default=1, help="questions envoyées en même temps (1 à 32, défaut 1)")
    args = parser.parse_args(argv)

    token = os.environ.get("ALIVE_TOKEN")
    if not token:
        print("ALIVE_TOKEN absent : export ALIVE_TOKEN=... d'abord.", file=sys.stderr)
        return 2

    with make_client(token, transport) as client:
        try:
            config = load_config(args.configs, args.config)
            questions = load_questions(args.questions)
            n_ok, n_failed = generate(client, questions, config, out_path=args.out,
                                      limit=args.limit, workers=args.workers, sleep=sleep)
        except (ValueError, OSError) as err:  # config, questions, limit, workers ou sortie invalide ;
            # OSError = fichier introuvable ou illisible
            print(f"REFUS : {err}", file=sys.stderr)
            return 2
        except (CampaignStopped, httpx.HTTPStatusError) as err:
            print(f"ARRÊT : {err}", file=sys.stderr)
            return 1
        except KeyboardInterrupt:
            print("INTERROMPU : relancer la même commande pour reprendre.", file=sys.stderr)
            return 130  # code de sortie habituel après Ctrl-C
    print(f"{n_ok} réponses gardées, {n_failed} échecs à ce lancement -> {args.out}")
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
