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

Ce fichier assemble trois pièces :
- files.py    : lire les questions, écrire les lignes, relire ce qui est fait ;
- pool.py     : envoyer plusieurs questions à la fois et s'arrêter proprement ;
- recorder.py : que faire de chaque résultat (garder, noter l'échec, arrêter).

Exemple sur 3 questions (3 runs payants, quelques millièmes de crédit) :

    (jeton dans le fichier .env : ALIVE_ACCESS_TOKEN=...)
    uv run python -m alive_drift.generate --config sonde --out outputs/sonde.jsonl --limit 3
"""

import argparse
import sys
import threading
from pathlib import Path

import httpx

from alive_drift.check import RunRejected
from alive_drift.client import TOKEN_HELP, find_token, make_client, run_with_retries
from alive_drift.config import Config, load_config
from alive_drift.files import load_done, load_questions
from alive_drift.pool import MAX_WORKERS, run_tasks
from alive_drift.recorder import CampaignStopped, Recorder


def generate(client: httpx.Client, questions: list[dict], config: Config, *,
             out_path, limit: int | None = None, workers: int = 1,
             sleep=None) -> tuple[int, int]:
    """Un run par question à faire. Renvoie (réponses gardées, échecs) DE CE LANCEMENT.

    `config` : où envoyer les runs et ce qu'on attend d'eux (voir config.py).
    `limit` : ne considérer que les N premières questions (au moins 1).
    `workers` : nombre de questions envoyées à Alive en même temps (1 à 32).
    `sleep` : fonction d'attente entre deux essais ; à ne passer que dans les tests.

    Lève ValueError, sans lancer de run, si une option est invalide ou si
    out_path existe mais n'est pas le fichier de cette campagne.
    Lève l'erreur qui a arrêté la campagne (401, CampaignStopped,
    KeyboardInterrupt), une fois les runs en cours rangés.
    """
    # --- 1. Vérifier les options --------------------------------------------------
    # questions[:-1] voudrait dire "toutes sauf la dernière" : on refuse.
    if limit is not None and limit < 1:
        raise ValueError(f"limit doit valoir au moins 1 (reçu {limit})")
    if not 1 <= workers <= MAX_WORKERS:
        raise ValueError(f"workers doit être entre 1 et {MAX_WORKERS} (reçu {workers})")

    # --- 2. Ce qui reste à faire (reprise) ----------------------------------------
    # Vérifié contre TOUTES les questions, pas seulement les `limit` premières :
    # un fichier de 10 réponses reste valide si on relance avec --limit 3.
    out_path = Path(out_path)
    done, executions = load_done(out_path, questions, config)
    todo = [q for q in questions[:limit] if q["id"] not in done]
    if done:
        print(f"reprise : {len(done)} réponses déjà dans {out_path}, "
              f"{len(todo)} questions à faire", file=sys.stderr)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- 3. Une tentative = une question envoyée à Alive --------------------------
    # Tourne dans un fil secondaire : elle n'écrit RIEN, elle renvoie seulement
    # (question, run, erreur). `stopping` est levé dès qu'un arrêt est décidé :
    # les fils qui attendent entre deux essais se réveillent alors et abandonnent.
    stopping = threading.Event()
    if sleep is None:
        sleep = stopping.wait  # attend N secondes, ou moins si l'arrêt est décidé

    def attempt(question: dict):
        try:
            run = run_with_retries(client, config.room, config.deployment,
                                   question["text"], config.model,
                                   config.reasoning_effort, sleep=sleep,
                                   give_up=stopping.is_set)
        except (RunRejected, httpx.HTTPStatusError, httpx.TransportError) as err:
            return question, None, err
        return question, run, None

    # --- 4. Lancer : pool distribue, recorder range -------------------------------
    recorder = Recorder(config, out_path, executions, total=len(todo))
    run_tasks(todo, attempt, recorder.record, workers=workers, stopping=stopping)

    if recorder.stop is not None:
        raise recorder.stop
    return recorder.n_ok, recorder.n_failed


# --- Ligne de commande -----------------------------------------------------------

def main(argv: list[str] | None = None, transport: httpx.BaseTransport | None = None,
         sleep=None, env_file=".env") -> int:
    parser = argparse.ArgumentParser(description="Génère les réponses d'une config sur Alive.")
    parser.add_argument("--config", required=True, help="nom d'une config de configs.json")
    parser.add_argument("--configs", default="configs.json", help="fichier des configs")
    parser.add_argument("--questions", default="data/healthbench.jsonl", help="JSONL {id, text}")
    parser.add_argument("--out", required=True, help="fichier de sortie, ex. outputs/A.jsonl")
    parser.add_argument("--limit", type=int, default=None, help="ne prendre que les N premières questions")
    parser.add_argument("--workers", type=int, default=1, help="questions envoyées en même temps (1 à 32, défaut 1)")
    args = parser.parse_args(argv)

    token = find_token(env_file)
    if not token:
        print(TOKEN_HELP, file=sys.stderr)
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
