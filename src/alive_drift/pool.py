"""Le moteur : exécuter des tâches, plusieurs à la fois, et s'arrêter proprement.

Ce fichier ne connaît ni Alive, ni les questions, ni les fichiers. Il sait
seulement distribuer des tâches à des fils (threads), récupérer les résultats
et gérer l'arrêt. Tout ce qui est propre à la campagne est dans recorder.py.
"""

import sys
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

# Plafond du nombre de tâches en même temps. À l'arrêt, les tâches en cours
# vont au bout : ce plafond borne donc aussi ce qu'un arrêt peut coûter.
MAX_WORKERS = 32


def run_tasks(tasks, work, handle, *, workers: int, stopping: threading.Event) -> None:
    """Exécute work(tâche) pour chaque tâche, au plus `workers` à la fois.

    - `work(tâche)` tourne dans un fil secondaire et renvoie un résultat.
    - `handle(résultat)` est appelé dans le fil principal, un résultat à la
      fois, dans l'ordre où ils arrivent. S'il renvoie True, on arrête.
    - `stopping` est levé dès que l'arrêt est décidé (par handle ou par Ctrl-C).
      `work` peut le consulter pour abandonner ce qu'il attend.

    Une tâche n'est distribuée que s'il y a une place libre ET que l'arrêt
    n'est pas décidé. À l'arrêt, il y a donc au plus `workers` tâches en cours ;
    elles vont au bout et leur résultat passe quand même par handle.

    Ctrl-C : même arrêt propre, puis KeyboardInterrupt une fois tout rangé.
    Un second Ctrl-C abandonne l'attente (les résultats en cours sont perdus).
    """
    pool = ThreadPoolExecutor(max_workers=workers)
    remaining = iter(tasks)  # les tâches pas encore distribuées
    in_flight = set()        # les tâches en cours (au plus `workers`)
    interrupted = False      # Ctrl-C reçu une fois
    forced = False           # Ctrl-C reçu deux fois : on n'attend plus
    try:
        while True:
            # --- 1. Distribuer : s'il y a de la place et pas d'arrêt décidé ------
            while not stopping.is_set() and len(in_flight) < workers:
                task = next(remaining, None)
                if task is None:
                    break
                in_flight.add(pool.submit(work, task))
            if not in_flight:
                break  # plus rien en cours, plus rien à distribuer

            # --- 2. Attendre qu'au moins une tâche se termine -------------------
            try:
                finished, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
            except KeyboardInterrupt:
                if interrupted:
                    forced = True
                    raise
                interrupted = True
                stopping.set()
                print(f"Ctrl-C : plus aucun envoi, j'attends les {len(in_flight)} runs en cours "
                      "(Ctrl-C à nouveau pour abandonner)", file=sys.stderr)
                continue

            # --- 3. Ranger chaque résultat, dans le fil principal ---------------
            for future in finished:
                if handle(future.result()):
                    stopping.set()
    finally:
        # Normalement plus rien n'est en cours ici. Sur une erreur imprévue, on
        # attend les tâches parties ; après un second Ctrl-C, on n'attend pas.
        stopping.set()
        pool.shutdown(wait=not forced, cancel_futures=True)

    if interrupted:
        raise KeyboardInterrupt
