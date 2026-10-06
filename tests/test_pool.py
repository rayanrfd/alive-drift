"""Tests de pool.py : le moteur seul, sans Alive ni fichiers.

Ici une "tâche" est un simple nombre et `work` le double. On vérifie qui
travaille quand, et comment ça s'arrête.
"""

import os
import signal
import threading
import time

import pytest

from alive_drift.pool import run_tasks


def run(tasks, work, workers, stop_when=lambda result: False):
    """Lance run_tasks. Renvoie (résultats dans l'ordre où handle les a reçus, fils de handle)."""
    results, handle_threads = [], set()

    def handle(result):
        results.append(result)
        handle_threads.add(threading.current_thread().name)
        return stop_when(result)

    run_tasks(tasks, work, handle, workers=workers, stopping=threading.Event())
    return results, handle_threads


def test_toutes_les_taches_sont_faites_une_fois():
    results, _ = run(range(50), lambda n: n * 2, workers=4)
    assert sorted(results) == [n * 2 for n in range(50)]


def test_handle_tourne_toujours_dans_le_fil_principal():
    # C'est ce qui garantit qu'un seul fil écrit dans les fichiers.
    work_threads = set()

    def work(n):
        work_threads.add(threading.current_thread().name)
        return n

    _, handle_threads = run(range(20), work, workers=4)
    assert handle_threads == {threading.main_thread().name}
    assert threading.main_thread().name not in work_threads


def test_un_worker_garde_l_ordre():
    results, _ = run(range(10), lambda n: n, workers=1)
    assert results == list(range(10))


def test_un_worker_la_tache_suivante_part_apres_handle():
    # Avec 1 worker : work(0), handle(0), work(1), handle(1)... jamais deux work de suite.
    journal = []

    def work(n):
        journal.append(f"work {n}")
        return n

    def handle(n):
        journal.append(f"handle {n}")

    run_tasks(range(3), work, handle, workers=1, stopping=threading.Event())
    assert journal == ["work 0", "handle 0", "work 1", "handle 1", "work 2", "handle 2"]


def test_jamais_plus_de_taches_en_cours_que_de_workers():
    lock = threading.Lock()
    state = {"now": 0, "max": 0}

    def work(n):
        with lock:
            state["now"] += 1
            state["max"] = max(state["max"], state["now"])
        time.sleep(0.02)
        with lock:
            state["now"] -= 1
        return n

    run(range(12), work, workers=3)
    assert state["max"] == 3


def test_arret_demande_par_handle_plus_rien_n_est_distribue():
    started = []

    def work(n):
        started.append(n)
        return n

    # handle demande l'arrêt dès le premier résultat, sur 1000 tâches instantanées.
    results, _ = run(range(1000), work, workers=4, stop_when=lambda result: True)
    assert len(started) <= 4                     # seulement les tâches déjà parties
    assert sorted(results) == sorted(started)    # et chacune passe quand même par handle


def test_stopping_est_leve_a_l_arret():
    stopping = threading.Event()
    run_tasks(range(5), lambda n: n, lambda result: True, workers=1, stopping=stopping)
    assert stopping.is_set()


def test_erreur_dans_work_remonte():
    def work(n):
        raise RuntimeError("bug dans work")

    with pytest.raises(RuntimeError, match="bug dans work"):
        run(range(3), work, workers=2)


def test_erreur_imprevue_leve_stopping():
    # Même sur un bug, les fils qui attendent (entre deux essais) doivent être réveillés.
    stopping = threading.Event()

    def handle(result):
        raise RuntimeError("bug dans handle")

    with pytest.raises(RuntimeError, match="bug dans handle"):
        run_tasks(range(5), lambda n: n, handle, workers=2, stopping=stopping)
    assert stopping.is_set()


def test_ctrl_c_range_ce_qui_est_parti_puis_leve():
    started, results = [], []

    def work(n):
        started.append(n)
        if n == 2:
            time.sleep(0.05)
            os.kill(os.getpid(), signal.SIGINT)  # le signal qu'envoie Ctrl-C
        time.sleep(0.05)
        return n

    interrupted = False
    try:
        run_tasks(range(30), work, results.append, workers=3, stopping=threading.Event())
    except KeyboardInterrupt:
        interrupted = True

    assert interrupted
    assert len(started) < 30                    # plus rien de distribué après Ctrl-C
    assert sorted(results) == sorted(started)   # tout ce qui est parti a été rangé
