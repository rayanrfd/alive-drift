"""Que faire d'un résultat : le ranger dans les fichiers et tenir les comptes.

Un résultat est (question, run, erreur), fabriqué par `attempt` dans generate.py.
Il y a trois issues possibles, toutes dans Recorder.record :
- réponse gardée        -> une ligne dans le fichier de sortie ;
- échec de CETTE question -> une ligne dans le journal d'erreurs, on continue ;
- erreur de notre côté  -> on arrête la campagne.
"""

import sys
from pathlib import Path

from alive_drift.check import RunRejected
from alive_drift.client import is_retryable
from alive_drift.config import Config
from alive_drift.files import alive_detail, append_line, errors_path_for, make_record, now_utc

# Après 5 questions ratées d'affilée, on arrête tout : le problème est général
# (modèle en panne, déploiement mal réglé). Un run rejeté est facturé : sans
# cette limite, un mauvais réglage coûterait 1000 runs inutiles.
MAX_CONSECUTIVE_FAILURES = 5


class CampaignStopped(Exception):
    """Trop d'échecs d'affilée : la campagne est arrêtée."""


class Recorder:
    """Range chaque résultat et tient les comptes d'UN lancement."""

    def __init__(self, config: Config, out_path: Path, executions: set[str], total: int):
        self.config = config
        self.out_path = out_path
        self.errors_path = errors_path_for(out_path)
        self.executions = executions  # les execution_id déjà enregistrés
        self.total = total            # nombre de questions de ce lancement
        self.n_ok = 0                 # réponses gardées
        self.n_failed = 0             # échecs notés
        self.consecutive = 0          # échecs d'affilée
        self.stop = None              # l'exception qui arrête la campagne, s'il y en a une

    def record(self, result) -> bool:
        """Range un résultat. Renvoie True s'il faut arrêter la campagne."""
        question, run, err = result

        if err is None and run["execution_id"] in self.executions:
            # Alive a rendu la réponse d'un run déjà enregistré.
            err = RunRejected(f"execution_id {run['execution_id']!r} déjà enregistré")

        if err is None:
            self._keep(question, run)
        elif isinstance(err, RunRejected) or is_retryable(err):
            self._note_failure(question, err)
        else:
            # Erreur de notre côté (401, 404...) : chaque question échouerait pareil.
            self.stop = self.stop or err
        return self.stop is not None

    def _keep(self, question: dict, run: dict) -> None:
        """Réponse gardée : une ligne dans le fichier de sortie."""
        append_line(self.out_path, make_record(question, run, self.config))
        self.executions.add(run["execution_id"])
        self.n_ok += 1
        self.consecutive = 0
        print(f"{self._progress()} question {question['id']} : ok", file=sys.stderr)

    def _note_failure(self, question: dict, err: Exception) -> None:
        """Échec de cette question : une ligne dans le journal, et arrêt au 5e d'affilée."""
        append_line(self.errors_path, {
            "question_id": question["id"],
            "config": self.config.name,
            "deployment_id": self.config.deployment,
            "error_type": type(err).__name__,
            "error": str(err),
            "detail": alive_detail(err),
            "written_at": now_utc(),
        })
        self.n_failed += 1
        self.consecutive += 1
        print(f"{self._progress()} question {question['id']} : ÉCHEC "
              f"({type(err).__name__}) {alive_detail(err) or ''}", file=sys.stderr)
        if self.consecutive >= MAX_CONSECUTIVE_FAILURES and self.stop is None:
            self.stop = CampaignStopped(
                f"{self.consecutive} échecs d'affilée, arrêt à la question {question['id']}")

    def _progress(self) -> str:
        """"[12/1000]" : numéro du résultat qu'on vient de ranger."""
        return f"[{self.n_ok + self.n_failed}/{self.total}]"
