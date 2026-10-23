"""MedAgentBench : noter les tâches faites par mab/loop.py avec le correcteur d'origine (morceau 32).

Le correcteur, c'est refsol.py de MedAgentBench (lien Box du README, pas dans leur dépôt
git). Il n'est PAS copié ici : on le charge depuis le clone de MedAgentBench
(--medagentbench, défaut ../MedAgentBench), avec son utils.py, et on enregistre leurs
empreintes avec chaque note.

On reproduit leur calcul (src/server/tasks/medagentbench/__init__.py, calculate_overall,
et eval.py, commit 9926011, licence MIT) :

- une tâche sans réponse (statut autre que « completed ») est fausse, sans appeler le
  correcteur ;
- sinon, correcteur du type de la tâche (task1... task10) : juste seulement s'il renvoie
  True ; s'il lève une exception, fausse (chez eux : print(e), return False) ;
- taux de réussite = tâches justes / toutes les tâches.

Le correcteur interroge lui-même le serveur FHIR (la bonne valeur, pour la plupart des
types), avec send_get_request de leur utils.py. Écarts volontaires, pour ne jamais compter
une panne comme une erreur de l'agent :
- chaque GET du correcteur a un délai (60 s) : chez eux, requests.get sans délai peut
  attendre indéfiniment ;
- si un GET du correcteur échoue (pas de données : serveur en panne, délai dépassé,
  erreur 5xx...), on arrête tout sans rien écrire, même si refsol a avalé l'erreur ;
- si le correcteur lève une exception et que le serveur ne répond plus, pareil.
Les autres exceptions sont gardées dans la ligne (grader_error) et la tâche est fausse.
Une réponse réussie est passée telle quelle à refsol : la note ne change pas.

Ce que le correcteur affiche (souvent « id, réponse attendue, réponse de l'agent ») est
gardé dans grader_output : c'est là qu'on lit la bonne réponse.

Sortie : un nouveau fichier, jamais écrasé (défaut : <runs>.graded.jsonl), écrit d'un
coup à la fin. Aucun appel à Alive.

    uv run python -m alive_drift.mab.grade --runs outputs/mab/mab_g54mini.jsonl
"""

import argparse
import importlib.util
import io
import itertools
import json
import sys
import types
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import requests

from alive_drift.files import append_line, now_utc, read_records
from alive_drift.mab.fhir import ServerUnreachable, check_server
from alive_drift.mab.tasks import FUNCS_PATH, TASKS_PATH, file_hash, tasks_fingerprint

MAB_DIR = "../MedAgentBench"
TASK_DIR = Path("src") / "server" / "tasks" / "medagentbench"   # dans le clone de MedAgentBench

GRADER_TIMEOUT_S = 60

_names = itertools.count(1)


class GraderGetFailed(ServerUnreachable):
    """Un GET du correcteur n'a pas eu de réponse : la note serait celle d'une panne."""


class Refsol(SimpleNamespace):
    """Le module refsol chargé, ses empreintes, et les GET ratés du dernier appel (failures)."""


class _RequestsWithTimeout:
    """Remplace `requests` dans leur utils.py : le même requests.get, avec un délai."""

    def __init__(self, timeout: float):
        self.timeout = timeout

    def get(self, url, params=None, headers=None):
        return requests.get(url, params=params, headers=headers, timeout=self.timeout)


def _watch_gets(module, timeout: float) -> list[str]:
    """Branche un délai et un témoin sur send_get_request. Renvoie la liste des GET ratés.

    Leur send_get_request renvoie {"error": ...} au lieu de lever : refsol plante plus
    loin (KeyError 'data'), ou avale l'erreur. Le témoin note chaque GET sans données,
    puis rend la réponse telle quelle : refsol se comporte exactement comme chez eux.
    """
    original = getattr(module, "send_get_request", None)
    if not callable(original) or "requests" not in getattr(original, "__globals__", {}):
        raise ValueError("refsol.py n'a pas le send_get_request de utils.py (avec requests) : "
                         "le correcteur a changé, à vérifier avant de noter")
    original.__globals__["requests"] = _RequestsWithTimeout(timeout)
    failures: list[str] = []

    def watched(url, params=None, headers=None):
        response = original(url, params, headers)
        if not isinstance(response, dict) or "data" not in response:
            failures.append(f"GET {url} : {response.get('error') if isinstance(response, dict) else response!r}")
        return response

    module.send_get_request = watched           # refsol appelle send_get_request par ce nom
    return failures


def load_refsol(mab_dir=MAB_DIR, timeout: float = GRADER_TIMEOUT_S) -> Refsol:
    """Charge refsol.py depuis le clone de MedAgentBench, avec son « from .utils import * ».

    On ne passe pas par leur paquet (son __init__.py importe tout leur serveur) : on crée
    un paquet vide qui pointe vers leur dossier, pour que .utils soit leur utils.py.
    Lève FileNotFoundError si refsol.py ou utils.py manque, ValueError si refsol n'a pas
    le send_get_request attendu.
    """
    task_dir = Path(mab_dir).expanduser() / TASK_DIR
    paths = {name: task_dir / f"{name}.py" for name in ("refsol", "utils")}
    for name, path in paths.items():
        if not path.is_file():
            hint = " (à télécharger depuis le lien Box du README de MedAgentBench)" if name == "refsol" else ""
            raise FileNotFoundError(f"{path} introuvable{hint}")
    package = f"_medagentbench_refsol_{next(_names)}"     # un nom neuf à chaque chargement
    pkg = types.ModuleType(package)
    pkg.__path__ = [str(task_dir)]
    sys.modules[package] = pkg
    try:
        spec = importlib.util.spec_from_file_location(f"{package}.refsol", paths["refsol"])
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    finally:                                    # le module reste utilisable, on ne le garde pas
        for name in [n for n in sys.modules if n == package or n.startswith(package + ".")]:
            del sys.modules[name]
    failures = _watch_gets(module, timeout)
    return Refsol(module=module, refsol_hash=file_hash(paths["refsol"]),
                  utils_hash=file_hash(paths["utils"]), task_dir=str(task_dir), failures=failures)


def load_cases(tasks_path=TASKS_PATH) -> dict[str, dict]:
    """Les tâches telles quelles (le case_data de MedAgentBench : id, sol, eval_MRN...)."""
    rows = json.loads(Path(tasks_path).read_text(encoding="utf-8"))
    return {row["id"]: row for row in rows}


def grade_one(refsol: Refsol, case: dict, rec: dict, api_base: str,
              server_ok: Callable[[str], bool]) -> dict:
    """La note d'une tâche : {correct, graded, grader_output, grader_error}.

    Lève GraderGetFailed si un GET du correcteur a échoué, ServerUnreachable si le
    correcteur a levé une exception et que le serveur ne répond plus.
    """
    if rec["result"] is None:                   # chez eux : seules les tâches avec un résultat
        return {"correct": False, "graded": False, "grader_output": "", "grader_error": None}
    # Ce que lit refsol : results.result, et results.history (objets avec .role et .content)
    results = SimpleNamespace(result=rec["result"],
                              history=[SimpleNamespace(role=h["role"], content=h["content"])
                                       for h in rec["history"]])
    grader = getattr(refsol.module, case["id"].split("_")[0])     # comme eval.py
    out, error = io.StringIO(), None
    refsol.failures.clear()
    with redirect_stdout(out):
        try:
            correct = grader(case, results, api_base) is True
        except Exception as err:                # comme eval.py : fausse
            correct, error = False, f"{type(err).__name__}: {err}"
    if refsol.failures:
        raise GraderGetFailed(f"{rec['task_id']} : le correcteur n'a pas eu de réponse du serveur "
                              f"({'; '.join(refsol.failures)})")
    if error is not None and not server_ok(api_base):
        raise ServerUnreachable(f"{rec['task_id']} : le correcteur a échoué ({error}) "
                                f"et le serveur FHIR {api_base} ne répond plus")
    return {"correct": correct, "graded": True, "grader_output": out.getvalue(), "grader_error": error}


def load_runs(runs_path, fingerprint: dict, cases: dict) -> list[dict]:
    """Les lignes de mab/loop.py. Lève ValueError si le fichier est vide, mélange des
    configs ou des serveurs, a une tâche en double ou inconnue, ou d'autres tâches."""
    records = read_records(Path(runs_path))
    if not records:
        raise ValueError(f"{runs_path} : aucune tâche (fichier vide ou absent)")
    first, seen = records[0][1], set()
    for where, rec in records:
        if rec.get("fingerprint") != fingerprint:
            raise ValueError(f"{where} : empreinte des tâches {rec.get('fingerprint')!r}, attendu {fingerprint!r}")
        for field in ("config", "deployment_id", "model", "reasoning_effort", "api_base"):
            if rec.get(field) != first.get(field):
                raise ValueError(f"{where} : {field} {rec.get(field)!r}, différent de la 1re ligne "
                                 f"({first.get(field)!r})")
        tid = rec.get("task_id")
        if tid not in cases:
            raise ValueError(f"{where} : tâche {tid!r} inconnue")
        if tid in seen:
            raise ValueError(f"{where} : tâche {tid!r} en double")
        if rec.get("task_type") != tid.split("_")[0]:
            raise ValueError(f"{where} : type {rec.get('task_type')!r} ne correspond pas à {tid!r}")
        seen.add(tid)
    return [rec for _, rec in records]


def grade_file(runs_path, out_path, refsol: Refsol, *, tasks_path=TASKS_PATH, funcs_path=FUNCS_PATH,
               server_ok: Callable[[str], bool] = check_server) -> list[dict]:
    """Note toutes les tâches de runs_path et écrit out_path (nouveau fichier). Renvoie les lignes.

    Lève ValueError (rien d'écrit) si out_path existe déjà ou si runs_path n'est pas
    cohérent, ServerUnreachable (rien d'écrit) si le serveur FHIR ne répond pas.
    """
    out_path = Path(out_path)
    if out_path.exists():
        raise ValueError(f"{out_path} existe déjà : on n'écrase jamais une sortie (choisir --out)")
    cases = load_cases(tasks_path)
    runs = load_runs(runs_path, tasks_fingerprint(tasks_path, funcs_path), cases)
    api_base = runs[0]["api_base"]
    if not server_ok(api_base):
        raise ServerUnreachable(f"le serveur FHIR {api_base} ne répond pas (Docker lancé ?)")
    graded = []
    for rec in runs:
        note = grade_one(refsol, cases[rec["task_id"]], rec, api_base, server_ok)
        graded.append({"task_id": rec["task_id"], "task_type": rec["task_type"], "config": rec["config"],
                       "deployment_id": rec["deployment_id"], "model": rec["model"],
                       "reasoning_effort": rec["reasoning_effort"], "status": rec["status"],
                       "result": rec["result"], **note, "n_rounds": rec["n_rounds"],
                       "execution_ids": [r["execution_id"] for r in rec["runs"]],
                       "runs_file": str(runs_path), "runs_written_at": rec["written_at"],
                       "api_base": api_base, "fingerprint": rec["fingerprint"],
                       "refsol": refsol.refsol_hash, "refsol_utils": refsol.utils_hash,
                       "graded_at": now_utc()})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for line in graded:                         # tout est noté : on écrit d'un coup
        append_line(out_path, line)
    return graded


def summary(graded: list[dict]) -> str:
    """Taux de réussite (justes / toutes, comme l'article), par type, et statuts de l'agent."""
    n, ok = len(graded), sum(g["correct"] for g in graded)
    lines = [f"réussite : {ok}/{n} = {ok / n:.1%}"]
    by_type = sorted({g["task_type"] for g in graded}, key=lambda t: int(t[4:]))
    for t in by_type:
        rows = [g for g in graded if g["task_type"] == t]
        lines.append(f"  {t:<7} {sum(g['correct'] for g in rows)}/{len(rows)}")
    statuses = Counter(g["status"] for g in graded)
    lines.append("statuts : " + ", ".join(f"{s} {c}" for s, c in statuses.most_common()))
    errors = sum(g["grader_error"] is not None for g in graded)
    if errors:
        lines.append(f"{errors} tâche(s) où le correcteur a levé une exception (grader_error)")
    return "\n".join(lines)


def default_out(runs_path) -> Path:
    runs_path = Path(runs_path)
    return runs_path.with_name(f"{runs_path.stem}.graded.jsonl")


def main(argv: list[str] | None = None, server_ok: Callable[[str], bool] = check_server) -> int:
    parser = argparse.ArgumentParser(description="Noter les tâches MedAgentBench avec refsol.py.")
    parser.add_argument("--runs", required=True, help="sortie de mab/loop.py, ex. outputs/mab/mab_g54mini.jsonl")
    parser.add_argument("--out", default=None, help="défaut : <runs>.graded.jsonl (jamais écrasé)")
    parser.add_argument("--medagentbench", default=MAB_DIR, help=f"clone de MedAgentBench (défaut {MAB_DIR})")
    parser.add_argument("--tasks", default=TASKS_PATH)
    parser.add_argument("--funcs", default=FUNCS_PATH)
    args = parser.parse_args(argv)
    out = Path(args.out) if args.out else default_out(args.runs)
    try:
        refsol = load_refsol(args.medagentbench)
        graded = grade_file(args.runs, out, refsol, tasks_path=args.tasks, funcs_path=args.funcs,
                            server_ok=server_ok)
    except (ValueError, OSError) as err:
        print(f"REFUS : {err}", file=sys.stderr)
        return 2
    except ServerUnreachable as err:
        print(f"ARRÊT : {err} ; rien n'a été écrit", file=sys.stderr)
        return 1
    print(summary(graded))
    print(f"refsol.py {refsol.refsol_hash}, utils.py {refsol.utils_hash} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
