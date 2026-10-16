"""MedAgentBench : les tâches et le premier message envoyé au modèle.

Lit les 300 tâches et les 9 fonctions FHIR (data/medagentbench/, copiés de
MedAgentBench, voir SOURCE.md) et fabrique, pour une tâche, le premier message de la
conversation, identique caractère pour caractère à celui de MedAgentBench : même
gabarit, mêmes remplacements, liste des fonctions écrite par json.dumps comme chez eux.
Comme chez eux, tout est dans ce message utilisateur (pas de prompt système).

Aucun appel réseau : ni Alive, ni serveur FHIR.

    uv run python -m alive_drift.mab.tasks                  # résumé et empreintes
    uv run python -m alive_drift.mab.tasks --task task5_1   # le message d'une tâche
    uv run python -m alive_drift.mab.tasks --template       # le gabarit brut
"""

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

TASKS_PATH = "data/medagentbench/test_data_v2.json"
FUNCS_PATH = "data/medagentbench/funcs_v1.json"
FHIR_API_BASE = "http://localhost:8080/fhir/"     # celui de configs/tasks/medagentbench.yaml
TASK_ID = re.compile(r"(task[1-9][0-9]*)_[1-9][0-9]*")

# Recopié tel quel de MedAgentBench, src/server/tasks/medagentbench/__init__.py
# (variable MedAgentBench_prompt), commit 9926011, licence MIT. Ne pas retoucher :
# tests/mab/test_tasks.py vérifie son empreinte.
PROMPT_TEMPLATE = """You are an expert in using FHIR functions to assist medical professionals. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.

1. If you decide to invoke a GET function, you MUST put it in the format of
GET url?param_name1=param_value1&param_name2=param_value2...

2. If you decide to invoke a POST function, you MUST put it in the format of
POST url
[your payload data in JSON format]

3. If you have got answers for all the questions and finished all the requested tasks, you MUST call to finish the conversation in the format of (make sure the list is JSON loadable.)
FINISH([answer1, answer2, ...])

Your response must be in the format of one of the three cases, and you can call only one function each time. You SHOULD NOT include any other text in the response.

Here is a list of functions in JSON format that you can invoke. Note that you should use {api_base} as the api_base.
{functions}

Context: {context}
Question: {question}"""


@dataclass(frozen=True)
class Task:
    id: str                 # ex. "task4_12"
    type: str               # ex. "task4" : le correcteur dépend du type
    instruction: str        # la question, telle quelle
    context: str            # heure figée, codes... ("" pour certaines tâches)
    eval_mrn: str | None    # le patient ; absent pour deux tâches 1 (« Patient not found »)
    sol: list | None        # réponse attendue, seulement pour les tâches 1


def _read_json_list(path, what: str) -> list:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} : fichier des {what} introuvable")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as err:
        raise ValueError(f"{path} : JSON illisible ({err})") from err
    if not isinstance(data, list) or not data:
        raise ValueError(f"{path} : une liste non vide de {what} est attendue")
    return data


def load_tasks(path=TASKS_PATH) -> list[Task]:
    """Les tâches, dans l'ordre du fichier.

    Lève ValueError si une tâche n'est pas un objet, s'il manque id, instruction ou
    context (textes), si un id n'a pas la forme taskN_M ou apparaît deux fois, si
    eval_MRN n'est pas un texte, s'il manque hors des tâches 1, ou si sol n'est pas
    une liste.
    """
    tasks, seen = [], set()
    for i, row in enumerate(_read_json_list(path, "tâches"), start=1):
        where = f"{path}, tâche n° {i}"
        if not isinstance(row, dict):
            raise ValueError(f"{where} : un objet est attendu")
        for key in ("id", "instruction", "context"):
            if not isinstance(row.get(key), str):
                raise ValueError(f"{where} : champ {key!r} absent ou pas un texte")
        match = TASK_ID.fullmatch(row["id"])
        if match is None:
            raise ValueError(f"{where} : id {row['id']!r} n'a pas la forme taskN_M")
        if row["id"] in seen:
            raise ValueError(f"{where} : id {row['id']!r} en double")
        mrn = row.get("eval_MRN")
        if mrn is not None and not isinstance(mrn, str):
            raise ValueError(f"{where} : eval_MRN n'est pas un texte")
        if mrn is None and match.group(1) != "task1":
            raise ValueError(f"{where} : eval_MRN absent (seules les tâches 1 peuvent s'en passer)")
        sol = row.get("sol")
        if sol is not None and not isinstance(sol, list):
            raise ValueError(f"{where} : sol n'est pas une liste")
        seen.add(row["id"])
        tasks.append(Task(id=row["id"], type=match.group(1), instruction=row["instruction"],
                          context=row["context"], eval_mrn=mrn, sol=sol))
    return tasks


def load_funcs(path=FUNCS_PATH) -> list[dict]:
    """Les fonctions FHIR, telles quelles (elles sont recopiées dans le message).

    Lève ValueError si une fonction n'est pas un objet avec name et description (textes).
    """
    funcs = _read_json_list(path, "fonctions")
    for i, func in enumerate(funcs, start=1):
        if (not isinstance(func, dict) or not isinstance(func.get("name"), str)
                or not isinstance(func.get("description"), str)):
            raise ValueError(f"{path}, fonction n° {i} : objet avec name et description attendu")
    return funcs


def first_message(task: Task, funcs: list[dict], api_base: str = FHIR_API_BASE) -> str:
    """Le premier message de la conversation, comme MedAgentBench le fabrique.

    `{api_base}` est remplacé dans la consigne, pas dans la liste des fonctions : leurs
    noms gardent le texte « {api_base} », exactement comme chez MedAgentBench.
    """
    return PROMPT_TEMPLATE.format(api_base=api_base, functions=json.dumps(funcs),
                                  context=task.context, question=task.instruction)


def file_hash(path) -> str:
    """sha256 du fichier, 12 premiers caractères (même format que drift-audit)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def tasks_fingerprint(tasks_path=TASKS_PATH, funcs_path=FUNCS_PATH) -> dict[str, str]:
    """Empreintes à enregistrer avec chaque résultat : tâches, fonctions et gabarit."""
    return {"tasks": file_hash(tasks_path), "funcs": file_hash(funcs_path),
            "template": hashlib.sha256(PROMPT_TEMPLATE.encode("utf-8")).hexdigest()[:12]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Tâches MedAgentBench et premier message envoyé au modèle. Aucun appel réseau.")
    parser.add_argument("--tasks", default=TASKS_PATH, help="les tâches (JSON)")
    parser.add_argument("--funcs", default=FUNCS_PATH, help="les fonctions FHIR (JSON)")
    parser.add_argument("--api-base", default=FHIR_API_BASE, help="adresse du serveur FHIR")
    shown = parser.add_mutually_exclusive_group()
    shown.add_argument("--task", default=None, help="affiche le premier message de cette tâche (ex. task5_1)")
    shown.add_argument("--template", action="store_true", help="affiche le gabarit brut")
    args = parser.parse_args(argv)
    if args.template:
        print(PROMPT_TEMPLATE)
        return 0
    try:
        tasks = load_tasks(args.tasks)
        funcs = load_funcs(args.funcs)
        prints = tasks_fingerprint(args.tasks, args.funcs)
    except (ValueError, OSError) as err:
        print(f"REFUS : {err}", file=sys.stderr)
        return 2
    if args.task is not None:
        found = [t for t in tasks if t.id == args.task]
        if not found:
            print(f"REFUS : tâche {args.task!r} introuvable dans {args.tasks}", file=sys.stderr)
            return 2
        print(first_message(found[0], funcs, args.api_base))
        return 0
    by_type = Counter(t.type for t in tasks)
    print(f"{len(tasks)} tâches, {len(funcs)} fonctions")
    for kind in sorted(by_type, key=lambda k: int(k[len("task"):])):
        print(f"  {kind:7} {by_type[kind]:4}")
    print(f"empreintes : tâches {prints['tasks']}, fonctions {prints['funcs']}, gabarit {prints['template']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
