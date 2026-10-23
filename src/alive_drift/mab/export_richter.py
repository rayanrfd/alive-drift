"""MedAgentBench : les notes (1 juste / 0 faux) au format que drift-audit attend pour Richter (morceau 33).

Même sortie que alive_drift.export_richter (HealthBench), pour réutiliser tel quel
pipeline/run_pipeline1.py -> drift-audit (phase3_clinical.py, fpr_permutation.py) :

    a_hb_clinical_adequacy.jsonl      <- --ref  (référence)
    a_bis_hb_clinical_adequacy.jsonl  <- --h0   (copie de la référence : rien n'a changé)
    b_hb_clinical_adequacy.jsonl      <- --h1   (l'agent dont on cherche la dérive)

    {"id": 17, "clinical_adequacy": 1.0, "task_id": "task4_12", "jury_fingerprint": "…", ...}

Les noms « hb » et « clinical_adequacy » sont ceux qu'attend drift-audit ; ici le score
est la note de refsol.py, 0.0 ou 1.0. Le test de Richter le prend tel quel : φ reçoit un
score dans [0, 1], qui ne vaut que 0 ou 1.

Deux adaptations :
- drift-audit veut des ids entiers et range les paires par id. Rangées par nom de tâche,
  les 30 tâches d'un même type se suivraient (lots de 25 presque d'un seul type). On
  numérote donc les 300 tâches dans un ordre tiré au hasard, avec une graine fixée
  (--order-seed, enregistrée sur chaque ligne) : même numéro pour une tâche dans les
  trois fichiers, donc l'appariement est intact.
- « jury_fingerprint » (drift-audit refuse de comparer deux empreintes différentes)
  résume ici l'instrument : refsol.py, utils.py, les tâches et le serveur FHIR.

Écrit outputs/richter/<--name>/ (jamais par-dessus) et affiche la commande de Richter.
Aucun appel à Alive ni au serveur FHIR. Aucun import de drift-audit.

    uv run python -m alive_drift.mab.export_richter --name mab_essai \\
        --ref outputs/mab/A.graded.jsonl --h0 outputs/mab/A_bis.graded.jsonl --h1 outputs/mab/B.graded.jsonl
"""

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

from alive_drift.files import append_line, read_records
from alive_drift.mab.tasks import TASKS_PATH, load_tasks

ORDER_SEED = 0
NAMES = {"ref": "a_hb", "h0": "a_bis_hb", "h1": "b_hb"}      # les noms de drift-audit
INSTRUMENT = ("refsol", "refsol_utils", "fingerprint", "api_base")
SAME_IN_FILE = ("config", "deployment_id", "model", "reasoning_effort") + INSTRUMENT


def task_numbers(task_ids: list[str], seed: int = ORDER_SEED) -> dict[str, int]:
    """{tâche: numéro} : les tâches numérotées 0, 1, 2... dans un ordre tiré avec `seed`.

    Ne dépend que de la liste des tâches et de la graine : les trois fichiers ont donc
    les mêmes numéros, quelles que soient les tâches faites par chaque agent.
    """
    order = list(task_ids)
    random.Random(seed).shuffle(order)
    return {tid: i for i, tid in enumerate(order)}


def instrument_fingerprint(line: dict) -> str:
    """12 caractères qui changent si refsol.py, utils.py, les tâches ou le serveur changent."""
    payload = json.dumps({k: line[k] for k in INSTRUMENT}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def read_graded(path, known: set[str]) -> list[dict]:
    """Les lignes de mab/grade.py, vérifiées.

    Lève FileNotFoundError si le fichier manque, ValueError s'il est vide, si une tâche
    est inconnue ou en double, si `correct` n'est pas un booléen, ou si les lignes ne
    viennent pas toutes de la même config et du même instrument.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} : fichier de notes introuvable")
    rows, seen = [], set()
    for where, row in read_records(path):
        tid = row.get("task_id")
        if tid not in known:
            raise ValueError(f"{where} : tâche {tid!r} inconnue")
        if tid in seen:
            raise ValueError(f"{where} : tâche {tid!r} en double")
        if type(row.get("correct")) is not bool:
            raise ValueError(f"{where} : correct {row.get('correct')!r} n'est pas vrai/faux")
        for field in SAME_IN_FILE:
            if field not in row:
                raise ValueError(f"{where} : champ {field!r} absent (fichier de mab/grade.py ?)")
            if rows and row[field] != rows[0][field]:
                raise ValueError(f"{where} : {field} {row[field]!r}, différent de la 1re ligne "
                                 f"({rows[0][field]!r})")
        seen.add(tid)
        rows.append(row)
    if not rows:
        raise ValueError(f"{path} : aucune note")
    return rows


def export_rows(graded: dict[str, list[dict]], numbers: dict[str, int],
                order_seed: int) -> tuple[str, dict[str, list[dict]]]:
    """(empreinte de l'instrument, {nom drift-audit: lignes à écrire, rangées par id}).

    `graded` : {"ref": lignes, "h0": lignes, "h1": lignes}. Lève ValueError si deux
    rôles ont la même config, si h0 n'a pas le modèle et l'effort de ref, ou si les
    notes ne viennent pas du même instrument.
    """
    configs = {role: rows[0]["config"] for role, rows in graded.items()}
    if len(set(configs.values())) != 3:
        raise ValueError(f"il faut trois configs différentes : {configs}")
    ref, h0 = graded["ref"][0], graded["h0"][0]
    for field in ("model", "reasoning_effort"):
        if ref[field] != h0[field]:
            raise ValueError(f"--h0 doit être une copie de --ref : {field} {h0[field]!r} "
                             f"au lieu de {ref[field]!r}")
    prints = {role: instrument_fingerprint(rows[0]) for role, rows in graded.items()}
    if len(set(prints.values())) != 1:
        raise ValueError(f"les notes ne viennent pas du même instrument (refsol, tâches, serveur) : {prints}")
    fingerprint = prints["ref"]
    out = {}
    for role, rows in graded.items():
        lines = [{"id": numbers[r["task_id"]], "clinical_adequacy": float(r["correct"]),
                  "task_id": r["task_id"], "task_type": r["task_type"], "status": r["status"],
                  "jury_fingerprint": fingerprint, "config": r["config"],
                  "execution_ids": r["execution_ids"], "order_seed": order_seed,
                  "refsol": r["refsol"], "tasks_fingerprint": r["fingerprint"]} for r in rows]
        out[NAMES[role]] = sorted(lines, key=lambda line: line["id"])
    return fingerprint, out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Notes MedAgentBench -> fichiers d'entrée de Richter (drift-audit). Aucun appel réseau.")
    parser.add_argument("--ref", required=True, help="notes de la référence (mab/grade.py)")
    parser.add_argument("--h0", required=True, help="notes de la copie de contrôle de la référence")
    parser.add_argument("--h1", required=True, help="notes de l'agent dont on cherche la dérive")
    parser.add_argument("--name", required=True, help="nom de l'export, ex. mab_g54mini_g55mini")
    parser.add_argument("--out-dir", default=None, help="défaut : outputs/richter/<name>")
    parser.add_argument("--order-seed", type=int, default=ORDER_SEED,
                        help=f"graine de l'ordre des tâches (défaut {ORDER_SEED} ; à fixer avant de voir les résultats)")
    parser.add_argument("--tasks", default=TASKS_PATH)
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir or Path("outputs") / "richter" / args.name)
    try:
        if out_dir.exists():
            raise ValueError(f"{out_dir} existe déjà : on n'écrit jamais par-dessus")
        task_ids = [t.id for t in load_tasks(args.tasks)]
        known = set(task_ids)
        graded = {role: read_graded(getattr(args, role), known) for role in NAMES}
        fingerprint, out = export_rows(graded, task_numbers(task_ids, args.order_seed), args.order_seed)
    except (ValueError, OSError) as err:
        print(f"REFUS : {err}", file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True)
    common = set.intersection(*({line["task_id"] for line in rows} for rows in out.values()))
    for name, rows in out.items():
        for row in rows:
            append_line(out_dir / f"{name}_clinical_adequacy.jsonl", row)
        right = sum(r["clinical_adequacy"] for r in rows)
        print(f"{name:9} ({rows[0]['config']}) {len(rows):4} tâches, {right:.0f} justes, "
              f"{len(rows) - sum(r['task_id'] in common for r in rows)} absentes d'un autre fichier")
    print(f"{len(common)} tâches dans les trois fichiers (les autres seront écartées par drift-audit)")
    print(f"empreinte de l'instrument : {fingerprint}, ordre des tâches : graine {args.order_seed}")
    print(f"-> {out_dir}")
    run = f"alive_{out_dir.name}"
    print("Richter (depuis le dossier pipeline) :\n"
          f"  uv run python run_pipeline1.py --scores-dir {out_dir.resolve()} "
          f"--out-dir results/{run} --run-id {run}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
