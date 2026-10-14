"""Les scores du jury au format que drift-audit attend pour le test de Richter.

drift-audit (experiments/phase3_clinical.py, experiments/fpr_permutation.py,
enchaînés par pipeline/run_pipeline1.py) lit un dossier de trois fichiers,
une ligne par question :

    a_hb_clinical_adequacy.jsonl      <- A      (référence)
    a_bis_hb_clinical_adequacy.jsonl  <- A_bis  (contrôle : rien n'a changé)
    b_hb_clinical_adequacy.jsonl      <- B      (dérive à détecter)

    {"id": 12, "clinical_adequacy": 0.75, "jury_fingerprint": "…", ...}

Un score null (tous les juges ont dit N/A) est écrit tel quel : drift-audit
écarte la question. drift-audit refuse aussi de comparer des fichiers dont
l'empreinte de jury diffère ; ici elle résume les juges (dans l'ordre) et la
grille, et on refuse dès l'export des scores qui n'ont pas le même jury.

    uv run python -m alive_drift.export_richter outputs/scores/20261014-144523

écrit outputs/richter/20261014-144523/ (jamais par-dessus) et affiche la
commande de Richter. Aucun appel à Alive. Aucun import de drift-audit : les
deux dépôts ne se parlent que par ces fichiers.
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

from alive_drift.files import append_line, read_records

# Nom de la config ici -> nom attendu par drift-audit (REF, H0, H1 de run_pipeline1).
NAMES = {"A": "a_hb", "A_bis": "a_bis_hb", "B": "b_hb"}


def jury_fingerprint(judges: list[str], grid: str) -> str:
    """12 caractères qui changent si un juge, leur ordre ou la grille change."""
    payload = json.dumps({"judges": judges, "grid": grid}, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def read_scores(path, config: str) -> list[dict]:
    """Les lignes d'un fichier de scores.py, vérifiées.

    Lève FileNotFoundError si le fichier manque, ValueError si une ligne est
    d'une autre config, n'a pas un id numérique (chiffres 0-9, sans zéro en tête), apparaît deux fois, ou n'a
    pas un score entre 0 et 1 (ou null), une liste de juges et une grille.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} : fichier de scores introuvable")
    rows, seen = [], set()
    for where, row in read_records(path):
        if row.get("config") != config:
            raise ValueError(f"{where} : config {row.get('config')!r}, attendu {config!r}")
        qid = row.get("question_id")
        # Chiffres 0-9 seulement, sans zéro en tête : "01" et "1" donneraient le même
        # id entier, et drift-audit garderait le premier sans rien dire.
        if not isinstance(qid, str) or re.fullmatch(r"0|[1-9][0-9]*", qid) is None:
            raise ValueError(f"{where} : question_id {qid!r} n'est pas un numéro")
        if qid in seen:
            raise ValueError(f"{where} : question {qid!r} en double")
        score = row.get("score")
        if score is not None and (type(score) not in (int, float) or not 0 <= score <= 1):
            raise ValueError(f"{where} : score {score!r} hors de [0, 1]")
        judges = row.get("judges")
        if (not isinstance(judges, list) or not judges
                or not all(isinstance(j, str) and j for j in judges)):
            raise ValueError(f"{where} : liste de juges absente")
        if not isinstance(row.get("grid"), str) or not row["grid"]:
            raise ValueError(f"{where} : grille absente")
        seen.add(qid)
        rows.append(row)
    if not rows:
        raise ValueError(f"{path} : aucun score")
    return rows


def export_rows(rows_by_config: dict[str, list[dict]]) -> tuple[str, dict[str, list[dict]]]:
    """(empreinte du jury, {nom drift-audit: lignes à écrire}).

    Lève ValueError si deux lignes, de la même config ou non, n'ont pas été
    notées par le même jury (mêmes juges dans le même ordre, même grille).
    """
    juries = {(tuple(r["judges"]), r["grid"]) for rows in rows_by_config.values() for r in rows}
    if len(juries) != 1:
        raise ValueError(f"les scores ne viennent pas du même jury : {sorted(juries)}")
    (judges, grid), = juries
    fingerprint = jury_fingerprint(list(judges), grid)
    out = {}
    for config, rows in rows_by_config.items():
        out[NAMES[config]] = [{"id": int(r["question_id"]), "clinical_adequacy": r["score"],
                               "jury_fingerprint": fingerprint, "config": config,
                               "response_execution_id": r.get("response_execution_id"),
                               "judges": list(judges), "grid": grid}
                              for r in rows]
    return fingerprint, out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Scores du jury -> fichiers d'entrée de Richter (drift-audit). Aucun appel à Alive.")
    parser.add_argument("scores_dir", help="un dossier écrit par scores.py, ex. outputs/scores/20261014-144523")
    parser.add_argument("--out-dir", default=None,
                        help="défaut : outputs/richter/<nom du dossier de scores>")
    args = parser.parse_args(argv)
    scores_dir = Path(args.scores_dir)
    out_dir = Path(args.out_dir or Path("outputs") / "richter" / scores_dir.name)
    try:
        if out_dir.exists():
            raise ValueError(f"{out_dir} existe déjà : on n'écrit jamais par-dessus")
        rows_by_config = {c: read_scores(scores_dir / f"{c}.jsonl", c) for c in NAMES}
        fingerprint, out = export_rows(rows_by_config)
    except (ValueError, OSError) as err:
        print(f"REFUS : {err}", file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True)
    for name, rows in out.items():
        for row in rows:
            append_line(out_dir / f"{name}_clinical_adequacy.jsonl", row)
        nulls = sum(r["clinical_adequacy"] is None for r in rows)
        print(f"{name:9} {len(rows):5} questions, dont {nulls} sans score (tout N/A)")
    print(f"empreinte du jury : {fingerprint}")
    print(f"-> {out_dir}")
    print("Richter (depuis le dossier pipeline) :\n"
          f"  uv run python run_pipeline1.py --scores-dir {out_dir.resolve()} "
          f"--out-dir results/alive_{scores_dir.name} --run-id alive_{scores_dir.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
