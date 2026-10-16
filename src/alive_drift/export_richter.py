"""Les scores du jury au format que drift-audit attend pour le test de Richter.

drift-audit (experiments/phase3_clinical.py, experiments/fpr_permutation.py,
enchaînés par pipeline/run_pipeline1.py) lit un dossier de trois fichiers,
une ligne par question :

    a_hb_clinical_adequacy.jsonl      <- A      (référence ; --ref pour une autre config)
    a_bis_hb_clinical_adequacy.jsonl  <- A_bis  (contrôle : rien n'a changé ; --h0)
    b_hb_clinical_adequacy.jsonl      <- B      (dérive à détecter ; --h1 C pour un autre agent)

    {"id": 12, "clinical_adequacy": 0.75, "jury_fingerprint": "…", ...}

Un score null (tous les juges ont dit N/A) est écrit tel quel : drift-audit
écarte la question. drift-audit refuse aussi de comparer des fichiers dont
l'empreinte de jury diffère ; ici elle résume les juges (dans l'ordre) et la
grille, et on refuse dès l'export des scores qui n'ont pas le même jury.

    uv run python -m alive_drift.export_richter outputs/scores/20261014-144523

    uv run python -m alive_drift.export_richter outputs/scores/jury3_C --h1 C

    uv run python -m alive_drift.export_richter outputs/scores/luna --ref L --h0 L_bis --h1 M

écrit outputs/richter/<nom du dossier de scores>/ (jamais par-dessus) et affiche la
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

def names_for(h1: str = "B", ref: str = "A", h0: str = "A_bis") -> dict[str, str]:
    """Nom de la config ici -> nom attendu par drift-audit (REF, H0, H1 de run_pipeline1).

    `ref` est la référence (A par défaut), `h0` sa copie de contrôle (A_bis), `h1`
    l'agent dont on cherche la dérive (B par défaut, ou C...). Lève ValueError si
    un nom est vide ou si deux des trois sont la même config.
    """
    if not ref or not h0 or ref == h0:
        raise ValueError(f"--ref {ref!r} et --h0 {h0!r} : il faut deux configs différentes")
    if not h1 or h1 in (ref, h0):
        raise ValueError(f"--h1 {h1!r} : la dérive à détecter ne peut pas être {ref} ni {h0}")
    return {ref: "a_hb", h0: "a_bis_hb", h1: "b_hb"}


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


def export_rows(rows_by_config: dict[str, list[dict]],
                names: dict[str, str] | None = None) -> tuple[str, dict[str, list[dict]]]:
    """(empreinte du jury, {nom drift-audit: lignes à écrire}).

    Lève ValueError si deux lignes, de la même config ou non, n'ont pas été
    notées par le même jury (mêmes juges dans le même ordre, même grille).
    """
    juries = {(tuple(r["judges"]), r["grid"]) for rows in rows_by_config.values() for r in rows}
    if len(juries) != 1:
        raise ValueError(f"les scores ne viennent pas du même jury : {sorted(juries)}")
    (judges, grid), = juries
    fingerprint = jury_fingerprint(list(judges), grid)
    names = names or names_for()
    out = {}
    for config, rows in rows_by_config.items():
        out[names[config]] = [{"id": int(r["question_id"]), "clinical_adequacy": r["score"],
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
    parser.add_argument("--h1", default="B",
                        help="l'agent dont on cherche la dérive, écrit en b_hb (défaut : B)")
    parser.add_argument("--ref", default="A", help="la référence, écrite en a_hb (défaut : A)")
    parser.add_argument("--h0", default="A_bis",
                        help="la copie de contrôle de la référence, écrite en a_bis_hb (défaut : A_bis)")
    args = parser.parse_args(argv)
    scores_dir = Path(args.scores_dir)
    out_dir = Path(args.out_dir or Path("outputs") / "richter" / scores_dir.name)
    try:
        if out_dir.exists():
            raise ValueError(f"{out_dir} existe déjà : on n'écrit jamais par-dessus")
        names = names_for(args.h1, args.ref, args.h0)
        rows_by_config = {c: read_scores(scores_dir / f"{c}.jsonl", c) for c in names}
        fingerprint, out = export_rows(rows_by_config, names)
    except (ValueError, OSError) as err:
        print(f"REFUS : {err}", file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True)
    for name, rows in out.items():
        for row in rows:
            append_line(out_dir / f"{name}_clinical_adequacy.jsonl", row)
        nulls = sum(r["clinical_adequacy"] is None for r in rows)
        print(f"{name:9} ({rows[0]['config']:5}) {len(rows):5} questions, dont {nulls} sans score (tout N/A)")
    print(f"empreinte du jury : {fingerprint}")
    print(f"-> {out_dir}")
    # Nom du run = nom du dossier d'export, pas du dossier de scores : un même dossier de
    # scores peut donner plusieurs exports (--h1 C, --ref L...), et run_pipeline1 écrase
    # son --out-dir sans prévenir. Deux exports différents -> deux commandes différentes.
    run = f"alive_{out_dir.name}"
    print("Richter (depuis le dossier pipeline) :\n"
          f"  uv run python run_pipeline1.py --scores-dir {out_dir.resolve()} "
          f"--out-dir results/{run} --run-id {run}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
