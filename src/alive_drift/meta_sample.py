"""L'échantillon figé de la méta-évaluation : des réponses déjà jugées par des médecins.

HealthBench publie, pour chaque critère consensus, des réponses (question +
réponse d'un modèle) que 2 à 4 médecins ont jugées : critère respecté ou non.
On en tire, pour chaque critère de la grille de travail, autant de « oui »
que de « non » (25 et 25 par défaut), pour mesurer ensuite si un juge retrouve
l'avis des médecins (meta_eval.py).

Lignes gardées : une question en un seul tour (comme nos 1 000 questions) et
un avis majoritaire net (2 médecins sur 2 d'accord, 2 sur 3...). Les égalités
sont écartées : on ne sait pas quelle est la bonne réponse.

Écrit deux fichiers, à commiter (ils fixent l'expérience) :

    data/meta_sample.jsonl      une ligne par réponse tirée, avec l'avis des médecins
    data/meta_reference.json    le fichier source, la graine, et l'accord entre
                                médecins par critère (sur toutes les lignes en un tour)

    uv run python -m alive_drift.meta_sample \\
        --source ../drift-audit/archive/clinical_adequacy/data/2025-05-07-06-14-12_oss_meta_eval.jsonl

Refuse si un des deux fichiers existe déjà, ou si le texte d'un critère dans la
source n'est pas exactement celui de data/criteria.json. Aucun appel à Alive.
"""

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

from alive_drift.files import append_line
from alive_drift.grid import Criterion, load_working_grid

CLUSTER_PREFIX = "cluster:"


def majority(labels: list) -> bool | None:
    """L'avis majoritaire des médecins : True, False, ou None en cas d'égalité."""
    yes = sum(1 for label in labels if label is True)
    no = sum(1 for label in labels if label is False)
    if yes == no:
        return None
    return yes > no


def agreeing_pairs(labels: list) -> tuple[int, int]:
    """(paires de médecins d'accord, paires en tout) pour une ligne."""
    yes = sum(1 for label in labels if label is True)
    no = sum(1 for label in labels if label is False)
    n = yes + no
    return yes * (yes - 1) // 2 + no * (no - 1) // 2, n * (n - 1) // 2


def read_source(path, grid: list[Criterion]) -> dict[str, list[dict]]:
    """{critère: lignes candidates} : un seul tour, critère de la grille.

    Une ligne candidate : {"criterion", "labels", "prompt_id", "completion_id",
    "question", "response", "line"}. Les égalités sont gardées ici (elles
    comptent pour l'accord entre médecins) ; le tirage les écarte.

    Lève ValueError si le texte d'un critère diffère de la grille, si une ligne
    est mal formée, si une réponse apparaît deux fois pour un même critère, ou
    si un critère de la grille n'a aucune ligne.
    """
    text_by_name = {c.name: c.text for c in grid}
    rows = {name: [] for name in text_by_name}
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, start=1):
            where = f"{path} ligne {n}"
            row = json.loads(line)
            name = str(row.get("category", "")).removeprefix(CLUSTER_PREFIX)
            if name not in text_by_name:
                continue
            if row.get("rubric") != text_by_name[name]:
                raise ValueError(f"{where} : le texte du critère {name} diffère de data/criteria.json")
            prompt = row.get("prompt")
            if not isinstance(prompt, list) or len(prompt) != 1:
                continue  # conversation en plusieurs tours : pas comme nos questions
            turn = prompt[0]
            labels = row.get("binary_labels")
            if (not isinstance(turn, dict) or turn.get("role") != "user"
                    or not isinstance(turn.get("content"), str) or not turn["content"].strip()
                    or not isinstance(row.get("completion"), str) or not row["completion"].strip()
                    or not isinstance(labels, list) or len(labels) < 2
                    or any(type(label) is not bool for label in labels)
                    or not row.get("completion_id") or not row.get("prompt_id")):
                raise ValueError(f"{where} : ligne mal formée")
            if any(r["completion_id"] == row["completion_id"] for r in rows[name]):
                raise ValueError(f"{where} : réponse {row['completion_id']} deux fois pour {name}")
            rows[name].append({"criterion": name, "labels": labels,
                               "prompt_id": row["prompt_id"], "completion_id": row["completion_id"],
                               "question": turn["content"], "response": row["completion"],
                               "line": n})
    empty = [name for name, found in rows.items() if not found]
    if empty:
        raise ValueError(f"aucune ligne en un tour pour : {empty}")
    return rows


def physician_agreement(rows: dict[str, list[dict]]) -> dict[str, dict]:
    """Par critère, sur toutes les lignes en un tour : avis oui / non / égalité, et
    la part de paires de médecins d'accord (la référence humaine)."""
    result = {}
    for name, candidates in rows.items():
        agree = pairs = 0
        counts = {"yes": 0, "no": 0, "tie": 0}
        for row in candidates:
            a, p = agreeing_pairs(row["labels"])
            agree, pairs = agree + a, pairs + p
            label = majority(row["labels"])
            counts["tie" if label is None else "yes" if label else "no"] += 1
        result[name] = {**counts, "agreeing_pairs": agree, "pairs": pairs,
                        "agreement": agree / pairs}
    return result


def draw(rows: dict[str, list[dict]], per_label: int, seed: int) -> list[dict]:
    """Le tirage : jusqu'à `per_label` « oui » et autant de « non » par critère.

    Les candidats sont triés avant le tirage (le résultat ne dépend pas de
    l'ordre du fichier source). Les lignes sont rendues entrelacées (critère 1
    oui, critère 1 non, critère 2 oui...) : les N premières couvrent tous les
    critères, ce qui rend utile un essai avec --limit.
    """
    rng = random.Random(seed)
    groups = []
    for name in sorted(rows):
        for label in (True, False):
            candidates = sorted((r for r in rows[name] if majority(r["labels"]) is label),
                                key=lambda r: (r["prompt_id"], r["completion_id"], r["line"]))
            picked = rng.sample(candidates, min(per_label, len(candidates)))
            groups.append([{**r, "label": label} for r in picked])
    sample = []
    for k in range(max(len(g) for g in groups)):
        sample.extend(g[k] for g in groups if k < len(g))
    return [{"id": f"m{i:03d}", "criterion": r["criterion"], "label": r["label"],
             "physician_labels": r["labels"], "prompt_id": r["prompt_id"],
             "completion_id": r["completion_id"], "question": r["question"],
             "response": r["response"], "source_line": r["line"]}
            for i, r in enumerate(sample, start=1)]


def file_hash(path) -> str:
    """sha256 du fichier, 12 premiers caractères (même format que drift-audit)."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:12]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tire l'échantillon figé de la méta-évaluation.")
    parser.add_argument("--source", required=True, help="le fichier oss_meta_eval de HealthBench")
    parser.add_argument("--per-label", type=int, default=25, help="« oui » et « non » par critère (défaut 25)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="data/meta_sample.jsonl")
    parser.add_argument("--reference", default="data/meta_reference.json")
    parser.add_argument("--grid", default="data/criteria.json")
    parser.add_argument("--selection", default="data/selection.json")
    args = parser.parse_args(argv)
    out, reference = Path(args.out), Path(args.reference)
    try:
        for path in (out, reference):
            if path.exists():
                raise ValueError(f"{path} existe déjà : l'échantillon est figé, on ne le retire pas")
        if args.per_label < 1:
            raise ValueError(f"--per-label doit valoir au moins 1 (reçu {args.per_label})")
        grid = load_working_grid(args.grid, args.selection)
        rows = read_source(args.source, grid)
        sample = draw(rows, args.per_label, args.seed)
        agreement = physician_agreement(rows)
        source_hash = file_hash(args.source)
    except (ValueError, OSError, KeyError) as err:
        print(f"REFUS : {err}", file=sys.stderr)
        return 2
    out.parent.mkdir(parents=True, exist_ok=True)
    for row in sample:
        append_line(out, row)
    doc = {"source": Path(args.source).name, "source_sha256_12": source_hash,
           "seed": args.seed, "per_label": args.per_label,
           "sample": out.name, "sample_sha256_12": file_hash(out),
           "physicians": agreement}
    reference.parent.mkdir(parents=True, exist_ok=True)
    reference.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    n_runs = len({r["completion_id"] for r in sample})
    print(f"{'critère':60} {'oui':>4} {'non':>4}   accord entre médecins")
    for number, criterion in enumerate(grid, start=1):
        yes = sum(1 for r in sample if r["criterion"] == criterion.name and r["label"])
        no = sum(1 for r in sample if r["criterion"] == criterion.name and not r["label"])
        print(f"{number} {criterion.name[:58]:58} {yes:>4} {no:>4}   "
              f"{agreement[criterion.name]['agreement']:.0%}")
    print(f"{len(sample)} lignes, {n_runs} réponses différentes (= runs par juge) -> {out}, {reference}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
