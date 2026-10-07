"""Reconstruit la grille depuis le jeu HealthBench, pour vérifier data/criteria.json.

Dans HealthBench Consensus, chaque conversation a une liste de critères
("rubrics"). Ceux qui portent une étiquette `cluster:<nom>` sont les critères
consensus : le même texte, réutilisé d'une conversation à l'autre. On les
regroupe par nom, et on écarte les 3 critères `*_language`.

    uv run python -m alive_drift.build_grid

télécharge le fichier (public, sur Hugging Face, aucun appel à Alive) et dit si
la grille reconstruite est identique à data/criteria.json. Avec un fichier
déjà téléchargé : --source chemin/vers/consensus.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

import httpx

from alive_drift.grid import theme_of

DATASET = "openai/healthbench"
FILE = "consensus_2025-05-09-20-00-46.jsonl"
URL = f"https://huggingface.co/datasets/{DATASET}/resolve/main/{FILE}"
CLUSTER_PREFIX = "cluster:"
EXCLUDED_SUFFIX = "_language"  # absents de la méta-évaluation : jury non validé dessus


def extract_criteria(rows: list[dict]) -> list[dict]:
    """Les critères consensus trouvés dans `rows`, triés par nom, sans les *_language.

    Lève ValueError si un même nom porte deux textes ou deux poids différents :
    une grille fixe exige un seul texte par critère.
    """
    texts: dict[str, set] = {}
    points: dict[str, set] = {}
    for row in rows:
        for item in row.get("rubrics") or []:
            for tag in item.get("tags") or []:
                if not tag.startswith(CLUSTER_PREFIX):
                    continue
                name = tag.removeprefix(CLUSTER_PREFIX)
                texts.setdefault(name, set()).add(str(item.get("criterion", "")))
                points.setdefault(name, set()).add(float(item.get("points", 0.0)))
    criteria = []
    for name in sorted(texts):
        if name.endswith(EXCLUDED_SUFFIX):
            continue
        if len(texts[name]) != 1:
            raise ValueError(f"{name} : {len(texts[name])} textes différents")
        if len(points[name]) != 1:
            raise ValueError(f"{name} : poids différents {sorted(points[name])}")
        criteria.append({"name": name, "theme": theme_of(name),
                         "points": points[name].pop(), "text": texts[name].pop()})
    return criteria


def read_rows(text: str) -> list[dict]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def compare(built: list[dict], stored: list[dict]) -> list[str]:
    """Les différences entre la grille reconstruite et celle du dépôt ([] si identiques)."""
    a = {c["name"]: c for c in built}
    b = {c["name"]: c for c in stored}
    problems = [f"absent du dépôt : {n}" for n in sorted(set(a) - set(b))]
    problems += [f"absent du jeu de données : {n}" for n in sorted(set(b) - set(a))]
    for name in sorted(set(a) & set(b)):
        for field in ("theme", "points", "text"):
            if a[name][field] != b[name][field]:
                problems.append(f"{name} : {field} différent")
    return problems


def main(argv: list[str] | None = None, transport: httpx.BaseTransport | None = None) -> int:
    parser = argparse.ArgumentParser(description="Vérifie data/criteria.json contre HealthBench.")
    parser.add_argument("--source", help="fichier consensus déjà téléchargé (sinon : téléchargement)")
    parser.add_argument("--grid", default="data/criteria.json", help="la grille du dépôt")
    args = parser.parse_args(argv)
    try:
        if args.source:
            text = Path(args.source).read_text(encoding="utf-8")
        else:
            print(f"téléchargement de {URL} ...", file=sys.stderr)
            with httpx.Client(follow_redirects=True, timeout=300.0, transport=transport) as client:
                response = client.get(URL)
                response.raise_for_status()
                text = response.text
        built = extract_criteria(read_rows(text))
        stored = json.loads(Path(args.grid).read_text(encoding="utf-8"))["criteria"]
    except (OSError, ValueError, KeyError, httpx.HTTPError) as err:
        print(f"REFUS : {type(err).__name__} : {err}", file=sys.stderr)
        return 2
    problems = compare(built, stored)
    for problem in problems:
        print(f"DIFFÉRENCE : {problem}")
    if problems:
        return 1
    print(f"{len(built)} critères reconstruits depuis {FILE} : identiques à {args.grid}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
