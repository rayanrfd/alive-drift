"""Résume un fichier de réponses et son journal d'erreurs. Ne lance aucun run.

    uv run python -m alive_drift.summary outputs/A.jsonl

À lancer après chaque campagne (ou pendant, dans un autre terminal) pour voir
ce qui manque, qui a servi les réponses, si elles semblent tronquées, ce que ça
a coûté, et pourquoi les échecs ont eu lieu.
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

from alive_drift.files import errors_path_for, load_questions


# --- Lecture -------------------------------------------------------------------

def read_jsonl(path) -> list[dict]:
    """Toutes les lignes d'un fichier JSONL. Lève ValueError sur une ligne illisible."""
    rows = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            row = None
        if not isinstance(row, dict):
            raise ValueError(f"{path} ligne {n} : ligne illisible")
        rows.append(row)
    return rows


# --- Petits calculs ---------------------------------------------------------------

def numbers(rows: list[dict], key: str) -> list:
    """Les valeurs numériques du champ `key`, triées. Les absentes (None) sont ignorées."""
    return sorted(r[key] for r in rows if type(r.get(key)) in (int, float))


def percentile(sorted_values: list, fraction: float):
    """La valeur située à `fraction` (0.5 = médiane, 0.95 = 95e centile) d'une liste triée."""
    return sorted_values[round(fraction * (len(sorted_values) - 1))]


def counts(rows: list[dict], key: str) -> dict:
    """{valeur: nombre de lignes} pour le champ `key`."""
    return dict(Counter(str(r.get(key)) for r in rows))


def short_detail(error: dict) -> str:
    """Le message d'une erreur, réduit à une ligne courte pour pouvoir regrouper."""
    detail = error.get("detail")
    if not detail:
        return "(pas de message)"
    title = re.search(r"<title>(.*?)</title>", detail, flags=re.S)
    if title:  # page HTML du serveur web (nginx) : on ne garde que son titre
        return f"page HTML : {' '.join(title.group(1).split())}"
    return " ".join(str(detail).split())[:80]


# --- Le résumé --------------------------------------------------------------------

def summarize(out_path, questions: list[dict] | None = None) -> dict:
    """Résume out_path et son journal d'erreurs (s'il existe).

    `questions` : la liste complète des questions, pour savoir lesquelles manquent.
    """
    out_path = Path(out_path)
    rows = read_jsonl(out_path)
    errors_path = errors_path_for(out_path)
    errors = read_jsonl(errors_path) if errors_path.exists() else []

    done = {r.get("question_id") for r in rows}
    tokens = numbers(rows, "completion_tokens")
    credits = numbers(rows, "credits_used")
    durations = numbers(rows, "duration_ms")
    reasoning = numbers(rows, "reasoning_tokens")
    times = sorted(r["written_at"] for r in rows + errors if r.get("written_at"))

    return {
        # Ce qui est fait
        "responses": len(rows),
        "distinct_questions": len(done),            # < responses : une question en double
        "distinct_executions": len({r.get("execution_id") for r in rows}),
        "missing": None if questions is None else [q["id"] for q in questions if q["id"] not in done],
        # D'où viennent les réponses : une seule valeur attendue par ligne
        "configs": counts(rows, "config"),
        "deployments": counts(rows, "deployment_id"),
        "models": counts(rows, "model"),
        "providers": counts(rows, "provider"),
        "reasoning_efforts": counts(rows, "reasoning_effort"),
        "reasoning_tokens_max": reasoning[-1] if reasoning else None,
        # Longueur des réponses. at_max = nombre de réponses à la longueur maximale :
        # s'il y en a plusieurs, elles sont peut-être coupées à un plafond.
        "completion_tokens": None if not tokens else {
            "min": tokens[0], "median": percentile(tokens, 0.5),
            "p95": percentile(tokens, 0.95), "max": tokens[-1],
            "at_max": tokens.count(tokens[-1]),
        },
        # Coût et durée
        "credits_total": round(sum(credits), 4),
        "credits_mean": round(sum(credits) / len(credits), 4) if credits else None,
        "duration_ms_mean": round(sum(durations) / len(durations)) if durations else None,
        "duration_ms_max": durations[-1] if durations else None,
        "first_written_at": times[0] if times else None,
        "last_written_at": times[-1] if times else None,
        # Échecs. never_succeeded = en erreur ET toujours sans réponse.
        "error_lines": len(errors),
        "never_succeeded": sorted({e.get("question_id") for e in errors} - done, key=str),
        "error_kinds": dict(Counter(f"{e.get('error_type')} | {short_detail(e)}" for e in errors)),
    }


def format_summary(s: dict) -> str:
    """Le résumé en texte, une information par ligne."""
    lines = [
        f"réponses            : {s['responses']} "
        f"({s['distinct_questions']} questions, {s['distinct_executions']} execution_id)",
    ]
    if s["distinct_questions"] != s["responses"] or s["distinct_executions"] != s["responses"]:
        lines.append("  ATTENTION : des questions ou des execution_id sont en double")
    if s["missing"] is not None:
        preview = ", ".join(s["missing"][:10]) + (" ..." if len(s["missing"]) > 10 else "")
        lines.append(f"questions manquantes : {len(s['missing'])}" + (f" ({preview})" if preview else ""))
    for label, key in (("config", "configs"), ("déploiement", "deployments"), ("modèle", "models"),
                       ("provider", "providers"), ("effort", "reasoning_efforts")):
        lines.append(f"{label:<20}: {s[key]}")
        if len(s[key]) > 1:
            lines.append(f"  ATTENTION : plusieurs valeurs pour {label}")
    lines.append(f"raisonnement max    : {s['reasoning_tokens_max']} tokens")
    t = s["completion_tokens"]
    if t:
        lines.append(f"tokens de sortie    : min {t['min']}, médiane {t['median']}, "
                     f"95 % sous {t['p95']}, max {t['max']}")
        if t["at_max"] > 1:
            lines.append(f"  ATTENTION : {t['at_max']} réponses font exactement {t['max']} tokens "
                         "(troncature possible)")
    lines.append(f"crédits             : {s['credits_total']} au total, {s['credits_mean']} par réponse")
    lines.append(f"durée d'un run      : {s['duration_ms_mean']} ms en moyenne, {s['duration_ms_max']} ms au plus")
    if s["first_written_at"]:
        lines.append(f"période             : {s['first_written_at']} -> {s['last_written_at']}")
    else:
        lines.append("période             : inconnue (lignes écrites avant l'ajout de l'heure)")
    lines.append(f"lignes d'erreur     : {s['error_lines']} "
                 f"({len(s['never_succeeded'])} questions toujours sans réponse)")
    for kind, n in sorted(s["error_kinds"].items(), key=lambda item: -item[1]):
        lines.append(f"  {n:>4} x {kind}")
    return "\n".join(lines)


# --- Ligne de commande -----------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Résume un fichier de réponses.")
    parser.add_argument("out", help="fichier de réponses, ex. outputs/A.jsonl")
    parser.add_argument("--questions", default="data/healthbench.jsonl",
                        help="JSONL {id, text}, pour lister les questions manquantes")
    args = parser.parse_args(argv)
    try:
        questions = load_questions(args.questions)
        summary = summarize(args.out, questions)
    except (ValueError, OSError) as err:
        print(f"REFUS : {err}", file=sys.stderr)
        return 2
    print(format_summary(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
