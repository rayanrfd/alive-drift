"""Les scores du jury : des votes des juges à une note par réponse.

Pour une réponse et un critère, chaque juge a voté oui (true), non (false) ou
non applicable (null). La valeur du critère est la part de « oui » parmi les
juges qui se sont prononcés : oui + non + N/A donne 1/2. Si tous les juges ont
dit N/A, le critère n'a pas de valeur pour cette réponse.

    score d'un thème = moyenne des valeurs des critères du thème qui en ont une
    score global     = moyenne des valeurs de tous les critères qui en ont une

Une réponse n'a un score que si TOUS les juges demandés l'ont notée : sinon
elle est écartée et comptée, pour que toutes les notes reposent sur le même
jury. Les votes de chaque juge restent aussi notés à part (by_judge).

    uv run python -m alive_drift.scores --judges juge_haiku_latest juge_glm juge_deepseek

lit outputs/jury/<config>.<juge>.jsonl, écrit outputs/scores/<date-heure>/<config>.jsonl
(jamais par-dessus un calcul précédent) et affiche un résumé. Aucun appel à Alive.

Juge « dans le fil » (morceau 36) : il a un déploiement par room, donc un nom par config
(juge_haiku_fil2_L, juge_haiku_fil2_M...). Un nom de juge peut contenir {config}, remplacé
par la config lue :

    uv run python -m alive_drift.scores --judges "juge_haiku_fil2_{config}" --configs L L_bis M

Les lignes de score gardent le nom avec {config} : pour export_richter, c'est le même jury
pour toutes les configs. Avant d'écrire, on vérifie dans les votes que chaque juge a le même
réglage (prompt, modèle servi, raisonnement, mode) dans toutes les configs : sinon ce ne
serait pas le même instrument des deux côtés de Richter.
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

from alive_drift.files import append_line, read_records
from alive_drift.grid import Criterion, grid_fingerprint, load_working_grid, names_by_theme


def read_votes(path, grid: list[Criterion], judge: str) -> dict[str, dict]:
    """{question_id: vote} d'un fichier de votes du jury.

    Lève FileNotFoundError si le fichier n'existe pas, ValueError si une ligne
    vient d'un autre juge ou d'une autre grille, si une question apparaît deux
    fois, ou si les votes ne sont pas un verdict (true, false, null) par critère
    de la grille.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} : fichier de votes introuvable")
    fingerprint = grid_fingerprint(grid)
    names = [c.name for c in grid]
    votes = {}
    for where, vote in read_records(path):
        if vote.get("judge") != judge:
            raise ValueError(f"{where} : juge {vote.get('judge')!r}, attendu {judge!r}")
        if vote.get("grid") != fingerprint:
            raise ValueError(f"{where} : grille {vote.get('grid')!r}, attendu {fingerprint!r}")
        qid = vote.get("question_id")
        if not isinstance(qid, str) or not qid:
            raise ValueError(f"{where} : question_id absent")
        if qid in votes:
            raise ValueError(f"{where} : question {qid!r} en double")
        verdicts = vote.get("verdicts")
        if (not isinstance(verdicts, dict) or list(verdicts) != names
                or any(type(met) not in (bool, type(None)) for met in verdicts.values())):
            raise ValueError(f"{where} : votes incomplets ou mal formés")
        votes[qid] = vote
    return votes


CONFIG_PLACEHOLDER = "{config}"
# Ce qui fait l'instrument d'un juge, lu dans chaque vote (votes.make_vote).
INSTRUMENT_FIELDS = ("judge_prompt", "model", "reasoning_effort", "mode")


def judge_name_for(judge: str, config: str) -> str:
    """Le nom du juge pour `config` : {config} remplacé par la config, sinon inchangé.

    Lève ValueError s'il reste une accolade (modèle de nom mal écrit, ex. {conf}).
    """
    name = judge.replace(CONFIG_PLACEHOLDER, config)
    if "{" in name or "}" in name:
        raise ValueError(f"nom de juge {judge!r} : seul {CONFIG_PLACEHOLDER} est accepté entre accolades")
    return name


def instrument_of(vote: dict) -> tuple:
    """(prompt, modèle servi, raisonnement, mode) d'un vote ; None pour un champ absent."""
    return tuple(vote.get(field) for field in INSTRUMENT_FIELDS)


def judge_instruments(config: str, judges: list[str], grid: list[Criterion],
                      jury_dir="outputs/jury") -> dict[str, tuple]:
    """{juge (nom tel que donné): son instrument} pour `config`.

    Lève ValueError si un juge n'a pas voté avec un seul et même réglage dans cette config.
    """
    instruments = {}
    for judge in judges:
        name = judge_name_for(judge, config)
        votes = read_votes(Path(jury_dir) / f"{config}.{name}.jsonl", grid, name)
        found = {instrument_of(v) for v in votes.values()}
        if len(found) != 1:
            raise ValueError(f"{config}.{name} : plusieurs réglages du juge dans les votes "
                             f"{sorted(map(str, found))}")
        instruments[judge] = found.pop()
    return instruments


def check_same_instruments(by_config: dict[str, dict[str, tuple]]) -> None:
    """Lève ValueError si un juge n'a pas le même instrument dans toutes les configs.

    `by_config` : {config: judge_instruments(config, ...)}. Avec {config}, ce sont des
    déploiements différents (un par room) : ils doivent être le même juge.
    """
    judges = next(iter(by_config.values()), {})
    for judge in judges:
        seen = {config: inst[judge] for config, inst in by_config.items()}
        if len(set(seen.values())) != 1:
            detail = "; ".join(f"{c} : {dict(zip(INSTRUMENT_FIELDS, i))}" for c, i in seen.items())
            raise ValueError(f"le juge {judge!r} n'est pas le même instrument d'une config à "
                             f"l'autre ({detail})")


def mean_or_none(values) -> float | None:
    """La moyenne des valeurs qui ne sont pas None ; None s'il n'y en a aucune."""
    kept = [v for v in values if v is not None]
    return sum(kept) / len(kept) if kept else None


def criterion_values(verdicts: list[dict], names: list[str]) -> dict[str, float | None]:
    """{critère: part de oui parmi les juges qui ne disent pas N/A}, None si tous disent N/A."""
    values = {}
    for name in names:
        said = [v[name] for v in verdicts if v[name] is not None]
        values[name] = sum(said) / len(said) if said else None
    return values


def score_response(votes_by_judge: dict[str, dict], grid: list[Criterion]) -> dict:
    """Les notes d'UNE réponse à partir des votes de plusieurs juges.

    Renvoie {"criteria": {critère: valeur}, "themes": {thème: score},
    "score": score global, "by_judge": {juge: score global de ce juge seul}}.
    Seuls les thèmes qui ont au moins un critère dans la grille apparaissent.
    """
    names = [c.name for c in grid]
    values = criterion_values([v["verdicts"] for v in votes_by_judge.values()], names)
    themes = {theme: mean_or_none(values[n] for n in members)
              for theme, members in names_by_theme(grid).items() if members}
    by_judge = {judge: mean_or_none(criterion_values([vote["verdicts"]], names).values())
                for judge, vote in votes_by_judge.items()}
    return {"criteria": values, "themes": themes, "score": mean_or_none(values.values()),
            "by_judge": by_judge}


def question_order(qid: str):
    """Ordre des questions : 0, 1, 2... puis les autres ids.

    Seuls les chiffres 0-9 comptent comme un numéro : str.isdigit() accepte
    aussi « ² » ou « ٢ », sur lesquels int() échoue.
    """
    numeric = re.fullmatch(r"[0-9]+", qid) is not None
    return (not numeric, int(qid) if numeric else 0, qid)


def score_config(config: str, judges: list[str], grid: list[Criterion],
                 jury_dir="outputs/jury") -> tuple[list[dict], int]:
    """(lignes de score de `config`, nombre de réponses écartées faute de tous les juges).

    Un juge avec {config} : on lit le juge de la room de `config` (judge_name_for),
    mais la ligne garde le nom tel que donné (même jury pour toutes les configs).

    Lève ValueError si la liste de juges est vide ou a un doublon, si un fichier
    contient des votes d'une autre config, ou si, pour une même question, les
    juges n'ont pas noté la même réponse. FileNotFoundError si un fichier manque.
    """
    if not judges or len(set(judges)) != len(judges):
        raise ValueError(f"liste de juges vide ou avec un doublon : {judges}")
    per_judge = {}
    for j in judges:
        name = judge_name_for(j, config)   # j avec {config} : le juge de la room de `config`
        per_judge[j] = read_votes(Path(jury_dir) / f"{config}.{name}.jsonl", grid, name)
    all_ids = set().union(*(set(v) for v in per_judge.values()))
    complete = [q for q in all_ids if all(q in votes for votes in per_judge.values())]
    rows = []
    for qid in sorted(complete, key=question_order):
        votes = {j: per_judge[j][qid] for j in judges}
        executions = {v.get("response_execution_id") for v in votes.values()}
        if len(executions) != 1:
            raise ValueError(f"{config} question {qid} : les juges n'ont pas noté la même réponse "
                             f"({sorted(map(str, executions))})")
        noted = {v.get("judged_config") for v in votes.values()}
        if noted != {config}:
            raise ValueError(f"{config} question {qid} : votes d'une autre config ({sorted(map(str, noted))})")
        rows.append({"question_id": qid, "config": config,
                     "response_execution_id": executions.pop(),
                     "grid": grid_fingerprint(grid), "judges": list(judges),
                     **score_response(votes, grid)})
    return rows, len(all_ids) - len(complete)


def summary(rows_by_config: dict[str, list[dict]], excluded: dict[str, int],
            judges: list[str], grid: list[Criterion]) -> str:
    """Le résumé affiché : moyenne du jury et de chaque juge, puis valeur moyenne par critère."""
    configs = list(rows_by_config)
    fmt = lambda x: "  –  " if x is None else f"{x:5.2f}"
    lines = [f"{'':28}" + "".join(f"{c:>9}" for c in configs),
             f"{'réponses notées':28}" + "".join(f"{len(rows_by_config[c]):>9}" for c in configs),
             f"{'écartées (juge manquant)':28}" + "".join(f"{excluded[c]:>9}" for c in configs),
             f"{'score du jury':28}" + "".join(
                 f"{fmt(mean_or_none(r['score'] for r in rows_by_config[c])):>9}" for c in configs)]
    for judge in judges:
        lines.append(f"{'  ' + judge:28}" + "".join(
            f"{fmt(mean_or_none(r['by_judge'][judge] for r in rows_by_config[c])):>9}" for c in configs))
    lines.append("par critère (part de oui), numéros du prompt des juges :")
    for number, criterion in enumerate(grid, start=1):
        lines.append(f"{'  ' + str(number):28}" + "".join(
            f"{fmt(mean_or_none(r['criteria'][criterion.name] for r in rows_by_config[c])):>9}"
            for c in configs) + f"   {criterion.name}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scores du jury à partir des votes. Aucun appel à Alive.")
    parser.add_argument("--judges", nargs="+", required=True,
                        help='ex. juge_haiku_latest juge_glm ; "juge_x_{config}" = un juge par room')
    parser.add_argument("--configs", nargs="+", default=["A", "A_bis", "B"], help="défaut : A A_bis B")
    parser.add_argument("--jury-dir", default="outputs/jury", help="où sont les votes")
    parser.add_argument("--out-dir", default=None, help="défaut : outputs/scores/<date-heure>")
    parser.add_argument("--grid", default="data/criteria.json", help="les 34 critères")
    parser.add_argument("--selection", default="data/selection.json", help="les critères gardés")
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir or f"outputs/scores/{datetime.now():%Y%m%d-%H%M%S}")
    try:
        if out_dir.exists():
            raise ValueError(f"{out_dir} existe déjà : on n'écrit jamais par-dessus un calcul")
        grid = load_working_grid(args.grid, args.selection)
        rows_by_config, excluded = {}, {}
        for config in args.configs:
            rows_by_config[config], excluded[config] = score_config(config, args.judges, grid,
                                                                    args.jury_dir)
        check_same_instruments({config: judge_instruments(config, args.judges, grid, args.jury_dir)
                                for config in args.configs})
    except (ValueError, OSError, KeyError) as err:
        print(f"REFUS : {err}", file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True)
    for config, rows in rows_by_config.items():
        for row in rows:
            append_line(out_dir / f"{config}.jsonl", row)
    print(summary(rows_by_config, excluded, args.judges, grid))
    print(f"-> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
