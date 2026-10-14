"""La méta-évaluation : un juge retrouve-t-il l'avis des médecins ?

On fait noter par le juge, TEL QU'IL EST DÉPLOYÉ (même prompt, mêmes 8
critères, « non applicable » permis), les réponses de data/meta_sample.jsonl.
Pour chaque ligne, on regarde son verdict sur LE critère que les médecins ont
jugé, et on le compare à leur avis majoritaire.

    oui des médecins, juge « respecté »          -> juste
    non des médecins, juge « non respecté »      -> juste
    tout le reste, « non applicable » compris     -> faux

Mesure par critère : la justesse équilibrée = moitié (part des « oui » retrouvés)
+ moitié (part des « non » retrouvés). 0.5 = pas mieux que le hasard, 1 = parfait.
Le « jury » est la règle des scores : part de « oui » parmi les juges qui ne
disent pas N/A, au-dessus de 1/2 -> oui, en dessous -> non, 1/2 ou tout N/A -> faux.
Référence humaine : la part de paires de médecins d'accord (data/meta_reference.json).
Ce n'est pas la même mesure (elle porte sur toutes les lignes, égalités comprises) :
elle dit seulement à quel point le critère est discutable.

    uv run python -m alive_drift.meta_eval run --judge juge_glm --limit 5
    uv run python -m alive_drift.meta_eval report --judges juge_haiku_latest juge_glm juge_deepseek

`run` : un run payant par réponse différente de l'échantillon, mêmes réessais,
même reprise et mêmes vérifications que jury.py (le juge est relu chez Alive
avant le premier run). Votes dans outputs/meta_eval/<juge>.jsonl.
`report` : aucun appel ; affiche le tableau et écrit
outputs/meta_eval/rapport-<date-heure>.json (jamais par-dessus).
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import httpx

from alive_drift.client import TOKEN_HELP, find_token, make_client
from alive_drift.config import load_config
from alive_drift.files import read_records
from alive_drift.grid import Criterion, load_working_grid
from alive_drift.jury import judge_responses
from alive_drift.recorder import CampaignStopped
from alive_drift.scores import criterion_values, read_votes

CONFIG = "meta"  # le nom de « config notée » dans les fichiers de votes de la méta-évaluation
JURY = "jury"


def load_sample(path, grid: list[Criterion]) -> list[dict]:
    """Les lignes de data/meta_sample.jsonl, vérifiées.

    Lève ValueError si le fichier est vide ou si une ligne est mal formée :
    id en double, critère hors de la grille, avis non booléen, ou une même
    réponse (completion_id) avec deux textes différents.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} : échantillon introuvable (voir meta_sample.py)")
    names = {c.name for c in grid}
    ids, texts, sample = set(), {}, []
    for where, row in read_records(path):
        if row.get("id") in ids or not isinstance(row.get("id"), str):
            raise ValueError(f"{where} : id absent ou en double")
        if row.get("criterion") not in names:
            raise ValueError(f"{where} : critère {row.get('criterion')!r} absent de la grille de travail")
        if type(row.get("label")) is not bool:
            raise ValueError(f"{where} : avis des médecins absent")
        cid = row.get("completion_id")
        text = (row.get("question"), row.get("response"))
        if (not isinstance(cid, str) or not cid
                or not all(isinstance(t, str) and t.strip() for t in text)):
            raise ValueError(f"{where} : réponse incomplète")
        if texts.setdefault(cid, text) != text:
            raise ValueError(f"{where} : la réponse {cid} a deux textes différents")
        ids.add(row["id"])
        sample.append(row)
    if not sample:
        raise ValueError(f"{path} : échantillon vide")
    return sample


def sample_responses(sample: list[dict]) -> list[dict]:
    """Les réponses à faire noter, une par completion_id, dans l'ordre de l'échantillon.

    Même forme que votes.load_responses : jury.judge_responses les note sans
    savoir que c'est une méta-évaluation. Une réponse jugée par les médecins
    sur deux critères n'est notée qu'une fois (le juge rend les 8 verdicts).
    """
    responses, seen = [], set()
    for row in sample:
        cid = row["completion_id"]
        if cid not in seen:
            seen.add(cid)
            responses.append({"id": cid, "config": CONFIG, "question": row["question"],
                              "response": row["response"], "execution_id": cid})
    return responses


def jury_verdict(verdicts: list[dict], name: str) -> bool | None:
    """Le verdict du jury sur un critère : oui si plus de la moitié des avis (hors N/A)
    disent oui, non si moins, None si exactement la moitié ou si tous disent N/A."""
    value = criterion_values(verdicts, [name])[name]
    if value is None or value == 0.5:
        return None
    return value > 0.5


def measure(sample: list[dict], votes: dict[str, dict[str, dict]], grid: list[Criterion]) -> dict:
    """Les mesures, par évaluateur (chaque juge, puis le jury) et par critère.

    `votes` : {juge: {completion_id: ligne de vote}}. Une ligne de l'échantillon
    sans vote d'un juge n'est pas comptée pour lui (essai partiel) ; le jury ne
    compte que les lignes notées par tous les juges.
    """
    evaluators = list(votes) + ([JURY] if len(votes) > 1 else [])
    result = {}
    for evaluator in evaluators:
        per = {c.name: {"yes": 0, "yes_ok": 0, "no": 0, "no_ok": 0, "na": 0} for c in grid}
        for row in sample:
            cid, name = row["completion_id"], row["criterion"]
            if evaluator == JURY:
                if not all(cid in v for v in votes.values()):
                    continue
                said = jury_verdict([v[cid]["verdicts"] for v in votes.values()], name)
            else:
                if cid not in votes[evaluator]:
                    continue
                said = votes[evaluator][cid]["verdicts"][name]
            counts = per[name]
            key = "yes" if row["label"] else "no"
            counts[key] += 1
            counts[key + "_ok"] += said is row["label"]
            counts["na"] += said is None
        for counts in per.values():
            n = counts["yes"] + counts["no"]
            counts["n"] = n
            counts["balanced_accuracy"] = (
                (counts["yes_ok"] / counts["yes"] + counts["no_ok"] / counts["no"]) / 2
                if counts["yes"] and counts["no"] else None)
            counts["na_rate"] = counts["na"] / n if n else None
        defined = [c["balanced_accuracy"] for c in per.values() if c["balanced_accuracy"] is not None]
        n_rows = sum(c["n"] for c in per.values())
        result[evaluator] = {
            "criteria": per,
            "mean_balanced_accuracy": sum(defined) / len(defined) if defined else None,
            "n": n_rows,
            "na_rate": sum(c["na"] for c in per.values()) / n_rows if n_rows else None,
        }
    return result


def format_table(result: dict, grid: list[Criterion], physicians: dict, n_sample: int) -> str:
    """Le tableau affiché : justesse équilibrée par critère, puis moyenne, N/A et couverture."""
    evaluators = list(result)
    fmt = lambda x: "    –" if x is None else f"{x:5.2f}"
    head = f"{'critère':44}" + "".join(f"{e[:12]:>13}" for e in evaluators) + f"{'médecins':>11}"
    lines = ["justesse équilibrée (0.5 = hasard, 1 = parfait) ; médecins = paires d'accord",
             head]
    for number, c in enumerate(grid, start=1):
        cells = "".join(f"{fmt(result[e]['criteria'][c.name]['balanced_accuracy']):>13}"
                        for e in evaluators)
        ref = physicians.get(c.name, {}).get("agreement")
        lines.append(f"{str(number) + ' ' + c.name[:41]:44}{cells}{fmt(ref):>11}")
    lines.append(f"{'moyenne des critères':44}"
                 + "".join(f"{fmt(result[e]['mean_balanced_accuracy']):>13}" for e in evaluators))
    lines.append(f"{'part de « non applicable »':44}"
                 + "".join(f"{fmt(result[e]['na_rate']):>13}" for e in evaluators))
    lines.append(f"{'lignes notées / ' + str(n_sample):44}"
                 + "".join(f"{result[e]['n']:>13}" for e in evaluators))
    return "\n".join(lines)


# --- Ligne de commande --------------------------------------------------------------------

def run(args, transport=None, sleep=None, env_file=".env") -> int:
    token = find_token(env_file)
    if not token:
        print(TOKEN_HELP, file=sys.stderr)
        return 2
    out_path = Path(args.out_dir) / f"{args.judge}.jsonl"
    with make_client(token, transport) as client:
        try:
            judge = load_config(args.configs, args.judge)
            grid = load_working_grid(args.grid, args.selection)
            responses = sample_responses(load_sample(args.sample, grid))
            n_ok, n_failed = judge_responses(client, responses, judge, grid, out_path=out_path,
                                             limit=args.limit, workers=args.workers, sleep=sleep)
        except (ValueError, OSError, KeyError) as err:
            print(f"REFUS : {err}", file=sys.stderr)
            return 2
        except (CampaignStopped, httpx.HTTPStatusError) as err:
            print(f"ARRÊT : {err}", file=sys.stderr)
            return 1
        except KeyboardInterrupt:
            print("INTERROMPU : relancer la même commande pour reprendre.", file=sys.stderr)
            return 130
    print(f"{n_ok} réponses notées, {n_failed} échecs à ce lancement -> {out_path}")
    return 0 if n_failed == 0 else 1


def report(args) -> int:
    out = Path(args.report_out or Path(args.out_dir) / f"rapport-{datetime.now():%Y%m%d-%H%M%S}.json")
    try:
        if out.exists():
            raise ValueError(f"{out} existe déjà : on n'écrit jamais par-dessus")
        if len(set(args.judges)) != len(args.judges):
            raise ValueError(f"juge en double : {args.judges}")
        grid = load_working_grid(args.grid, args.selection)
        sample = load_sample(args.sample, grid)
        wanted = {r["completion_id"] for r in sample}
        votes = {}
        for judge in args.judges:
            found = read_votes(Path(args.out_dir) / f"{judge}.jsonl", grid, judge)
            stray = [qid for qid, vote in found.items()
                     if qid not in wanted or vote.get("judged_config") != CONFIG
                     or vote.get("response_execution_id") != qid]
            if stray:
                raise ValueError(f"{judge} : votes qui ne sont pas de cet échantillon : {stray[:3]}")
            votes[judge] = found
        physicians = json.loads(Path(args.reference).read_text(encoding="utf-8"))["physicians"]
    except (ValueError, OSError, KeyError) as err:
        print(f"REFUS : {err}", file=sys.stderr)
        return 2
    result = measure(sample, votes, grid)
    print(format_table(result, grid, physicians, len(sample)))
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = {"judges": args.judges, "sample": str(args.sample), "n_sample": len(sample),
           "result": result, "physicians": physicians}
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"-> {out}")
    return 0


def main(argv: list[str] | None = None, transport: httpx.BaseTransport | None = None,
         sleep=None, env_file=".env") -> int:
    parser = argparse.ArgumentParser(description="Méta-évaluation des juges contre l'avis des médecins.")
    commands = parser.add_subparsers(dest="command", required=True)
    run_p = commands.add_parser("run", help="faire noter l'échantillon par un juge (runs payants)")
    run_p.add_argument("--judge", required=True)
    run_p.add_argument("--limit", type=int, default=None, help="ne noter que les N premières réponses")
    run_p.add_argument("--workers", type=int, default=1, help="runs en même temps (1 à 32, défaut 1)")
    run_p.add_argument("--configs", default="configs.json")
    report_p = commands.add_parser("report", help="comparer les juges aux médecins (aucun appel)")
    report_p.add_argument("--judges", nargs="+", required=True)
    report_p.add_argument("--reference", default="data/meta_reference.json")
    report_p.add_argument("--report-out", default=None,
                          help="défaut : <out-dir>/rapport-<date-heure>.json")
    for p in (run_p, report_p):
        p.add_argument("--sample", default="data/meta_sample.jsonl")
        p.add_argument("--out-dir", default="outputs/meta_eval", help="où sont les votes")
        p.add_argument("--grid", default="data/criteria.json")
        p.add_argument("--selection", default="data/selection.json")
    args = parser.parse_args(argv)
    if args.command == "run":
        return run(args, transport, sleep, env_file)
    return report(args)


if __name__ == "__main__":
    sys.exit(main())
