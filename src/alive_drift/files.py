"""Lire et écrire les fichiers d'une campagne. Aucun appel à Alive ici.

- load_questions : le fichier de questions ;
- make_record, append_line : une ligne de réponse, ajoutée à la fin du fichier ;
- errors_path_for, alive_detail : le journal d'erreurs ;
- load_done : ce qui est déjà fait, pour la reprise.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import httpx

from alive_drift.config import Config


# --- Lecture des questions ---------------------------------------------------

def load_questions(path) -> list[dict]:
    """Lit un JSONL de questions {"id": ..., "text": "..."}.

    Renvoie une liste de {"id": texte, "text": texte}, dans l'ordre du fichier.
    Lève ValueError si une ligne est mal formée, si un id est en double, ou si
    le fichier ne contient aucune question.
    """
    questions = []
    seen = set()
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, start=1):
            if not line.strip():
                continue  # ligne vide tolérée
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path} ligne {n} : doit être un objet {{id, text}}")
            qid, text = row.get("id"), row.get("text")
            # Le fichier HealthBench a des ids entiers (0, 1, 2...) : on en fait du texte.
            # `type(...) is int` et non isinstance : True/False sont aussi des int en Python.
            if type(qid) is int:
                qid = str(qid)
            if not isinstance(qid, str) or not qid:
                raise ValueError(f"{path} ligne {n} : 'id' manquant")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{path} ligne {n} : 'text' vide")
            if qid in seen:
                raise ValueError(f"{path} ligne {n} : id {qid!r} en double")
            seen.add(qid)
            questions.append({"id": qid, "text": text})
    if not questions:
        raise ValueError(f"{path} : aucune question")
    return questions


# --- Écriture ----------------------------------------------------------------

def make_record(question: dict, run: dict, config: Config) -> dict:
    """La ligne enregistrée pour une réponse gardée.

    `run` a passé check_run : le bloc model et served[0] existent forcément.
    """
    served = run["model"]["served"][0]
    tokens = run.get("tokens")
    if not isinstance(tokens, dict):
        tokens = {}
    return {
        "question_id": question["id"],
        "config": config.name,
        "room_id": config.room,
        "deployment_id": config.deployment,
        "reasoning_effort": config.reasoning_effort,
        "question": question["text"],
        "response": run["content"],
        "model": served["model"],
        "provider": served["provider"],
        "execution_id": run["execution_id"],
        "prompt_tokens": tokens.get("prompt"),
        "completion_tokens": tokens.get("completion"),
        "reasoning_tokens": tokens.get("reasoning"),
        "credits_used": run.get("credits_used"),
        "duration_ms": run.get("duration_ms"),
        "written_at": now_utc(),
    }


def now_utc() -> str:
    """L'heure courante, en UTC, à la seconde : "2026-10-06T09:24:31+00:00"."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def append_line(path: Path, record: dict) -> None:
    """Ajoute une ligne JSON à la fin du fichier, puis le ferme (donc sur disque)."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def alive_detail(err: Exception) -> str | None:
    """Le message qu'Alive a mis dans sa réponse d'erreur (502, 429...), s'il y en a un.

    str(err) ne contient que le code et l'URL : sans ce message, on ne sait pas
    POURQUOI Alive a refusé (limite de débit, panne du modèle...).
    """
    if not isinstance(err, httpx.HTTPStatusError):
        return None
    try:
        body = err.response.json()
    except ValueError:
        return err.response.text[:500] or None
    if isinstance(body, dict) and "detail" in body:
        return str(body["detail"])[:500]
    return str(body)[:500]


def errors_path_for(out_path: Path) -> Path:
    """outputs/A.jsonl -> outputs/A.errors.jsonl"""
    return out_path.with_name(out_path.stem + ".errors.jsonl")


# --- Reprise -------------------------------------------------------------------

def read_records(path: Path) -> list[tuple[str, dict]]:
    """Les lignes d'un fichier JSONL qu'on complète, avec leur position.

    Renvoie [("<chemin> ligne <n>", ligne décodée), ...], ou [] si le fichier
    n'existe pas encore. Sert à la reprise (generate et jury).
    Lève ValueError si la dernière ligne a été coupée ou si une ligne n'est pas
    un objet JSON : on ne répare jamais un fichier tout seul.
    """
    if not path.exists():
        return []
    content = path.read_text(encoding="utf-8")
    # Chaque ligne est écrite avec son retour à la ligne. S'il manque à la fin,
    # la dernière écriture a été coupée : la prochaine ligne s'y collerait.
    if content and not content.endswith("\n"):
        raise ValueError(f"{path} : dernière ligne incomplète (écriture interrompue ?)")
    records = []
    for n, line in enumerate(content.splitlines(), start=1):
        where = f"{path} ligne {n}"
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            record = None
        if not isinstance(record, dict):
            raise ValueError(f"{where} : ligne illisible")
        records.append((where, record))
    return records


def load_done(out_path: Path, questions: list[dict], config: Config) -> tuple[set[str], set[str]]:
    """Ce qui est déjà dans out_path : (ids des questions faites, execution_id utilisés).

    Deux ensembles vides si le fichier n'existe pas (premier lancement).

    Avant de compléter un fichier, on vérifie qu'il est bien celui de CETTE
    campagne. Lève ValueError à la première ligne qui ne colle pas : on ne
    mélange jamais deux configs, deux déploiements, deux modèles, deux efforts
    de raisonnement ou deux jeux de questions, et on ne répare pas le fichier
    tout seul.
    """
    text_by_id = {q["id"]: q["text"] for q in questions}
    done = set()
    executions = set()
    for where, record in read_records(out_path):
        qid = record.get("question_id")
        # Chaque ligne doit venir de la même config, champ par champ.
        for label, key, expected in (
            ("config", "config", config.name),
            ("room", "room_id", config.room),
            ("déploiement", "deployment_id", config.deployment),
            ("modèle", "model", config.served()),
            ("effort de raisonnement", "reasoning_effort", config.reasoning_effort),
        ):
            if record.get(key) != expected:
                raise ValueError(f"{where} : {label} {record.get(key)!r}, attendu {expected!r}")
        # Même règle que check_run : "none" veut dire 0 token de raisonnement.
        reasoning = record.get("reasoning_tokens")
        if config.reasoning_effort == "none" and (type(reasoning) is not int or reasoning != 0):
            raise ValueError(f"{where} : {reasoning!r} tokens de raisonnement alors qu'il est coupé")
        if qid not in text_by_id:
            raise ValueError(f"{where} : question {qid!r} absente du fichier de questions")
        if record.get("question") != text_by_id[qid]:
            raise ValueError(f"{where} : le texte de la question {qid!r} a changé")
        if qid in done:
            raise ValueError(f"{where} : question {qid!r} en double")
        response = record.get("response")
        if not isinstance(response, str) or not response.strip():
            raise ValueError(f"{where} : question {qid!r} sans réponse")
        execution = record.get("execution_id")
        if not isinstance(execution, str) or not execution:
            raise ValueError(f"{where} : execution_id absent")
        if execution in executions:
            raise ValueError(f"{where} : execution_id {execution!r} en double")
        done.add(qid)
        executions.add(execution)
    return done, executions
