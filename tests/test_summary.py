"""Tests de summary.py. Les fichiers sont écrits à la main, ligne par ligne,
pour ne pas dépendre de generate : on vérifie le résumé contre des valeurs connues."""

import json

import pytest

from alive_drift.files import errors_path_for
from alive_drift.summary import format_summary, main, summarize
from test_generate import make_questions, run_generate


def row(qid, **changes):
    """Une ligne de réponse plausible pour la question `qid`."""
    record = {
        "question_id": str(qid), "config": "A", "room_id": "th-1", "deployment_id": "dp-A",
        "reasoning_effort": "none", "question": f"question {qid}", "response": "texte",
        "model": "openai/gpt-4.1-mini", "provider": "OpenAI", "execution_id": f"ex-{qid}",
        "prompt_tokens": 40, "completion_tokens": 300, "reasoning_tokens": 0,
        "credits_used": 0.1, "duration_ms": 4000, "written_at": "2026-10-06T09:20:00+00:00",
    }
    record.update(changes)
    return record


def error(qid, detail, error_type="HTTPStatusError", **changes):
    record = {"question_id": str(qid), "config": "A", "deployment_id": "dp-A",
              "error_type": error_type, "error": "...", "detail": detail,
              "written_at": "2026-10-06T09:24:00+00:00"}
    record.update(changes)
    return record


def write_jsonl(path, records):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
                    encoding="utf-8")
    return path


NGINX = "<html>\r\n<head><title>502 Bad Gateway</title></head>\r\n<body>nginx</body></html>"


# --- Ce qui est fait, ce qui manque ---------------------------------------------

def test_comptes_et_questions_manquantes(tmp_path):
    out = write_jsonl(tmp_path / "A.jsonl", [row(0), row(2), row(3)])
    s = summarize(out, make_questions(6))
    assert s["responses"] == 3
    assert s["distinct_questions"] == 3
    assert s["distinct_executions"] == 3
    assert s["missing"] == ["1", "4", "5"]  # dans l'ordre du fichier de questions


def test_sans_liste_de_questions_pas_de_manquantes(tmp_path):
    out = write_jsonl(tmp_path / "A.jsonl", [row(0)])
    assert summarize(out)["missing"] is None


def test_doublons_visibles(tmp_path):
    out = write_jsonl(tmp_path / "A.jsonl", [row(0), row(0, execution_id="ex-autre"),
                                             row(1, execution_id="ex-autre")])
    s = summarize(out)
    assert (s["responses"], s["distinct_questions"], s["distinct_executions"]) == (3, 2, 2)
    assert "ATTENTION : des questions ou des execution_id sont en double" in format_summary(s)


# --- D'où viennent les réponses -------------------------------------------------

def test_provenance(tmp_path):
    out = write_jsonl(tmp_path / "A.jsonl", [
        row(0), row(1), row(2, provider="Azure"), row(3, reasoning_tokens=12)])
    s = summarize(out)
    assert s["providers"] == {"OpenAI": 3, "Azure": 1}
    assert s["models"] == {"openai/gpt-4.1-mini": 4}
    assert s["configs"] == {"A": 4}
    assert s["deployments"] == {"dp-A": 4}
    assert s["reasoning_efforts"] == {"none": 4}
    assert s["reasoning_tokens_max"] == 12
    text = format_summary(s)
    assert "ATTENTION : plusieurs valeurs pour provider" in text
    assert "plusieurs valeurs pour modèle" not in text


# --- Longueur, coût, durée -------------------------------------------------------

def test_longueur_des_reponses(tmp_path):
    lengths = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100]
    out = write_jsonl(tmp_path / "A.jsonl",
                      [row(i, completion_tokens=n) for i, n in enumerate(lengths)])
    s = summarize(out)
    assert s["completion_tokens"] == {"min": 100, "median": 600, "p95": 1100, "max": 1100, "at_max": 1}
    assert "troncature" not in format_summary(s)


def test_plusieurs_reponses_a_la_longueur_max_signalees(tmp_path):
    # Trois réponses à exactement 1500 tokens : sans doute un plafond.
    out = write_jsonl(tmp_path / "A.jsonl",
                      [row(i, completion_tokens=n) for i, n in enumerate([300, 1500, 1500, 400, 1500])])
    s = summarize(out)
    assert s["completion_tokens"]["at_max"] == 3
    assert "ATTENTION : 3 réponses font exactement 1500 tokens (troncature possible)" in format_summary(s)


def test_cout_et_duree(tmp_path):
    out = write_jsonl(tmp_path / "A.jsonl", [
        row(0, credits_used=0.1, duration_ms=3000),
        row(1, credits_used=0.3, duration_ms=5000),
        row(2, credits_used=None, duration_ms=None),  # absents : ignorés, pas comptés pour 0
    ])
    s = summarize(out)
    assert s["credits_total"] == 0.4
    assert s["credits_mean"] == 0.2
    assert s["duration_ms_mean"] == 4000
    assert s["duration_ms_max"] == 5000


def test_periode(tmp_path):
    out = write_jsonl(tmp_path / "A.jsonl", [
        row(0, written_at="2026-10-06T09:23:10+00:00"),
        row(1, written_at="2026-10-06T09:21:00+00:00"),
        row(2, written_at=None)])  # anciennes lignes sans heure
    write_jsonl(errors_path_for(out), [error(5, "x", written_at="2026-10-06T09:33:31+00:00")])
    s = summarize(out)
    assert s["first_written_at"] == "2026-10-06T09:21:00+00:00"
    assert s["last_written_at"] == "2026-10-06T09:33:31+00:00"  # les erreurs comptent aussi
    assert "période             : 2026-10-06T09:21:00+00:00 -> 2026-10-06T09:33:31+00:00" in format_summary(s)


def test_periode_inconnue_pour_les_anciennes_lignes(tmp_path):
    out = write_jsonl(tmp_path / "A.jsonl", [row(0, written_at=None)])
    assert "période             : inconnue" in format_summary(summarize(out))


# --- Échecs ------------------------------------------------------------------------

def test_erreurs_regroupees_par_message(tmp_path):
    out = write_jsonl(tmp_path / "A.jsonl", [row(0), row(1)])
    write_jsonl(errors_path_for(out), [
        error(1, NGINX),                               # la question 1 a réussi ensuite
        error(2, NGINX), error(3, NGINX),
        error(4, "[Errno 24] Too many open files"),
        error(5, None, error_type="RunRejected"),
        error(6, None),
    ])
    s = summarize(out)
    assert s["error_lines"] == 6
    assert s["never_succeeded"] == ["2", "3", "4", "5", "6"]  # pas la 1
    assert s["error_kinds"] == {
        "HTTPStatusError | page HTML : 502 Bad Gateway": 3,
        "HTTPStatusError | [Errno 24] Too many open files": 1,
        "RunRejected | (pas de message)": 1,
        "HTTPStatusError | (pas de message)": 1,
    }
    text = format_summary(s)
    assert "lignes d'erreur     : 6 (5 questions toujours sans réponse)" in text
    assert "   3 x HTTPStatusError | page HTML : 502 Bad Gateway" in text


def test_ancienne_ligne_d_erreur_sans_champ_detail(tmp_path):
    # Les erreurs écrites avant le morceau 7 bis n'ont pas de champ "detail".
    out = write_jsonl(tmp_path / "A.jsonl", [row(0)])
    old = error(1, "x"); del old["detail"]
    write_jsonl(errors_path_for(out), [old])
    assert summarize(out)["error_kinds"] == {"HTTPStatusError | (pas de message)": 1}


def test_sans_journal_d_erreurs(tmp_path):
    out = write_jsonl(tmp_path / "A.jsonl", [row(0)])
    s = summarize(out)
    assert (s["error_lines"], s["never_succeeded"], s["error_kinds"]) == (0, [], {})


# --- Cas limites -------------------------------------------------------------------

def test_fichier_vide(tmp_path):
    out = write_jsonl(tmp_path / "A.jsonl", [])
    s = summarize(out, make_questions(2))
    assert s["responses"] == 0
    assert s["missing"] == ["0", "1"]
    assert s["completion_tokens"] is None
    assert s["credits_mean"] is None
    format_summary(s)  # ne doit pas planter


def test_ligne_illisible_refusee(tmp_path):
    out = tmp_path / "A.jsonl"
    out.write_text(json.dumps(row(0)) + "\npas du JSON\n", encoding="utf-8")
    with pytest.raises(ValueError, match="ligne 2 : ligne illisible"):
        summarize(out)


# --- Sur un vrai fichier produit par generate --------------------------------------

def test_resume_d_une_sortie_de_generate(tmp_path):
    out = tmp_path / "A.jsonl"
    run_generate(make_questions(6), out, plan={"question 2": ["gemini"]}, limit=4)
    s = summarize(out, make_questions(6))
    assert s["responses"] == 3
    assert s["missing"] == ["2", "4", "5"]
    assert s["never_succeeded"] == ["2"]
    assert s["configs"] == {"A": 3}
    assert s["first_written_at"] is not None


# --- Ligne de commande ------------------------------------------------------------

def test_main(tmp_path, capsys):
    out = write_jsonl(tmp_path / "A.jsonl", [row(0), row(1)])
    questions = tmp_path / "q.jsonl"
    questions.write_text("".join(json.dumps(q) + "\n" for q in make_questions(3)), encoding="utf-8")
    assert main([str(out), "--questions", str(questions)]) == 0
    printed = capsys.readouterr().out
    assert "réponses            : 2 (2 questions, 2 execution_id)" in printed
    assert "questions manquantes : 1 (2)" in printed
    assert "crédits             : 0.2 au total, 0.1 par réponse" in printed


def test_main_fichier_absent_code_2(tmp_path, capsys):
    assert main([str(tmp_path / "absent.jsonl"), "--questions", str(tmp_path / "absent2")]) == 2
    assert capsys.readouterr().out == ""
