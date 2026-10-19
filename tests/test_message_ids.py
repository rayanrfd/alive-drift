"""Tests de message_ids.py, sans réseau : un faux Alive répond aux GET /executions/{id}.

REAL_EXECUTION est la vraie réponse d'Alive du 19/10 (première réponse de L).
"""

import copy
import json

import httpx

from alive_drift.message_ids import main, message_record, messages_path_for, read_responses

ROOM = "th-ai6FIal2CDNSJdQr"
REAL_EXECUTION = {
    "execution_id": "ex-MsIDY6DgAPmi8ZPO", "status": "completed", "credits_used": 0.049532,
    "started_at": 1792134101236, "completed_at": 1792134106206, "duration_ms": 4970,
    "tokens": {"prompt": 44, "completion": 217, "reasoning": 0},
    "model": {"requested": "openai/gpt-5.6-luna",
              "served": [{"model": "openai/gpt-5.6-luna", "provider": "OpenAI", "calls": 1,
                          "substituted": False}],
              "substituted": False, "fallback": False, "reasoning_effort": "none"},
    "thread_id": ROOM, "message_id": "mg-c0d99AGPLOwBpAle"}


def response_row(qid, /, **changes):
    row = {"question_id": qid, "config": "L", "room_id": ROOM, "execution_id": f"ex-{qid}",
           "response": "..."}
    row.update(changes)
    return row


def write_responses(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def fake_alive(overrides=None, status_codes=None):
    """GET /executions/ex-<q> -> exécution de la question q (mg-<q>). Renvoie (transport, urls)."""
    urls = []
    overrides, status_codes = overrides or {}, list(status_codes or [])

    def handler(request):
        urls.append(str(request.url))
        if status_codes:
            return httpx.Response(status_codes.pop(0), json={"detail": "rate limited"})
        ex = request.url.path.rsplit("/", 1)[1]
        qid = ex.removeprefix("ex-")
        body = dict(copy.deepcopy(REAL_EXECUTION), execution_id=ex, message_id=f"mg-{qid}")
        body.update(overrides.get(qid, {}))
        return httpx.Response(200, json=body)

    return httpx.MockTransport(handler), urls


def run(tmp_path, monkeypatch, args, transport):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("ALIVE_ACCESS_TOKEN=jeton-de-test\n", encoding="utf-8")
    monkeypatch.delenv("ALIVE_TOKEN", raising=False)
    monkeypatch.delenv("ALIVE_ACCESS_TOKEN", raising=False)
    return main(args, transport=transport, sleep=lambda s: None)


def read_out(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# --- La vraie réponse d'Alive -------------------------------------------------------

def test_vraie_reponse_d_alive():
    record = message_record(response_row("0", execution_id="ex-MsIDY6DgAPmi8ZPO"), REAL_EXECUTION)
    assert record == {"question_id": "0", "config": "L", "execution_id": "ex-MsIDY6DgAPmi8ZPO",
                      "message_id": "mg-c0d99AGPLOwBpAle", "thread_id": ROOM}


def test_nom_du_fichier_de_sortie():
    assert str(messages_path_for("outputs/L.jsonl")) == "outputs/L.messages.jsonl"


# --- De bout en bout, puis reprise --------------------------------------------------

def test_tout_le_fichier_puis_reprise(tmp_path, monkeypatch, capsys):
    write_responses(tmp_path / "L.jsonl", [response_row(q) for q in ("0", "1", "2")])
    transport, urls = fake_alive()
    assert run(tmp_path, monkeypatch, ["L.jsonl", "--limit", "2"], transport) == 0
    assert [r["message_id"] for r in read_out(tmp_path / "L.messages.jsonl")] == ["mg-0", "mg-1"]
    assert urls[0] == "https://brain-bzh.alive.dev/api/v1/executions/ex-0"
    assert "2 ids écrits, 0 déjà là : 2/3" in capsys.readouterr().out
    # relancer : seule la 3e est demandée, rien n'est réécrit
    assert run(tmp_path, monkeypatch, ["L.jsonl"], transport) == 0
    assert urls[2:] == ["https://brain-bzh.alive.dev/api/v1/executions/ex-2"]
    assert [r["question_id"] for r in read_out(tmp_path / "L.messages.jsonl")] == ["0", "1", "2"]
    assert "1 ids écrits, 2 déjà là : 3/3" in capsys.readouterr().out


def test_429_reessaye_puis_ecrit(tmp_path, monkeypatch):
    write_responses(tmp_path / "L.jsonl", [response_row("0")])
    transport, urls = fake_alive(status_codes=[429, 502])
    assert run(tmp_path, monkeypatch, ["L.jsonl"], transport) == 0
    assert len(urls) == 3
    assert read_out(tmp_path / "L.messages.jsonl")[0]["message_id"] == "mg-0"


# --- Arrêts : jamais un id douteux dans le fichier ------------------------------------

def test_arret_si_le_message_est_dans_une_autre_room(tmp_path, monkeypatch, capsys):
    write_responses(tmp_path / "L.jsonl", [response_row(q) for q in ("0", "1", "2")])
    transport, _ = fake_alive(overrides={"1": {"thread_id": "th-autre"}})
    assert run(tmp_path, monkeypatch, ["L.jsonl"], transport) == 2
    assert [r["question_id"] for r in read_out(tmp_path / "L.messages.jsonl")] == ["0"]
    assert "th-autre" in capsys.readouterr().err


def test_arret_si_le_run_n_est_pas_termine_ou_sans_message(tmp_path, monkeypatch, capsys):
    write_responses(tmp_path / "L.jsonl", [response_row("0"), response_row("1")])
    transport, _ = fake_alive(overrides={"0": {"status": "failed"}})
    assert run(tmp_path, monkeypatch, ["L.jsonl"], transport) == 2
    assert not (tmp_path / "L.messages.jsonl").exists()
    assert "failed" in capsys.readouterr().err
    transport, _ = fake_alive(overrides={"0": {"message_id": None}})
    assert run(tmp_path, monkeypatch, ["L.jsonl"], transport) == 2
    assert "n'est pas un id de message" in capsys.readouterr().err


def test_arret_sur_401_sans_reessai(tmp_path, monkeypatch, capsys):
    write_responses(tmp_path / "L.jsonl", [response_row("0")])
    transport, urls = fake_alive(status_codes=[401])
    assert run(tmp_path, monkeypatch, ["L.jsonl"], transport) == 2
    assert len(urls) == 1
    assert "ARRÊT" in capsys.readouterr().err


def test_refus_d_un_fichier_de_sortie_d_une_autre_campagne(tmp_path, monkeypatch, capsys):
    write_responses(tmp_path / "L.jsonl", [response_row("0")])
    (tmp_path / "L.messages.jsonl").write_text(json.dumps(
        {"question_id": "0", "config": "L", "execution_id": "ex-AUTRE", "message_id": "mg-x",
         "thread_id": ROOM}) + "\n", encoding="utf-8")
    transport, urls = fake_alive()
    assert run(tmp_path, monkeypatch, ["L.jsonl"], transport) == 2
    assert urls == []
    assert "execution_id différent" in capsys.readouterr().err


def test_refus_d_un_fichier_de_reponses_abime(tmp_path):
    import pytest
    with pytest.raises(ValueError, match="plusieurs configs"):
        read_responses(write_responses(tmp_path / "x.jsonl", [response_row("0"), response_row("1", config="M")]))
    with pytest.raises(ValueError, match="en double"):
        read_responses(write_responses(tmp_path / "y.jsonl", [response_row("0"), response_row("0")]))
    with pytest.raises(ValueError, match="'execution_id'"):
        read_responses(write_responses(tmp_path / "z.jsonl", [response_row("0", execution_id=None)]))
