"""Morceau 37 : generate garde l'id du message de chaque réponse, et le jury « dans le fil »
s'en sert comme parent, sans fichier de message_ids. Les anciennes réponses (sans message_id)
passent toujours par ce fichier. Sans réseau."""

import json

import httpx
import pytest

from alive_drift.files import make_record, message_id_of
from alive_drift.grid import load_working_grid
from alive_drift.judge import prompt_fingerprint, system_prompt_thread
from alive_drift.jury import main, thread_parents
from alive_drift.votes import load_responses
from test_check import REAL_RUN
from test_generate import make_config, make_questions, read_jsonl, run_generate
from test_jury import (MODEL, QUESTIONS, REPO, TOKEN, alive_deployment, deployments_response,
                       generation_line, judge_run, write_lines)

ROOM = "th-A"


# --- generate ----------------------------------------------------------------------------

def test_generate_enregistre_l_id_du_message(tmp_path):
    out = tmp_path / "A.jsonl"
    result, _ = run_generate(make_questions(2), out)
    assert result == (2, 0)
    assert [r["message_id"] for r in read_jsonl(out)] == ["mg-sBZ3LMug33mOUX7O"] * 2


@pytest.mark.parametrize("value", [None, "", "ex-123", 42])
def test_pas_d_id_de_message_valide_donne_none(value):
    run = dict(REAL_RUN, id=value)
    assert message_id_of(run) is None
    record = make_record({"id": "0", "text": "q"}, run, make_config())
    assert record["message_id"] is None


# --- load_responses ----------------------------------------------------------------------

def test_load_responses_garde_message_id_et_room(tmp_path):
    path = write_lines(tmp_path / "A.jsonl", [
        generation_line(0, message_id="mg-0", room_id=ROOM),
        generation_line(1)])                                   # ancienne ligne, sans message_id
    first, second = load_responses(path, QUESTIONS)
    assert (first["message_id"], first["room_id"]) == ("mg-0", ROOM)
    assert "message_id" not in second


# --- thread_parents ----------------------------------------------------------------------

def response(i, **fields):
    return {"id": str(i), "execution_id": f"ex-gen-A-{i}", **fields}


def test_ids_pris_dans_les_reponses_sans_fichier(tmp_path):
    responses = [response(0, message_id="mg-0", room_id=ROOM),
                 response(1, message_id="mg-1", room_id=ROOM)]
    absent = tmp_path / "absent.messages.jsonl"
    assert thread_parents(responses, ROOM, absent) == {"0": "mg-0", "1": "mg-1"}


@pytest.mark.parametrize("fields, words", [
    ({"message_id": "mg-0", "room_id": "th-AUTRE"}, "le juge est dans"),
    ({"message_id": "ex-0", "room_id": ROOM}, "invalide"),
])
def test_refus_room_ou_id_invalide(tmp_path, fields, words):
    with pytest.raises(ValueError, match=words):
        thread_parents([response(0, **fields)], ROOM, tmp_path / "absent.jsonl")


def test_melange_les_anciennes_reponses_passent_par_le_fichier(tmp_path):
    path = tmp_path / "A.messages.jsonl"
    write_lines(path, [{"question_id": "1", "execution_id": "ex-gen-A-1",
                        "message_id": "mg-du-fichier", "thread_id": ROOM}])
    responses = [response(0, message_id="mg-0", room_id=ROOM), response(1)]
    assert thread_parents(responses, ROOM, path) == {"0": "mg-0", "1": "mg-du-fichier"}


def test_anciennes_reponses_sans_fichier_refus(tmp_path):
    with pytest.raises(FileNotFoundError, match="message_ids"):
        thread_parents([response(0)], ROOM, tmp_path / "absent.jsonl")


# --- De bout en bout : jury --in-thread sans fichier de message_ids -------------------------

def test_jury_dans_le_fil_avec_les_ids_de_generate(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ALIVE_TOKEN", TOKEN)
    criteria, selection = REPO / "data" / "criteria.json", REPO / "data" / "selection.json"
    grid = load_working_grid(criteria, selection)
    prompt = system_prompt_thread(grid)
    configs = {"juge_fil": {"room": ROOM, "deployment": "dp-JUGE", "model": MODEL,
                            "reasoning_effort": "none", "judge_prompt": prompt_fingerprint(prompt)},
               "A": {"room": ROOM, "deployment": "dp-A", "model": MODEL, "reasoning_effort": "none"}}
    (tmp_path / "configs.json").write_text(json.dumps(configs), encoding="utf-8")
    write_lines(tmp_path / "questions.jsonl", [{"id": str(i), "text": f"question {i}"} for i in range(2)])
    write_lines(tmp_path / "A.jsonl", [generation_line(i, message_id=f"mg-rep-{i}", room_id=ROOM)
                                       for i in range(2)])
    runs = []

    def handler(request):
        if request.method == "GET":
            deployment = alive_deployment(prompt=prompt)
            deployment["run_defaults"]["conversation"] = "transcript"
            return deployments_response([deployment])
        body = json.loads(request.content)
        runs.append((body["input"], body["parent_id"]))
        content = json.dumps({"results": [{"criterion_id": n, "explanation": "r", "verdict": "met"}
                                          for n in range(1, len(grid) + 1)]})
        return httpx.Response(200, json={**judge_run(content), "parent_id": body["parent_id"]})

    code = main(["--judge", "juge_fil", "--responses", str(tmp_path / "A.jsonl"), "--in-thread",
                 "--configs", str(tmp_path / "configs.json"),
                 "--questions", str(tmp_path / "questions.jsonl"),
                 "--grid", str(criteria), "--selection", str(selection)],
                httpx.MockTransport(handler))
    assert code == 0, capsys.readouterr().err
    assert runs == [(None, "mg-rep-0"), (None, "mg-rep-1")]     # aucun fichier .messages.jsonl
    assert not (tmp_path / "A.messages.jsonl").exists()
