"""Morceaux 34 et 34bis : jury « dans le fil ». Sans réseau.

Le faux Alive se comporte comme le vrai (essais du 19/10) : un run SANS input est placé
directement sous parent_id ; un run AVEC input est placé sous un message utilisateur
intermédiaire qu'Alive crée sous parent_id. Le juge doit être en conversation=transcript.
"""

import json

import httpx
import pytest

from alive_drift.check import RunRejected
from alive_drift.client import make_client, run_once
from alive_drift.config import Config
from alive_drift.judge import prompt_fingerprint, system_prompt_thread
from alive_drift.jury import judge_responses, load_thread_parents
from test_jury import CONTENTS, GRID, MODEL, TOKEN, alive_deployment, deployments_response, \
    judge_run, make_responses

ROOM = "th-JUGE"          # la room de la campagne, où vit aussi le juge dans le fil
THREAD_JUDGE = Config(name="juge_fil", room=ROOM, deployment="dp-JUGE", model=MODEL,
                      reasoning_effort="none", judge_prompt=prompt_fingerprint(system_prompt_thread(GRID)))


def thread_deployment(conversation="transcript"):
    deployment = alive_deployment(prompt=system_prompt_thread(GRID))
    deployment["run_defaults"]["conversation"] = conversation
    return deployment


def fake_alive(wrong_parent_for=None, conversation="transcript"):
    """Renvoie (transport, runs reçus : (input, parent_id))."""
    runs = []

    def handler(request):
        if request.method == "GET":
            return deployments_response([thread_deployment(conversation)])
        body = json.loads(request.content)
        runs.append((body["input"], body["parent_id"]))
        run = judge_run(CONTENTS["ok"])
        if body["parent_id"] == wrong_parent_for:
            run["parent_id"] = "mg-AILLEURS"
        elif body["input"] is not None:
            run["parent_id"] = f"mg-DEMANDE-sous-{body['parent_id']}"   # message intermédiaire
        else:
            run["parent_id"] = body["parent_id"]
        return httpx.Response(200, json=run)

    return httpx.MockTransport(handler), runs


def parents_for(responses):
    return {r["id"]: f"mg-{r['id']}" for r in responses}


def run(responses, out_path, parents, judge=THREAD_JUDGE, wrong_parent_for=None, limit=None,
        skip=0, conversation="transcript"):
    transport, runs = fake_alive(wrong_parent_for, conversation)
    with make_client(TOKEN, transport) as client:
        try:
            result = judge_responses(client, responses, judge, GRID, out_path=out_path, limit=limit,
                                     skip=skip, sleep=lambda s: None, parents=parents)
        except Exception as err:
            result = err
    return result, runs


def read_out(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# --- Le cas normal ---------------------------------------------------------------------------

def test_chaque_run_part_sans_message_sous_sa_reponse(tmp_path):
    responses = make_responses(3)
    out = tmp_path / "votes.jsonl"
    result, runs = run(responses, out, parents_for(responses))
    assert result == (3, 0)
    assert runs == [(None, "mg-0"), (None, "mg-1"), (None, "mg-2")]   # input null : rien entre réponse et vote
    votes = read_out(out)
    assert [(v["question_id"], v["mode"], v["parent_message_id"]) for v in votes] == [
        ("0", "thread", "mg-0"), ("1", "thread", "mg-1"), ("2", "thread", "mg-2")]
    # même format de votes que le jury habituel : scores s'en sert tel quel
    assert list(votes[0]["verdicts"]) == [c.name for c in GRID]
    assert votes[0]["judge_prompt"] == THREAD_JUDGE.judge_prompt


def test_reprise_dans_le_fil(tmp_path):
    responses = make_responses(3)
    out = tmp_path / "votes.jsonl"
    run(responses, out, parents_for(responses), limit=2)
    result, runs = run(responses, out, parents_for(responses))
    assert result == (1, 0) and runs == [(None, "mg-2")]


def test_skip_saute_les_premieres_reponses(tmp_path):
    responses = make_responses(5)
    result, runs = run(responses, tmp_path / "votes.jsonl", parents_for(responses), skip=2, limit=2)
    assert result == (2, 0) and runs == [(None, "mg-2"), (None, "mg-3")]


def test_skip_negatif_refuse(tmp_path):
    responses = make_responses(2)
    result, runs = run(responses, tmp_path / "v.jsonl", parents_for(responses), skip=-1)
    assert isinstance(result, ValueError) and "skip" in str(result)
    assert runs == []


# --- Ce qui est refusé avant tout run ---------------------------------------------------------

def test_refus_si_une_reponse_n_a_pas_d_id_de_message(tmp_path):
    responses = make_responses(3)
    parents = parents_for(responses)
    del parents["1"]
    result, runs = run(responses, tmp_path / "v.jsonl", parents)
    assert isinstance(result, ValueError) and "sans id de message" in str(result)
    assert runs == []


def test_refus_d_un_juge_habituel_en_mode_fil(tmp_path):
    # le juge habituel (prompt dd478…) n'est pas un juge « dans le fil »
    from test_jury import JUDGE
    responses = make_responses(1)
    result, runs = run(responses, tmp_path / "v.jsonl", parents_for(responses), judge=JUDGE)
    assert isinstance(result, ValueError) and "mauvais mode" in str(result)
    assert runs == []


def test_refus_d_un_juge_encore_en_mode_messages(tmp_path):
    # en "messages", le juge prendrait la réponse pour la sienne et la prolongerait
    responses = make_responses(1)
    result, runs = run(responses, tmp_path / "v.jsonl", parents_for(responses), conversation="messages")
    assert isinstance(result, ValueError) and "conversation" in str(result)
    assert runs == []


# --- Ce qui est rejeté après un run --------------------------------------------------------------

def test_run_place_ailleurs_rejete_et_journalise(tmp_path):
    responses = make_responses(2)
    out = tmp_path / "votes.jsonl"
    result, _ = run(responses, out, parents_for(responses), wrong_parent_for="mg-1")
    assert result == (1, 1)                                   # 1 gardé, 1 échec
    assert [v["question_id"] for v in read_out(out)] == ["0"]
    errors = (tmp_path / "votes.errors.jsonl").read_text(encoding="utf-8")
    assert "mg-AILLEURS" in errors


def test_run_avec_un_input_rejete_car_place_sous_un_message_intermediaire():
    # l'erreur du 19/10 (morceau 34) : avec un input, Alive crée un message utilisateur sous la
    # réponse et place le run sous CE message ; le parent direct n'est pas la réponse.
    transport, _ = fake_alive()
    with make_client(TOKEN, transport) as client:
        with pytest.raises(RunRejected, match="mg-DEMANDE-sous-mg-0"):
            run_once(client, ROOM, "dp-JUGE", "Evaluate…", MODEL, parent_id="mg-0")
        run = run_once(client, ROOM, "dp-JUGE", None, MODEL, parent_id="mg-0")
    assert run["parent_id"] == "mg-0"


# --- Le fichier des ids de messages ---------------------------------------------------------------

def write_messages(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def message_row(qid, /, **changes):
    row = {"question_id": qid, "config": "A", "execution_id": f"ex-gen-A-{qid}",
           "message_id": f"mg-{qid}", "thread_id": ROOM}
    row.update(changes)
    return row


def test_load_thread_parents(tmp_path):
    responses = make_responses(2)
    path = write_messages(tmp_path / "A.messages.jsonl", [message_row("0"), message_row("1"),
                                                           message_row("9")])   # 9 : pas à noter, ignorée
    assert load_thread_parents(path, responses, ROOM) == {"0": "mg-0", "1": "mg-1"}


@pytest.mark.parametrize("row, message", [
    (message_row("0", thread_id="th-AUTRE"), "le juge est dans"),
    (message_row("0", execution_id="ex-autre"), "execution_id différent"),
    (message_row("0", message_id="ex-0"), "message_id"),
])
def test_load_thread_parents_refus(tmp_path, row, message):
    path = write_messages(tmp_path / "A.messages.jsonl", [row])
    with pytest.raises(ValueError, match=message):
        load_thread_parents(path, make_responses(1), ROOM)


def test_load_thread_parents_fichier_absent(tmp_path):
    with pytest.raises(FileNotFoundError, match="message_ids"):
        load_thread_parents(tmp_path / "absent.jsonl", make_responses(1), ROOM)
