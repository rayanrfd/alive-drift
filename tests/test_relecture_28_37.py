"""Morceau 38 : les trous trouvés par la relecture indépendante des morceaux 28 à 37.
Sans réseau."""

import json

import httpx
import pytest

from alive_drift.message_ids import message_record
from test_judge_thread import cli, configs_file
from test_jury_thread import parents_for, read_out, run
from test_jury import make_responses
from test_message_ids import REAL_EXECUTION, response_row
import test_rooms


# --- deploy : un juge dans le fil sans room de campagne ---------------------------------------

def test_judge_thread_sans_in_room_of_refuse_sans_rien_creer(tmp_path, monkeypatch, capsys):
    configs_file(tmp_path)
    code, requests = cli(tmp_path, monkeypatch, "--judge-thread")
    assert code == 2 and requests == []
    assert "--judge-thread demande --in-room-of" in capsys.readouterr().err


# --- jury : la reprise ne mélange jamais votes habituels et votes dans le fil ---------------

def rewrite_first_vote(path, **changes):
    lines = path.read_text(encoding="utf-8").splitlines()
    first = {**json.loads(lines[0]), **changes}
    path.write_text("\n".join([json.dumps(first), *lines[1:]]) + "\n", encoding="utf-8")


@pytest.mark.parametrize("changes, words", [
    ({"mode": None}, "mode None, attendu 'thread'"),                 # un vote habituel
    ({"parent_message_id": "mg-AUTRE"}, "placé sous 'mg-AUTRE'"),    # sous un autre message
])
def test_reprise_dans_le_fil_refuse_un_vote_etranger(tmp_path, changes, words):
    responses = make_responses(3)
    out = tmp_path / "votes.jsonl"
    assert run(responses, out, parents_for(responses), limit=2)[0] == (2, 0)
    rewrite_first_vote(out, **changes)
    result, runs = run(responses, out, parents_for(responses))
    assert isinstance(result, ValueError) and words in str(result)
    assert runs == []                                                 # rien n'est relancé


def test_reprise_habituelle_refuse_un_vote_dans_le_fil(tmp_path):
    # jury habituel : un vote marqué "thread" dans son fichier est refusé à la reprise
    from test_jury import run_jury
    responses = make_responses(3)
    out = tmp_path / "votes.jsonl"
    assert run_jury(responses, out, limit=2)[0] == (2, 0)
    rewrite_first_vote(out, mode="thread", parent_message_id="mg-0")
    result, requests = run_jury(responses, out)
    assert isinstance(result, ValueError) and "mode 'thread', attendu None" in str(result)
    assert len(requests) == 0                                         # aucun run relancé


# --- message_ids : les deux vérifications non exercées -----------------------------------------

@pytest.mark.parametrize("changes, words", [
    ({"message_id": "ex-MsIDY6DgAPmi8ZPO"}, "n'est pas un id de message"),
    ({"execution_id": "ex-AUTRE"}, "renvoie l'exécution 'ex-AUTRE'"),
])
def test_message_record_refuse(changes, words):
    with pytest.raises(ValueError, match=words):
        message_record(response_row("0", execution_id="ex-MsIDY6DgAPmi8ZPO"),
                       {**REAL_EXECUTION, **changes})


# --- rooms : arrêt au milieu d'un --apply ----------------------------------------------------

def test_rooms_arret_au_milieu_dit_ce_qui_est_deja_fait(tmp_path, monkeypatch, capsys):
    test_rooms.setup(tmp_path, monkeypatch)
    rooms = test_rooms.fresh_rooms()
    inner, _ = test_rooms.fake_alive(rooms)

    def handler(request):
        if request.method == "PATCH" and request.url.path.endswith("/th-B"):
            return httpx.Response(403, json={"detail": "hors périmètre"})
        return inner.handle_request(request)

    assert test_rooms.run(["--apply"], httpx.MockTransport(handler)) == 2
    assert rooms["th-A"]["title"] == "HB · A · référence"             # A reste fait
    err = capsys.readouterr().err
    assert "ARRÊT : B" in err and "déjà modifiées à ce lancement : A" in err
