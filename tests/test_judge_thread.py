"""Morceaux 33 et 34bis : juge « dans le fil ». Prompt système, déploiement dans la room
d'une campagne avec son propre agent et en conversation=transcript, et rooms qui accepte ce
juge sans étiquette. Sans réseau."""

import json
from pathlib import Path

import httpx
import pytest

from alive_drift.client import make_client
from alive_drift.config import load_config, read_configs
from alive_drift.deploy import DEFAULT_AGENT, DeployError, deploy, main
from alive_drift.grid import load_working_grid
from alive_drift.judge import (INSTRUCTIONS, THREAD_INSTRUCTIONS, prompt_fingerprint,
                               system_prompt, system_prompt_thread)
from alive_drift.rooms import load_labels, unlabeled

GRID = load_working_grid()       # la grille de travail du dépôt (8 critères)
TOKEN = "jeton-de-test"
EXISTING = {"L": {"room": "th-L", "deployment": "dp-L", "model": "openai/gpt-5.6-luna",
                  "reasoning_effort": "none", "system_prompt": "You are a helpful assistant."}}


# --- Le prompt système -----------------------------------------------------------------

def test_le_prompt_des_juges_actuels_ne_change_pas():
    # empreinte enregistrée pour tous les juges actuels (configs.json) : surtout ne pas la casser
    assert prompt_fingerprint(system_prompt(GRID)) == "dd478ccfe671"


def test_prompt_dans_le_fil_seul_le_premier_paragraphe_change():
    normal, thread = system_prompt(GRID), system_prompt_thread(GRID)
    first = INSTRUCTIONS.split("\n\n", 1)[0]
    assert first in normal and first not in thread
    assert "ANOTHER assistant" in thread and "You did not write that response" in thread
    # tout ce qui suit le premier paragraphe (règles + 8 critères) est identique
    assert normal.split("\n\n", 1)[1] == thread.split("\n\n", 1)[1]
    assert thread.startswith(THREAD_INSTRUCTIONS.split("\n\n", 1)[0])
    assert prompt_fingerprint(thread) != prompt_fingerprint(normal)


def test_prompt_dans_le_fil_decrit_le_transcript_d_alive():
    # 34bis : plus de demande dans le fil ; le juge reçoit <thread_events> [User ...] / [Agent ...]
    thread = system_prompt_thread(GRID)
    assert "<thread_events>" in thread and "[User ...]" in thread and "[Agent ...]" in thread
    assert "last user message" not in thread
    # pas le prompt du juge du morceau 34 (mode messages + demande « Evaluate… »), abandonné
    assert prompt_fingerprint(thread) != "c1f1524b9dec"


# --- Le déploiement dans la room d'une campagne ------------------------------------------

def fake_alive(conversation=None):
    """conversation : ce qu'Alive renvoie pour ce réglage (défaut : ce qui a été demandé)."""
    requests, asked = [], {}

    def handler(request):
        body = json.loads(request.content)
        path = request.url.path.removeprefix("/api/v1")
        requests.append((request.method, path, body))
        if request.method == "POST" and path == "/agents":
            return httpx.Response(200, json={"id": "ag-JUGE", "name": body["name"]})
        if request.method == "POST" and path == "/threads":
            return httpx.Response(200, json={"id": "th-NEW"})
        if request.method == "POST" and path.endswith("/deployments"):
            asked.update(body)
            return httpx.Response(200, json={"id": "dp-JUGE"})
        if request.method == "PATCH" and path.endswith("/deployments/dp-JUGE"):
            return httpx.Response(200, json={
                "id": "dp-JUGE", "agent_id": asked["agent_id"], "status": "active",
                "model": asked["model"], "reasoning_effort": body["reasoning_effort"],
                "run_defaults": {**{k: asked[k] for k in ("context", "conversation", "model_fallback",
                                                          "platform_tools", "system_prompt")},
                                 **({"conversation": conversation} if conversation else {})}})
        return httpx.Response(404, json={"detail": f"inattendu : {request.method} {path}"})

    return httpx.MockTransport(handler), requests


def configs_file(tmp_path):
    path = tmp_path / "configs.json"
    path.write_text(json.dumps(EXISTING), encoding="utf-8")
    return path


def test_dans_une_room_existante_avec_un_agent_a_part(tmp_path):
    path = configs_file(tmp_path)
    transport, requests = fake_alive()
    prompt = system_prompt_thread(GRID)
    with make_client(TOKEN, transport) as client:
        config = deploy(client, path, name="juge_fil_L", model="m", system_prompt=prompt,
                        judge=True, room="th-L", agent_name="Juge Haiku 5.5")
    assert [(m, p) for m, p, _ in requests] == [
        ("POST", "/agents"), ("POST", "/threads/th-L/deployments"),
        ("PATCH", "/threads/th-L/deployments/dp-JUGE")]          # aucune room créée
    assert requests[0][2] == {"name": "Juge Haiku 5.5", "prompt": "", "aliveness": "none"}
    assert requests[1][2]["agent_id"] == "ag-JUGE"                # l'agent du juge, pas l'agent neutre
    assert requests[1][2]["conversation"] == "messages"
    assert requests[1][2]["system_prompt"] == prompt
    assert (config.room, config.judge_prompt) == ("th-L", prompt_fingerprint(prompt))
    assert read_configs(path)["L"] == EXISTING["L"]               # la campagne n'est pas touchée


def test_deploiement_en_transcript(tmp_path):
    transport, requests = fake_alive()
    with make_client(TOKEN, transport) as client:
        deploy(client, configs_file(tmp_path), name="juge_fil_L", model="m",
               system_prompt=system_prompt_thread(GRID), judge=True, room="th-L",
               conversation="transcript")
    assert requests[0][2]["conversation"] == "transcript"
    assert requests[0][2]["platform_tools"] is False              # le reste de la recette est gardé


def test_transcript_demande_mais_alive_garde_messages(tmp_path):
    path = configs_file(tmp_path)
    transport, _ = fake_alive(conversation="messages")
    with make_client(TOKEN, transport) as client:
        with pytest.raises(DeployError, match="conversation"):
            deploy(client, path, name="juge_fil_L", model="m", system_prompt="p", judge=True,
                   room="th-L", conversation="transcript")
    assert "juge_fil_L" not in read_configs(path)                  # config non ajoutée


def test_format_d_historique_inconnu_refuse_sans_rien_creer(tmp_path):
    transport, requests = fake_alive()
    with make_client(TOKEN, transport) as client:
        with pytest.raises(ValueError, match="conversation"):
            deploy(client, configs_file(tmp_path), name="X", model="m", system_prompt="p",
                   conversation="events")
    assert requests == []


def test_sans_option_rien_ne_change(tmp_path):
    transport, requests = fake_alive()
    with make_client(TOKEN, transport) as client:
        deploy(client, configs_file(tmp_path), name="B", model="m", system_prompt="p")
    assert [(m, p) for m, p, _ in requests][:2] == [("POST", "/threads"), ("POST", "/threads/th-NEW/deployments")]
    assert requests[1][2]["agent_id"] == DEFAULT_AGENT


def cli(tmp_path, monkeypatch, *args):
    monkeypatch.setenv("ALIVE_TOKEN", TOKEN)
    transport, requests = fake_alive()
    code = main(["--name", "juge_fil_L", "--model", "m", "--configs", str(tmp_path / "configs.json"),
                 *args], transport)
    return code, requests


def test_commande_juge_dans_le_fil(tmp_path, monkeypatch, capsys):
    configs_file(tmp_path)
    code, requests = cli(tmp_path, monkeypatch, "--judge-thread", "--in-room-of", "L",
                         "--agent-name", "Juge Haiku 5.5")
    assert code == 0
    assert requests[1][1] == "/threads/th-L/deployments"
    assert requests[1][2]["system_prompt"] == system_prompt_thread(GRID)
    assert requests[1][2]["conversation"] == "transcript"
    config = load_config(tmp_path / "configs.json", "juge_fil_L")
    assert config.system_prompt is None
    assert config.judge_prompt == prompt_fingerprint(system_prompt_thread(GRID))
    out = capsys.readouterr().out
    assert "(celle de L)" in out and "des juges dans le fil" in out and "transcript" in out


def test_commande_refuse_une_room_inconnue_sans_rien_creer(tmp_path, monkeypatch, capsys):
    configs_file(tmp_path)
    code, requests = cli(tmp_path, monkeypatch, "--judge-thread", "--in-room-of", "Z")
    assert code == 2 and requests == []
    assert "--in-room-of 'Z'" in capsys.readouterr().err


def test_agent_et_agent_name_sont_exclusifs(tmp_path, monkeypatch):
    configs_file(tmp_path)
    with pytest.raises(SystemExit):
        cli(tmp_path, monkeypatch, "--judge", "--agent", "ag-X", "--agent-name", "Y")


# --- rooms : le juge partage la room d'une campagne étiquetée ------------------------------

def test_rooms_accepte_un_juge_dans_une_room_etiquetee(tmp_path):
    configs = {"L": {"room": "th-L"}, "juge_fil_L": {"room": "th-L"}, "seul": {"room": "th-S"}}
    (tmp_path / "configs.json").write_text(json.dumps(configs), encoding="utf-8")
    (tmp_path / "labels.json").write_text(json.dumps(
        {"L": {"title": "HB · L", "metadata": {"kind": "campagne"}}}), encoding="utf-8")
    labels = load_labels(tmp_path / "labels.json", tmp_path / "configs.json")
    assert unlabeled(labels, tmp_path / "configs.json") == ["seul"]
