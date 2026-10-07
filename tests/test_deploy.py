"""Tests de deploy.py, sans réseau.

Le faux Alive imite les trois appels de création :
  POST  /threads                       -> {"id": "th-NEW"}
  POST  /threads/th-NEW/deployments    -> {"id": "dp-NEW"}
  PATCH /threads/th-NEW/deployments/dp-NEW -> les réglages du déploiement
Pour le PATCH, il renvoie ce qu'on lui a demandé, dans la même forme que le
vrai Alive (relevée le 07/10). `tamper` permet de fausser un réglage, et
`fail` de faire échouer une étape.
"""

import json
from pathlib import Path

import httpx
import pytest

from alive_drift.client import make_client
from alive_drift.config import Config, load_config, read_configs
from alive_drift.deploy import DEFAULT_AGENT, DEFAULT_WORKSPACE, DeployError, deploy, main
from alive_drift.grid import load_grid
from alive_drift.judge import prompt_fingerprint, system_prompt as judge_system_prompt

TOKEN = "jeton-de-test"
PROMPT = "You are a helpful assistant. Sois bref."
EXISTING = {"A": {"room": "th-1", "deployment": "dp-A", "model": "openai/gpt-4.1-mini",
                  "reasoning_effort": "none"}}


def fake_alive(tamper=None, fail=None):
    """Renvoie (transport, requests). requests = liste de (méthode, chemin, corps).

    fail = (numéro de l'appel à faire échouer, code HTTP, message d'Alive).
    """
    requests = []
    asked = {}

    def handler(request):
        body = json.loads(request.content)
        path = request.url.path.removeprefix("/api/v1")
        requests.append((request.method, path, body))
        if fail and len(requests) == fail[0]:
            return httpx.Response(fail[1], json={"detail": fail[2]})
        if request.method == "POST" and path == "/threads":
            return httpx.Response(200, json={"id": "th-NEW", "title": body["title"]})
        if request.method == "POST" and path == "/threads/th-NEW/deployments":
            asked.update(body)
            return httpx.Response(200, json={"id": "dp-NEW"})
        if request.method == "PATCH" and path == "/threads/th-NEW/deployments/dp-NEW":
            deployment = {
                "id": "dp-NEW", "thread_id": "th-NEW", "agent_id": asked["agent_id"],
                "status": "active", "trigger": {"type": "manual"},
                "model": asked["model"], "reasoning_effort": body["reasoning_effort"],
                "run_defaults": {
                    "context": dict(asked["context"]), "conversation": asked["conversation"],
                    "model_fallback": asked["model_fallback"],
                    "platform_tools": asked["platform_tools"],
                    "system_prompt": asked["system_prompt"],
                },
            }
            if tamper:
                tamper(deployment)
            return httpx.Response(200, json=deployment)
        return httpx.Response(404, json={"detail": f"chemin inattendu : {request.method} {path}"})

    return httpx.MockTransport(handler), requests


def configs_file(tmp_path):
    path = tmp_path / "configs.json"
    path.write_text(json.dumps(EXISTING, indent=2) + "\n", encoding="utf-8")
    return path


def run_deploy(path, tamper=None, fail=None, **options):
    """Lance deploy pour une config "B". Renvoie (config ou exception, requêtes)."""
    transport, requests = fake_alive(tamper, fail)
    options = {"name": "B", "model": "openai/gpt-4.1-mini", "system_prompt": PROMPT, **options}
    with make_client(TOKEN, transport) as client:
        try:
            result = deploy(client, path, **options)
        except Exception as err:
            result = err
    return result, requests


# --- Cas normal ------------------------------------------------------------------

def test_les_trois_appels_et_leur_contenu(tmp_path):
    path = configs_file(tmp_path)
    _, requests = run_deploy(path)
    assert requests == [
        ("POST", "/threads", {"workspace_id": DEFAULT_WORKSPACE, "title": "campagne-B"}),
        ("POST", "/threads/th-NEW/deployments", {
            "agent_id": DEFAULT_AGENT,
            "model": "openai/gpt-4.1-mini",
            "system_prompt": PROMPT,
            "platform_tools": False,
            "context": {"room": False, "aliveness": False, "notice": False},
            "conversation": "messages",
            "skills": [],
            "model_fallback": False,
        }),
        ("PATCH", "/threads/th-NEW/deployments/dp-NEW", {"reasoning_effort": "none"}),
    ]


def test_la_config_est_ajoutee(tmp_path):
    path = configs_file(tmp_path)
    config, _ = run_deploy(path)
    expected = Config(name="B", room="th-NEW", deployment="dp-NEW", model="openai/gpt-4.1-mini",
                      reasoning_effort="none", system_prompt=PROMPT)
    assert config == expected
    assert load_config(path, "B") == expected       # relue depuis le fichier
    assert load_config(path, "A").deployment == "dp-A"  # l'ancienne est intacte


def test_effort_de_raisonnement_demande_et_enregistre(tmp_path):
    path = configs_file(tmp_path)
    config, requests = run_deploy(path, reasoning_effort="high")
    assert requests[2][2] == {"reasoning_effort": "high"}
    assert load_config(path, "B").reasoning_effort == "high"


def test_cree_configs_json_s_il_n_existe_pas(tmp_path):
    path = tmp_path / "configs.json"
    config, _ = run_deploy(path)
    assert list(read_configs(path)) == ["B"]


# --- Refus avant tout appel -----------------------------------------------------

@pytest.mark.parametrize("name", ["A", "", " B", "B "])
def test_nom_pris_ou_invalide_rien_n_est_cree(tmp_path, name):
    path = configs_file(tmp_path)
    before = path.read_text(encoding="utf-8")
    result, requests = run_deploy(path, name=name)
    assert isinstance(result, ValueError)
    assert requests == []  # aucun appel à Alive
    assert path.read_text(encoding="utf-8") == before


# --- Alive n'a pas les réglages demandés : la config n'est PAS ajoutée ----------

def set_default(key, value):
    return lambda deployment: deployment["run_defaults"].__setitem__(key, value)


@pytest.mark.parametrize("tamper, setting", [
    (lambda d: d.__setitem__("model", "google/gemini-3.1-flash-lite"), "model"),
    (lambda d: d.__setitem__("reasoning_effort", "high"), "reasoning_effort"),
    (set_default("system_prompt", PROMPT + " Et autre chose."), "system_prompt"),
    (set_default("model_fallback", True), "model_fallback"),
    (set_default("platform_tools", True), "platform_tools"),
    (set_default("conversation", "events"), "conversation"),
    (set_default("context", {"room": False, "aliveness": False, "notice": True}), "context"),
    (lambda d: d["run_defaults"].pop("model_fallback"), "model_fallback"),  # réglage absent
    (set_default("model_fallback", 0), "model_fallback"),  # 0 n'est pas false
    (lambda d: d.pop("run_defaults"), "run_defaults"),
    (lambda d: d.__setitem__("id", "dp-AUTRE"), "réglage id"),  # un autre déploiement
    (lambda d: d.__setitem__("agent_id", "ag-AUTRE"), "agent_id"),
    (lambda d: d.__setitem__("status", "inactive"), "status"),
    (lambda d: d.pop("status"), "status"),
])
def test_reglage_different_chez_alive(tmp_path, tamper, setting):
    path = configs_file(tmp_path)
    before = path.read_text(encoding="utf-8")
    result, requests = run_deploy(path, tamper=tamper)

    assert isinstance(result, DeployError)
    assert setting in str(result)
    assert "room th-NEW et déploiement dp-NEW créés, mais config non ajoutée" in str(result)
    assert path.read_text(encoding="utf-8") == before


# --- Une étape échoue : on dit ce qui existe déjà, on ne réessaie pas ----------

def test_jeton_refuse_des_la_premiere_etape(tmp_path):
    path = configs_file(tmp_path)
    result, requests = run_deploy(path, fail=(1, 401, "Authentication required"))
    assert isinstance(result, DeployError)
    assert "Authentication required" in str(result)
    assert "rien n'a été créé" in str(result)
    assert len(requests) == 1
    assert "B" not in read_configs(path)


def test_modele_inconnu_a_la_deuxieme_etape(tmp_path):
    path = configs_file(tmp_path)
    result, requests = run_deploy(path, fail=(2, 400, "Unknown model: 'deepseek/x'"))
    assert isinstance(result, DeployError)
    assert "Unknown model: 'deepseek/x'" in str(result)
    assert "room th-NEW créée" in str(result)
    assert len(requests) == 2  # pas de PATCH après l'échec
    assert "B" not in read_configs(path)


@pytest.mark.parametrize("step", [1, 2, 3])
def test_un_502_n_est_jamais_reessaye(tmp_path, step):
    # Refaire une création ferait une room ou un déploiement en double.
    path = configs_file(tmp_path)
    result, requests = run_deploy(path, fail=(step, 502, "Bad Gateway"))
    assert isinstance(result, DeployError)
    assert len(requests) == step
    assert "B" not in read_configs(path)


def test_le_jeton_n_apparait_jamais_dans_l_erreur(tmp_path):
    result, _ = run_deploy(configs_file(tmp_path), fail=(1, 401, "Authentication required"))
    assert TOKEN not in str(result)


# --- Ligne de commande ----------------------------------------------------------

def cli(path, *extra, name="B"):
    return ["--name", name, "--model", "openai/gpt-4.1-mini", "--system-prompt", PROMPT,
            "--configs", str(path), *extra]


def test_main_cree_et_affiche(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ALIVE_TOKEN", TOKEN)
    path = configs_file(tmp_path)
    transport, requests = fake_alive()
    assert main(cli(path), transport) == 0
    printed = capsys.readouterr().out
    assert "config 'B' ajoutée" in printed
    assert "dp-NEW" in printed and PROMPT in printed
    assert TOKEN not in printed
    assert load_config(path, "B").deployment == "dp-NEW"


def test_main_options_workspace_agent_effort(tmp_path, monkeypatch):
    monkeypatch.setenv("ALIVE_TOKEN", TOKEN)
    path = configs_file(tmp_path)
    transport, requests = fake_alive()
    args = cli(path, "--workspace", "ws-X", "--agent", "ag-Y", "--reasoning-effort", "low")
    assert main(args, transport) == 0
    assert requests[0][2]["workspace_id"] == "ws-X"
    assert requests[1][2]["agent_id"] == "ag-Y"
    assert requests[2][2] == {"reasoning_effort": "low"}


def test_main_nom_deja_pris_code_2(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ALIVE_TOKEN", TOKEN)
    transport, requests = fake_alive()
    assert main(cli(configs_file(tmp_path), name="A"), transport) == 2
    assert requests == []
    assert "REFUS" in capsys.readouterr().err


def test_main_echec_alive_code_1(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ALIVE_TOKEN", TOKEN)
    transport, _ = fake_alive(fail=(2, 400, "Unknown model: 'x'"))
    assert main(cli(configs_file(tmp_path)), transport) == 1
    assert "ÉCHEC" in capsys.readouterr().err


def test_main_sans_jeton_code_2(tmp_path, monkeypatch):
    for name in ("ALIVE_TOKEN", "ALIVE_ACCESS_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    transport, requests = fake_alive()
    assert main(cli(configs_file(tmp_path)), transport, env_file=tmp_path / "absent.env") == 2
    assert requests == []


# --- Un juge : --judge ---------------------------------------------------------------

REPO_GRID = Path(__file__).parent.parent / "data" / "criteria.json"


def judge_cli(path, *extra):
    # argparse garde le dernier --grid : un test peut en donner un autre dans `extra`
    return ["--name", "juge_qwen", "--model", "qwen/qwen3.8-flash", "--judge",
            "--configs", str(path), "--grid", str(REPO_GRID), *extra]


def test_juge_le_prompt_envoye_est_celui_de_la_grille(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ALIVE_TOKEN", TOKEN)
    path = configs_file(tmp_path)
    transport, requests = fake_alive()
    expected = judge_system_prompt(load_grid(REPO_GRID))

    assert main(judge_cli(path), transport) == 0
    assert requests[1][2]["system_prompt"] == expected  # envoyé à Alive, entier
    assert requests[1][2]["model"] == "qwen/qwen3.8-flash"
    # la config garde l'empreinte, pas les 36 000 caractères
    entry = read_configs(path)["juge_qwen"]
    assert entry == {"room": "th-NEW", "deployment": "dp-NEW", "model": "qwen/qwen3.8-flash",
                     "reasoning_effort": "none", "judge_prompt": prompt_fingerprint(expected)}
    config = load_config(path, "juge_qwen")
    assert config.system_prompt is None and config.judge_prompt == prompt_fingerprint(expected)
    printed = capsys.readouterr().out
    assert f"{len(expected)} caractères, empreinte {prompt_fingerprint(expected)}" in printed
    assert "rubric_item" not in printed  # le prompt n'est pas déversé dans le terminal


def test_juge_prompt_tronque_par_alive_config_non_ajoutee(tmp_path, monkeypatch, capsys):
    # Si Alive coupe un prompt aussi long, la vérification le voit.
    monkeypatch.setenv("ALIVE_TOKEN", TOKEN)
    path = configs_file(tmp_path)

    def truncate(deployment):
        deployment["run_defaults"]["system_prompt"] = deployment["run_defaults"]["system_prompt"][:20000]
    transport, _ = fake_alive(tamper=truncate)
    assert main(judge_cli(path), transport) == 1
    assert "juge_qwen" not in read_configs(path)
    error = capsys.readouterr().err
    assert "réglage system_prompt" in error and "caractères)" in error
    assert len(error) < 1000  # les 36 000 caractères ne sont pas déversés dans le terminal


def test_juge_grille_abimee_rien_n_est_cree(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ALIVE_TOKEN", TOKEN)
    grid = tmp_path / "criteria.json"
    grid.write_text('{"criteria": []}', encoding="utf-8")
    transport, requests = fake_alive()
    assert main(judge_cli(configs_file(tmp_path), "--grid", str(grid)), transport) == 2
    assert requests == []
    assert "0 critères au lieu de 34" in capsys.readouterr().err


@pytest.mark.parametrize("args", [
    ["--name", "X", "--model", "m"],                                         # ni l'un ni l'autre
    ["--name", "X", "--model", "m", "--judge", "--system-prompt", "Sois bref."],  # les deux
])
def test_il_faut_exactement_un_prompt(tmp_path, monkeypatch, args):
    monkeypatch.setenv("ALIVE_TOKEN", TOKEN)
    transport, requests = fake_alive()
    with pytest.raises(SystemExit):
        main(args + ["--configs", str(configs_file(tmp_path))], transport)
    assert requests == []


def test_empreinte_de_prompt():
    assert prompt_fingerprint("abc") == prompt_fingerprint("abc")
    assert prompt_fingerprint("abc") != prompt_fingerprint("abc ")
    assert len(prompt_fingerprint("abc")) == 12
