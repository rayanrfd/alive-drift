"""Crée un déploiement de mesure sur Alive et l'ajoute à configs.json.

Remplace les trois commandes curl qu'on tapait à la main :
1. une room (une conversation) pour la campagne ;
2. un déploiement dans cette room, avec la recette de mesure : pas d'outils,
   pas de contexte ajouté, pas de fallback ;
3. le raisonnement réglé.
Puis Alive renvoie les réglages du déploiement : on les compare un par un à ce
qu'on a demandé, et seulement s'ils sont tous bons on ajoute la config.

Aucun run n'est lancé : créer un déploiement est gratuit.

    uv run python -m alive_drift.deploy --name B \\
        --model openai/gpt-4.1-mini \\
        --system-prompt "You are a helpful assistant."

Pour un juge, le prompt système est fabriqué depuis la grille (judge.py) :

    uv run python -m alive_drift.deploy --name juge_qwen --model qwen/qwen3.8-flash --judge

Juge « dans le fil » (morceau 33) : il répond SOUS les réponses d'une campagne, donc il vit
dans la room de cette campagne (Alive refuse un parent d'une autre room), avec son propre
agent pour que ses messages s'affichent sous son nom :

    uv run python -m alive_drift.deploy --name juge_haiku_fil_L --model ~anthropic/claude-haiku-latest \\
        --served-model anthropic/claude-haiku-5.5 --reasoning-effort low \\
        --judge-thread --in-room-of L --agent-name "Juge Haiku 5.5"
"""

import argparse
import sys
from pathlib import Path

import httpx

from alive_drift.client import TOKEN_HELP, find_token, make_client
from alive_drift.config import Config, add_config, read_configs
from alive_drift.files import alive_detail
from alive_drift.grid import load_working_grid
from alive_drift.judge import (prompt_fingerprint, system_prompt as judge_system_prompt,
                               system_prompt_thread as judge_thread_system_prompt)

# Le workspace de test (options d'évolution des agents coupées) et l'agent
# neutre (prompt vide), créés à la main le 01/10.
DEFAULT_WORKSPACE = "ws-jWFCoN8jkSGxVQLk"
DEFAULT_AGENT = "ag-V6NYoxUFdFEEQcSg"

# La recette de mesure : ce qu'on impose à tout déploiement, et pourquoi.
RECIPE = {
    "platform_tools": False,   # aucun outil Alive
    "context": {"room": False, "aliveness": False, "notice": False},  # rien d'ajouté au prompt
    "conversation": "messages",  # pas d'enveloppe <thread_events>
    "skills": [],              # pas de skills hérités de l'agent
    "model_fallback": False,   # si le modèle échoue, le run échoue (pas de remplaçant)
}


class DeployError(Exception):
    """Le déploiement n'a pas pu être créé, ou n'a pas les réglages demandés."""


def short(value) -> str:
    """`value` pour un message d'erreur : coupé s'il est long (le prompt d'un juge)."""
    text = repr(value)
    return text if len(text) <= 120 else f"{text[:120]}… ({len(text)} caractères)"


def check_deployment(deployment: dict, *, deployment_id: str, agent: str, model: str,
                     system_prompt: str, reasoning_effort: str) -> None:
    """Compare les réglages renvoyés par Alive à ceux demandés. Lève DeployError au premier écart.

    On vérifie aussi que c'est bien CE déploiement (id), lié à NOTRE agent,
    et qu'il est actif. Un réglage absent de la réponse compte comme un écart,
    sauf `skills`, qu'Alive ne renvoie pas.
    """
    defaults = deployment.get("run_defaults")
    if not isinstance(defaults, dict):
        raise DeployError("la réponse d'Alive n'a pas de bloc 'run_defaults'")
    expected = {
        "id": (deployment.get("id"), deployment_id),
        "agent_id": (deployment.get("agent_id"), agent),
        "status": (deployment.get("status"), "active"),
        "model": (deployment.get("model"), model),
        "reasoning_effort": (deployment.get("reasoning_effort"), reasoning_effort),
        "system_prompt": (defaults.get("system_prompt"), system_prompt),
        "model_fallback": (defaults.get("model_fallback"), RECIPE["model_fallback"]),
        "platform_tools": (defaults.get("platform_tools"), RECIPE["platform_tools"]),
        "conversation": (defaults.get("conversation"), RECIPE["conversation"]),
        "context": (defaults.get("context"), RECIPE["context"]),
    }
    for setting, (got, wanted) in expected.items():
        if got != wanted or type(got) is not type(wanted):
            raise DeployError(f"réglage {setting} : Alive a {short(got)}, demandé {short(wanted)}")


def deploy(client: httpx.Client, configs_path, *, name: str, model: str, system_prompt: str,
           reasoning_effort: str = "none", workspace: str = DEFAULT_WORKSPACE,
           agent: str = DEFAULT_AGENT, judge: bool = False,
           served_model: str | None = None, room: str | None = None,
           agent_name: str | None = None) -> Config:
    """Crée la room et le déploiement, vérifie les réglages, ajoute la config. Renvoie la config.

    judge=True : la config garde l'empreinte du prompt (champ judge_prompt)
    au lieu du prompt entier. Ce qui est envoyé à Alive est identique.
    served_model : si `model` est un alias d'Alive, le vrai nom du modèle qui
    doit répondre ; noté dans la config, vérifié à chaque run.
    room : une room EXISTANTE (celle d'une campagne) au lieu d'en créer une.
    agent_name : crée d'abord un agent à ce nom (prompt vide, aliveness none) et
    l'utilise à la place de `agent` : ses messages s'affichent sous ce nom.

    Lève ValueError, sans rien créer, si le nom est vide ou déjà pris.
    Lève DeployError si Alive refuse une étape ou si les réglages ne sont pas
    ceux demandés ; le message dit ce qui a déjà été créé. Dans ce cas la
    config n'est PAS ajoutée.
    Aucune étape n'est réessayée : refaire une création ferait un doublon.
    """
    if not name or name != name.strip():
        raise ValueError(f"nom de config invalide : {name!r}")
    if Path(configs_path).exists() and name in read_configs(configs_path):
        raise ValueError(f"{configs_path} : la config {name!r} existe déjà")

    created = "rien n'a été créé"

    def call(method: str, path: str, body: dict) -> dict:
        try:
            response = client.request(method, path, json=body)
            response.raise_for_status()
            result = response.json()
        except (httpx.HTTPError, ValueError) as err:
            detail = alive_detail(err) or type(err).__name__
            raise DeployError(f"{method} {path} a échoué ({detail}) ; {created}") from None
        if not isinstance(result, dict) or not isinstance(result.get("id"), str):
            raise DeployError(f"{method} {path} : réponse sans 'id' ; {created}")
        return result

    # 0. Un agent à part, si demandé (juge « dans le fil »).
    if agent_name is not None:
        if not agent_name.strip():
            raise ValueError("nom d'agent vide")
        agent = call("POST", "/agents", {"name": agent_name, "prompt": "", "aliveness": "none"})["id"]
        created = f"agent {agent} créé"

    # 1. La room : une neuve, ou celle d'une campagne.
    if room is None:
        room = call("POST", "/threads", {"workspace_id": workspace, "title": f"campagne-{name}"})["id"]
        created = (f"agent {agent} et " if agent_name else "") + f"room {room} créée, sans déploiement utilisable"
    else:
        created = (f"agent {agent} créé, " if agent_name else "") + f"room existante {room} inchangée"

    # 2. Le déploiement, avec la recette.
    body = {"agent_id": agent, "model": model, "system_prompt": system_prompt, **RECIPE}
    deployment_id = call("POST", f"/threads/{room}/deployments", body)["id"]
    created = f"room {room} et déploiement {deployment_id} créés, mais config non ajoutée"

    # 3. Le raisonnement. La réponse contient tous les réglages du déploiement.
    deployment = call("PATCH", f"/threads/{room}/deployments/{deployment_id}",
                      {"reasoning_effort": reasoning_effort})

    # 4. Vérifier, puis seulement alors enregistrer.
    try:
        check_deployment(deployment, deployment_id=deployment_id, agent=agent, model=model,
                         system_prompt=system_prompt, reasoning_effort=reasoning_effort)
    except DeployError as err:
        raise DeployError(f"{err} ; {created}") from None
    config = Config(name=name, room=room, deployment=deployment_id, model=model,
                    reasoning_effort=reasoning_effort,
                    system_prompt=None if judge else system_prompt,
                    judge_prompt=prompt_fingerprint(system_prompt) if judge else None,
                    served_model=served_model)
    add_config(configs_path, config)
    return config


def main(argv: list[str] | None = None, transport: httpx.BaseTransport | None = None,
         env_file=".env") -> int:
    parser = argparse.ArgumentParser(description="Crée un déploiement de mesure sur Alive.")
    parser.add_argument("--name", required=True, help="nom de la config à créer, ex. B")
    parser.add_argument("--model", required=True, help="ex. openai/gpt-4.1-mini")
    parser.add_argument("--served-model", default=None,
                        help="si --model est un alias (~...) : le modèle qui doit répondre")
    prompt = parser.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--system-prompt", help="le prompt système, entre guillemets")
    prompt.add_argument("--judge", action="store_true",
                        help="un juge : prompt système = consignes + grille (voir judge.py)")
    prompt.add_argument("--judge-thread", action="store_true",
                        help="un juge « dans le fil » : comme --judge, mais il juge la réponse "
                             "au-dessus de lui (à utiliser avec --in-room-of)")
    parser.add_argument("--in-room-of", default=None, metavar="CONFIG",
                        help="crée le déploiement dans la room de cette config (ex. L)")
    parser.add_argument("--grid", default="data/criteria.json", help="les 34 critères, avec --judge")
    parser.add_argument("--selection", default="data/selection.json",
                        help="les critères envoyés au juge, avec --judge")
    parser.add_argument("--reasoning-effort", default="none", help="défaut : none")
    parser.add_argument("--configs", default="configs.json", help="fichier des configs")
    parser.add_argument("--workspace", default=DEFAULT_WORKSPACE)
    agent = parser.add_mutually_exclusive_group()
    agent.add_argument("--agent", default=DEFAULT_AGENT)
    agent.add_argument("--agent-name", default=None,
                       help="crée un agent à ce nom pour ce déploiement (ex. \"Juge Haiku 5.5\")")
    args = parser.parse_args(argv)

    system_prompt = args.system_prompt
    judge = args.judge or args.judge_thread
    room = None
    try:
        if judge:
            make_prompt = judge_thread_system_prompt if args.judge_thread else judge_system_prompt
            system_prompt = make_prompt(load_working_grid(args.grid, args.selection))
        if args.in_room_of is not None:
            configs = read_configs(args.configs) if Path(args.configs).exists() else {}
            if args.in_room_of not in configs:
                raise ValueError(f"--in-room-of {args.in_room_of!r} : config absente de {args.configs}")
            room = configs[args.in_room_of]["room"]
    except (OSError, ValueError, KeyError) as err:
        print(f"REFUS : {err}", file=sys.stderr)
        return 2

    token = find_token(env_file)
    if not token:
        print(TOKEN_HELP, file=sys.stderr)
        return 2

    with make_client(token, transport) as client:
        try:
            config = deploy(client, args.configs, name=args.name, model=args.model,
                            system_prompt=system_prompt,
                            reasoning_effort=args.reasoning_effort,
                            workspace=args.workspace, agent=args.agent, judge=judge,
                            served_model=args.served_model, room=room,
                            agent_name=args.agent_name)
        except (ValueError, OSError) as err:
            print(f"REFUS : {err}", file=sys.stderr)
            return 2
        except DeployError as err:
            print(f"ÉCHEC : {err}", file=sys.stderr)
            return 1
    print(f"config {config.name!r} ajoutée à {args.configs}")
    print(f"  room             : {config.room}"
          + (f" (celle de {args.in_room_of})" if args.in_room_of else ""))
    if args.agent_name:
        print(f"  agent            : {args.agent_name!r} (créé pour ce déploiement)")
    print(f"  déploiement      : {config.deployment}")
    print(f"  modèle           : {config.model}")
    if config.served_model:
        print(f"  modèle servi     : {config.served_model} (exigé à chaque run)")
    print(f"  raisonnement     : {config.reasoning_effort}")
    if judge:
        kind = "des juges dans le fil" if args.judge_thread else "des juges"
        print(f"  prompt système   : celui {kind}, {len(system_prompt)} caractères, "
              f"empreinte {config.judge_prompt}")
    else:
        print(f"  prompt système   : {config.system_prompt}")
    print("  vérifié chez Alive : pas d'outils, pas de contexte, pas de fallback")
    return 0


if __name__ == "__main__":
    sys.exit(main())
