"""Titres et métadonnées des rooms Alive, pour que les campagnes se lisent dans Alive.

Applique data/room_labels.json : pour chaque config de configs.json, le titre de sa room
et ses métadonnées (kind, dataset, rôle, modèle...). PATCH /threads/{id} ne touche ni
aux messages ni aux réponses. Aucun run : gratuit.

    uv run python -m alive_drift.rooms            # affiche seulement ce qui changerait
    uv run python -m alive_drift.rooms --apply    # applique, puis relit chaque room

Après --apply, chaque room est relue : si Alive ne renvoie pas le titre et les
métadonnées demandés, on s'arrête (code 2).
"""

import argparse
import json
import sys
import time

import httpx

from alive_drift.client import RETRY_DELAYS_S, TOKEN_HELP, find_token, is_retryable, make_client
from alive_drift.config import read_configs
from alive_drift.files import alive_detail

LABELS_PATH = "data/room_labels.json"
SCALARS = (str, int, float, bool)


def load_labels(path=LABELS_PATH, configs_path="configs.json") -> dict[str, dict]:
    """{config: {"room", "title", "metadata"}} pour chaque config étiquetée.

    Les clés qui commencent par « _ » sont des commentaires. Lève ValueError si une
    config étiquetée n'existe pas dans configs.json, si un titre est vide, si les
    métadonnées ne sont pas un objet de valeurs simples (texte, nombre, booléen),
    ou si deux configs étiquetées partagent une room.
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    configs = read_configs(configs_path)
    labels, rooms = {}, {}
    for name, label in raw.items():
        if name.startswith("_"):
            continue
        if name not in configs:
            raise ValueError(f"{path} : config {name!r} absente de {configs_path}")
        if not isinstance(label, dict) or not isinstance(label.get("title"), str) \
                or not label["title"].strip():
            raise ValueError(f"{path} : {name!r} sans titre")
        meta = label.get("metadata")
        if not isinstance(meta, dict) or not meta \
                or not all(isinstance(v, SCALARS) for v in meta.values()):
            raise ValueError(f"{path} : {name!r}, metadata doit être un objet de textes, nombres ou booléens")
        room = configs[name]["room"]
        if room in rooms:
            raise ValueError(f"{path} : {name!r} et {rooms[room]!r} ont la même room {room}")
        rooms[room] = name
        labels[name] = {"room": room, "title": label["title"], "metadata": meta}
    return labels


def unlabeled(labels: dict, configs_path="configs.json") -> list[str]:
    """Les configs sans étiquette dont la room n'est pas déjà étiquetée (signalées, pas refusées).

    Un juge « dans le fil » vit dans la room d'une campagne : cette room a déjà son titre,
    il n'a pas besoin d'étiquette à lui.
    """
    labeled_rooms = {label["room"] for label in labels.values()}
    return sorted(name for name, cfg in read_configs(configs_path).items()
                  if name not in labels and cfg.get("room") not in labeled_rooms)


def is_applied(room: dict, label: dict) -> bool:
    """Vrai si la room a le titre voulu et au moins ces métadonnées, avec ces valeurs."""
    meta = room.get("metadata") or {}
    return room.get("title") == label["title"] and all(
        meta.get(k) == v for k, v in label["metadata"].items())


def call(client: httpx.Client, method: str, path: str, body=None, sleep=time.sleep) -> dict:
    """Une requête à Alive, réessayée si Alive ou le réseau flanche (429, 5xx)."""
    attempts = len(RETRY_DELAYS_S) + 1
    for attempt in range(1, attempts + 1):
        try:
            response = client.request(method, path, json=body)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPStatusError, httpx.TransportError) as err:
            if attempt == attempts or not is_retryable(err):
                raise
            sleep(RETRY_DELAYS_S[attempt - 1])
    raise AssertionError("inatteignable")


def main(argv: list[str] | None = None, transport: httpx.BaseTransport | None = None,
         sleep=time.sleep, env_file=".env") -> int:
    parser = argparse.ArgumentParser(description="Titres et métadonnées des rooms Alive. Aucun run.")
    parser.add_argument("--labels", default=LABELS_PATH)
    parser.add_argument("--configs", default="configs.json")
    parser.add_argument("--apply", action="store_true", help="applique (sinon : affiche seulement)")
    args = parser.parse_args(argv)
    try:
        labels = load_labels(args.labels, args.configs)
        missing = unlabeled(labels, args.configs)
    except (ValueError, OSError) as err:
        print(f"REFUS : {err}", file=sys.stderr)
        return 2
    token = find_token(env_file)
    if not token:
        print(TOKEN_HELP, file=sys.stderr)
        return 2
    changed = 0
    with make_client(token, transport) as client:
        for name, label in labels.items():
            path = f"/threads/{label['room']}"
            try:
                room = call(client, "GET", path, sleep=sleep)
                if is_applied(room, label):
                    print(f"{name:18} {label['room']}  déjà à jour")
                    continue
                print(f"{name:18} {label['room']}  {room.get('title')!r} -> {label['title']!r}")
                if not args.apply:
                    continue
                call(client, "PATCH", path, {"title": label["title"], "metadata": label["metadata"]}, sleep)
                if not is_applied(call(client, "GET", path, sleep=sleep), label):
                    print(f"ARRÊT : {name} : la room relue n'a pas le titre ou les métadonnées "
                          f"demandés", file=sys.stderr)
                    return 2
                changed += 1
            except (httpx.HTTPStatusError, httpx.TransportError) as err:
                detail = alive_detail(err)
                print(f"ARRÊT : {name} : {err}" + (f" ({detail})" if detail else ""), file=sys.stderr)
                return 2
    if missing:
        print(f"sans étiquette : {', '.join(missing)}")
    print(f"{changed} rooms modifiées" if args.apply else "rien n'a été modifié (ajouter --apply)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
