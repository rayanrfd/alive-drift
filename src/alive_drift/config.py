"""Les configs testées (A, A_bis, B...), décrites une fois dans configs.json.

Une config = un nom + le déploiement Alive qui la réalise + ce qu'on attend
de lui (modèle, effort de raisonnement). Le prompt système n'est pas ici :
il est porté par le déploiement, côté Alive.

    {
      "A": {"room": "th-...", "deployment": "dp-...",
            "model": "openai/gpt-4.1-mini", "reasoning_effort": "none"}
    }

On choisit une config par son NOM. On ne tape donc jamais le déploiement à la
main : impossible de lancer "A" avec le déploiement de B.
"""

import json
from dataclasses import dataclass

FIELDS = ("room", "deployment", "model", "reasoning_effort")


@dataclass(frozen=True)
class Config:
    name: str
    room: str
    deployment: str
    model: str
    reasoning_effort: str


def _no_duplicate_keys(pairs):
    """Pour json.load : refuse un objet qui contient deux fois la même clé.

    Sans ça, Python garde la dernière en silence : deux entrées "A" dans le
    fichier, et la campagne A partirait sur le déploiement de la seconde.
    """
    keys = [key for key, _ in pairs]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise ValueError(f"clés en double dans le fichier de configs : {duplicates}")
    return dict(pairs)


def load_config(path, name: str) -> Config:
    """Lit configs.json et renvoie la config `name`.

    Lève ValueError si le nom n'existe pas (comparaison exacte : "A " ou "a"
    ne sont pas "A"), si un des quatre champs manque ou est vide, ou si une
    clé apparaît deux fois dans le fichier.
    """
    with open(path, encoding="utf-8") as f:
        configs = json.load(f, object_pairs_hook=_no_duplicate_keys)
    if not isinstance(configs, dict):
        raise ValueError(f"{path} : doit contenir un objet {{nom: config}}")
    if name not in configs:
        raise ValueError(f"{path} : config {name!r} inconnue (disponibles : {sorted(configs)})")
    entry = configs[name]
    if not isinstance(entry, dict):
        raise ValueError(f"{path} : la config {name!r} doit être un objet")
    for field in FIELDS:
        value = entry.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{path} : config {name!r}, champ {field!r} manquant ou vide")
    unknown = sorted(set(entry) - set(FIELDS))
    if unknown:
        raise ValueError(f"{path} : config {name!r}, champs inconnus {unknown}")
    return Config(name=name, **{field: entry[field] for field in FIELDS})
