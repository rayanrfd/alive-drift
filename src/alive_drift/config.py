"""Les configs testées (A, A_bis, B...), décrites une fois dans configs.json.

Une config = un nom + le déploiement Alive qui la réalise + ce qu'on attend
de lui (modèle, effort de raisonnement).

    {
      "A": {"room": "th-...", "deployment": "dp-...",
            "model": "openai/gpt-4.1-mini", "reasoning_effort": "none",
            "system_prompt": "You are a helpful assistant."}
    }

Le prompt système est celui avec lequel le déploiement a été créé. C'est une
trace : c'est le déploiement, côté Alive, qui l'applique. Il est facultatif
(les configs écrites avant le 07/10 ne l'avaient pas).

Un juge a un prompt système trop long pour ce fichier. Sa config porte à la
place "judge_prompt" : l'empreinte de ce prompt.

Si "model" est un alias d'Alive (ex. "~anthropic/claude-haiku-latest"),
"served_model" donne le vrai nom du modèle qui doit répondre
("anthropic/claude-haiku-5.5"). Chaque run servi par un autre modèle est rejeté.

On choisit une config par son NOM. On ne tape donc jamais le déploiement à la
main : impossible de lancer "A" avec le déploiement de B.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path

FIELDS = ("room", "deployment", "model", "reasoning_effort")  # obligatoires
OPTIONAL_FIELDS = ("system_prompt", "judge_prompt", "served_model")


@dataclass(frozen=True)
class Config:
    name: str
    room: str
    deployment: str
    model: str
    reasoning_effort: str
    system_prompt: str | None = None
    judge_prompt: str | None = None  # juges seulement : empreinte du prompt système
    served_model: str | None = None  # si model est un alias : le modèle qui répond vraiment

    def served(self) -> str:
        """Le nom du modèle qui doit répondre : served_model s'il y en a un, sinon model."""
        return self.served_model or self.model


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


def read_configs(path) -> dict:
    """Tout le contenu de configs.json : {nom: {champ: valeur}}.

    Lève ValueError si une clé apparaît deux fois ou si ce n'est pas un objet.
    """
    with open(path, encoding="utf-8") as f:
        configs = json.load(f, object_pairs_hook=_no_duplicate_keys)
    if not isinstance(configs, dict):
        raise ValueError(f"{path} : doit contenir un objet {{nom: config}}")
    return configs


def load_config(path, name: str) -> Config:
    """Lit configs.json et renvoie la config `name`.

    Lève ValueError si le nom n'existe pas (comparaison exacte : "A " ou "a"
    ne sont pas "A"), si un des quatre champs obligatoires manque ou est vide,
    ou si une clé apparaît deux fois dans le fichier.
    """
    configs = read_configs(path)
    if name not in configs:
        raise ValueError(f"{path} : config {name!r} inconnue (disponibles : {sorted(configs)})")
    entry = configs[name]
    if not isinstance(entry, dict):
        raise ValueError(f"{path} : la config {name!r} doit être un objet")
    for field in FIELDS:
        value = entry.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{path} : config {name!r}, champ {field!r} manquant ou vide")
    unknown = sorted(set(entry) - set(FIELDS) - set(OPTIONAL_FIELDS))
    if unknown:
        raise ValueError(f"{path} : config {name!r}, champs inconnus {unknown}")
    for field in OPTIONAL_FIELDS:
        if entry.get(field) is not None and not isinstance(entry[field], str):
            raise ValueError(f"{path} : config {name!r}, {field!r} doit être un texte")
    return Config(name=name, system_prompt=entry.get("system_prompt"),
                  judge_prompt=entry.get("judge_prompt"), served_model=entry.get("served_model"),
                  **{field: entry[field] for field in FIELDS})


def add_config(path, config: Config) -> None:
    """Ajoute une config à configs.json. Les configs déjà présentes ne sont pas touchées.

    Lève ValueError si le nom existe déjà : on ne remplace jamais une config.
    Le fichier est réécrit d'un coup (fichier temporaire puis remplacement), pour
    ne jamais laisser un configs.json à moitié écrit.
    """
    path = Path(path)
    configs = read_configs(path) if path.exists() else {}
    if config.name in configs:
        raise ValueError(f"{path} : la config {config.name!r} existe déjà")
    entry = {field: getattr(config, field) for field in FIELDS}
    for field in OPTIONAL_FIELDS:
        if getattr(config, field) is not None:
            entry[field] = getattr(config, field)
    configs[config.name] = entry
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(configs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)
