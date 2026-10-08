"""La grille de notation : les 34 critères HealthBench Consensus, et la sélection.

data/criteria.json garde les 34 critères tels que HealthBench les donne
(vérifiés par build_grid.py). data/selection.json choisit ceux qu'on envoie
aux juges (décision du 08/10 : 8 critères, sans doublon ni paire
contradictoire). load_working_grid() rend cette grille de travail.

La même grille sert pour TOUTES les réponses (décision du 29/09) : un critère
qui ne s'applique pas à la question est noté oui ou non comme les autres, et
pèse pareil pour les configs comparées.

Un critère a un nom en trois parties, par exemple
    emergency_referrals_conditionally-emergent_emergency_behavior
    └── thème ────────┘ └── catégorie ───────┘ └── exigence ─────┘
Il y a 7 thèmes. Le fichier data/criteria.json est produit par build_grid.py.
"""

import hashlib
import json
import math
from dataclasses import dataclass

THEMES = ("communication", "complex_responses", "context_seeking", "emergency_referrals",
          "global_health", "health_data_tasks", "hedging")
N_CRITERIA = 34


@dataclass(frozen=True)
class Criterion:
    name: str      # identifiant unique, c'est lui qu'on enregistre avec chaque vote
    theme: str     # un des 7 thèmes
    text: str      # le texte donné au juge, tel qu'il est dans HealthBench
    points: float  # le poids du critère (5.0 pour tous)


def no_duplicate_keys(pairs):
    """Pour json.load(s) : refuse un objet qui contient deux fois la même clé.

    Sans ça, Python garde la dernière en silence.
    """
    keys = [key for key, _ in pairs]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise ValueError(f"clé en double : {duplicates}")
    return dict(pairs)


def theme_of(name: str) -> str:
    """Le thème d'un critère : le début de son nom. Lève ValueError s'il est inconnu."""
    for theme in THEMES:
        if name.startswith(theme + "_"):
            return theme
    raise ValueError(f"critère {name!r} : thème inconnu")


def load_grid(path="data/criteria.json") -> list[Criterion]:
    """Lit la grille. Toujours dans le même ordre : alphabétique par nom.

    Lève ValueError si la grille n'est pas exactement celle attendue : 34
    critères, noms uniques, thème cohérent avec le nom, texte non vide,
    points positifs, aucune clé en double. Une grille abîmée fausserait
    toutes les notes.
    """
    with open(path, encoding="utf-8") as f:
        entries = json.load(f, object_pairs_hook=no_duplicate_keys)["criteria"]
    grid = []
    for entry in entries:
        name, text, points = entry.get("name"), entry.get("text"), entry.get("points")
        if not isinstance(name, str) or not name:
            raise ValueError(f"{path} : critère sans nom")
        if entry.get("theme") != theme_of(name):
            raise ValueError(f"{path} : critère {name!r}, thème {entry.get('theme')!r} incohérent")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{path} : critère {name!r}, texte vide")
        if type(points) not in (int, float) or not math.isfinite(points) or points <= 0:
            raise ValueError(f"{path} : critère {name!r}, points invalides ({points!r})")
        grid.append(Criterion(name=name, theme=entry["theme"], text=text, points=float(points)))
    names = [c.name for c in grid]
    if len(set(names)) != len(names):
        raise ValueError(f"{path} : noms de critères en double")
    if len(grid) != N_CRITERIA:
        raise ValueError(f"{path} : {len(grid)} critères au lieu de {N_CRITERIA}")
    return sorted(grid, key=lambda c: c.name)


def load_selection(path="data/selection.json") -> list[str]:
    """Les noms des critères gardés : la liste "keep" du fichier de sélection.

    Lève ValueError si "keep" manque, est vide, n'est pas une liste de noms,
    ou contient un nom en double.
    """
    with open(path, encoding="utf-8") as f:
        doc = json.load(f, object_pairs_hook=no_duplicate_keys)
    if not isinstance(doc, dict):
        raise ValueError(f"{path} : doit être un objet avec une liste 'keep'")
    keep = doc.get("keep")
    if not isinstance(keep, list) or not keep or not all(isinstance(n, str) and n for n in keep):
        raise ValueError(f"{path} : 'keep' doit être une liste non vide de noms de critères")
    if len(set(keep)) != len(keep):
        raise ValueError(f"{path} : critère en double dans 'keep'")
    return keep


def select_criteria(grid: list[Criterion], names: list[str]) -> list[Criterion]:
    """Les critères de `grid` dont le nom est dans `names`, dans l'ordre de la grille.

    Lève ValueError si un nom n'existe pas dans la grille : une faute de frappe
    retirerait un critère sans bruit.
    """
    unknown = sorted(set(names) - {c.name for c in grid})
    if unknown:
        raise ValueError(f"critères inconnus dans la sélection : {unknown}")
    if not names:
        raise ValueError("sélection vide")
    return [c for c in grid if c.name in set(names)]


def load_working_grid(criteria_path="data/criteria.json",
                      selection_path="data/selection.json") -> list[Criterion]:
    """La grille envoyée aux juges : les 34 critères vérifiés, réduits à la sélection."""
    return select_criteria(load_grid(criteria_path), load_selection(selection_path))


def grid_fingerprint(grid: list[Criterion]) -> str:
    """Empreinte courte de la grille (noms, textes, points).

    Enregistrée avec chaque note : si la grille change d'une virgule,
    l'empreinte change, et on ne mélange pas des notes de deux grilles.
    """
    payload = json.dumps([[c.name, c.text, c.points] for c in grid], ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def names_by_theme(grid: list[Criterion]) -> dict[str, list[str]]:
    """{thème: [noms des critères de ce thème]}, pour les scores par thème."""
    result = {theme: [] for theme in THEMES}
    for criterion in grid:
        result[criterion.theme].append(criterion.name)
    return result
