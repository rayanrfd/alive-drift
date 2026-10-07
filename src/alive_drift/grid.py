"""La grille de notation : les 34 critères HealthBench Consensus.

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
    points positifs. Une grille abîmée fausserait toutes les notes.
    """
    with open(path, encoding="utf-8") as f:
        entries = json.load(f)["criteria"]
    grid = []
    for entry in entries:
        name, text, points = entry.get("name"), entry.get("text"), entry.get("points")
        if not isinstance(name, str) or not name:
            raise ValueError(f"{path} : critère sans nom")
        if entry.get("theme") != theme_of(name):
            raise ValueError(f"{path} : critère {name!r}, thème {entry.get('theme')!r} incohérent")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{path} : critère {name!r}, texte vide")
        if type(points) not in (int, float) or points <= 0:
            raise ValueError(f"{path} : critère {name!r}, points invalides ({points!r})")
        grid.append(Criterion(name=name, theme=entry["theme"], text=text, points=float(points)))
    names = [c.name for c in grid]
    if len(set(names)) != len(names):
        raise ValueError(f"{path} : noms de critères en double")
    if len(grid) != N_CRITERIA:
        raise ValueError(f"{path} : {len(grid)} critères au lieu de {N_CRITERIA}")
    return sorted(grid, key=lambda c: c.name)


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
