"""Tests de config.py : lecture de configs.json."""

import json
from pathlib import Path

import pytest

from alive_drift.config import Config, load_config

GOOD = {
    "A": {"room": "th-1", "deployment": "dp-A", "model": "openai/gpt-4.1-mini",
          "reasoning_effort": "none"},
    "B": {"room": "th-1", "deployment": "dp-B", "model": "openai/gpt-4.1-mini",
          "reasoning_effort": "none"},
}


def write_configs(tmp_path, content):
    path = tmp_path / "configs.json"
    path.write_text(json.dumps(content), encoding="utf-8")
    return path


def test_charge_la_config_demandee(tmp_path):
    path = write_configs(tmp_path, GOOD)
    assert load_config(path, "B") == Config(
        name="B", room="th-1", deployment="dp-B",
        model="openai/gpt-4.1-mini", reasoning_effort="none",
    )


@pytest.mark.parametrize("name", ["C", "a", "A ", ""])
def test_nom_inconnu_refuse(tmp_path, name):
    # Comparaison exacte : une faute de frappe ne crée pas une nouvelle config.
    path = write_configs(tmp_path, GOOD)
    with pytest.raises(ValueError, match="inconnue"):
        load_config(path, name)


@pytest.mark.parametrize("field", ["room", "deployment", "model", "reasoning_effort"])
def test_champ_manquant_refuse(tmp_path, field):
    configs = json.loads(json.dumps(GOOD))  # copie
    del configs["A"][field]
    path = write_configs(tmp_path, configs)
    with pytest.raises(ValueError) as err:
        load_config(path, "A")
    assert f"champ {field!r} manquant ou vide" in str(err.value)


def test_champ_vide_refuse(tmp_path):
    configs = json.loads(json.dumps(GOOD))
    configs["A"]["deployment"] = "  "
    with pytest.raises(ValueError, match="'deployment' manquant ou vide"):
        load_config(write_configs(tmp_path, configs), "A")


def test_champ_inconnu_refuse(tmp_path):
    # Une faute dans un nom de champ ("modele") ne doit pas passer inaperçue.
    configs = json.loads(json.dumps(GOOD))
    configs["A"]["modele"] = "x"
    with pytest.raises(ValueError, match="champs inconnus"):
        load_config(write_configs(tmp_path, configs), "A")


def test_une_config_abimee_ne_bloque_pas_les_autres(tmp_path):
    configs = json.loads(json.dumps(GOOD))
    del configs["A"]["model"]
    assert load_config(write_configs(tmp_path, configs), "B").deployment == "dp-B"


def test_fichier_pas_un_objet_refuse(tmp_path):
    with pytest.raises(ValueError, match="doit contenir un objet"):
        load_config(write_configs(tmp_path, ["A"]), "A")


def test_configs_json_du_depot():
    # Le fichier livré avec le dépôt se charge, et "sonde" est le déploiement de test.
    path = Path(__file__).parent.parent / "configs.json"
    config = load_config(path, "sonde")
    assert config.deployment == "dp-E8dfWyMHc4Q7oXnN"
    assert config.reasoning_effort == "none"
