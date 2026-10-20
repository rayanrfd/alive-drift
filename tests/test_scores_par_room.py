"""Morceau 36 : scores avec un juge « dans le fil », qui a un déploiement (donc un nom) par
room : --judges "juge_fil_{config}". Aucun appel à Alive."""

import json

import pytest

from alive_drift.export_richter import export_rows
from alive_drift.scores import judge_name_for, main, score_config
from test_scores import REPO_GRID, REPO_SELECTION, GRID, vote, write_votes
from alive_drift.grid import load_working_grid

TEMPLATE = "juge_fil_{config}"
CONFIGS = ("L", "L_bis", "M")
INSTRUMENT = {"judge_prompt": "54f276947f1f", "model": "anthropic/claude-haiku-5.5",
              "reasoning_effort": "low", "mode": "thread"}


def thread_campaign(jury_dir, grid, changes_for=None):
    """Un juge par room : L.juge_fil_L.jsonl, L_bis.juge_fil_L_bis.jsonl, M.juge_fil_M.jsonl.

    L et L_bis : oui partout ; M : non au 1er critère. `changes_for` : {config: champs du vote
    à changer pour toute cette config}.
    """
    n = len(grid)
    for config in CONFIGS:
        name = f"juge_fil_{config}"
        letters = ("n" + "o" * (n - 1)) if config == "M" else "o" * n
        changes = {**INSTRUMENT, **(changes_for or {}).get(config, {})}
        write_votes(jury_dir, config, name,
                    [vote(q, letters, name, config, grid, **changes) for q in ("0", "1", "2")])


def run_main(jury_dir, out_dir, *judges):
    return main(["--judges", *(judges or (TEMPLATE,)), "--configs", *CONFIGS,
                 "--jury-dir", str(jury_dir), "--out-dir", str(out_dir),
                 "--grid", str(REPO_GRID), "--selection", str(REPO_SELECTION)])


def read_rows(out_dir):
    return {c: [json.loads(l) for l in (out_dir / f"{c}.jsonl").read_text().splitlines()]
            for c in CONFIGS}


# --- Le nom du juge ----------------------------------------------------------------------

def test_nom_du_juge_par_config():
    assert judge_name_for(TEMPLATE, "L_bis") == "juge_fil_L_bis"
    assert judge_name_for("juge_haiku_latest", "M") == "juge_haiku_latest"   # sans {config} : inchangé
    with pytest.raises(ValueError, match="accolades"):
        judge_name_for("juge_fil_{conf}", "M")


def test_score_config_lit_le_juge_de_la_room(tmp_path):
    write_votes(tmp_path, "M", "juge_fil_M", [vote("0", "on-", "juge_fil_M", "M")])
    rows, excluded = score_config("M", [TEMPLATE], GRID, tmp_path)
    assert excluded == 0
    assert rows[0]["judges"] == [TEMPLATE]                 # le nom tel que donné
    assert rows[0]["by_judge"] == {TEMPLATE: 0.5}
    assert rows[0]["config"] == "M"


# --- La commande -------------------------------------------------------------------------

def test_main_un_juge_par_room(tmp_path, capsys):
    grid = load_working_grid(REPO_GRID, REPO_SELECTION)
    thread_campaign(tmp_path / "jury", grid)
    out_dir = tmp_path / "scores"
    assert run_main(tmp_path / "jury", out_dir) == 0
    rows = read_rows(out_dir)
    assert {c: len(r) for c, r in rows.items()} == {"L": 3, "L_bis": 3, "M": 3}
    assert rows["L"][0]["score"] == 1.0
    assert rows["M"][0]["score"] == (7 / 8)
    # export_richter y voit UN seul jury pour les trois configs
    fingerprint, out = export_rows(rows, {"L": "a_hb", "L_bis": "a_bis_hb", "M": "b_hb"})
    assert {r["jury_fingerprint"] for rs in out.values() for r in rs} == {fingerprint}
    assert TEMPLATE in capsys.readouterr().out


@pytest.mark.parametrize("field, value", [
    ("judge_prompt", "dd478ccfe671"),           # le juge habituel dans la room de M
    ("model", "anthropic/claude-haiku-4.5"),
    ("reasoning_effort", "none"),
    ("mode", None),                            # votes du jury habituel, pas dans le fil
])
def test_refus_si_le_juge_d_une_room_n_est_pas_le_meme_instrument(tmp_path, capsys, field, value):
    grid = load_working_grid(REPO_GRID, REPO_SELECTION)
    thread_campaign(tmp_path / "jury", grid, {"M": {field: value}})
    out_dir = tmp_path / "scores"
    assert run_main(tmp_path / "jury", out_dir) == 2
    err = capsys.readouterr().err
    assert "pas le même instrument" in err and field in err
    assert not out_dir.exists()                # rien d'écrit


def test_refus_si_un_juge_change_de_reglage_dans_une_config(tmp_path, capsys):
    grid = load_working_grid(REPO_GRID, REPO_SELECTION)
    thread_campaign(tmp_path / "jury", grid)
    path = tmp_path / "jury" / "L.juge_fil_L.jsonl"
    lines = path.read_text().splitlines()
    row = json.loads(lines[1])
    row["judge_prompt"] = "autre"
    path.write_text("\n".join([lines[0], json.dumps(row), lines[2]]) + "\n")
    assert run_main(tmp_path / "jury", tmp_path / "scores") == 2
    assert "plusieurs réglages" in capsys.readouterr().err


def test_meme_nom_de_juge_mais_autre_modele_dans_une_config_refuse(tmp_path, capsys):
    # Sans {config} aussi : un même nom de juge doit être le même instrument partout.
    grid = load_working_grid(REPO_GRID, REPO_SELECTION)
    n = len(grid)
    for config in CONFIGS:
        model = "anthropic/claude-haiku-4.5" if config == "M" else INSTRUMENT["model"]
        write_votes(tmp_path / "jury", config, "j1",
                    [vote(q, "o" * n, "j1", config, grid, **{**INSTRUMENT, "model": model})
                     for q in ("0", "1")])
    assert run_main(tmp_path / "jury", tmp_path / "scores", "j1") == 2
    assert "pas le même instrument" in capsys.readouterr().err
