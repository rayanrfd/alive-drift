"""Tests de rooms.py, sans réseau : un faux Alive garde des rooms en mémoire."""

import json

import httpx
import pytest

from alive_drift.rooms import is_applied, load_labels, main


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


def setup(tmp_path, monkeypatch, labels=None, configs=None):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ALIVE_TOKEN", raising=False)
    monkeypatch.delenv("ALIVE_ACCESS_TOKEN", raising=False)
    (tmp_path / ".env").write_text("ALIVE_ACCESS_TOKEN=jeton-de-test\n", encoding="utf-8")
    write(tmp_path / "configs.json", configs or {
        "A": {"room": "th-A", "deployment": "dp-A", "model": "m", "reasoning_effort": "none"},
        "B": {"room": "th-B", "deployment": "dp-B", "model": "m", "reasoning_effort": "none"},
        "sonde": {"room": "th-S", "deployment": "dp-S", "model": "m", "reasoning_effort": "none"}})
    write(tmp_path / "labels.json", labels or {
        "_lisez_moi": "commentaire ignoré",
        "A": {"title": "HB · A · référence", "metadata": {"kind": "campagne", "n_questions": 1000}},
        "B": {"title": "HB · B · dérive", "metadata": {"kind": "campagne", "role": "derive"}}})


def fake_alive(rooms, patch_ignored=False, merge=False):
    """GET et PATCH /threads/{id}. Renvoie (transport, requêtes reçues)."""
    seen = []

    def handler(request):
        room_id = request.url.path.rsplit("/", 1)[1]
        seen.append((request.method, room_id, json.loads(request.content) if request.content else None))
        if request.method == "PATCH" and not patch_ignored:
            body = json.loads(request.content)
            rooms[room_id]["title"] = body["title"]
            old = rooms[room_id].get("metadata") or {}
            rooms[room_id]["metadata"] = {**old, **body["metadata"]} if merge else body["metadata"]
        return httpx.Response(200, json=rooms[room_id])

    return httpx.MockTransport(handler), seen


def fresh_rooms():
    return {"th-A": {"id": "th-A", "title": None, "metadata": None},
            "th-B": {"id": "th-B", "title": "ancien titre", "metadata": {"vieux": "x"}}}


def run(args, transport):
    return main(["--labels", "labels.json", *args], transport=transport, sleep=lambda s: None)


# --- Sans --apply : on regarde seulement -------------------------------------------------

def test_sans_apply_rien_n_est_modifie(tmp_path, monkeypatch, capsys):
    setup(tmp_path, monkeypatch)
    rooms = fresh_rooms()
    transport, seen = fake_alive(rooms)
    assert run([], transport) == 0
    assert [m for m, _, _ in seen] == ["GET", "GET"]
    assert rooms["th-B"]["title"] == "ancien titre"
    out = capsys.readouterr().out
    assert "'ancien titre' -> 'HB · B · dérive'" in out
    assert "sans étiquette : sonde" in out
    assert "rien n'a été modifié" in out


# --- Avec --apply ------------------------------------------------------------------------

def test_apply_envoie_titre_et_metadonnees_puis_relit(tmp_path, monkeypatch, capsys):
    setup(tmp_path, monkeypatch)
    rooms = fresh_rooms()
    transport, seen = fake_alive(rooms)
    assert run(["--apply"], transport) == 0
    assert seen[:3] == [("GET", "th-A", None),
                        ("PATCH", "th-A", {"title": "HB · A · référence",
                                           "metadata": {"kind": "campagne", "n_questions": 1000}}),
                        ("GET", "th-A", None)]
    assert rooms["th-B"]["title"] == "HB · B · dérive"
    assert "2 rooms modifiées" in capsys.readouterr().out
    # relancer : tout est déjà à jour, aucun PATCH
    transport, seen = fake_alive(rooms)
    assert run(["--apply"], transport) == 0
    assert [m for m, _, _ in seen] == ["GET", "GET"]
    assert "0 rooms modifiées" in capsys.readouterr().out


def test_apply_accepte_qu_alive_complete_les_metadonnees(tmp_path, monkeypatch):
    # si Alive fusionne au lieu de remplacer, la clé « vieux » reste : c'est accepté
    setup(tmp_path, monkeypatch)
    rooms = fresh_rooms()
    transport, _ = fake_alive(rooms, merge=True)
    assert run(["--apply"], transport) == 0
    assert rooms["th-B"]["metadata"] == {"vieux": "x", "kind": "campagne", "role": "derive"}


def test_arret_si_la_room_relue_n_a_pas_change(tmp_path, monkeypatch, capsys):
    setup(tmp_path, monkeypatch)
    transport, _ = fake_alive(fresh_rooms(), patch_ignored=True)
    assert run(["--apply"], transport) == 2
    assert "n'a pas le titre ou les métadonnées" in capsys.readouterr().err


def test_arret_sur_erreur_d_alive(tmp_path, monkeypatch, capsys):
    setup(tmp_path, monkeypatch)
    transport = httpx.MockTransport(lambda r: httpx.Response(403, json={"detail": "hors périmètre"}))
    assert run(["--apply"], transport) == 2
    assert "hors périmètre" in capsys.readouterr().err


# --- Le fichier d'étiquettes ----------------------------------------------------------------

@pytest.mark.parametrize("labels, message", [
    ({"Z": {"title": "t", "metadata": {"k": "v"}}}, "absente de"),
    ({"A": {"title": " ", "metadata": {"k": "v"}}}, "sans titre"),
    ({"A": {"title": "t", "metadata": {"k": ["liste"]}}}, "textes, nombres ou booléens"),
    ({"A": {"title": "t", "metadata": {}}}, "textes, nombres ou booléens"),
])
def test_refus_d_un_fichier_d_etiquettes_faux(tmp_path, monkeypatch, labels, message):
    setup(tmp_path, monkeypatch, labels=labels)
    with pytest.raises(ValueError, match=message):
        load_labels("labels.json")


def test_refus_si_deux_configs_partagent_une_room(tmp_path, monkeypatch):
    configs = {"A": {"room": "th-X", "deployment": "d", "model": "m", "reasoning_effort": "none"},
               "B": {"room": "th-X", "deployment": "d", "model": "m", "reasoning_effort": "none"}}
    setup(tmp_path, monkeypatch, configs=configs)
    with pytest.raises(ValueError, match="même room"):
        load_labels("labels.json")


def test_is_applied():
    label = {"title": "T", "metadata": {"kind": "juge"}}
    assert is_applied({"title": "T", "metadata": {"kind": "juge", "autre": 1}}, label)
    assert not is_applied({"title": "T", "metadata": {"kind": "campagne"}}, label)
    assert not is_applied({"title": "t", "metadata": {"kind": "juge"}}, label)
    assert not is_applied({"title": "T", "metadata": None}, label)


# --- Le vrai fichier du dépôt ---------------------------------------------------------------

def test_le_fichier_du_depot_etiquette_toutes_les_configs():
    labels = load_labels()            # data/room_labels.json et configs.json du dépôt
    configs = json.loads(open("configs.json", encoding="utf-8").read())
    assert set(labels) == set(configs)
    for name, label in labels.items():
        assert label["metadata"]["kind"] in ("campagne", "juge", "sonde")
        assert label["metadata"]["model"] == configs[name].get("served_model", configs[name]["model"])
