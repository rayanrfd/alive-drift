"""Tests de mab/fhir.py avec un faux serveur : ni Docker, ni réseau.

Les réponses sont de vrais objets requests.Response : raise_for_status() et les
messages d'erreur sont donc ceux de requests, comme chez MedAgentBench.
"""

import pytest
import requests

from alive_drift.mab.fhir import ServerUnreachable, check_server, fhir_get, main

BASE = "http://localhost:8080/fhir/"


def response(status=200, body=b'{"resourceType": "Bundle", "total": 0}',
             content_type="application/fhir+json;charset=utf-8", reason="OK", url=BASE):
    r = requests.Response()
    r.status_code, r.reason, r.url, r._content = status, reason, url, body
    r.headers["Content-Type"] = content_type
    return r


class FakeServer:
    """Remplace requests : rend une réponse prévue, ou lève l'erreur prévue."""

    def __init__(self, result):
        self.result, self.calls = result, []

    def get(self, url, timeout):
        self.calls.append((url, timeout))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


# --- Réponses du serveur ------------------------------------------------------------

def test_reponse_fhir_json_reste_du_texte():
    # HAPI annonce application/fhir+json : le modèle reçoit le JSON en texte brut
    server = FakeServer(response())
    assert fhir_get(f"{BASE}Patient?x=1&_format=json", session=server) == {
        "status_code": 200, "data": '{"resourceType": "Bundle", "total": 0}'}
    assert server.calls == [(f"{BASE}Patient?x=1&_format=json", 60)]


def test_reponse_application_json_exacte_devient_un_objet():
    server = FakeServer(response(content_type="application/json"))
    assert fhir_get(f"{BASE}Patient", session=server)["data"] == {"resourceType": "Bundle", "total": 0}


def test_application_json_avec_charset_reste_du_texte():
    # chez eux : égalité EXACTE avec "application/json", pas « commence par »
    server = FakeServer(response(content_type="application/json;charset=utf-8"))
    assert fhir_get(f"{BASE}Patient", session=server)["data"] == '{"resourceType": "Bundle", "total": 0}'


def test_404_message_de_requests_renvoye_au_modele():
    url = f"{BASE}Observatio?patient=S1&_format=json"
    server = FakeServer(response(404, b"not found", reason="Not Found", url=url))
    assert fhir_get(url, session=server) == {"error": f"404 Client Error: Not Found for url: {url}"}


def test_500_message_de_requests_renvoye_au_modele():
    server = FakeServer(response(500, b"boom", reason="Server Error", url=BASE))
    assert fhir_get(BASE, session=server) == {"error": f"500 Server Error: Server Error for url: {BASE}"}


def test_json_annonce_mais_illisible_renvoye_au_modele():
    server = FakeServer(response(body=b"pas du json", content_type="application/json"))
    assert "error" in fhir_get(f"{BASE}Patient", session=server)


def test_adresse_mal_formee_par_le_modele_renvoyee_au_modele():
    # « GET Patient?x=1 » (sans http://) : requests refuse, comme chez eux le modèle le lit
    assert "error" in fhir_get("Patient?x=1&_format=json")


# --- Panne de notre côté : on arrête -------------------------------------------------

@pytest.mark.parametrize("err", [requests.ConnectionError("refused"), requests.ReadTimeout("slow"),
                                 requests.ConnectTimeout("slow")])
def test_notre_serveur_injoignable_arrete_tout(err):
    with pytest.raises(ServerUnreachable, match="injoignable"):
        fhir_get(f"{BASE}Patient?x=1", session=FakeServer(err))


def test_un_autre_serveur_injoignable_reste_une_erreur_de_l_agent():
    # le modèle a inventé une adresse : ce n'est pas une panne de notre côté
    result = fhir_get("http://autre-serveur:9999/fhir/Patient", session=FakeServer(requests.ConnectionError("refused")))
    assert result == {"error": "refused"}


# --- Vérification du serveur et commande ------------------------------------------------

def test_check_server():
    assert check_server(session=FakeServer(response())) is True
    assert check_server(session=FakeServer(response(404, b"", reason="Not Found"))) is False
    assert check_server(session=FakeServer(requests.ConnectionError("refused"))) is False
    server = FakeServer(response())
    check_server(session=server)
    assert server.calls[0][0] == f"{BASE}metadata"


def test_commande_check(capsys):
    assert main(["--check"], session=FakeServer(response())) == 0
    assert "OK" in capsys.readouterr().out
    assert main(["--check"], session=FakeServer(requests.ConnectionError("refused"))) == 2
    assert "Docker" in capsys.readouterr().err


def test_commande_get(capsys):
    assert main(["--get", f"{BASE}Patient"], session=FakeServer(response())) == 0
    out = capsys.readouterr()
    assert out.out == '{"resourceType": "Bundle", "total": 0}\n'
    assert "38 caractères" in out.err
    assert main(["--get", f"{BASE}Patient"], session=FakeServer(requests.ConnectionError("x"))) == 2
