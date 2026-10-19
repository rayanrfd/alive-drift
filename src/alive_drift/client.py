"""Lance UN run sur Alive et renvoie sa réponse JSON, vérifiée par check_run.

run_once : un appel, une réponse. Équivalent Python du curl de alive_quickref.md §4.
run_with_retries : run_once, réessayé si l'échec vient du serveur ou du réseau.
Pas encore de parallélisme.

Essai réel (un run, coût ~0,003 crédit) :

    (jeton dans le fichier .env : ALIVE_ACCESS_TOKEN=...)
    uv run python -m alive_drift.client th-... dp-... mistralai/mistral-small-24b-instruct-2501 \
        "What is a normal resting heart rate?"
"""

import argparse
import json
import os
import sys
import time

import httpx

from alive_drift.check import RunRejected, check_run

BASE_URL = "https://brain-bzh.alive.dev/api/v1"

# Un run peut durer longtemps (Alive coupe à 1800 s). 300 s suffit pour une réponse.
TIMEOUT_S = 300.0

# Codes HTTP qui valent la peine d'un nouvel essai : serveur surchargé (429)
# ou en panne passagère (5xx). Un 502 d'Alive n'est pas facturé.
# Les autres erreurs (400, 401, 403, 404, 423...) viennent de NOTRE requête
# ou de notre jeton : réessayer ne changerait rien.
RETRYABLE_STATUS = {429, 500, 502, 503, 504}

# Attente avant chaque nouvel essai. 4 délais = 5 essais au total, étalés sur
# près de 2 minutes : assez pour laisser le serveur d'Alive redémarrer. Le
# 06/10, il saturait ("Too many open files") environ tous les 100 runs et
# revenait en quelques dizaines de secondes ; 3 essais en 7 s n'y suffisaient pas.
RETRY_DELAYS_S = (5.0, 15.0, 30.0, 60.0)


# Les deux noms acceptés pour le jeton, dans le terminal comme dans .env.
TOKEN_NAMES = ("ALIVE_TOKEN", "ALIVE_ACCESS_TOKEN")
TOKEN_HELP = ("Jeton Alive introuvable : mets une ligne ALIVE_ACCESS_TOKEN=... dans le "
              "fichier .env du dépôt (il est ignoré par git).")


def read_env_file(path) -> dict[str, str]:
    """Lit un fichier .env : une ligne NOM=valeur par variable. {} si le fichier n'existe pas.

    Tolère les lignes vides, les commentaires (#), le préfixe "export " et les
    guillemets autour de la valeur.
    """
    values = {}
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines()
    except FileNotFoundError:
        return values
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.removeprefix("export ").split("=", 1)
        values[name.strip()] = value.strip().strip("\"'")
    return values


def find_token(env_file=".env") -> str | None:
    """Le jeton Alive, ou None s'il est introuvable.

    Cherché d'abord dans les variables du terminal, puis dans le fichier .env.
    Le jeton n'est jamais affiché ni écrit ailleurs.
    """
    for source in (os.environ, read_env_file(env_file)):
        for name in TOKEN_NAMES:
            if source.get(name):
                return source[name]
    return None


def make_client(token: str, transport: httpx.BaseTransport | None = None) -> httpx.Client:
    """Client HTTP qui ajoute le jeton à chaque requête.

    `transport` sert seulement aux tests : on y branche un faux serveur.
    """
    return httpx.Client(
        base_url=BASE_URL,
        headers={"Authorization": f"Bearer {token}"},
        timeout=TIMEOUT_S,
        transport=transport,
    )


def run_once(client: httpx.Client, room_id: str, deployment_id: str,
             question: str | None, expected_model: str, expected_effort: str = "none",
             served_model: str | None = None, parent_id: str | None = None) -> dict:
    """Envoie la question, renvoie le JSON d'Alive s'il passe check_run.

    question : None = aucun message (juge « dans le fil » en conversation=transcript :
    Alive ne crée pas de message utilisateur, le run répond directement sous parent_id).
    parent_id : None = conversation neuve ; sinon l'id du message (mg-...) sous lequel
    répondre. Le run renvoyé doit alors avoir ce message pour parent DIRECT, sinon
    RunRejected. (Avec un input, Alive crée d'abord un message utilisateur sous parent_id
    et place le run sous ce message-là : parent direct différent, rejeté.)

    Lève httpx.HTTPStatusError si Alive répond une erreur (401, 502...),
    RunRejected si la réponse ne doit pas être gardée.
    """
    url = f"/threads/{room_id}/deployments/{deployment_id}/runs"
    body = {
        "input": question,  # None : pas de message utilisateur
        "parent_id": parent_id,  # None : conversation neuve ; mg-... : réponse sous ce message
        "stream": False,    # une seule réponse JSON, pas un flux
    }
    response = client.post(url, json=body)
    response.raise_for_status()  # code HTTP 4xx ou 5xx -> exception
    try:
        run = response.json()
    except ValueError:  # code 200 mais le corps n'est pas du JSON
        raise RunRejected("réponse 200 illisible (pas du JSON)") from None
    check_run(run, expected_model, expected_effort, served_model)
    if parent_id is not None and run.get("parent_id") != parent_id:
        raise RunRejected(f"run placé sous {run.get('parent_id')!r}, attendu sous {parent_id!r}")
    return run


def is_retryable(err: Exception) -> bool:
    """Vrai si l'échec vient du serveur ou du réseau, pas de notre requête."""
    if isinstance(err, httpx.HTTPStatusError):
        return err.response.status_code in RETRYABLE_STATUS
    # TransportError : connexion impossible, coupée, ou délai dépassé (timeout).
    return isinstance(err, httpx.TransportError)


def run_with_retries(client: httpx.Client, room_id: str, deployment_id: str,
                     question: str | None, expected_model: str, expected_effort: str = "none",
                     sleep=time.sleep, give_up=lambda: False,
                     served_model: str | None = None, parent_id: str | None = None) -> dict:
    """run_once, avec jusqu'à 5 essais si l'échec est passager.

    RunRejected n'est JAMAIS réessayé : le run a abouti (donc il est facturé)
    mais a été servi autrement que demandé. On veut le voir, pas le masquer.

    `sleep` : la fonction d'attente (remplacée dans les tests pour ne pas attendre).
    `give_up` : fonction sans argument ; si elle renvoie vrai, on ne réessaie plus
    et l'erreur en cours est levée. Sert à abandonner quand la campagne s'arrête.
    """
    attempts = len(RETRY_DELAYS_S) + 1
    for attempt in range(1, attempts + 1):
        try:
            return run_once(client, room_id, deployment_id, question,
                            expected_model, expected_effort, served_model, parent_id)
        except (httpx.HTTPStatusError, httpx.TransportError) as err:
            if attempt == attempts or not is_retryable(err) or give_up():
                raise  # dernier essai, erreur sans espoir, ou campagne en cours d'arrêt
            delay = RETRY_DELAYS_S[attempt - 1]
            print(f"essai {attempt}/{attempts} échoué ({type(err).__name__}), "
                  f"nouvel essai dans {delay:g} s", file=sys.stderr)
            sleep(delay)
            if give_up():
                raise  # l'arrêt a été décidé pendant l'attente
    raise AssertionError("inatteignable")  # la boucle rend ou lève toujours


def main(argv: list[str] | None = None, transport: httpx.BaseTransport | None = None,
         sleep=time.sleep, env_file=".env") -> int:
    """Lance un run, affiche le JSON brut puis le verdict. Renvoie le code de sortie."""
    parser = argparse.ArgumentParser(description="Un run Alive, sortie brute + verdict.")
    parser.add_argument("room_id")
    parser.add_argument("deployment_id")
    parser.add_argument("expected_model")
    parser.add_argument("question")
    args = parser.parse_args(argv)

    token = find_token(env_file)
    if not token:
        print(TOKEN_HELP, file=sys.stderr)
        return 2

    with make_client(token, transport) as client:
        try:
            run = run_with_retries(client, args.room_id, args.deployment_id,
                                   args.question, args.expected_model, sleep=sleep)
        except RunRejected as err:
            print(f"REJETÉ : {err}")
            return 1
    print(json.dumps(run, indent=2, ensure_ascii=False))
    print("ACCEPTÉ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
