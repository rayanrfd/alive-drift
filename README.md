# alive-drift

Mesure de dérive d'agents LLM médicaux, construite uniquement autour d'Alive.
Dépôt indépendant : aucun import de `drift-audit`, `healthbench_judges` ou `pipeline`.

Construit petit à petit, un morceau relu à la fois. La première version du
client (plus grosse) reste sur la branche `step1-alive-client`, comme référence.

## Contenu

- `src/alive_drift/check.py` : décide si une réponse de run Alive est gardée.
  Rejette si le run n'a pas abouti, si la réponse est vide, ou si le modèle
  servi n'est pas exactement celui demandé (fallback, substitution, plusieurs
  appels, raisonnement non coupé).
- `src/alive_drift/client.py` : lance UN run et renvoie le JSON d'Alive vérifié
  par `check_run`. Jusqu'à 5 essais (attentes de 5, 15, 30 puis 60 s) si Alive répond 429 ou 5xx, ou si le réseau
  coupe ; jamais pour une erreur de notre requête ni pour un run rejeté.
- `configs.json` + `src/alive_drift/config.py` : les configs, décrites une fois
  (room, déploiement, modèle, effort de raisonnement) et choisies par leur nom.
- `src/alive_drift/deploy.py` : crée un déploiement de mesure sur Alive (room,
  déploiement, raisonnement), vérifie les réglages renvoyés par Alive, puis ajoute
  la config à `configs.json`. Gratuit, aucun run :
  `uv run python -m alive_drift.deploy --name B --model openai/gpt-4.1-mini --system-prompt "..."`.
- `src/alive_drift/generate.py` : un run par question pour UNE config (A, A_bis, B).
  Il assemble trois pièces : `files.py` (lire les questions, écrire les lignes,
  relire ce qui est fait), `pool.py` (envoyer plusieurs questions à la fois et
  s'arrêter proprement) et `recorder.py` (garder la réponse, noter l'échec, ou
  arrêter la campagne).
  Chaque réponse est écrite tout de suite dans un JSONL ; les échecs vont dans
  `<sortie>.errors.jsonl`. Arrêt après 5 échecs d'affilée ou sur une erreur de
  notre côté (401, 404...).
  Reprise : relancer la même commande saute les questions déjà dans la sortie
  et retente les autres. Rien n'est jamais écrasé ; un fichier qui n'est pas
  celui de la campagne (autre config, autre modèle, autres questions, ligne
  abîmée) est refusé sans lancer de run.
  `--workers N` envoie N questions en même temps (défaut 1) ; un seul fil écrit
  les fichiers. Les lignes sont alors dans l'ordre d'arrivée des réponses.
  Commande : `uv run python -m alive_drift.generate --config sonde --out outputs/sonde.jsonl --limit 3`.
- `src/alive_drift/summary.py` : résume un fichier de réponses et son journal
  d'erreurs (questions manquantes, providers, longueurs, coût, échecs par message).
  Ne lance aucun run : `uv run python -m alive_drift.summary outputs/A.jsonl`.
- `data/healthbench.jsonl` : les 1000 questions (copie de `drift-audit/outputs/prompts/`).
  Essai réel : `uv run python -m alive_drift.client th-… dp-… <modèle> "<question>"`.

## Tests

Aucun appel réseau. Les tests partent d'une vraie sortie d'Alive.

```zsh
uv run pytest -q
```

## Jeton

Une ligne `ALIVE_ACCESS_TOKEN=...` dans le fichier `.env` à la racine du dépôt
(ignoré par git). Une variable du terminal `ALIVE_TOKEN` ou `ALIVE_ACCESS_TOKEN`
passe avant le fichier. Jamais dans un fichier commité.

## Règles d'usage

- Ctrl-C une fois pour arrêter proprement, puis relancer la même commande pour
  reprendre.
- Ne jamais lancer deux fois la même commande en parallèle (deux processus
  écriraient les mêmes questions).
- Le fichier de sortie n'est pas trié après une reprise : toujours retrouver une
  réponse par son `question_id`, jamais par sa position.

## Limites connues, acceptées

- Après un timeout (300 s côté client, 1800 s côté Alive), le nouvel essai peut
  créer un second run facturé.
- Avec `--workers N` (32 au plus), un arrêt (401, 5 échecs d'affilée, Ctrl-C)
  n'envoie plus rien et laisse finir les runs en cours (N au plus), tous
  enregistrés. Un run en attente entre deux essais abandonne ses réessais.
- Un second Ctrl-C abandonne l'attente : les runs en cours sont payés mais pas
  écrits ; la reprise les refait.
- Un run rejeté est facturé, et retenté à chaque relance ; le compteur des 5
  échecs d'affilée repart de zéro à chaque lancement.
