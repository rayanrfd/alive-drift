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
  par `check_run`. Jusqu'à 3 essais si Alive répond 429 ou 5xx, ou si le réseau
  coupe ; jamais pour une erreur de notre requête ni pour un run rejeté.
- `src/alive_drift/generate.py` : un run par question pour UNE config (A, A_bis, B).
  Chaque réponse est écrite tout de suite dans un JSONL ; les échecs vont dans
  `<sortie>.errors.jsonl`. Arrêt après 5 échecs d'affilée ou sur une erreur de
  notre côté (401, 404...).
  Reprise : relancer la même commande saute les questions déjà dans la sortie
  et retente les autres. Rien n'est jamais écrasé ; un fichier qui n'est pas
  celui de la campagne (autre config, autre modèle, autres questions, ligne
  abîmée) est refusé sans lancer de run. Pas encore de parallélisme.
- `data/healthbench.jsonl` : les 1000 questions (copie de `drift-audit/outputs/prompts/`).
  Essai réel : `uv run python -m alive_drift.client th-… dp-… <modèle> "<question>"`.

## Tests

Aucun appel réseau. Les tests partent d'une vraie sortie d'Alive.

```zsh
uv run pytest -q
```

## Jeton

`export ALIVE_TOKEN=...` avant tout appel réel. Jamais dans un fichier commité.
