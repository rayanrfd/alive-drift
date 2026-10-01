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

## Tests

Aucun appel réseau. Les tests partent d'une vraie sortie d'Alive.

```zsh
uv run pytest -q
```

## Jeton

`export ALIVE_TOKEN=...` avant tout appel réel. Jamais dans un fichier commité.
