# archive

Ce qui ne sert plus, gardé comme trace. Rien ici n'est lu par le code.

- `juges_avant_na.json` : les six juges créés le 07/10 avec la première consigne
  (oui/non, sans « non applicable »), retirés de `configs.json` le même jour.
  Leurs déploiements existent toujours chez Alive. DeepSeek et Qwen ont été
  recréés avec la consigne N/A sous les mêmes noms. Gemini disait oui à tout ;
  MiniMax et Kimi raisonnaient malgré `none` ; GLM refuse de couper le raisonnement.
  Leurs votes d'essai sont dans `outputs/jury/avant_na/` (non suivi par git).
- `juges_sans_raisonnement.json` : Gemini 3.8 Flash et Claude Sonnet 5.5 créés le 07/10
  avec `reasoning_effort: none` et la consigne N/A. Aucun run n'a abouti : le fournisseur
  répond « Reasoning is mandatory for this endpoint and cannot be disabled » (502, non
  facturé). Recréés sous les mêmes noms avec un raisonnement faible.
- `juges_grille_34.json` : les onze juges créés les 07 et 08/10 avec la consigne N/A et
  les 34 critères (prompt `402721efd0f3`) : DeepSeek V4.1 Flash, Qwen3.8 Flash,
  Gemini 3.8 Flash, Sonnet 5.5, Haiku 4.5, Kimi K2.5, GLM 5.3 Flash, MiniMax M3,
  Grok 4.3, Qwen3.7 Plus, DeepSeek V4 Pro. Retirés le 08/10 quand la grille est passée
  à 8 critères (`data/selection.json`). Sonnet, Gemini 3.8 et GLM, les seuls à voir la
  dérive de B sur la question 0, sont recréés sous les mêmes noms avec la nouvelle grille.
  Leurs votes d'essai sont dans `outputs/jury/grille_34/` (non suivi par git).
- `agents_sans_raisonnement.json` : l'agent C (GPT-5 mini, même prompt que A) créé le 15/10
  avec `reasoning_effort: none`. Aucun run n'a abouti : « Reasoning is mandatory for this
  endpoint and cannot be disabled » (502, non facturé). Recréé sous le même nom avec un
  raisonnement faible. Ses échecs sont dans `outputs/C_none.errors.jsonl` (non suivi par git).
