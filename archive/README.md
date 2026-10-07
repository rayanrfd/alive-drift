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
