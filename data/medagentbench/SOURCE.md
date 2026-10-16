# MedAgentBench : copie des tâches et des fonctions

Copie exacte de deux fichiers du dépôt MedAgentBench (Stanford ML Group), licence MIT :
https://github.com/stanfordmlgroup/MedAgentBench, commit `9926011` (21 novembre 2025).

| Fichier ici | Fichier d'origine | sha256 (12 premiers caractères) |
|---|---|---|
| `test_data_v2.json` | `data/medagentbench/test_data_v2.json` | `b6e89b2ef82f` |
| `funcs_v1.json` | `data/medagentbench/funcs_v1.json` | `c977266db5ec` |

- `test_data_v2.json` : les 300 tâches (10 types × 30) de l'article et de la config par
  défaut de MedAgentBench (`configs/tasks/medagentbench.yaml`). `test_data_v1.json`
  (100 tâches, ancienne version) n'est pas copié.
- `funcs_v1.json` : les 9 fonctions FHIR décrites au modèle.
- Le correcteur (`refsol.py`) n'est pas dans le dépôt d'origine (lien Box du README) ;
  il n'est pas copié ici.

Ne pas modifier ces fichiers : `alive_drift.mab.tasks` enregistre leur empreinte avec
chaque résultat. Article : Jiang et al., *MedAgentBench*, NEJM AI 2025 (arXiv 2501.14654).
