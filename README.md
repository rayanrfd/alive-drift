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
  Pour un juge, `--judge` remplace `--system-prompt` : le prompt est fabriqué depuis la
  grille, et la config garde son empreinte (`judge_prompt`) au lieu du texte entier.
  Pour un alias d'Alive (modèle en `~…-latest`), ajouter `--served-model <vrai nom>` :
  Alive répond avec le vrai nom du modèle, qui est exigé à chaque run (champ
  `served_model` de la config). Un alias redirigé vers un autre modèle est donc rejeté.
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
- `data/criteria.json` + `src/alive_drift/grid.py` : les 34 critères HealthBench Consensus
  (7 thèmes), tels que HealthBench les donne. `load_grid()` refuse une grille abîmée ;
  `grid_fingerprint()` donne l'empreinte à enregistrer avec chaque note.
- `data/selection.json` : les critères envoyés aux juges (décision du 08/10 : 8 critères,
  sans doublon, un seul par paire contradictoire ; les raisons sont dans le fichier).
  `load_working_grid()` rend cette grille de travail. Pour changer de critères, on change
  la liste `keep`, puis on recrée les juges.
- `src/alive_drift/build_grid.py` : reconstruit les 34 critères depuis le jeu HealthBench
  (Hugging Face, public) et les compare à `data/criteria.json`. Aucun appel à Alive :
  `uv run python -m alive_drift.build_grid`.
- `src/alive_drift/judge.py` : ce qu'on envoie à un juge et comment on lit sa réponse.
  `system_prompt(grid)` (consignes + critères numérotés, fixé à la création du
  déploiement du juge), `user_message(question, réponse)` (le message de chaque run),
  `parse_verdicts(contenu, grid)` (un verdict par critère, rangé par nom, tout ou rien).
  Trois verdicts : `met`, `not_met`, `not_applicable` (N/A, décidé d'après la question
  seule). Dans les fichiers de votes : `true`, `false`, `null`.
  Relire le prompt : `uv run python -m alive_drift.judge`.
- `src/alive_drift/jury.py` : fait noter les réponses d'UNE config par UN juge. Même
  mécanique que `generate` (parallèle, réessais, arrêt après 5 échecs, reprise). Une ligne
  par réponse dans `outputs/jury/<config>.<juge>.jsonl`, avec un vote par critère rangé
  par nom. Une réponse de juge incomplète n'est jamais gardée : elle va au journal
  d'erreurs. Refuse de démarrer si le juge n'a pas le prompt de la grille de travail.
  Comme `generate`, il s'appuie sur un fichier qui lit et écrit les lignes :
  `votes.py` (réponses à noter, ligne de vote, reprise).
  `uv run python -m alive_drift.jury --judge juge_glm --responses outputs/A.jsonl --limit 3`.
- `src/alive_drift/scores.py` : des votes du jury à une note par réponse. Aucun appel à
  Alive. Pour un critère : part de « oui » parmi les juges qui ne disent pas N/A (oui, non,
  N/A -> 1/2 ; tous N/A -> pas de valeur). Score d'un thème, score global : moyenne des
  critères qui ont une valeur. Une réponse n'est notée que si TOUS les juges demandés l'ont
  notée (sinon écartée et comptée). Écrit `outputs/scores/<date-heure>/<config>.jsonl`
  (jamais par-dessus un calcul) et affiche un résumé par config, par juge et par critère :
  `uv run python -m alive_drift.scores --judges juge_haiku_latest juge_glm juge_deepseek`.
  Juge « dans le fil » (morceau 36) : un déploiement, donc un nom, par room ; `{config}` dans
  le nom est remplacé par la config lue :
  `uv run python -m alive_drift.scores --judges "juge_haiku_fil2_{config}" --configs L L_bis M`.
  Refuse (sans rien écrire) si un juge n'a pas le même réglage (prompt, modèle servi,
  raisonnement, mode) dans toutes les configs.
- `src/alive_drift/export_richter.py` : écrit les scores au format que drift-audit lit pour
  le test de Richter : `a_hb` (A), `a_bis_hb` (A_bis), `b_hb` (B), fichiers
  `<nom>_clinical_adequacy.jsonl` avec `id`, `clinical_adequacy` et `jury_fingerprint`
  (juges + grille). Refuse des configs notées par des jurys différents. `--h1 C` prend C au lieu de B
  comme dérive à détecter ; `--ref L --h0 L_bis` change la référence et son contrôle (défaut A et
  A_bis), pour une campagne sur d'autres modèles. Aucun appel à Alive :
  `uv run python -m alive_drift.export_richter outputs/scores/<date-heure>`.
- `src/alive_drift/message_ids.py` : retrouve l'id du message (`mg-...`) de chaque réponse
  à partir de son `execution_id` (`GET /executions/{id}`), pour qu'un juge puisse répondre
  sous la réponse dans Alive (jury « dans le fil »). Lectures seules, gratuit ; reprise.
  `uv run python -m alive_drift.message_ids outputs/L.jsonl` -> `outputs/L.messages.jsonl`.
  Depuis le morceau 37, `generate` enregistre directement ce `message_id` (champ `id` du run) :
  `message_ids` ne sert plus que pour les réponses générées avant.
- `data/room_labels.json` + `src/alive_drift/rooms.py` : titre et métadonnées de la room de
  chaque config dans Alive (kind campagne / juge / sonde, rôle, modèle...), pour que les
  campagnes se lisent dans Alive. `PATCH /threads/{id}`, aucun run, gratuit.
  `uv run python -m alive_drift.rooms` (affiche) puis `--apply` (applique et relit).
- Juge « dans le fil » (morceau 33) : un juge qui répond SOUS les réponses d'une campagne,
  dans sa room. `deploy --judge-thread --in-room-of L --agent-name "Juge Haiku 5.5"` crée son
  agent puis son déploiement dans la room de L ; son prompt système (`judge.system_prompt_thread`)
  ne diffère du juge habituel que par le premier paragraphe, qui dit que la réponse a été écrite
  par un AUTRE assistant. `rooms` n'exige pas d'étiquette pour ce juge (la room en a déjà une).
- Jury « dans le fil » (morceaux 34 et 34bis) : `jury --in-thread` lance le juge SANS message
  (`input: null`), avec pour parent l'id du message de la réponse (enregistré par `generate`, ou fichier de
  `message_ids` pour les réponses plus anciennes). Le
  juge est déployé en `conversation: "transcript"` : Alive lui envoie la question et la réponse
  comme un texte `<thread_events>` (`[User …]` / `[Agent …]`), et son vote s'affiche juste sous
  la réponse, sans rien entre les deux. (En `"messages"`, le juge prend la réponse pour la sienne
  et la prolonge ; avec un input, Alive crée un message intermédiaire : le juge du morceau 34,
  `juge_haiku_fil_L`, est abandonné et refusé.) Refuse avant tout run si le juge n'est pas en
  `transcript`, si une réponse n'a pas d'id de message ou n'est pas dans la room du juge ; un run
  qui n'est pas DIRECTEMENT sous la réponse est rejeté. Votes au même format (+ `mode`,
  `parent_message_id`, vérifiés à la reprise : un fichier ne mélange jamais votes habituels et votes
  dans le fil). `--skip N` saute les N premières réponses :
  `uv run python -m alive_drift.jury --judge juge_haiku_fil2_L --responses outputs/L.jsonl --in-thread --skip 4 --limit 3`.
- `data/meta_sample.jsonl` + `data/meta_reference.json` + `src/alive_drift/meta_sample.py` :
  l'échantillon figé de la méta-évaluation, tiré du fichier `oss_meta_eval` de HealthBench
  (réponses jugées par 2 à 4 médecins, critère par critère). Pour chacun des 8 critères :
  25 « oui » et 25 « non » des médecins (21 « non » seulement pour le critère 3), questions en
  un tour, égalités écartées, graine 0 ; soit 396 lignes et 393 réponses différentes.
  `meta_reference.json` garde l'empreinte de la source et l'accord entre médecins par critère.
  Refait à l'identique par `uv run python -m alive_drift.meta_sample --source <oss_meta_eval.jsonl>
  --out /tmp/x.jsonl --reference /tmp/x.json` (même empreinte attendue).
- `src/alive_drift/meta_eval.py` : la méta-évaluation. `run` fait noter l'échantillon par un
  juge tel qu'il est déployé (même mécanique et mêmes vérifications que `jury`, votes dans
  `outputs/meta_eval/<juge>.jsonl`, 393 runs payants par juge) ; `report` compare aux médecins
  sans aucun appel : justesse équilibrée par critère (0.5 = hasard, « non applicable » compte
  faux), part de N/A, juges seuls et jury (règle des scores), accord entre médecins en regard.
  `uv run python -m alive_drift.meta_eval run --judge juge_glm --limit 5`, puis
  `uv run python -m alive_drift.meta_eval report --judges juge_haiku_latest juge_glm juge_deepseek`.
- `archive/` : les juges qui ne servent plus (anciennes consignes, ancienne grille), gardés
  comme trace ; voir `archive/README.md`.
- `data/healthbench.jsonl` : les 1000 questions (copie de `drift-audit/outputs/prompts/`).
  Essai réel : `uv run python -m alive_drift.client th-… dp-… <modèle> "<question>"`.

## MedAgentBench (dossier `src/alive_drift/mab/`, tests dans `tests/mab/`)

Faire passer à un agent déployé sur Alive les tâches de MedAgentBench (Stanford) : un
faux dossier patient (serveur FHIR dans Docker) où l'agent cherche des données et
prescrit. Séparé du reste (HealthBench, jury). Construit morceau par morceau, comme le reste.

- `data/medagentbench/` : copie des 300 tâches et des 9 fonctions FHIR (voir `SOURCE.md`).
- `src/alive_drift/mab/tasks.py` : lit les tâches et fabrique le premier message envoyé au
  modèle, identique à celui de MedAgentBench. Aucun appel réseau :
  `uv run python -m alive_drift.mab.tasks --task task5_1`.
- `src/alive_drift/mab/actions.py` : lit la réponse du modèle à chaque tour (GET, POST,
  FINISH ou action invalide) et fabrique la phrase à lui renvoyer, exactement comme
  l'orchestrateur de MedAgentBench, bizarreries comprises (voir `tests/mab/test_actions.py`).
- `src/alive_drift/mab/fhir.py` : interroge le serveur FHIR (Docker), comme MedAgentBench
  (même bibliothèque `requests`, donc mêmes messages d'erreur envoyés au modèle). Si notre
  serveur ne répond pas, tout s'arrête (`ServerUnreachable`) au lieu de compter un échec
  de l'agent : `uv run python -m alive_drift.mab.fhir --check`.

## Du jury à Richter

```zsh
uv run python -m alive_drift.scores --judges juge_haiku_latest juge_glm juge_deepseek
uv run python -m alive_drift.export_richter outputs/scores/<date-heure>
cd ../pipeline
uv run python run_pipeline1.py --scores-dir <dossier affiché par export_richter> \
    --out-dir results/alive_<date-heure> --run-id alive_<date-heure>
```

`run_pipeline1.py` enchaîne l'alignement, Richter (A contre A_bis, puis A contre B) et les
permutations de fausses alarmes. Il écrit avec `--overwrite` : toujours un `--out-dir` neuf.

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
- Jury : la reprise reconnaît une réponse déjà notée par son `execution_id`. Un texte de
  réponse modifié à la main dans `outputs/` (interdit par les règles du projet) ne serait
  pas détecté.
- Scores : les fichiers de votes sont relus tels quels (même juge, même grille, même
  réponse pour tous les juges) ; le déploiement et le prompt du juge ont déjà été vérifiés
  par `jury` à l'écriture et ne sont pas revérifiés ici.
- Richter : l'empreinte du jury couvre les juges et la grille, pas les consignes du prompt
  des juges (vérifiées par `jury` au moment du vote, pas recopiées dans les scores).
- Méta-évaluation : l'accord entre médecins porte sur toutes les lignes en un tour (égalités
  comprises), la justesse des juges sur les lignes sans égalité : ce ne sont pas les mêmes
  lignes, la comparaison est indicative.
- Jury : le numéro de critère donné par le juge est cru. Un juge qui se trompe de numéro
  range son verdict sous un autre critère ; la méta-évaluation mesurera ce genre d'erreur.
