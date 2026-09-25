# Revue v3 — synthèse de validation

Chaque décision est listée avec l'avis du compagnon ML et les faits mesurés.
Le statut est à valider / invalider une par une avec l'utilisateur.

## Résultats mesurés (test grandeur nature, CSV 116 Mo, 1 M lignes)

- Big-file pipeline : upload 5,4 s, train réel F1 0,833 ± 0,0003 (Crop), R² 0,913 (Yield).
- Chat réel contre Ollama local (`gemma4:e4b`) : 4 scénarios / 4 réussis
  (describe 45,6 s · ml 38,8 s · train 37,4 s · figure 19,2 s), ~141 s au total.
- Tokens LLM sur la session complète : **8 requêtes, 43 409 tokens dont 39 858 de prompt (87 %)**.
- Suite : 95/95 tests verts, ruff propre. Rien poussé, `master` intact.

## Décisions à valider

| # | Décision | Mon avis | Recommandation |
|---|---|---|---|
| 1 | GGUF / mode embedded supprimé — providers Ollama + API distante | Aligné sur l'option C (local OU API) validée plus tôt. `llama-cpp-python` était la première friction d'install. | **Valider** |
| 2 | Sessions SQLite conservées telles quelles | Rien de cassé, zéro dette perçue lors de la migration. | **Valider** |
| 3 | Entraînement ML réel borné (baseline sklearn + CV 5-fold) | Tient la promesse produit « pipeline ML ». Borné volontairement : garder le scope. | **Valider** |
| 4 | Live tests consolidés en `scripts/manual/smoke_*.py` | Les 4 fichiers `test_*_live.py` étaient cassés et invisibles hors CI. | **Valider** |
| 5 | Setup universel : uv + lockfile, `setup.ps1/sh`, Docker all-in-one, CI, backend sert le frontend buildé | C'est la promesse « une commande, < 5 min ». Vérifié en conditions réelles sur cette machine. | **Valider** |
| 6 | MLflow optionnel (extra `[mlflow]`, backend SQLite, kill-switch) + instrumentation tokens `/api/metrics` | Ne concurrence pas MLflow : positionné au-dessus (UX conversationnelle). L'arbitrage de pertinence repose désormais sur une vraie mesure (87 % de prompt = marge d'optimisation évidente). | **Valider** |
| 7 | Branche `v3-dev`, `master` préservé, pas de push sans feu vert | Hygiène conservée tout au long. | **Valider** |

## Points de vigilance honnêtes (à trancher, pas des blocages)

- **F1 0,833 (Crop)** : les colonnes `*_ref` sont des moyennes *par culture* — suspect de
  leakage. Test de valeur proposé : ré-entraîner sans les `*_ref` ; si le F1 s'effondre,
  c'était de la fuite, pas une découverte. À intégrer comme garde dans le trainer v3.1.
- **Rendements négatifs** (min −1,15 t/ha) dans le CSV source : problème de qualité
  de données du jeu, pas de datamind. Le LLM l'a signalé tout seul — bon signe UX.
- **Latences 20–45 s par réponse** sur CPU local : acceptable en local, mais à
  documenter honnêtement dans le README (déjà partiellement fait).
- **`data-test/`** : 116 Mo gitignored, volontairement hors dépôt.

## Prochaine étape après validation

Feu vert explicite → push `v3-dev`, ouverture PR vers `master`, puis traitement
du test de leakage `*_ref` en v3.1.
