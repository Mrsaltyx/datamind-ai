# Refonte « nouvelle V1 » — v3-dev

## Baseline (état v2.1 au moment du fork, mesurée par exécution)

- Tests : **59/62 passent** (3 échecs dans `utils/preprocessing.py`, heuristiques edge cases)
- 33 × `Pandas4Warning` (rupture pandas 3 en vue sur `select_dtypes(include=["object"])`)
- Code mort : `n_folds` (3 branches identiques) dans `ml_advisor.py` ; init client dupliqué ×4 dans `agent.py`
- `llama-cpp-python` en dépendance principale (friction d'installation) — **retiré** en v3
- 4 fichiers `test_*_live.py` hors CI, cassés en l'état (lisent un `train.csv` absent)
- Promesse produit non tenue : « pipeline ML complet » = conseils uniquement, aucun entraînement (scikit-learn absent)

## Décisions validées

| Sujet | Décision |
|---|---|
| GGUF / mode embedded | **Supprimé** — 2 providers : Ollama (local) / API distante |
| Sessions SQLite | Conservées telles quelles |
| Entraînement ML réel | Oui — version bornée (baseline sklearn + CV 5-fold + métriques) |
| Live tests | Consolidation en `scripts/manual/smoke_*.py` |
| Setup | uv + lockfile, backend sert le frontend buildé, scripts one-command |
| Workflow | Branche `v3-dev`, `master` préservé, pas de push sans feu vert |

## Phasage

1. Fondations : uv, `src/datamind/`, couche providers, bootstrap auto-détection, static serving
2. Migration code éprouvé + fix racine des 3 tests + pandas 3-proofing
3. Endpoint `/api/ml/{session_id}/train` + outil agent `train_model`
4. Setup universel : `setup.ps1/sh`, CI release avec `dist/`, README
5. Qualité finale : gates verts (ruff/pytest/vue-tsc), smoke scripts, mémoire skill

## Métrique de succès

Setup utilisateur final : **une commande, < 5 min, zéro prérequis exotique**. Suite de tests 100 % verte.
