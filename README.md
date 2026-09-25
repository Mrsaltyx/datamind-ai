# DataMind AI

[![CI](https://github.com/Mrsaltyx/datamind-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/Mrsaltyx/datamind-ai/actions)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Assistant d'analyse de données propulsé par l'IA : chargez un CSV, discutez avec vos données en français, obtenez visualisations interactives, statistiques — et **entraînez de vrais modèles ML** en un clic.

## Démarrage en une commande

**Windows** — `powershell -ExecutionPolicy Bypass -File scripts/setup.ps1` puis `scripts\run.ps1`

**Linux/macOS** — `bash scripts/setup.sh` puis `bash scripts/run.sh`

C'est tout. Ouvrez <http://localhost:8000>.

- Prérequis : seulement **Python 3.12+** (uv s'installe tout seul). Node.js est optionnel (le frontend pré-buildé peut être téléchargé depuis les [releases](https://github.com/Mrsaltyx/datamind-ai/releases)).
- Sans LLM configuré, l'app fonctionne quand même : EDA, visualisations et entraînement ML. Le chat IA s'active dès qu'un provider est disponible.

## Ce que fait le projet

- **EDA automatique** : statistiques descriptives, corrélations, distributions, outliers — en un clic, sans LLM.
- **Chat en langage naturel** : l'agent LLM sélectionne 11 outils d'analyse et produit des visualisations Plotly.
- **ML réel** : `POST /api/ml/{session_id}/train` entraîne une baseline scikit-learn (Logistic Regression / Ridge) et retourne des métriques mesurées par validation croisée — F1, ROC-AUC, RMSE, R² ± écart-type. L'outil `train_model` permet aussi à l'agent de lancer l'entraînement dans la conversation.
- **Deux providers LLM** : local via [Ollama](https://ollama.com) (défaut, `gemma4:e4b`) ou API distante compatible OpenAI — basculez depuis la sidebar ou `.env`.
- **Sessions persistées** (SQLite asynchrone) et déploiement Docker.

## Stack technique

| Composant | Technologie |
|---|---|
| Frontend | Vue 3, TypeScript, Pinia, Tailwind CSS 4 — **servi par le backend** |
| Backend | Python 3.12+, FastAPI, Pydantic v2, package `src/datamind/` |
| LLM | API compatible OpenAI (Ollama local / distant) |
| ML | scikit-learn (baselines mesurées en CV) |
| Data | Pandas 2/3, NumPy, SciPy, Plotly |
| Tooling | uv (lockfile), ruff, pytest, GitHub Actions |

## Architecture

```text
src/datamind/
  api/        # FastAPI : routers, schemas ; sert aussi le frontend buildé
  agent/      # boucle agent tool-calling (prompt system inclus)
  providers/  # abstraction LLM : OllamaProvider / RemoteProvider
  analysis/   # 11 outils d'analyse, EDA, preprocessing, advisor ML
  ml/         # entrainement de baselines (trainer.py)
  core/       # config, sessions SQLite, bootstrap auto-detection LLM
```

Le bootstrap détecte au démarrage ce qui est disponible (Ollama ? clé API ?) et s'adapte : **mode dégradé** = EDA + ML sans LLM, chat désactivé avec message d'aide.

## Configuration

Copiez `.env.example` vers `.env` :

| Variable | Description | Défaut |
|---|---|---|
| `LLM_PROVIDER` | `ollama` (local) ou `remote` (API) | `ollama` |
| `OLLAMA_MODEL` | Modèle Ollama | `gemma4:e4b` |
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL` | Provider distant | z.ai `glm-5.1` |

Basculable en temps réel depuis la sidebar de l'application.

## API

```text
POST /api/data/upload                 # Upload d'un CSV -> session_id
POST /api/chat/{session_id}/send      # Message au chat (LLM requis)
POST /api/chat/{session_id}/auto-eda  # EDA automatique (LLM requis)
POST /api/ml/{session_id}/train       # Entraine une baseline (aucun LLM requis)
POST /api/ml/{session_id}/suggest     # Rapport de recommandation ML
GET  /api/config/llm-status           # Etat du provider
GET  /api/health                      # Health check
GET  /docs                            # Swagger UI
```

## Développement

```bash
uv sync --extra dev        # environnement complet
uv run pytest              # 84 tests
uv run ruff check src tests
cd frontend && npm ci && npx vue-tsc -b --noEmit
```

Scripts manuels (appels LLM réels) : voir `scripts/manual/`.

## Docker

```bash
docker compose up --build   # app + Ollama
```

## Licence

MIT
