# DataMind AI

[![CI](https://github.com/Mrsaltyx/datamind-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/Mrsaltyx/datamind-ai/actions)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub stars](https://img.shields.io/github/stars/Mrsaltyx/datamind-ai)](https://github.com/Mrsaltyx/datamind-ai)

Assistant d'analyse de données propulsé par l'IA : chargez un CSV, posez vos questions en français, obtenez visualisations interactives, statistiques et recommandations Machine Learning.

## Ce que fait le projet

- Automatise l'analyse exploratoire (EDA) : statistiques descriptives, corrélations, distributions, variables catégorielles — en un clic.
- Propose un chat en langage naturel : l'agent LLM sélectionne les outils pertinents et produit des visualisations Plotly.
- Fournit 10 outils d'analyse : describe, distribution, corrélation, outliers (IQR), tendances temporelles, comparaison de groupes, catégories, scatter, détection de cible ML, pipeline ML complet.
- Conseille sur le ML : détection automatique de la variable cible et du type de tâche (classification / régression), modèles recommandés avec hyperparamètres, preprocessing et métriques.
- Supporte 3 fournisseurs LLM : modèle embarqué (GGUF local), Ollama, ou API distante compatible OpenAI.
- Persiste les sessions (SQLite asynchrone) et se déploie via Docker / docker-compose.

## Stack technique

| Composant | Technologie |
|---|---|
| Frontend | Vue 3 (Composition API), TypeScript, Pinia, Vue Router, Tailwind CSS 4 |
| Backend | Python 3.12+, FastAPI, Pydantic v2 |
| Base de données | SQLite async (SQLAlchemy + aiosqlite) |
| Agent LLM | API compatible OpenAI (Ollama / GGUF embarqué / distant) |
| Data | Pandas, NumPy, SciPy, Plotly |
| Build | Vite 6, vue-tsc |
| Conteneurisation | Docker, docker-compose |
| CI/CD | GitHub Actions (ruff, pytest, vue-tsc, docker build) |

## Prérequis

- Python 3.12 ou plus récent
- Node.js 20 ou plus récent
- Ollama (optionnel, pour le mode LLM local) : [ollama.com](https://ollama.com)

## Installation

```bash
git clone https://github.com/Mrsaltyx/datamind-ai.git
cd datamind-ai

cp .env.example .env
# Éditer .env selon le provider LLM choisi

# Backend
pip install -e ".[dev]"
uvicorn backend.main:app --reload --port 8000

# Frontend (dans un autre terminal)
cd frontend
npm install
npm run dev
```

L'application est accessible sur `http://localhost:5173`.

Avec Docker :

```bash
docker-compose up --build
```

Services démarrés : frontend (`:3000`), backend API (`:8000`), Ollama (`:11434`).

## Configuration

Principales variables d'environnement :

| Variable | Description | Défaut |
|---|---|---|
| `LLM_PROVIDER` | Provider LLM : `embedded`, `ollama`, `remote` | `ollama` |
| `EMBEDDED_MODEL_PATH` | Chemin du modèle GGUF (mode embarqué) | `models/gemma-4-4b-it-Q4_K_M.gguf` |
| `OLLAMA_BASE_URL` | URL du serveur Ollama | `http://localhost:11434/v1` |
| `OLLAMA_MODEL` | Modèle Ollama | `gemma4:e4b` |
| `OPENAI_API_KEY` | Clé API (mode distant) | — |
| `OPENAI_BASE_URL` | URL de l'API distante | `https://api.z.ai/api/coding/paas/v4/` |
| `OPENAI_MODEL` | Modèle distant | `glm-5.1` |

Le provider peut aussi être changé en temps réel depuis la sidebar de l'application.

## Utilisation

1. Ouvrir l'application dans le navigateur.
2. Sélectionner le provider LLM dans la sidebar (Ollama, embarqué ou distant).
3. Charger un fichier CSV (drag & drop supporté, détection automatique encodage / délimiteur).
4. Lancer l'EDA automatique ou discuter avec les données via le chat.
5. Générer un rapport ML complet en un clic.

Endpoints API principaux :

```text
POST /api/data/upload                  # Upload d'un CSV
POST /api/chat/{session_id}/send       # Message au chat
POST /api/chat/{session_id}/auto-eda   # EDA automatique
POST /api/ml/{session_id}/suggest      # Suggestion de pipeline ML
GET  /api/health                       # Health check
```

## Structure du projet

```text
datamind-ai/
  backend/      # API REST FastAPI (routers, schémas, sessions)
  frontend/     # SPA Vue 3 (components, stores Pinia, Dockerfile nginx)
  agent/        # Agent LLM (3 providers, tool loop) et 10 outils d'analyse
  utils/        # Chargement CSV, graphiques Plotly, preprocessing, ML advisor
  prompts/      # System prompt de l'agent
  scripts/      # Téléchargement du modèle GGUF, setup Windows
  tests/        # 62 tests pytest
  docker-compose.yml
  pyproject.toml
```

## Développement

Lancer les tests et les contrôles qualité :

```bash
pytest                    # 62 tests unitaires et d'intégration
ruff check .              # Linting backend
cd frontend && npx vue-tsc --noEmit   # Type-check frontend
```

## État du projet

- Version 2 fonctionnelle : EDA automatique, chat, conseiller ML et déploiement Docker opérationnels.
- CI GitHub Actions en place (lint, tests, type-check, build Docker).

## Licence

MIT
