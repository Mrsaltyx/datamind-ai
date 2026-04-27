# DataMind AI v2

Agent intelligent d'analyse de données propulsé par l'IA. Chargez un CSV, posez vos questions en français, et obtenez des visualisations interactives, des statistiques et des recommandations Machine Learning.

## Aperçu

DataMind AI est un assistant d'analyse de données propulsé par une architecture **FastAPI + Vue 3** avec un agent LLM multi-provider. Il automatise l'analyse exploratoire de données (EDA) et fournit des conseils Machine Learning en langage naturel.

## Fonctionnalités

- **3 fournisseurs LLM** : Mode embarqué (GGUF local), Ollama, ou API distante (OpenAI-compatible)
- **EDA automatique** : En un clic, génération d'un rapport complet (statistiques descriptives, corrélations, distributions, variables catégorielles)
- **Chat interactif** : Posez vos questions en français, l'agent sélectionne les outils pertinents et produit des visualisations Plotly
- **10 outils d'analyse** : describe, distribution, corrélation, outliers (IQR), tendances temporelles, comparaison de groupes, catégories, scatter, détection de cible ML, pipeline ML complet
- **Conseiller ML intégré** : Détection automatique de la variable cible, type de tâche (classification/régression), modèles recommandés avec hyperparamètres, preprocessing et métriques d'évaluation
- **Chargement CSV robuste** : Détection automatique de l'encodage et du délimiteur, fallback multi-encodages (UTF-8, Latin-1, CP1252)
- **Interface sombre et moderne** avec métriques en temps réel
- **Persistance des sessions** : SQLite asynchrone pour sauvegarder les analyses
- **Déploiement Docker** : docker-compose prêt pour la production

## Stack technique

| Composant | Technologie |
|---|---|
| Frontend | Vue 3 (Composition API), TypeScript, Pinia, Vue Router |
| UI / Styles | Tailwind CSS 4 |
| Backend | Python 3.12+, FastAPI, Pydantic v2 |
| Base de données | SQLite (async via SQLAlchemy + aiosqlite) |
| Agent LLM | OpenAI API (Ollama / GGUF embarqué / API distante) |
| Analyse de données | Pandas, NumPy, SciPy |
| Visualisation | Plotly |
| Build Frontend | Vite 6, vue-tsc |
| Conteneurisation | Docker, docker-compose |
| CI/CD | GitHub Actions (ruff, pytest, vue-tsc, docker build) |

## Architecture

```
datamind-ai/
├── backend/                          # API REST FastAPI
│   ├── main.py                       # App FastAPI, CORS, routers, lifespan
│   ├── config.py                     # Pydantic Settings (3 providers LLM)
│   ├── database.py                   # SQLAlchemy async engine
│   ├── models.py                     # ORM : SessionModel, MessageModel
│   ├── routers/
│   │   ├── data.py                   # Upload, summary, preview, statistics
│   │   ├── chat.py                   # Chat, auto-EDA, historique
│   │   ├── tools.py                  # Exécution d'outils
│   │   ├── ml.py                     # Suggestion ML, détection cible
│   │   └── config.py                 # Configuration LLM runtime
│   ├── schemas/
│   │   └── responses.py              # Modèles Pydantic v2
│   └── services/
│       └── session.py                # SessionManager (cache + SQLite)
├── frontend/                         # SPA Vue 3
│   ├── src/
│   │   ├── api/client.ts             # Client Axios (API REST)
│   │   ├── components/
│   │   │   ├── Sidebar.vue           # Config LLM + upload CSV
│   │   │   ├── ChatView.vue          # Chat avec rendu Plotly
│   │   │   ├── DataPreview.vue       # Aperçu tableau + stats
│   │   │   ├── AutoEda.vue           # EDA en un clic
│   │   │   ├── MlSuggestion.vue      # Rapport ML complet
│   │   │   └── ...                   # PlotlyChart, ToastContainer, etc.
│   │   ├── stores/                   # Pinia (data, chat, config)
│   │   ├── composables/              # useToast
│   │   └── types/                    # Interfaces TypeScript
│   ├── Dockerfile                    # Build Node + nginx
│   └── nginx.conf                    # Proxy /api/ vers backend
├── agent/
│   ├── agent.py                      # Agent LLM (3 providers, retry, tool loop)
│   └── tools.py                      # 10 outils d'analyse (schémas + exécution)
├── utils/
│   ├── data_loader.py                # Chargement CSV, résumés, dtype optimization
│   ├── charts.py                     # 7 types de visualisations Plotly
│   ├── preprocessing.py              # Détection cible, type de tâche, preprocessing
│   └── ml_advisor.py                 # Recommandation modèles + rapports ML
├── prompts/
│   └── system_prompt.py              # System prompt de l'agent
├── scripts/
│   ├── download-model.py             # Téléchargement modèle GGUF
│   └── dev.bat                       # Setup environnement Windows
├── tests/                            # 62 tests pytest
├── docker-compose.yml                # Ollama + Backend + Frontend
├── pyproject.toml                    # Config Python (deps, ruff, pytest)
└── .github/workflows/ci.yml          # CI GitHub Actions
```

## Installation rapide

### Prérequis

- **Python 3.12+** et **Node.js 20+**
- **Ollama** (optionnel, pour le mode local) : [ollama.com](https://ollama.com)

### Mode développement

```bash
# Cloner le repo
git clone https://github.com/Mrsaltyx/datamind-ai.git
cd datamind-ai

# Configuration
cp .env.example .env
# Éditez .env selon votre provider LLM préféré

# Backend
pip install -e ".[dev]"
uvicorn backend.main:app --reload --port 8000

# Frontend (dans un autre terminal)
cd frontend
npm install
npm run dev
```

L'application est accessible sur `http://localhost:5173`.

### Mode Docker

```bash
docker-compose up --build
```

Services démarrés :
- **Frontend** : `http://localhost:3000` (nginx)
- **Backend API** : `http://localhost:8000` (FastAPI)
- **Ollama** : `http://localhost:11434` (LLM local)

## Configuration

Copiez `.env.example` en `.env` et configurez votre provider LLM :

| Variable | Description | Défaut |
|---|---|---|
| `LLM_PROVIDER` | Provider LLM : `embedded`, `ollama`, `remote` | `ollama` |
| `EMBEDDED_MODEL_PATH` | Chemin vers le modèle GGUF (mode embarqué) | `models/gemma-4-4b-it-Q4_K_M.gguf` |
| `OLLAMA_BASE_URL` | URL du serveur Ollama | `http://localhost:11434/v1` |
| `OLLAMA_MODEL` | Modèle Ollama à utiliser | `gemma4:e4b` |
| `OPENAI_API_KEY` | Clé API (mode distant) | — |
| `OPENAI_BASE_URL` | URL de base API distante | `https://api.z.ai/api/coding/paas/v4/` |
| `OPENAI_MODEL` | Modèle distant | `glm-5.1` |

Vous pouvez aussi changer de provider en temps réel depuis la sidebar de l'application.

## Utilisation

1. Ouvrez l'application dans votre navigateur
2. Sélectionnez votre provider LLM dans la sidebar (Ollama, embarqué ou distant)
3. Chargez un fichier CSV (drag & drop supporté)
4. Lancez l'**EDA automatique** ou discutez avec vos données via le **chat**
5. Générez un **rapport ML complet** en un clic

## API REST

| Endpoint | Méthode | Description |
|---|---|---|
| `/api/data/upload` | POST | Upload d'un fichier CSV |
| `/api/data/{session_id}/summary` | GET | Résumé du dataset |
| `/api/data/{session_id}/preview` | GET | Aperçu des données |
| `/api/data/{session_id}/statistics` | GET | Statistiques détaillées |
| `/api/chat/{session_id}/send` | POST | Envoyer un message au chat |
| `/api/chat/{session_id}/auto-eda` | POST | Lancer l'EDA automatique |
| `/api/tools/{session_id}/execute` | POST | Exécuter un outil spécifique |
| `/api/ml/{session_id}/suggest` | POST | Suggestion de pipeline ML |
| `/api/ml/{session_id}/detect-target` | POST | Détecter la variable cible |
| `/api/config/update` | POST | Mettre à jour la config LLM |
| `/api/config/llm-status` | GET | Statut du provider LLM |
| `/api/health` | GET | Health check |

## Outils disponibles

| Outil | Description |
|---|---|
| `describe_data` | Statistiques descriptives du dataset ou d'une colonne |
| `show_distribution` | Histogramme + box plot pour une colonne numérique |
| `show_correlation` | Matrice de corrélation avec heatmap |
| `detect_outliers` | Détection des valeurs aberrantes (méthode IQR) |
| `show_trends` | Série temporelle avec moyenne mobile |
| `compare_groups` | Comparaison d'une variable numérique par catégorie |
| `show_categorical` | Distribution d'une variable catégorielle (barres + camembert) |
| `show_scatter` | Nuage de points avec droite de tendance et R² |
| `detect_target_and_task` | Détection automatique de la cible ML et du type de tâche |
| `suggest_ml_pipeline` | Rapport ML complet avec modèles, hyperparamètres et évaluation |

## Tests

```bash
# Tests unitaires et d'intégration (62 tests)
pytest

# Linting backend
ruff check .

# Type-check frontend
cd frontend && npx vue-tsc --noEmit
```

## Licence

MIT
