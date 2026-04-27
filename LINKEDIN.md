# DataMind AI v2 — De Streamlit à une architecture Full-Stack FastAPI + Vue 3

## Update majeure : DataMind AI passe en v2

Après avoir construit la première version de DataMind AI avec Streamlit, j'ai décidé de faire passer le projet au niveau supérieur avec une refonte complète de l'architecture. Le résultat : une application full-stack production-ready avec FastAPI + Vue 3.

## Ce qui a changé

### Architecture complètement repensée

| | v1 (Streamlit) | v2 (FastAPI + Vue 3) |
|---|---|---|
| Frontend | Streamlit (Python monolithique) | Vue 3 + TypeScript + Tailwind CSS |
| Backend | Script Python unique | API REST FastAPI avec SQLAlchemy async |
| State management | st.session_state | Pinia stores |
| Déploiement | streamlit run | Docker (nginx + FastAPI + Ollama) |
| CI/CD | Aucun | GitHub Actions (lint, test, build, Docker) |
| Tests | Basiques | 62 tests pytest + type-check TypeScript |

### Nouvelles fonctionnalités v2

- **3 providers LLM** : Modèle embarqué (GGUF local via llama.cpp), Ollama, ou API distante (OpenAI-compatible) — switchable à chaud depuis l'interface
- **API REST complète** : 12 endpoints documentés pour l'upload, le chat, les outils d'analyse et la configuration
- **Persistance des sessions** : SQLite async pour reprendre ses analyses
- **Interface Vue 3** : Sidebar, chat interactif, rendu Plotly, preview tabulaire, notifications toast
- **Déploiement Docker** : docker-compose avec Ollama + Backend + Frontend (nginx)
- **CI/CD GitHub Actions** : ruff (lint), pytest (62 tests), vue-tsc (type-check), Docker build

### Ce qui reste (et s'améliore)

Les fonctionnalités clés de la v1 sont toujours là, améliorées :

- **EDA automatique** : En un clic, rapport complet avec statistiques, corrélations, distributions
- **Chat en langage naturel** : Posez vos questions en français, l'agent sélectionne les 10 outils pertinents et produit des visualisations Plotly interactives
- **Conseiller ML intégré** : Détection automatique de la variable cible, classification/régression, modèles recommandés avec hyperparamètres
- **Chargement CSV robuste** : Détection auto d'encodage (UTF-8, Latin-1, CP1252) et de délimiteur

## Stack technique v2

| Composant | Technologie |
|---|---|
| Frontend | Vue 3, TypeScript, Pinia, Tailwind CSS 4 |
| Backend | Python 3.12+, FastAPI, Pydantic v2, SQLAlchemy async |
| Agent LLM | OpenAI API (multi-provider : GGUF / Ollama / distant) |
| Analyse | Pandas, NumPy, SciPy |
| Visualisation | Plotly |
| Base de données | SQLite (async) |
| Déploiement | Docker, nginx |
| CI/CD | GitHub Actions |

## Ce que j'ai appris avec cette v2

- **Architecture full-stack** : Séparation propre frontend/backend avec API REST
- **Vue 3 Composition API** : Stores Pinia, composants réactifs, TypeScript
- **FastAPI asynchrone** : SQLAlchemy async, lifespan, middleware CORS
- **Multi-provider LLM** : Abstraction entre llama.cpp, Ollama et API distante
- **DevOps** : Docker multi-stage, nginx reverse proxy, CI/CD GitHub Actions
- **Qualité code** : ruff linting, 62 tests pytest, type-check TypeScript, coverage

## Essayer DataMind AI v2

```bash
git clone https://github.com/Mrsaltyx/datamind-ai.git
cd datamind-ai
docker-compose up --build
```

Ou en mode développement :
```bash
# Backend
pip install -e ".[dev]"
uvicorn backend.main:app --reload

# Frontend
cd frontend && npm install && npm run dev
```

## Liens

- **Repo GitHub** : [github.com/Mrsaltyx/datamind-ai](https://github.com/Mrsaltyx/datamind-ai)

## Tags

`#DataScience` `#IA` `#LLM` `#Python` `#VueJS` `#FastAPI` `#TypeScript` `#DataAnalysis` `#MachineLearning` `#AgentIA` `#Docker` `#Plotly` `#EDA` `#FullStack` `#OpenSource`
