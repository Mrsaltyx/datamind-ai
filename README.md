# DataMind AI

Agent intelligent d'analyse de données propulsé par l'IA. Chargez un CSV, posez vos questions en français, et obtenez des visualisations interactives, des statistiques et des recommandations Machine Learning.

## Aperçu

DataMind AI est un assistant d'analyse de données construit avec Streamlit et un agent LLM (OpenAI / compatible). Il automatise l'analyse exploratoire de données (EDA) et fournit des conseils Machine Learning en langage naturel.

## Fonctionnalités

- **EDA automatique** : En un clic, génération d'un rapport complet (statistiques descriptives, corrélations, distributions, variables catégorielles)
- **Chat interactif** : Posez vos questions en français, l'agent sélectionne les outils pertinents et produit des visualisations Plotly
- **10 outils d'analyse** : describe, distribution, corrélation, outliers (IQR), tendances temporelles, comparaison de groupes, catégories, scatter, détection de cible ML, pipeline ML complet
- **Conseiller ML intégré** : Détection automatique de la variable cible, type de tâche (classification/régression), modèles recommandés avec hyperparamètres, preprocessing et métriques d'évaluation
- **Chargement CSV robuste** : Détection automatique de l'encodage et du délimiteur, fallback multi-encodages
- **Interface sombre et moderne** avec métriques en temps réel

## Stack technique

| Composant | Technologie |
|---|---|
| Interface utilisateur | Streamlit |
| Agent LLM | OpenAI API (compatible z.Ai, GPT-4o, etc.) |
| Analyse de données | Pandas, NumPy, SciPy |
| Visualisation | Plotly |
| Architecture | Function Calling (tool use) |
| Langage | Python 3.10+ |

## Architecture

```
datamind-ai/
├── app.py                    # Interface Streamlit principale
├── agent/
│   ├── agent.py              # Agent LLM (contexte, retry, orchestration)
│   └── tools.py              # 10 outils d'analyse (schémas + exécution)
├── utils/
│   ├── data_loader.py        # Chargement CSV et résumés statistiques
│   ├── charts.py             # Visualisations Plotly
│   ├── preprocessing.py      # Détection cible, type de tâche, preprocessing
│   └── ml_advisor.py         # Recommandation de modèles et stratégies ML
├── prompts/
│   └── system_prompt.py      # System prompt de l'agent
├── tests/                    # Tests d'intégration
├── .env.example
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/Mrsaltyx/datamind-ai.git
cd datamind-ai
pip install -r requirements.txt
```

## Configuration

Copiez le fichier `.env.example` en `.env` et renseignez vos clés :

```bash
cp .env.example .env
```

Variables d'environnement :

| Variable | Description | Défaut |
|---|---|---|
| `OPENAI_API_KEY` | Votre clé API (z.Ai ou OpenAI) | — |
| `OPENAI_BASE_URL` | URL de base de l'API | `https://api.z.ai/api/coding/paas/v4/` |
| `OPENAI_MODEL` | Modèle à utiliser | `glm-5.1` |

Vous pouvez aussi configurer ces paramètres directement dans la sidebar de l'application.

## Utilisation

```bash
streamlit run app.py
```

1. Ouvrez l'application dans votre navigateur
2. Entrez votre clé API dans la sidebar
3. Chargez un fichier CSV
4. Lancez l'EDA automatique ou discutez avec vos données via le chat
5. Générez un rapport ML complet en un clic

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

## Licence

MIT
