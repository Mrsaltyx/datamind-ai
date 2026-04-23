# DataMind AI - Assistant d'analyse de données propulsé par l'IA

## Le projet en quelques mots

DataMind AI est un agent intelligent d'analyse de données qui permet d'explorer, visualiser et comprendre n'importe quel jeu de données en langage naturel. Chargez un CSV, posez vos questions, et l'IA s'occupe du reste.

## Le problème

L'analyse exploratoire de données (EDA) est une étape chronophage dans tout projet data science. Entre le nettoyage, les statistiques descriptives, les visualisations et la détection d'anomalies, les data analysts passent des heures sur des tâches répétitives avant même de commencer la modélisation.

## La solution

J'ai construit DataMind AI, un agent conversationnel qui automatise l'ensemble du workflow d'analyse de données :

- **EDA automatique** : En un clic, l'agent génère un rapport complet avec statistiques descriptives, corrélations, distributions et recommandations
- **Chat interactif** : Posez vos questions en français, l'agent sélectionne les outils pertinents et produit des visualisations Plotly
- **Conseiller ML intégré** : Détecte automatiquement la variable cible, identifie le type de tâche (classification/régression), recommande des modèles avec hyperparamètres et stratégies d'évaluation

## Stack technique

| Composant | Technologie |
|---|---|
| Interface | Streamlit |
| Agent LLM | OpenAI API (compatible z.Ai / GPT) |
| Analyse de données | Pandas, NumPy, SciPy |
| Visualisation | Plotly |
| Architecture | Function Calling (tool use) |
| Langage | Python |

## Architecture

L'application repose sur une architecture d'agent avec function calling :

1. **app.py** - Interface Streamlit avec sidebar de configuration, preview des données, EDA automatique et chat
2. **agent/agent.py** - Agent LLM avec gestion du contexte, retry logic et orchestration des outils
3. **agent/tools.py** - 10 outils d'analyse (describe, distribution, corrélation, outliers, tendances, scatter, ML pipeline...)
4. **utils/** - Modules de data loading, charts Plotly, preprocessing et conseil ML
5. **prompts/** - System prompt structuré guidant le comportement de l'agent

## Fonctionnalités clés

- Chargement de CSV avec détection automatique d'encodage et de délimiteur
- Analyse exploratoire en un clic (describe, corrélation, distribution, catégories)
- Chat en langage naturel avec visualisations interactives
- Détection automatique de la cible ML et du type de tâche
- Rapport ML complet : modèles recommandés, hyperparamètres, preprocessing, métriques d'évaluation
- Détection de valeurs aberrantes (IQR), analyse de tendances, comparaisons de groupes
- Interface sombre et moderne avec métriques en temps réel

## Ce que j'ai appris

- Conception d'agents LLM avec function calling et gestion de contexte
- Orchestration de pipelines d'analyse de données complexes via des outils
- Building d'interfaces Streamlit avec state management avancé
- Visualisation de données interactives avec Plotly
- Stratégies de retry et gestion d'erreurs pour API LLM

## Liens

- **Repo GitHub** : [github.com/Mrsaltyx/datamind-ai](https://github.com/Mrsaltyx/datamind-ai)

## Tags

`#DataScience` `#IA` `#LLM` `#Python` `#Streamlit` `#DataAnalysis` `#MachineLearning` `#AgentIA` `#OpenAI` `#Plotly` `#EDA` `#FunctionCalling`
