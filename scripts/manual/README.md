# Scripts manuels — à lancer à la main, jamais en CI

Ces scripts appellent un **LLM réel** (Ollama ou API distante selon `.env`).
Ils ne sont pas des tests automatisés : ce sont des outils de terrain pour
déboguer l'agent et vérifier le comportement de bout en bout.

```bash
# 0. Prérequis : dependances installées (uv sync) et un LLM disponible
uv run python scripts/manual/smoke_chat.py tests/fixtures/sample_heart.csv
uv run python scripts/manual/smoke_auto_eda.py tests/fixtures/sample_heart.csv

# e2e_check.py : vérifie l'API sans LLM (mode dégradé, train, upload).
# Nécessite un serveur tournant : uv run uvicorn datamind.api.main:app --port 8124
uv run python scripts/manual/e2e_check.py
```

Scénarios couverts par `smoke_chat.py` : chat simple, chat avec figure,
question ML, bascule de provider (reload). `smoke_auto_eda.py` : l'EDA
automatique complet avec chronométrage.
