"""Bootstrap commun des scripts manuels : charge un CSV et initialise l'agent."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Script manuel DataMind (appelle un LLM reel)")
    parser.add_argument("csv", help="Chemin vers un fichier CSV")
    parser.add_argument("--rows", type=int, default=None, help="Limiter le nombre de lignes")
    return parser.parse_args()


def load_df() -> pd.DataFrame:
    args = parse_args()
    path = Path(args.csv)
    if not path.exists():
        sys.exit(f"Fichier introuvable : {path}")
    df = pd.read_csv(path, nrows=args.rows)
    print(f"Dataset : {df.shape[0]} lignes x {df.shape[1]} colonnes (source : {path})")
    return df


def make_agent(df: pd.DataFrame):
    import asyncio

    from datamind.agent.agent import DataMindAgent
    from datamind.core.bootstrap import refresh_status

    status = asyncio.run(refresh_status())
    print(f"LLM : {status.active} | {status.message}")
    if status.provider is None or status.degraded:
        print("ATTENTION : aucun LLM utilisable, le chat va echouer.")
        print("Lancez Ollama (ollama serve) ou configurez une cle API.\n")
    agent = DataMindAgent(provider=status.provider)
    agent.set_data(df)
    return agent
