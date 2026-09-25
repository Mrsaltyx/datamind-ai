"""Smoke test manuel : EDA automatique complet, chronometre.

Usage : uv run python scripts/manual/smoke_auto_eda.py <fichier.csv>
Necessite un LLM reel (Ollama lance ou cle API dans .env).
"""

from __future__ import annotations

import time

from _bootstrap import load_df, make_agent


def main() -> None:
    df = load_df()
    agent = make_agent(df)

    print("=== EDA automatique ===")
    t0 = time.time()
    try:
        result = agent.auto_eda()
        elapsed = time.time() - t0
        print(f"OK en {elapsed:.1f}s")
        print(f"  message : {len(result['message'])} chars")
        print(f"  figures : {len(result['figures'])}")
        print(f"  messages en contexte : {len(agent.messages)}")
        print()
        print(result["message"][:1500])
    except Exception as e:
        elapsed = time.time() - t0
        print(f"ECHEC apres {elapsed:.1f}s : {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
