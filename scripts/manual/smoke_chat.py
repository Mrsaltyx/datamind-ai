"""Smoke test manuel : chat simple, chat avec figure, question ML, reload config.

Usage : uv run python scripts/manual/smoke_chat.py <fichier.csv> [--rows N]
Necessite un LLM reel (Ollama lance ou cle API dans .env).
"""

from __future__ import annotations

import time

from _bootstrap import load_df, make_agent


def run_scenario(agent, label: str, message: str) -> None:
    print(f"=== {label} ===")
    t0 = time.time()
    try:
        result = agent.chat(message)
        elapsed = time.time() - t0
        print(
            f"OK en {elapsed:.1f}s | {len(result['message'])} chars | "
            f"{len(result['figures'])} figure(s) | {len(agent.messages)} messages"
        )
    except Exception as e:
        elapsed = time.time() - t0
        print(f"ECHEC apres {elapsed:.1f}s : {type(e).__name__}: {e}")
    print()


def main() -> None:
    df = load_df()
    agent = make_agent(df)

    run_scenario(agent, "Chat simple", "Decris le jeu de donnees en 3 phrases")
    run_scenario(agent, "Chat avec figure", "Affiche la distribution de la premiere colonne numerique")
    run_scenario(agent, "Question ML", "Quels modeles ML recommandes-tu pour ce dataset ?")
    run_scenario(agent, "Entrainement", "Entraine un modele et donne-moi les metriques")


if __name__ == "__main__":
    main()
