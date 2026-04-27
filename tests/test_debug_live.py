# ruff: noqa: E402
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv

load_dotenv()

import pandas as pd

from agent.agent import DataMindAgent

df = pd.read_csv("train.csv")
print(f"Dataset: {df.shape}")

agent = DataMindAgent()
agent.set_data(df)
print(f"Agent initialised, model: {agent.model}")

print()
print("=== Test 1: Chat simple ===")
t0 = time.time()
try:
    result = agent.chat("Decris le dataset en 3 phrases")
    elapsed = time.time() - t0
    print(
        f"OK: {elapsed:.1f}s, msg={len(result['message'])} chars, figs={len(result['figures'])}, msgs={len(agent.messages)}"
    )
except Exception as e:
    elapsed = time.time() - t0
    print(f"FAILED apres {elapsed:.1f}s: {type(e).__name__}: {e}")

print()
print("=== Test 2: Chat avec figure ===")
t0 = time.time()
try:
    result = agent.chat("Affiche la distribution de l'age")
    elapsed = time.time() - t0
    print(
        f"OK: {elapsed:.1f}s, msg={len(result['message'])} chars, figs={len(result['figures'])}, msgs={len(agent.messages)}"
    )
except Exception as e:
    elapsed = time.time() - t0
    print(f"FAILED apres {elapsed:.1f}s: {type(e).__name__}: {e}")

print()
print("=== Test 3: Quick action 'Decrire le jeu de donnees' ===")
t0 = time.time()
try:
    result = agent.chat("Decrire le jeu de donnees")
    elapsed = time.time() - t0
    print(
        f"OK: {elapsed:.1f}s, msg={len(result['message'])} chars, figs={len(result['figures'])}, msgs={len(agent.messages)}"
    )
except Exception as e:
    elapsed = time.time() - t0
    print(f"FAILED apres {elapsed:.1f}s: {type(e).__name__}: {e}")
