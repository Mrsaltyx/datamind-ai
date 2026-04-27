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
print("=== Test auto_eda (full 630K rows, optimized prompt) ===")
t0 = time.time()
try:
    result = agent.auto_eda()
    elapsed = time.time() - t0
    print(f"OK: {elapsed:.1f}s")
    print(f"  msg length: {len(result['message'])} chars")
    print(f"  figures: {len(result['figures'])}")
    print(f"  total agent messages: {len(agent.messages)}")
except Exception as e:
    elapsed = time.time() - t0
    print(f"FAILED apres {elapsed:.1f}s: {type(e).__name__}: {e}")
