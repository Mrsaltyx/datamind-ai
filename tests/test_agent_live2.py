import os
import sys
import json

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

import pandas as pd
from agent.agent import DataMindAgent

df = pd.read_csv("train.csv", nrows=500)
print(f"Dataset: {df.shape}")

agent = DataMindAgent()
agent.set_data(df)
print(f"Agent initialised")
print(f"Client base_url: {agent.client.base_url}")
print(f"Model: {agent.model}")

print()
print("=== Test 1: Chat avec tool (distribution Age) ===")
result = agent.chat("Affiche la distribution de l'age")
print(f"Message length: {len(result['message'])} chars")
print(f"Has content: {len(result['message']) > 0}")
print(f"Figures: {len(result['figures'])}")
print(f"Messages in context: {len(agent.messages)}")

print()
print("=== Test 2: Chat ML pipeline ===")
result2 = agent.chat("Quels modeles ML recommandes-tu pour ce dataset ?")
print(f"Message length: {len(result2['message'])} chars")
print(f"Has content: {len(result2['message']) > 0}")
print(f"Figures: {len(result2['figures'])}")

print()
print("=== Test 3: reload_config ===")
os.environ["OPENAI_MODEL"] = "gpt-4o"
agent.reload_config()
print(f"Model apres reload: {agent.model}")
os.environ["OPENAI_MODEL"] = "glm-5.1"
agent.reload_config()
print(f"Model restaure: {agent.model}")

print()
print("ALL LIVE TESTS PASSED")
