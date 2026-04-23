import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

import pandas as pd
from agent.agent import DataMindAgent

df = pd.read_csv("train.csv", nrows=100)
print(f"Dataset: {df.shape}")

agent = DataMindAgent()
agent.set_data(df)
print(f"Agent initialised, messages: {len(agent.messages)}")
print(f"Client base_url: {agent.client.base_url}")
print(f"Model: {agent.model}")

print()
print("Appel chat (description)...")
result = agent.chat("Decris le dataset en 3 phrases")
print(f"Success: {len(result['message'])} chars")
print(f"Message: {result['message'][:500]}")
print(f"Figures: {len(result['figures'])}")
print(f"Total messages: {len(agent.messages)}")
