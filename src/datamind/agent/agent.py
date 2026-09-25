"""Boucle agent : le LLM choisit les outils d'analyse, on execute, on repond."""

from __future__ import annotations

import json
import logging
import re

import pandas as pd

from datamind.agent.prompts import SYSTEM_PROMPT
from datamind.analysis.tools import TOOLS_SCHEMA, execute_tool
from datamind.providers import LLMError, OpenAICompatibleProvider

logger = logging.getLogger(__name__)

MAX_CONTEXT_MESSAGES = 30
MAX_TOOL_ITERATIONS = 10


class DataMindAgent:
    """Agent analyste. Necessite un provider LLM (sinon le chat leve LLMError)."""

    def __init__(self, provider: OpenAICompatibleProvider | None = None):
        if provider is None:
            # Provider courant du process (bootstrap), si deja resolu
            from datamind.core.bootstrap import current_status

            provider = current_status.provider if current_status else None
        self.provider = provider
        self.df: pd.DataFrame | None = None
        self.messages: list[dict] = []

    def set_provider(self, provider: OpenAICompatibleProvider | None) -> None:
        self.provider = provider

    def set_data(self, df: pd.DataFrame) -> None:
        self.df = df
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "system",
                "content": (
                    f"Jeu de donnees charge : {df.shape[0]} lignes x {df.shape[1]} colonnes.\n"
                    f"Colonnes : {', '.join(df.columns.tolist())}\n"
                    f"Types : {json.dumps({col: str(df[col].dtype) for col in df.columns})}\n"
                    f"Echantillon :\n{df.head(3).to_string()}"
                ),
            },
        ]

    def _trim_context(self) -> None:
        if len(self.messages) <= MAX_CONTEXT_MESSAGES:
            return
        system_msgs = [m for m in self.messages if m.get("role") == "system"]
        other_msgs = [m for m in self.messages if m.get("role") != "system"]
        keep = MAX_CONTEXT_MESSAGES - len(system_msgs)
        self.messages = system_msgs + other_msgs[-keep:]

    def _call_llm(self) -> object:
        if self.provider is None:
            raise LLMError(
                "Aucun provider LLM configure. "
                "Lancez Ollama ou configurez une cle API dans la barre laterale."
            )
        self._trim_context()
        return self.provider.chat(self.messages, tools=TOOLS_SCHEMA)

    def chat(self, user_message: str) -> dict:
        self.messages.append({"role": "user", "content": user_message})

        figures: list = []
        for _ in range(MAX_TOOL_ITERATIONS):
            result = self._call_llm()

            assistant_msg: dict = {"role": "assistant", "content": result.content or ""}
            if result.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": tc.arguments},
                    }
                    for tc in result.tool_calls
                ]
            self.messages.append(assistant_msg)

            if not result.tool_calls:
                return {"message": self._strip_thinking(result.content or ""), "figures": figures}

            for tc in result.tool_calls:
                try:
                    arguments = json.loads(tc.arguments)
                except json.JSONDecodeError:
                    arguments = {}

                tool_result = execute_tool(tc.name, arguments, self.df)
                if tool_result.get("figure"):
                    figures.append(tool_result["figure"])

                observation = tool_result.get("text", "Aucun resultat")
                if not tool_result.get("success"):
                    observation = f"Erreur : {observation}"

                self.messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": observation}
                )

        return {
            "message": "Analyse terminee (nombre maximal d'iterations atteint).",
            "figures": figures,
        }

    def auto_eda(self) -> dict:
        prompt = (
            "Realise une analyse exploratoire rapide. "
            "Fais EXACTEMENT ces 4 appels d'outils, pas plus :\n"
            "1. describe_data\n"
            "2. show_correlation\n"
            "3. show_distribution sur la meilleure colonne numerique\n"
            "4. show_categorical sur la meilleure colonne categorielle\n\n"
            "Puis fournis un resume en 5 points :\n"
            "- Dimensions et types\n"
            "- Statistiques cles\n"
            "- Correlations notables\n"
            "- Problemes detectes\n"
            "- 3 recommandations\n"
        )
        return self.chat(prompt)

    @staticmethod
    def _strip_thinking(text: str) -> str:
        """Retire les jetons de raisonnement des modeles type Gemma."""
        text = re.sub(r"<channel>thought\n.*?<channel\|>", "", text, flags=re.DOTALL)
        text = text.replace("<|think|>", "").replace("<|/think|>", "")
        return text.strip()
