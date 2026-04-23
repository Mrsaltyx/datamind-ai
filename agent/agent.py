import json
import os
import time
import pandas as pd
from openai import OpenAI, APITimeoutError, RateLimitError, APIError
from dotenv import load_dotenv
from agent.tools import TOOLS_SCHEMA, execute_tool
from prompts.system_prompt import SYSTEM_PROMPT

load_dotenv()

API_TIMEOUT = 60
MAX_CONTEXT_MESSAGES = 30
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2


class DataMindAgent:
    def __init__(self):
        self._init_client()
        self.df: pd.DataFrame | None = None
        self.messages: list[dict] = []

    def _init_client(self):
        load_dotenv()
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv(
                "OPENAI_BASE_URL", "https://api.z.ai/api/coding/paas/v4/"
            ),
            timeout=API_TIMEOUT,
            max_retries=0,
        )
        self.model = os.getenv("OPENAI_MODEL", "glm-5.1")

    def reload_config(self):
        self._init_client()

    def set_data(self, df: pd.DataFrame):
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

    def _trim_context(self):
        if len(self.messages) <= MAX_CONTEXT_MESSAGES:
            return
        system_msgs = [m for m in self.messages if m.get("role") == "system"]
        other_msgs = [m for m in self.messages if m.get("role") != "system"]
        keep = MAX_CONTEXT_MESSAGES - len(system_msgs)
        self.messages = system_msgs + other_msgs[-keep:]

    def _call_api(self, use_tools: bool = True) -> dict:
        self._trim_context()
        kwargs = {
            "model": self.model,
            "messages": self.messages,
            "temperature": 0.3,
        }
        if use_tools:
            kwargs["tools"] = TOOLS_SCHEMA
            kwargs["tool_choice"] = "auto"

        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(**kwargs)
                return response
            except RateLimitError as e:
                err_msg = str(e)
                if (
                    "1113" in err_msg
                    or "Insufficient balance" in err_msg
                    or "recharge" in err_msg.lower()
                ):
                    raise RuntimeError(
                        "Credit insuffisant sur votre compte z.ai. "
                        "Veuillez recharger votre compte sur https://z.ai "
                        "ou verifier votre cle API dans la barre laterale."
                    ) from e
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BASE_DELAY * (2**attempt))
            except APITimeoutError as e:
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BASE_DELAY * (2**attempt))
            except APIError as e:
                if hasattr(e, "status_code") and e.status_code and e.status_code >= 500:
                    last_error = e
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_BASE_DELAY * (2**attempt))
                else:
                    raise RuntimeError(
                        f"Erreur API ({getattr(e, 'status_code', '?')}) : {getattr(e, 'message', str(e))}"
                    ) from e
            except Exception as e:
                raise RuntimeError(
                    f"Erreur inattendue lors de l'appel API : {e}"
                ) from e

        if isinstance(last_error, RateLimitError):
            raise RuntimeError(
                f"Limite d'appels atteinte apres {MAX_RETRIES} tentatives : {last_error}"
            ) from last_error
        if isinstance(last_error, APITimeoutError):
            raise RuntimeError(
                "L'API ne repond pas (timeout apres retries). Verifiez votre connexion et l'URL de base."
            ) from last_error
        raise RuntimeError(
            f"Erreur API apres {MAX_RETRIES} tentatives : {last_error}"
        ) from last_error

    def chat(self, user_message: str) -> dict:
        self.messages.append({"role": "user", "content": user_message})

        figures = []
        max_iterations = 10

        for iteration in range(max_iterations):
            response = self._call_api(use_tools=True)

            assistant_msg = response.choices[0].message
            self.messages.append(assistant_msg.to_dict())

            if not assistant_msg.tool_calls:
                return {
                    "message": assistant_msg.content or "",
                    "figures": figures,
                }

            for tool_call in assistant_msg.tool_calls:
                tool_name = tool_call.function.name
                try:
                    arguments = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    arguments = {}

                result = execute_tool(tool_name, arguments, self.df)

                if result.get("figure"):
                    figures.append(result["figure"])

                observation = result.get("text", "Aucun resultat")
                if not result.get("success"):
                    observation = f"Erreur : {observation}"

                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": observation,
                    }
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
