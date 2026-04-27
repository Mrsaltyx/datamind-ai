import json
import logging
import os
import re
import time
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from openai import APIError, APITimeoutError, OpenAI, RateLimitError

from agent.tools import TOOLS_SCHEMA, execute_tool
from prompts.system_prompt import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

API_TIMEOUT = 120  # Local models can be slower
MAX_CONTEXT_MESSAGES = 30
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2


class DataMindAgent:
    """LLM Agent that supports 3 providers: embedded (llama-cpp), ollama, and remote (OpenAI API)."""

    def __init__(self):
        self._init_client()
        self.df: pd.DataFrame | None = None
        self.messages: list[dict] = []
        self.provider: str = "ollama"

    def _init_client(self):
        load_dotenv()

        # Check for explicit env vars first (set by config endpoint or session restore)
        api_key = os.getenv("OPENAI_API_KEY", "")
        base_url = os.getenv("OPENAI_BASE_URL", "")
        model = os.getenv("OPENAI_MODEL", "")
        provider_override = os.getenv("LLM_PROVIDER", "")

        if provider_override == "embedded" or base_url == "embedded":
            self.provider = "embedded"
            self._init_embedded()
            return

        if not base_url and not api_key:
            # Fall back to Settings defaults
            from backend.config import get_settings

            settings = get_settings()
            config = settings.get_active_llm_config()
            api_key = config["api_key"]
            base_url = config["base_url"]
            model = config["model"]
            self.provider = config["provider"]

            if self.provider == "embedded":
                self._init_embedded()
                return
        else:
            self.provider = "remote" if api_key and api_key != "ollama" else "ollama"

        # For ollama and remote: use OpenAI SDK (Ollama is API-compatible)
        self.client = OpenAI(
            api_key=api_key or "ollama",
            base_url=base_url or "http://localhost:11434/v1",
            timeout=API_TIMEOUT,
            max_retries=0,
        )
        self.model = model or "gemma4:e4b"
        self._llm = None  # No local LLM instance

    def _init_embedded(self):
        """Initialize the embedded LLM using llama-cpp-python."""
        from backend.config import get_settings

        settings = get_settings()
        model_path = settings.embedded_model_path

        # Resolve relative path from project root
        if not os.path.isabs(model_path):
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(project_root, model_path)

        if not os.path.exists(model_path):
            logger.warning(
                "Modele GGUF non trouve a %s. Mode embedded indisponible, fallback vers Ollama.",
                model_path,
            )
            self.provider = "ollama"
            self.client = OpenAI(
                api_key="ollama",
                base_url="http://localhost:11434/v1",
                timeout=API_TIMEOUT,
                max_retries=0,
            )
            self.model = settings.ollama_model
            self._llm = None
            return

        try:
            from llama_cpp import Llama

            gpu_layers = settings.embedded_gpu_layers
            logger.info("Chargement du modele embarque: %s (gpu_layers=%d)", model_path, gpu_layers)
            self._llm = Llama(
                model_path=model_path,
                n_ctx=settings.embedded_max_seq_length,
                n_gpu_layers=gpu_layers,
                verbose=False,
            )
            self.provider = "embedded"
            self.model = os.path.basename(model_path)
            self.client = None  # Not using OpenAI SDK
            logger.info("Modele embarque charge avec succes")
        except ImportError:
            logger.error(
                "llama-cpp-python n'est pas installe. "
                "Installez-le avec: pip install llama-cpp-python"
            )
            self.provider = "ollama"
            self.client = OpenAI(
                api_key="ollama",
                base_url="http://localhost:11434/v1",
                timeout=API_TIMEOUT,
                max_retries=0,
            )
            self.model = "gemma4:e4b"
            self._llm = None
        except Exception as e:
            logger.error("Erreur lors du chargement du modele embarque: %s", e)
            self.provider = "ollama"
            self.client = OpenAI(
                api_key="ollama",
                base_url="http://localhost:11434/v1",
                timeout=API_TIMEOUT,
                max_retries=0,
            )
            self.model = "gemma4:e4b"
            self._llm = None

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

    def _call_embedded(self, use_tools: bool = True) -> dict:
        """Call the embedded LLM using llama-cpp-python with tool support."""
        from llama_cpp import ChatCompletionMessage

        # Build messages for llama-cpp
        llama_messages = []
        for msg in self.messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "tool":
                role = "user"  # llama-cpp doesn't have tool role, use user
                content = f"[Resultat outil] {content}"
            llama_messages.append(ChatCompletionMessage(role=role, content=content))

        # Add tools as system context if needed
        if use_tools:
            tools_desc = self._tools_to_text()
            tool_system = {
                "role": "system",
                "content": (
                    "Tu as acces aux outils suivants. Pour les utiliser, "
                    "reponds EXACTEMENT sous ce format JSON:\n"
                    '{"tool_call": true, "name": "nom_outil", "arguments": {"param": "valeur"}}\n\n'
                    f"Outils disponibles:\n{tools_desc}\n\n"
                    "Si tu n'as pas besoin d'outil, reponds normalement en texte."
                ),
            }
            llama_messages.insert(0, ChatCompletionMessage(**tool_system))

        try:
            response = self._llm.create_chat_completion(
                messages=llama_messages,
                temperature=0.3,
                max_tokens=2048,
            )
        except Exception as e:
            raise RuntimeError(f"Erreur du modele embarque: {e}") from e

        content = response.choices[0].message.content or ""

        # Check if the model wants to call a tool
        tool_call = self._parse_tool_call(content)
        if tool_call:
            return {
                "tool_calls": [tool_call],
                "content": None,
            }

        return {
            "tool_calls": None,
            "content": self._strip_thinking(content),
        }

    def _tools_to_text(self) -> str:
        """Convert tools schema to text description for embedded model."""
        tools_text = []
        for tool in TOOLS_SCHEMA:
            func = tool["function"]
            params = func.get("parameters", {}).get("properties", {})
            required = func.get("parameters", {}).get("required", [])
            desc = f"- {func['name']}: {func['description']}\n  Parametres:"
            for pname, pinfo in params.items():
                req = "(requis)" if pname in required else "(optionnel)"
                desc += f"\n    - {pname}: {pinfo.get('description', '')} {req}"
            tools_text.append(desc)
        return "\n".join(tools_text)

    @staticmethod
    def _parse_tool_call(content: str) -> dict | None:
        """Try to parse a tool call from the model response."""
        import re

        # Try JSON parse first
        try:
            data = json.loads(content.strip())
            if isinstance(data, dict) and data.get("tool_call"):
                return {
                    "id": f"call_{hash(content) % 10000}",
                    "name": data["name"],
                    "arguments": json.dumps(data.get("arguments", {})),
                }
        except (json.JSONDecodeError, TypeError):
            pass

        # Try to find JSON in the response
        json_match = re.search(r'\{[^{}]*"tool_call"\s*:\s*true[^{}]*\}', content, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return {
                    "id": f"call_{hash(content) % 10000}",
                    "name": data["name"],
                    "arguments": json.dumps(data.get("arguments", {})),
                }
            except (json.JSONDecodeError, TypeError):
                pass

        return None

    def _call_api(self, use_tools: bool = True) -> dict:
        """Call LLM via OpenAI-compatible API (Ollama or remote)."""
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
                logger.warning("LLM timeout (attempt %d/%d)", attempt + 1, MAX_RETRIES)
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BASE_DELAY * (2**attempt))
            except APIError as e:
                if hasattr(e, "status_code") and e.status_code and e.status_code >= 500:
                    last_error = e
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_BASE_DELAY * (2**attempt))
                elif hasattr(e, "status_code") and e.status_code == 404:
                    raise RuntimeError(
                        f"Modele '{self.model}' non trouve. "
                        f"Verifiez qu'Ollama est lance et le modele est telecharge : ollama pull {self.model}"
                    ) from e
                else:
                    raise RuntimeError(
                        f"Erreur API ({getattr(e, 'status_code', '?')}) : {getattr(e, 'message', str(e))}"
                    ) from e
            except ConnectionError as e:
                raise RuntimeError(
                    "Impossible de se connecter au serveur LLM. "
                    "Verifiez qu'Ollama est lance (ollama serve) ou que l'URL est correcte."
                ) from e
            except Exception as e:
                raise RuntimeError(f"Erreur inattendue lors de l'appel LLM : {e}") from e

        if isinstance(last_error, RateLimitError):
            raise RuntimeError(
                f"Limite d'appels atteinte apres {MAX_RETRIES} tentatives : {last_error}"
            ) from last_error
        if isinstance(last_error, APITimeoutError):
            raise RuntimeError(
                "Le modele LLM ne repond pas (timeout apres retries). "
                "Si vous utilisez Ollama, verifiez qu'il est lance et le modele est charge."
            ) from last_error
        raise RuntimeError(
            f"Erreur LLM apres {MAX_RETRIES} tentatives : {last_error}"
        ) from last_error

    def _call_llm(self, use_tools: bool = True) -> dict:
        """Unified LLM call that dispatches to the right provider."""
        if self.provider == "embedded" and self._llm is not None:
            return self._call_embedded(use_tools=use_tools)
        return self._call_api(use_tools=use_tools)

    def chat(self, user_message: str) -> dict:
        self.messages.append({"role": "user", "content": user_message})

        figures: list[Any] = []
        max_iterations = 10

        for iteration in range(max_iterations):
            response = self._call_llm(use_tools=True)

            if self.provider == "embedded" and self._llm is not None:
                # Handle embedded model response
                tool_calls = response.get("tool_calls")
                content = response.get("content", "")

                if not tool_calls:
                    self.messages.append({"role": "assistant", "content": content})
                    return {
                        "message": content,
                        "figures": figures,
                    }

                # Process tool call
                for tc in tool_calls:
                    tool_name = tc["name"]
                    try:
                        arguments = json.loads(tc.get("arguments", "{}"))
                    except json.JSONDecodeError:
                        arguments = {}

                    result = execute_tool(tool_name, arguments, self.df)
                    if result.get("figure"):
                        figures.append(result["figure"])

                    observation = result.get("text", "Aucun resultat")
                    if not result.get("success"):
                        observation = f"Erreur : {observation}"

                    # For embedded: send tool result back as user message
                    self.messages.append(
                        {
                            "role": "user",
                            "content": f"[Resultat de {tool_name}]\n{observation}",
                        }
                    )
            else:
                # Handle OpenAI SDK response (Ollama / Remote)
                assistant_msg = response.choices[0].message
                self.messages.append(assistant_msg.to_dict())

                if not assistant_msg.tool_calls:
                    content = assistant_msg.content or ""
                    content = self._strip_thinking(content)
                    return {
                        "message": content,
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

    @staticmethod
    def _strip_thinking(text: str) -> str:
        """Remove Gemma 4 thinking tokens from output."""
        text = re.sub(r"<channel>thought\n.*?<channel\|>", "", text, flags=re.DOTALL)
        text = text.replace("<|think|>", "").replace("<|/think|>", "")
        return text.strip()

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
