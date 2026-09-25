"""Abstraction LLM : providers locaux (Ollama) et distants (API compatible OpenAI)."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from openai import APIConnectionError, APIError, APITimeoutError, OpenAI, RateLimitError

logger = logging.getLogger(__name__)

API_TIMEOUT = 120  # les modeles locaux peuvent etre lents
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2


class LLMError(RuntimeError):
    """Erreur LLM exploitable cote API (message destine a l'utilisateur)."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # JSON string, format API OpenAI


@dataclass
class ChatResult:
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)


class OpenAICompatibleProvider:
    """Client LLM via une API compatible OpenAI (Ollama ou service distant)."""

    name: str = "generic"

    def __init__(self, *, base_url: str, api_key: str, model: str) -> None:
        self.client = OpenAI(
            api_key=api_key or "none",
            base_url=base_url,
            timeout=API_TIMEOUT,
            max_retries=0,
        )
        self.model = model
        self.base_url = base_url

    @staticmethod
    def _backoff(attempt: int) -> None:
        time.sleep(RETRY_BASE_DELAY * (2**attempt))

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> ChatResult:
        kwargs: dict = {"model": self.model, "messages": messages, "temperature": 0.3}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(**kwargs)
                return self._to_result(response)
            except RateLimitError as e:
                err_msg = str(e)
                if (
                    "1113" in err_msg
                    or "Insufficient balance" in err_msg
                    or "recharge" in err_msg.lower()
                ):
                    raise LLMError(
                        "Credit insuffisant sur votre compte distant. "
                        "Rechargez votre compte ou verifiez votre cle API."
                    ) from e
                last_error = e
                self._log_retry("limite d'appels", attempt)
            except APITimeoutError as e:
                last_error = e
                self._log_retry("timeout", attempt)
            except APIConnectionError as e:
                raise LLMError(
                    "Impossible de se connecter au serveur LLM. "
                    "Verifiez qu'Ollama est lance (ollama serve) ou que l'URL est correcte."
                ) from e
            except APIError as e:
                status = getattr(e, "status_code", None)
                if status and status >= 500:
                    last_error = e
                    self._log_retry(f"erreur serveur {status}", attempt)
                elif status == 404:
                    raise LLMError(
                        f"Modele '{self.model}' non trouve. "
                        f"Si vous utilisez Ollama : ollama pull {self.model}"
                    ) from e
                else:
                    raise LLMError(
                        f"Erreur API ({status or '?'}) : {getattr(e, 'message', str(e))}"
                    ) from e
            except Exception as e:
                raise LLMError(f"Erreur inattendue lors de l'appel LLM : {e}") from e

        raise LLMError(f"Erreur LLM apres {MAX_RETRIES} tentatives : {last_error}")

    def _log_retry(self, reason: str, attempt: int) -> None:
        if attempt < MAX_RETRIES - 1:
            logger.warning("LLM %s (tentative %d/%d)", reason, attempt + 1, MAX_RETRIES)
            self._backoff(attempt)

    @staticmethod
    def _to_result(response) -> ChatResult:
        message = response.choices[0].message
        tool_calls = [
            ToolCall(id=tc.id, name=tc.function.name, arguments=tc.function.arguments or "{}")
            for tc in (message.tool_calls or [])
        ]
        return ChatResult(content=message.content, tool_calls=tool_calls)


class OllamaProvider(OpenAICompatibleProvider):
    """Provider local via un serveur Ollama (API compatible OpenAI)."""

    name = "ollama"

    def __init__(self, *, base_url: str, model: str) -> None:
        super().__init__(base_url=base_url, api_key="ollama", model=model)


class RemoteProvider(OpenAICompatibleProvider):
    """Provider distant via une API compatible OpenAI (z.ai, OpenAI, ...)."""

    name = "remote"

    def __init__(self, *, base_url: str, api_key: str, model: str) -> None:
        super().__init__(base_url=base_url, api_key=api_key, model=model)
