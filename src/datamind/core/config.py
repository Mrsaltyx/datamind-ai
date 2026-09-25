"""Application settings (pydantic-settings, charge .env)."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Provider actif : "ollama" (local) ou "remote" (API compatible OpenAI)
    llm_provider: str = "ollama"

    # Ollama (LLM local)
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "gemma4:e4b"

    # API distante (compatible OpenAI)
    openai_api_key: str = ""
    openai_base_url: str = "https://api.z.ai/api/coding/paas/v4/"
    openai_model: str = "glm-5.1"

    # Application
    host: str = "0.0.0.0"
    port: int = 8000
    max_session_age_seconds: int = 7200
    max_upload_size_mb: float = 200
    database_url: str = "sqlite+aiosqlite:///./data/sessions.db"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    def get_llm_config(self) -> dict:
        """Configuration du provider actif."""
        if self.llm_provider == "ollama":
            return {
                "provider": "ollama",
                "api_key": "ollama",
                "base_url": self.ollama_base_url,
                "model": self.ollama_model,
            }
        return {
            "provider": "remote",
            "api_key": self.openai_api_key,
            "base_url": self.openai_base_url,
            "model": self.openai_model,
        }


@lru_cache
def get_settings() -> Settings:
    return Settings()
