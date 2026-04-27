from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM Provider: "embedded" (Unsloth/in-process), "ollama" (Ollama server), or "remote" (OpenAI-compatible API)
    llm_provider: str = "ollama"

    # Embedded LLM (Unsloth) settings
    embedded_model_path: str = "models/gemma-4-4b-it-Q4_K_M.gguf"
    embedded_max_seq_length: int = 4096
    embedded_gpu_layers: int = 0  # 0 = CPU only, -1 = all layers on GPU

    # Ollama LLM settings
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "gemma4:e4b"

    # Remote LLM (OpenAI-compatible) settings
    openai_api_key: str = ""
    openai_base_url: str = "https://api.z.ai/api/coding/paas/v4/"
    openai_model: str = "glm-5.1"

    # App settings
    host: str = "0.0.0.0"
    port: int = 8000
    max_session_age_seconds: int = 7200
    max_upload_size_mb: float = 200
    database_url: str = "sqlite+aiosqlite:///./data/sessions.db"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    def get_active_llm_config(self) -> dict:
        """Return the active LLM configuration based on provider setting."""
        if self.llm_provider == "embedded":
            return {
                "api_key": "not-needed",
                "base_url": "embedded",
                "model": self.embedded_model_path,
                "provider": "embedded",
            }
        elif self.llm_provider == "ollama":
            return {
                "api_key": "ollama",
                "base_url": self.ollama_base_url,
                "model": self.ollama_model,
                "provider": "ollama",
            }
        return {
            "api_key": self.openai_api_key,
            "base_url": self.openai_base_url,
            "model": self.openai_model,
            "provider": "remote",
        }


@lru_cache
def get_settings() -> Settings:
    return Settings()
