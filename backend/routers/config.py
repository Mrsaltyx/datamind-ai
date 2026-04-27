from __future__ import annotations

import logging
import os
import sys

from fastapi import APIRouter
from pydantic import BaseModel

from backend.services.session import session_manager

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/config", tags=["config"])


class ConfigPayload(BaseModel):
    api_key: str = ""
    base_url: str = ""
    model: str = ""
    llm_provider: str = ""  # "embedded", "ollama", or "remote"


class LlmStatusResponse(BaseModel):
    provider: str
    model: str
    base_url: str
    available: bool
    message: str


@router.post("/update")
async def update_config(payload: ConfigPayload) -> dict:
    """Update API config and reload agent for all active sessions."""
    if payload.llm_provider:
        os.environ["LLM_PROVIDER"] = payload.llm_provider
        from backend.config import get_settings

        get_settings.cache_clear()

        settings = get_settings()

        if payload.llm_provider == "embedded":
            os.environ["OPENAI_API_KEY"] = "not-needed"
            os.environ["OPENAI_BASE_URL"] = "embedded"
            os.environ["OPENAI_MODEL"] = settings.embedded_model_path
        elif payload.llm_provider == "ollama":
            os.environ["OPENAI_API_KEY"] = "ollama"
            os.environ["OPENAI_BASE_URL"] = settings.ollama_base_url
            os.environ["OPENAI_MODEL"] = settings.ollama_model
        elif payload.llm_provider == "remote":
            if payload.api_key:
                os.environ["OPENAI_API_KEY"] = payload.api_key
            if payload.base_url:
                os.environ["OPENAI_BASE_URL"] = payload.base_url
            if payload.model:
                os.environ["OPENAI_MODEL"] = payload.model

    if payload.api_key and payload.llm_provider not in ("embedded", "ollama"):
        os.environ["OPENAI_API_KEY"] = payload.api_key
    if payload.base_url and payload.llm_provider not in ("embedded", "ollama"):
        os.environ["OPENAI_BASE_URL"] = payload.base_url
    if payload.model and payload.llm_provider not in ("embedded", "ollama"):
        os.environ["OPENAI_MODEL"] = payload.model

    # Reload config on all active sessions
    for session in list(session_manager._sessions.values()):
        session.agent.reload_config()

    return {"status": "ok"}


@router.get("/llm-status", response_model=LlmStatusResponse)
async def get_llm_status() -> LlmStatusResponse:
    """Check LLM provider status and availability."""
    from backend.config import get_settings

    settings = get_settings()
    config = settings.get_active_llm_config()

    available = False
    message = ""

    if config["provider"] == "embedded":
        # Check if GGUF file exists
        model_path = settings.embedded_model_path
        if not os.path.isabs(model_path):
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(project_root, model_path)

        if os.path.exists(model_path):
            try:
                from llama_cpp import Llama  # noqa: F401

                available = True
                message = f"Modele embarque pret ({os.path.basename(model_path)})"
            except ImportError:
                message = (
                    "llama-cpp-python n'est pas installe. "
                    "Installez-le avec: pip install llama-cpp-python"
                )
        else:
            message = (
                f"Modele GGUF non trouve a {model_path}. "
                "Telechargez un modele Gemma 4 quantifie dans le dossier models/. "
                "Voir README pour les instructions."
            )

    elif config["provider"] == "ollama":
        try:
            import httpx

            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{settings.ollama_base_url.replace('/v1', '')}/api/tags")
                if resp.status_code == 200:
                    models = resp.json().get("models", [])
                    model_names = [m.get("name", "") for m in models]
                    model_loaded = any(config["model"].split(":")[0] in n for n in model_names)
                    if model_loaded:
                        available = True
                        message = f"Modele {config['model']} disponible via Ollama"
                    else:
                        message = (
                            f"Ollama fonctionne mais le modele '{config['model']}' n'est pas telecharge. "
                            f"Executez : ollama pull {config['model']}"
                        )
                else:
                    message = "Ollama ne repond pas. Verifiez qu'il est lance (ollama serve)"
        except Exception:
            message = (
                "Impossible de se connecter a Ollama. "
                "Installez-le sur https://ollama.com puis lancez 'ollama serve' "
                "et 'ollama pull gemma4:e4b'"
            )
    else:
        if config["api_key"]:
            available = True
            message = f"Mode distant configure : {config['model']}"
        else:
            message = "Mode distant selectionne mais aucune cle API configuree."

    return LlmStatusResponse(
        provider=config["provider"],
        model=config["model"],
        base_url=config["base_url"],
        available=available,
        message=message,
    )
