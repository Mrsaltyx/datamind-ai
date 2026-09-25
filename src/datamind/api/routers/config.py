"""Configuration LLM : statut, bascule provider, mode degrade."""

from __future__ import annotations

import logging
import os

from fastapi import APIRouter
from pydantic import BaseModel

from datamind.core.bootstrap import refresh_status
from datamind.core.config import get_settings
from datamind.core.sessions import session_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/config", tags=["config"])


class ConfigPayload(BaseModel):
    api_key: str = ""
    base_url: str = ""
    model: str = ""
    llm_provider: str = ""  # "ollama" ou "remote"


class LlmStatusResponse(BaseModel):
    provider: str
    model: str
    base_url: str
    available: bool
    degraded: bool
    message: str


@router.post("/update")
async def update_config(payload: ConfigPayload) -> dict:
    """Met a jour la config LLM, re-resolve le provider, recharge les sessions."""
    if payload.llm_provider in ("ollama", "remote"):
        os.environ["LLM_PROVIDER"] = payload.llm_provider

    if payload.llm_provider == "ollama":
        settings = get_settings()
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

    get_settings.cache_clear()
    status = await refresh_status(get_settings())

    # Reinjecte le nouveau provider dans les agents des sessions actives
    for session in list(session_manager._sessions.values()):
        session.agent.set_provider(status.provider)

    return {"status": "ok", "message": status.message, "degraded": status.degraded}


@router.get("/llm-status", response_model=LlmStatusResponse)
async def get_llm_status() -> LlmStatusResponse:
    """Etat du provider LLM (re-probe a chaque appel : le statut est volatil)."""
    status = await refresh_status(get_settings())
    config = get_settings().get_llm_config()
    return LlmStatusResponse(
        provider=config["provider"],
        model=config["model"],
        base_url=config["base_url"],
        available=status.provider is not None and not status.degraded,
        degraded=status.degraded,
        message=status.message,
    )
