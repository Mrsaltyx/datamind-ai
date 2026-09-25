"""Bootstrap : detection automatique des providers LLM disponibles.

Mode degrade : sans LLM configure, l'application reste utilisable pour
l'EDA et l'entrainement ML ; seul le chat necessite un provider.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import httpx

from datamind.core.config import Settings
from datamind.providers import OllamaProvider, OpenAICompatibleProvider, RemoteProvider

logger = logging.getLogger(__name__)

PROBE_TIMEOUT = 2.0


@dataclass
class ProviderStatus:
    """Etat de la couche LLM au demarrage (ou apres reconfiguration)."""

    active: str  # "ollama" | "remote" | "none"
    provider: OpenAICompatibleProvider | None
    ollama_reachable: bool = False
    ollama_models: list[str] = field(default_factory=list)
    remote_configured: bool = False
    degraded: bool = False
    message: str = ""


async def probe_ollama(base_url: str, timeout: float = PROBE_TIMEOUT) -> list[str]:
    """Retourne la liste des modeles Ollama, ou [] si injoignable."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(f"{base_url.removesuffix('/v1')}/api/tags")
            if resp.status_code == 200:
                return [m.get("name", "") for m in resp.json().get("models", [])]
    except Exception:
        pass
    return []


async def resolve_provider(settings: Settings) -> ProviderStatus:
    """Construit le provider actif selon la config et ce qui existe reellement."""
    config = settings.get_llm_config()

    if config["provider"] == "ollama":
        models = await probe_ollama(settings.ollama_base_url)
        if not models:
            return ProviderStatus(
                active="none",
                provider=None,
                ollama_reachable=False,
                message=(
                    "Ollama introuvable. L'analyse de donnees fonctionne, "
                    "mais le chat necessite Ollama (https://ollama.com) ou une cle API."
                ),
                degraded=True,
            )
        model_loaded = any(config["model"].split(":")[0] in m for m in models)
        message = (
            f"Modele {config['model']} disponible via Ollama"
            if model_loaded
            else (
                f"Ollama fonctionne mais le modele '{config['model']}' n'est pas telecharge. "
                f"Executez : ollama pull {config['model']}"
            )
        )
        return ProviderStatus(
            active="ollama",
            provider=OllamaProvider(base_url=settings.ollama_base_url, model=settings.ollama_model),
            ollama_reachable=True,
            ollama_models=models,
            message=message,
            degraded=not model_loaded,
        )

    # remote
    if not config["api_key"]:
        return ProviderStatus(
            active="none",
            provider=None,
            remote_configured=False,
            message=(
                "Aucune cle API configuree. L'analyse de donnees fonctionne, "
                "mais le chat necessite une cle API (barre laterale) ou Ollama."
            ),
            degraded=True,
        )
    return ProviderStatus(
        active="remote",
        provider=RemoteProvider(
            base_url=config["base_url"], api_key=config["api_key"], model=config["model"]
        ),
        remote_configured=True,
        message=f"Mode distant configure : {config['model']}",
    )


# Statut global, re-resolu a la config update. Les sessions y accedent pour
# injecter le provider courant dans leurs agents.
current_status: ProviderStatus | None = None


async def refresh_status(settings: Settings | None = None) -> ProviderStatus:
    """Re-resout le provider et memorise le statut global."""
    global current_status
    settings = settings or Settings()
    current_status = await resolve_provider(settings)
    logger.info("LLM: %s | %s", current_status.active, current_status.message)
    return current_status
