"""API REST FastAPI de DataMind AI."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from datamind import __version__
from datamind.api.routers import chat, config, data, ml, tools
from datamind.core import metrics
from datamind.core.bootstrap import refresh_status
from datamind.core.config import get_settings
from datamind.core.sessions import session_manager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Frontend buildé servi par l'API (mode "une seule commande")
FRONTEND_DIST = Path(__file__).resolve().parents[3] / "frontend" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    session_manager._max_age = settings.max_session_age_seconds
    await session_manager.init()

    status = await refresh_status(settings)
    if status.degraded:
        logger.warning("Mode degrade : %s", status.message)
    else:
        logger.info("LLM actif : %s", status.message)

    yield
    session_manager._sessions.clear()


app = FastAPI(
    title="DataMind AI API",
    description="API REST pour l'agent d'analyse de donnees DataMind AI",
    version=__version__,
    lifespan=lifespan,
)

# CORS (utile en dev avec Vite ; inoffensif en production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(data.router)
app.include_router(tools.router)
app.include_router(chat.router)
app.include_router(ml.router)
app.include_router(config.router)


@app.get("/api/health")
async def health_check() -> dict:
    from datamind.core.bootstrap import current_status

    return {
        "status": "ok",
        "version": __version__,
        "llm": (current_status.active if current_status else "unknown"),
        "llm_message": (current_status.message if current_status else ""),
    }


@app.get("/api/metrics")
async def metrics_endpoint() -> dict:
    """Tokens LLM consommes (pour arbitrer les optimisations de contexte)."""
    return metrics.store.snapshot()


# Le backend sert le frontend buildé s'il est présent (setup utilisateur final).
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
    logger.info("Frontend statique servi depuis %s", FRONTEND_DIST)
else:
    logger.info(
        "Frontend non buildé (%s introuvable). "
        "API seule : utilisez 'npm run dev' dans frontend/ ou lancez scripts/setup.",
        FRONTEND_DIST,
    )


def run() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "datamind.api.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )
