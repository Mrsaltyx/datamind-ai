from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure project root is importable
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.config import get_settings  # noqa: E402
from backend.routers import chat, config, data, ml, tools  # noqa: E402
from backend.services.session import session_manager  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    session_manager._max_age = settings.max_session_age_seconds

    # Initialize database
    await session_manager.init()

    # Log LLM provider info
    llm_config = settings.get_active_llm_config()
    logger.info(
        "LLM Provider: %s | Model: %s | Base URL: %s",
        llm_config["provider"],
        llm_config["model"],
        llm_config["base_url"],
    )
    if llm_config["provider"] == "embedded":
        logger.info("Mode embarque active. Le modele GGUF sera charge au besoin.")
    elif llm_config["provider"] == "ollama":
        logger.info(
            "Mode Ollama active. Assurez-vous qu'Ollama est lance et le modele est telecharge."
        )

    yield
    # Cleanup on shutdown
    session_manager._sessions.clear()


app = FastAPI(
    title="DataMind AI API",
    description="API REST pour l'agent d'analyse de donnees DataMind AI",
    version="2.1.0",
    lifespan=lifespan,
)

# CORS
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

# Routers
app.include_router(data.router)
app.include_router(tools.router)
app.include_router(chat.router)
app.include_router(ml.router)
app.include_router(config.router)


@app.get("/api/health")
async def health_check() -> dict:
    return {"status": "ok", "version": "2.1.0"}


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "backend.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )
