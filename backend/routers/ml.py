from __future__ import annotations

import os
import sys

from fastapi import APIRouter, HTTPException

from agent.tools import execute_tool
from backend.schemas.responses import MlResponse
from backend.services.session import session_manager

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

router = APIRouter(prefix="/api/ml", tags=["ml"])


@router.post("/{session_id}/suggest", response_model=MlResponse)
async def suggest_ml(session_id: str) -> MlResponse:
    session = session_manager.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session introuvable ou expiree.")

    result = execute_tool("suggest_ml_pipeline", {}, session.df)
    return MlResponse(
        success=result.get("success", False),
        text=result.get("text", ""),
    )


@router.post("/{session_id}/detect-target", response_model=MlResponse)
async def detect_target(session_id: str) -> MlResponse:
    session = session_manager.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session introuvable ou expiree.")

    result = execute_tool("detect_target_and_task", {}, session.df)
    return MlResponse(
        success=result.get("success", False),
        text=result.get("text", ""),
    )
