from __future__ import annotations

from fastapi import APIRouter, HTTPException

from datamind.analysis.tools import execute_tool
from datamind.api.schemas import MlResponse
from datamind.core.sessions import session_manager

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
