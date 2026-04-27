from __future__ import annotations

import os
import sys

from fastapi import APIRouter, HTTPException

from agent.tools import execute_tool
from backend.schemas.responses import ToolExecuteRequest, ToolExecuteResponse
from backend.services.session import session_manager

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

router = APIRouter(prefix="/api/tools", tags=["tools"])


@router.post("/{session_id}/execute", response_model=ToolExecuteResponse)
async def execute_tool_endpoint(session_id: str, body: ToolExecuteRequest) -> ToolExecuteResponse:
    session = session_manager.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session introuvable ou expiree.")

    result = execute_tool(body.tool_name, body.arguments, session.df)

    figure_json = None
    if result.get("figure") is not None:
        figure_json = result["figure"].to_json()

    return ToolExecuteResponse(
        success=result.get("success", False),
        text=result.get("text", ""),
        figure_json=figure_json,
    )
