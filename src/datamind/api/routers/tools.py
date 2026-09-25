from __future__ import annotations

from fastapi import APIRouter, HTTPException

from datamind.analysis.tools import execute_tool
from datamind.api.schemas import ToolExecuteRequest, ToolExecuteResponse
from datamind.core.sessions import session_manager

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
