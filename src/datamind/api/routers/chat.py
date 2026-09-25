from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException

from datamind.api.schemas import ChatRequest, ChatResponse
from datamind.core.sessions import session_manager

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("/{session_id}/send", response_model=ChatResponse)
async def send_chat(session_id: str, body: ChatRequest) -> ChatResponse:
    session = session_manager.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session introuvable ou expiree.")

    try:
        result = await asyncio.to_thread(session.agent.chat, body.message)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    figures: list[str] = []
    for fig in result.get("figures", []):
        if fig is not None:
            figures.append(fig.to_json())

    return ChatResponse(
        message=result.get("message", ""),
        figures=figures,
    )


@router.post("/{session_id}/auto-eda", response_model=ChatResponse)
async def auto_eda(session_id: str) -> ChatResponse:
    session = session_manager.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session introuvable ou expiree.")

    try:
        result = await asyncio.to_thread(session.agent.auto_eda)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    figures: list[str] = []
    for fig in result.get("figures", []):
        if fig is not None:
            figures.append(fig.to_json())

    return ChatResponse(
        message=result.get("message", ""),
        figures=figures,
    )


@router.delete("/{session_id}/history")
async def clear_history(session_id: str) -> dict:
    session = session_manager.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session introuvable ou expiree.")

    session.agent.messages = [m for m in session.agent.messages if m.get("role") == "system"]
    return {"status": "ok"}
