from __future__ import annotations

import asyncio
import os
import sys

from fastapi import APIRouter, HTTPException, UploadFile

from agent.agent import DataMindAgent
from backend.schemas.responses import (
    DataPreviewResponse,
    DataSummary,
    StatisticsResponse,
    UploadResponse,
)
from backend.services.session import session_manager
from utils.data_loader import get_data_summary, load_csv

# Ensure project root is in path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

router = APIRouter(prefix="/api/data", tags=["data"])


class _FileWrapper:
    """Thin wrapper so load_csv() can call .getvalue()."""

    def __init__(self, content: bytes) -> None:
        self._content = content

    def getvalue(self) -> bytes:
        return self._content


def _load_csv_sync(wrapper: _FileWrapper):
    """Synchronous CSV loading (run in thread pool)."""
    return load_csv(wrapper)


@router.post("/upload", response_model=UploadResponse)
async def upload_csv(file: UploadFile) -> UploadResponse:
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Le fichier doit etre un CSV.")

    raw_bytes = await file.read()

    try:
        df = await asyncio.to_thread(_load_csv_sync, _FileWrapper(raw_bytes))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    agent = DataMindAgent()
    agent.set_data(df)

    session = session_manager.create(df=df, agent=agent)

    summary_raw = get_data_summary(df)
    summary = DataSummary(
        shape=list(summary_raw["shape"]),
        columns=summary_raw["columns"],
        dtypes=summary_raw["dtypes"],
        numeric_cols=summary_raw["numeric_cols"],
        categorical_cols=summary_raw["categorical_cols"],
        datetime_cols=summary_raw["datetime_cols"],
        missing_pct=summary_raw["missing_pct"],
        memory_mb=summary_raw["memory_mb"],
    )

    return UploadResponse(session_id=session.session_id, summary=summary)


@router.get("/{session_id}/summary", response_model=DataSummary)
async def get_summary(session_id: str) -> DataSummary:
    session = session_manager.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session introuvable ou expiree.")

    raw = get_data_summary(session.df)
    return DataSummary(
        shape=list(raw["shape"]),
        columns=raw["columns"],
        dtypes=raw["dtypes"],
        numeric_cols=raw["numeric_cols"],
        categorical_cols=raw["categorical_cols"],
        datetime_cols=raw["datetime_cols"],
        missing_pct=raw["missing_pct"],
        memory_mb=raw["memory_mb"],
    )


@router.get("/{session_id}/preview", response_model=DataPreviewResponse)
async def get_preview(session_id: str, rows: int = 20) -> DataPreviewResponse:
    session = session_manager.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session introuvable ou expiree.")

    preview = session.df.head(rows)
    return DataPreviewResponse(
        columns=preview.columns.tolist(),
        rows=preview.to_dict(orient="records"),
    )


@router.get("/{session_id}/statistics", response_model=StatisticsResponse)
async def get_statistics(session_id: str) -> StatisticsResponse:
    session = session_manager.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session introuvable ou expiree.")

    numeric_df = session.df.describe()
    return StatisticsResponse(
        columns=numeric_df.columns.tolist(),
        stats=numeric_df.to_dict(orient="list"),
    )
