"""Endpoints ML : suggestions (advisor) et entrainement reel de baselines."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException

from datamind.analysis.tools import execute_tool
from datamind.api.schemas import MlResponse, TrainResponse
from datamind.core.sessions import session_manager
from datamind.ml.tracker import log_training
from datamind.ml.trainer import format_training_result, train_baseline

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


@router.post("/{session_id}/train", response_model=TrainResponse)
async def train(session_id: str, target_column: str | None = None) -> TrainResponse:
    """Entraine une baseline scikit-learn sur les donnees de la session.

    Ne necessite AUCUN LLM : fonctionne en mode degrade.
    """
    session = session_manager.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session introuvable ou expiree.")

    result = await asyncio.to_thread(train_baseline, session.df, target_column)

    text = format_training_result(result)
    tracking = log_training(result, pipeline=result.get("pipeline"), report_text=text)

    metrics = [
        {"name": m["name"], "mean": m["mean"], "std": m["std"]} for m in result.get("metrics", [])
    ]
    return TrainResponse(
        success=result.get("success", False),
        task_type=result.get("task_type", ""),
        model_name=result.get("model_name", ""),
        target_column=result.get("target_column", ""),
        n_samples=result.get("n_samples", 0),
        n_features=result.get("n_features", 0),
        n_folds=result.get("n_folds", 0),
        metrics=metrics,
        warnings=result.get("warnings", []),
        text=text,
        tracked=tracking.get("tracked", False),
        mlflow_run_id=tracking.get("run_id", ""),
    )
