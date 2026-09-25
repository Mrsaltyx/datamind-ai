from __future__ import annotations

from typing import Any

from pydantic import BaseModel

# --- Data ---


class DataSummary(BaseModel):
    shape: list[int]
    columns: list[str]
    dtypes: dict[str, str]
    numeric_cols: list[str]
    categorical_cols: list[str]
    datetime_cols: list[str]
    missing_pct: dict[str, float]
    memory_mb: float


class UploadResponse(BaseModel):
    session_id: str
    summary: DataSummary


class DataPreviewResponse(BaseModel):
    columns: list[str]
    rows: list[dict[str, Any]]


class StatisticsResponse(BaseModel):
    columns: list[str]
    stats: dict[str, dict[str, Any]]


# --- Tools ---


class ToolExecuteRequest(BaseModel):
    tool_name: str
    arguments: dict[str, Any] = {}


class ToolExecuteResponse(BaseModel):
    success: bool
    text: str
    figure_json: str | None = None


# --- Chat ---


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    message: str
    figures: list[str]


# --- ML ---


class MlResponse(BaseModel):
    success: bool
    text: str


# --- Config ---


class ConfigUpdate(BaseModel):
    api_key: str = ""
    base_url: str = ""
    model: str = ""
    llm_provider: str = ""


class LlmStatusResponse(BaseModel):
    provider: str
    model: str
    base_url: str
    available: bool
    message: str
