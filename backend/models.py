"""SQLAlchemy models for session persistence."""

from __future__ import annotations

import time

import pandas as pd
from sqlalchemy import Column, Float, Integer, LargeBinary, String, Text
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class SessionModel(Base):
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(36), unique=True, nullable=False, index=True)
    created_at = Column(Float, nullable=False, default=lambda: time.time())
    last_accessed = Column(Float, nullable=False, default=lambda: time.time())
    # Serialized DataFrame (parquet bytes)
    df_blob = Column(LargeBinary, nullable=False)
    # OpenAI config snapshot at creation time
    config_api_key = Column(String(512), default="")
    config_base_url = Column(String(512), default="")
    config_model = Column(String(128), default="")


class MessageModel(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(36), nullable=False, index=True)
    role = Column(String(32), nullable=False)
    content = Column(Text, nullable=False)


def dataframe_to_bytes(df: pd.DataFrame) -> bytes:
    """Serialize a DataFrame to parquet bytes."""
    return df.to_parquet(index=False)


def bytes_to_dataframe(data: bytes) -> pd.DataFrame:
    """Deserialize parquet bytes back to a DataFrame."""
    return pd.read_parquet(data)
