"""Session management with SQLite persistence via SQLAlchemy."""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from dataclasses import dataclass, field

import pandas as pd

from agent.agent import DataMindAgent

logger = logging.getLogger(__name__)


@dataclass
class Session:
    """In-memory session wrapper. Backed by DB for persistence."""

    session_id: str
    df: pd.DataFrame
    agent: DataMindAgent
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)

    def touch(self) -> None:
        self.last_accessed = time.time()

    def is_expired(self, max_age: int) -> bool:
        return (time.time() - self.last_accessed) > max_age


class SessionManager:
    """Manages sessions with in-memory cache + SQLite persistence."""

    def __init__(self, max_age: int = 7200) -> None:
        self._sessions: dict[str, Session] = {}
        self._max_age = max_age

    # ---- sync public API (used by routers) ----

    def create(self, df: pd.DataFrame, agent: DataMindAgent) -> Session:
        session_id = str(uuid.uuid4())
        session = Session(session_id=session_id, df=df, agent=agent)
        self._sessions[session_id] = session

        # Persist to DB in background
        asyncio.get_event_loop().create_task(self._persist_session(session))
        return session

    def get(self, session_id: str) -> Session | None:
        session = self._sessions.get(session_id)
        if session is not None:
            if session.is_expired(self._max_age):
                self.delete(session_id)
                return None
            session.touch()
            return session

        # Try to restore from DB
        restored = self._restore_from_db(session_id)
        if restored is not None:
            if restored.is_expired(self._max_age):
                self.delete(session_id)
                return None
            restored.touch()
            self._sessions[session_id] = restored
        return restored

    def delete(self, session_id: str) -> bool:
        removed = self._sessions.pop(session_id, None) is not None
        if removed:
            asyncio.get_event_loop().create_task(self._delete_session_db(session_id))
        return removed

    def cleanup_expired(self) -> int:
        expired = [sid for sid, s in self._sessions.items() if s.is_expired(self._max_age)]
        for sid in expired:
            self.delete(sid)
        return len(expired)

    # ---- async DB operations ----

    async def _persist_session(self, session: Session) -> None:
        """Save session metadata to SQLite."""
        try:
            from backend.database import async_session_factory
            from backend.models import SessionModel, dataframe_to_bytes

            async with async_session_factory() as db:
                db_row = SessionModel(
                    session_id=session.session_id,
                    created_at=session.created_at,
                    last_accessed=session.last_accessed,
                    df_blob=dataframe_to_bytes(session.df),
                    config_api_key=os.getenv("OPENAI_API_KEY", ""),
                    config_base_url=os.getenv("OPENAI_BASE_URL", ""),
                    config_model=os.getenv("OPENAI_MODEL", ""),
                )
                db.add(db_row)
                await db.commit()
        except Exception:
            logger.exception("Failed to persist session %s", session.session_id)

    def _restore_from_db(self, session_id: str) -> Session | None:
        """Synchronously restore a session from SQLite. Returns None on failure."""
        try:
            import sqlite3

            from backend.models import bytes_to_dataframe

            db_path = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./data/sessions.db").split(
                "///"
            )[-1]

            conn = sqlite3.connect(db_path)
            try:
                row = conn.execute(
                    "SELECT session_id, created_at, last_accessed, df_blob, "
                    "config_api_key, config_base_url, config_model "
                    "FROM sessions WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
            finally:
                conn.close()

            if row is None:
                return None

            sid, created_at, last_accessed, df_blob, api_key, base_url, model = row

            # Restore env vars if they were set during this session
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
            if base_url:
                os.environ["OPENAI_BASE_URL"] = base_url
            if model:
                os.environ["OPENAI_MODEL"] = model

            df = bytes_to_dataframe(df_blob)
            agent = DataMindAgent()
            agent.set_data(df)

            return Session(
                session_id=sid,
                df=df,
                agent=agent,
                created_at=created_at,
                last_accessed=last_accessed,
            )
        except Exception:
            logger.exception("Failed to restore session %s", session_id)
            return None

    async def _delete_session_db(self, session_id: str) -> None:
        """Remove session from SQLite."""
        try:
            from backend.database import async_session_factory
            from backend.models import MessageModel, SessionModel

            async with async_session_factory() as db:
                await db.execute(
                    MessageModel.__table__.delete().where(MessageModel.session_id == session_id)
                )
                await db.execute(
                    SessionModel.__table__.delete().where(SessionModel.session_id == session_id)
                )
                await db.commit()
        except Exception:
            logger.exception("Failed to delete session %s from DB", session_id)

    async def init(self) -> None:
        """Initialize DB tables. Called at startup."""
        try:
            from backend.database import init_db

            await init_db()
            logger.info("Database initialized successfully")
        except Exception:
            logger.exception("Failed to initialize database")


session_manager = SessionManager()
