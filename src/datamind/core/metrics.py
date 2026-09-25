"""Metriques d'usage LLM (tokens par requete).

But : arbitrer sur donnees reelles l'apport des optimisations de contexte
(ex. outils adosses au tracking MLflow) au lieu de decider a l'intuition.
Chaque appel LLM rapporte son usage (prompt/completion tokens) ; les
compteurs sont exposes via /api/metrics et persistes en JSONL.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

DEFAULT_PATH = "data/token_metrics.jsonl"


@dataclass
class Totals:
    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


class MetricsStore:
    """Compteurs globaux, thread-safe (les appels LLM tournent en threads)."""

    def __init__(self, path: str = DEFAULT_PATH) -> None:
        self._lock = threading.Lock()
        self._totals = Totals()
        self._path = path

    def record(self, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        with self._lock:
            self._totals.requests += 1
            self._totals.prompt_tokens += prompt_tokens
            self._totals.completion_tokens += completion_tokens
        self._append_jsonl(
            {
                "ts": time.time(),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }
        )

    def snapshot(self) -> dict:
        with self._lock:
            t = self._totals
            return {
                "requests": t.requests,
                "prompt_tokens": t.prompt_tokens,
                "completion_tokens": t.completion_tokens,
                "total_tokens": t.prompt_tokens + t.completion_tokens,
            }

    def _append_jsonl(self, record: dict) -> None:
        try:
            directory = os.path.dirname(self._path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except OSError:
            logger.warning("Metriques : ecriture JSONL impossible vers %s", self._path)


store = MetricsStore()
