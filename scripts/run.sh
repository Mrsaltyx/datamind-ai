#!/usr/bin/env bash
# Lance DataMind AI (backend + frontend statique servi sur le meme port)
set -euo pipefail
cd "$(dirname "$0")/.."

echo "DataMind AI sur http://localhost:8000 (Ctrl+C pour arreter)"
( sleep 2; xdg-open "http://localhost:8000" 2>/dev/null || open "http://localhost:8000" 2>/dev/null ) &
exec uv run uvicorn datamind.api.main:app --host 0.0.0.0 --port 8000
