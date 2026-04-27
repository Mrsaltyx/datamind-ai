FROM python:3.12-slim AS base

WORKDIR /app

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install system dependencies for llama-cpp-python
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first (layer caching)
COPY backend/requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

# Copy backend code
COPY backend/ ./backend/
COPY agent/ ./agent/
COPY utils/ ./utils/
COPY prompts/ ./prompts/

# Copy model files if they exist (embedded mode)
COPY models/ ./models/

EXPOSE 8000

# Health check: verify backend is responsive
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
