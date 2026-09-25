# Image unique : frontend buildé (stage 1) + backend FastAPI (stage 2)
# qui sert l'interface statique. Un seul conteneur pour toute l'app.

# --- Stage 1 : build du frontend ---
FROM node:20-alpine AS frontend-build
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --silent
COPY frontend/ ./
RUN npm run build

# --- Stage 2 : backend ---
FROM python:3.12-slim AS runtime
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Dependances (cache layer dédié)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Code + frontend buildé
COPY src/ src/
RUN uv sync --frozen --no-dev

COPY --from=frontend-build /app/frontend/dist ./frontend/dist

ENV HOST=0.0.0.0 PORT=8000
EXPOSE 8000

CMD ["uv", "run", "uvicorn", "datamind.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
