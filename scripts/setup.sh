#!/usr/bin/env bash
# DataMind AI — setup en une commande (Linux/macOS)
# Usage : bash scripts/setup.sh
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== DataMind AI — setup ==="

# 1. uv
if ! command -v uv >/dev/null 2>&1; then
    echo "[1/4] Installation de uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "[1/4] uv present : $(uv --version)"
fi

# 2. Dependances
echo "[2/4] Installation des dependances (uv sync)..."
uv sync

# 3. Configuration
if [ ! -f .env ]; then
    cp .env.example .env
    echo "[3/4] .env cree depuis .env.example"
else
    echo "[3/4] .env deja present"
fi

# 4. Frontend buildé
if [ -f frontend/dist/index.html ]; then
    echo "[4/4] Frontend deja buildé (frontend/dist present)"
elif command -v npm >/dev/null 2>&1; then
    echo "[4/4] Build du frontend (npm ci + build)..."
    (cd frontend && npm ci --silent && npm run build --silent)
else
    echo "[4/4] Node absent : API seule (Swagger sur /docs)."
    echo "      Pour l'interface complete : installez Node.js ou telechargez la release"
    echo "      (frontend-dist.zip) depuis https://github.com/Mrsaltyx/datamind-ai/releases"
fi

# 5. Ollama (optionnel)
if command -v ollama >/dev/null 2>&1; then
    if ! ollama list 2>/dev/null | grep -q gemma4; then
        echo "Telechargement du modele local gemma4:e4b (une fois)..."
        ollama pull gemma4:e4b
    fi
fi

echo ""
echo "=== Setup termine — lancez : bash scripts/run.sh ==="
