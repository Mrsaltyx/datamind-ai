# DataMind AI — setup en une commande (Windows)
# Usage : powershell -ExecutionPolicy Bypass -File scripts/setup.ps1

$ErrorActionPreference = "Stop"

Write-Host "=== DataMind AI — setup ===" -ForegroundColor Cyan

# 1. uv (gestionnaire d'environnement Python)
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "[1/4] Installation de uv..."
    powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
} else {
    Write-Host "[1/4] uv present : $(uv --version)"
}

# 2. Environnement Python + dependances
Write-Host "[2/4] Installation des dependances (uv sync)..."
uv sync --extra mlflow

# 3. Configuration
if (-not (Test-Path .env)) {
    Copy-Item .env.example .env
    Write-Host "[3/4] .env cree depuis .env.example"
} else {
    Write-Host "[3/4] .env deja present"
}

# 4. Frontend buildé (optionnel mais recommandé)
if (Test-Path frontend/dist/index.html) {
    Write-Host "[4/4] Frontend deja buildé (frontend/dist present)"
} elseif (Get-Command npm -ErrorAction SilentlyContinue) {
    Write-Host "[4/4] Build du frontend (npm ci + build)..."
    Push-Location frontend
    npm ci --silent
    npm run build --silent
    Pop-Location
} else {
    Write-Host "[4/4] Node absent : API seule ( Swagger sur /docs )." -ForegroundColor Yellow
    Write-Host "      Pour l'interface complete : installez Node.js ou telechargez la release" -ForegroundColor Yellow
    Write-Host "      (frontend-dist.zip) depuis https://github.com/Mrsaltyx/datamind-ai/releases" -ForegroundColor Yellow
}

# 5. Ollama (optionnel, pour le LLM local)
if (Get-Command ollama -ErrorAction SilentlyContinue) {
    $tags = ollama list 2>$null | Out-String
    if ($tags -notmatch "gemma4") {
        Write-Host "Telechargement du modele local gemma4:e4b (une fois)..."
        ollama pull gemma4:e4b
    }
}

Write-Host ""
Write-Host "=== Setup termine ===" -ForegroundColor Green
Write-Host "Lancez : scripts\run.ps1" -ForegroundColor Green
