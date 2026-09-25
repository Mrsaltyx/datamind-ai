# Lance DataMind AI (backend + frontend statique servi sur le meme port)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $MyInvocation.MyCommand.Path | Split-Path)

Write-Host "DataMind AI sur http://localhost:8000 (Ctrl+C pour arreter)" -ForegroundColor Cyan
Start-Process "http://localhost:8000"
uv run uvicorn datamind.api.main:app --host 0.0.0.0 --port 8000
