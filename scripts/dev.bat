@echo off
echo ========================================
echo   DataMind AI - Environnement de dev
echo ========================================
echo.

:: Check Python
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERREUR] Python n'est pas installe. Installez Python 3.12+ depuis https://python.org
    pause
    exit /b 1
)

:: Check Node.js
where node >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERREUR] Node.js n'est pas installe. Installez Node.js depuis https://nodejs.org
    pause
    exit /b 1
)

:: Check if Ollama is running
echo [1/4] Verification d'Ollama...
curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] Ollama n'est pas lance. Lancement en arriere-plan...
    start "Ollama" /min ollama serve
    timeout /t 5 /nobreak >nul
    echo [OK] Ollama demarre
) else (
    echo [OK] Ollama est en cours d'execution
)

:: Check if Gemma 4 model is available
echo [2/4] Verification du modele Gemma 4...
curl -s http://localhost:11434/api/tags | findstr "gemma4" >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] Modele gemma4:e4b non trouve. Telechargement (cela peut prendre quelques minutes)...
    ollama pull gemma4:e4b
    if %errorlevel% neq 0 (
        echo [ERREUR] Impossible de telecharger le modele. Verifiez votre connexion.
        pause
        exit /b 1
    )
    echo [OK] Modele gemma4:e4b telecharge
) else (
    echo [OK] Modele gemma4:e4b disponible
)

:: Install backend dependencies
echo [3/4] Installation des dependances backend...
pip install -e ".[dev]" --quiet 2>nul
if %errorlevel% neq 0 (
    pip install -r backend/requirements.txt --quiet 2>nul
)

:: Install frontend dependencies
echo [4/4] Installation des dependances frontend...
cd frontend
call npm install --silent 2>nul
cd ..

echo.
echo ========================================
echo   Tout est pret ! Lancement...
echo ========================================
echo.
echo   Backend :  http://localhost:8000
echo   Frontend : http://localhost:5173
echo   Ollama :   http://localhost:11434
echo.
echo   Ctrl+C dans chaque fenetre pour arreter
echo ========================================
echo.

:: Start backend and frontend in separate windows
start "DataMind Backend" cmd /c "python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload"
timeout /t 3 /nobreak >nul
start "DataMind Frontend" cmd /c "cd frontend && npm run dev"

echo Appuyez sur une touche dans cette fenetre pour arreter tous les services...
pause >nul
taskkill /f /fi "WINDOWTITLE eq DataMind Backend" >nul 2>&1
taskkill /f /fi "WINDOWTITLE eq DataMind Frontend" >nul 2>&1
echo Services arretes.
