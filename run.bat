@echo off
REM ═══════════════════════════════════════════════════════════════════════════
REM  AI Hedge Fund — Windows Launcher
REM  Starts both the FastAPI backend and Vite frontend dev server
REM ═══════════════════════════════════════════════════════════════════════════

set "INFO=[INFO]"
set "SUCCESS=[OK]"
set "ERROR=[ERROR]"

REM ── Pre-flight checks ──────────────────────────────────────────────────────
where node >nul 2>&1
if %errorlevel% neq 0 (
    echo %ERROR% Node.js is not installed. Please install from https://nodejs.org/
    pause & exit /b 1
)

where python >nul 2>&1
if %errorlevel% neq 0 (
    echo %ERROR% Python is not installed. Please install from https://python.org/
    pause & exit /b 1
)


echo %SUCCESS% Prerequisites OK

REM ── Check for .env ─────────────────────────────────────────────────────────
if not exist ".env" (
    if exist ".env.example" (
        echo %INFO% No .env file found. Creating from .env.example...
        copy ".env.example" ".env"
        echo %INFO% Please edit .env to add your API keys.
    ) else (
        echo %ERROR% No .env or .env.example file found.
        pause & exit /b 1
    )
) else (
    echo %SUCCESS% Environment file (.env)
)

REM ── Install backend dependencies ───────────────────────────────────────────
echo %INFO% Checking backend dependencies...
if not exist ".venv\Scripts\python.exe" python -m venv .venv
.venv\Scripts\python.exe -c "import uvicorn; import fastapi" >nul 2>&1
if %errorlevel% neq 0 (
    echo %INFO% Installing Python dependencies from backend\requirements.txt...
    .venv\Scripts\python.exe -m pip install -r backend\requirements.txt
    if %errorlevel% neq 0 (
        echo %ERROR% Failed to install backend dependencies
        pause & exit /b 1
    )
)
echo %SUCCESS% Backend dependencies ready

REM ── Install frontend dependencies ──────────────────────────────────────────
echo %INFO% Checking frontend dependencies...
if not exist "frontend\node_modules" (
    echo %INFO% Installing Node.js dependencies...
    pushd frontend
    npm install
    popd
    if %errorlevel% neq 0 (
        echo %ERROR% Failed to install frontend dependencies
        pause & exit /b 1
    )
)
echo %SUCCESS% Frontend dependencies ready

REM ── Start services ─────────────────────────────────────────────────────────
echo.
echo %INFO% Starting AI Hedge Fund web application...
echo %INFO% Press Ctrl+C to stop all services
echo.

REM Start backend (from project root so Python resolves "backend" package)
echo %INFO% Launching backend server on :8080 ...
start /b .venv\Scripts\python.exe -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8080

timeout /t 3 /nobreak >nul

REM Start frontend
echo %INFO% Launching frontend dev server on :3000 ...
pushd frontend
start /b npm run dev
popd

timeout /t 5 /nobreak >nul

echo %INFO% Opening browser...
start http://localhost:3000

echo.
echo %SUCCESS% AI Hedge Fund web application is running!
echo %INFO% Frontend: http://localhost:3000
echo %INFO% Backend:  http://localhost:8080
echo %INFO% Docs:     http://localhost:8080/docs
echo.
echo %INFO% Press any key to stop both services...
pause >nul

taskkill /f /im "uvicorn.exe" >nul 2>&1
taskkill /f /im "node.exe" >nul 2>&1

echo %SUCCESS% Services stopped. Goodbye!
pause
