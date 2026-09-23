#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
#  AI Hedge Fund — Linux / macOS Launcher
#  Starts both the FastAPI backend and Vite frontend dev server
# ═══════════════════════════════════════════════════════════════════════════

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[OK]${NC} $1"; }
error()   { echo -e "${RED}[ERROR]${NC} $1"; }

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# ── Pre-flight checks ────────────────────────────────────────────────────
for cmd in node npm python3; do
    command -v "$cmd" >/dev/null 2>&1 || { error "$cmd is not installed"; exit 1; }
done
success "Prerequisites OK"

# ── Environment ──────────────────────────────────────────────────────────
if [[ ! -f ".env" ]]; then
    if [[ -f ".env.example" ]]; then
        info "Creating .env from .env.example..."
        cp .env.example .env
    else
        error "No .env or .env.example found"; exit 1
    fi
fi

# ── Dependencies ─────────────────────────────────────────────────────────
info "Checking backend dependencies..."
[[ -d ".venv" ]] || python3 -m venv .venv
.venv/bin/python -c "import uvicorn; import fastapi" >/dev/null 2>&1 || .venv/bin/pip install -r backend/requirements.txt
success "Backend ready"

info "Checking frontend dependencies..."
[[ -d "frontend/node_modules" ]] || (cd frontend && npm install)
success "Frontend ready"

# ── Cleanup handler ──────────────────────────────────────────────────────
cleanup() {
    info "Shutting down..."
    kill "$BACKEND_PID" 2>/dev/null || true
    kill "$FRONTEND_PID" 2>/dev/null || true
    success "Stopped. Goodbye!"
    exit 0
}
trap cleanup SIGINT SIGTERM

# ── Start services ───────────────────────────────────────────────────────
info "Starting backend on :8080..."
PYTHONPATH="." .venv/bin/python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8080 &
BACKEND_PID=$!
sleep 3

info "Starting frontend on :3000..."
(cd frontend && npm run dev) &
FRONTEND_PID=$!
sleep 3

echo ""
success "🚀 AI Hedge Fund is running!"
info "Frontend: http://localhost:3000"
info "Backend:  http://localhost:8080"
info "Docs:     http://localhost:8080/docs"
echo ""
info "Press Ctrl+C to stop"

# Wait
while kill -0 "$BACKEND_PID" 2>/dev/null && kill -0 "$FRONTEND_PID" 2>/dev/null; do
    sleep 1
done
cleanup