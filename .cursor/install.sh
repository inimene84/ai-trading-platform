#!/usr/bin/env bash
# QuantumTrade Pro — Cloud Agent install phase.
# Idempotent: safe to re-run. Prepares the Python backend venv, the frontend
# node_modules, and a local .env so the FastAPI backend + Vite dev server can
# start. Runtime services live in `terminals` (see .cursor/environment.json).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# 1. System packages the default image lacks: the stdlib venv builder
#    (ensurepip) and a C toolchain for any source-built wheels.
if ! python3 -c "import ensurepip" >/dev/null 2>&1; then
  PY_MINOR="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  sudo apt-get update -qq
  sudo apt-get install -y -qq "python${PY_MINOR}-venv" python3-dev build-essential
fi

# 2. Python backend virtualenv (single source of truth: backend/requirements.txt).
if [ ! -x ".venv/bin/python" ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
. .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
pip install -r backend/requirements.txt

# 3. Local .env with safe paper-trading defaults (never overwrite an existing one).
if [ ! -f ".env" ]; then
  cp .env.example .env
fi

# 4. Frontend dependencies (deterministic from package-lock.json).
cd frontend
npm ci
cd "$REPO_ROOT"

echo "install.sh: environment ready."
