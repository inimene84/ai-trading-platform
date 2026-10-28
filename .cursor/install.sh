#!/usr/bin/env bash
# Cloud Agent environment install for QuantumTrade Pro.
# Idempotent: safe to run repeatedly. Runs after the repo is checked out.
set -euo pipefail

# Resolve repo root (this script lives in <repo>/.cursor/).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "==> Installing system packages (venv + build toolchain for psycopg2/native wheels)"
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
  python3.12-venv \
  python3-dev \
  build-essential \
  libpq-dev

echo "==> Creating/refreshing backend virtualenv (.venv)"
if [ ! -x ".venv/bin/python" ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
. .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools

echo "==> Installing backend dependencies (production source of truth)"
pip install -r backend/requirements.txt
# cTrader transport is installed --no-deps in production (see Dockerfile.backend);
# mirror that here so the broker adapter can import its protobuf messages.
pip install "ctrader-open-api==0.9.2" --no-deps
# Ruff is the CI linter but is intentionally not in requirements.txt (prod deps).
pip install ruff

echo "==> Installing frontend dependencies"
cd frontend
npm install

echo "==> Install complete."
