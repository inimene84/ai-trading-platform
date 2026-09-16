#!/usr/bin/env bash
# 8.5h B200 campaign. Stops before the 9h Hostinger GPU window expires.
# Does not talk to brokers. Writes artifacts under storage/models/.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export JESSE_MODELS_DIR="$ROOT/storage/models"
export JESSE_TRAIN_DEVICE=cuda
mkdir -p logs storage/models storage/candles
DEADLINE=$(( $(date +%s) + 8 * 3600 + 20 * 60 ))
LOG="$ROOT/logs/campaign.log"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG"; }
still_time() { [[ $(date +%s) -lt $DEADLINE ]]; }

log "B200 campaign start. deadline_unix=$DEADLINE"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv | tee -a "$LOG"
df -h / | tee -a "$LOG"
echo '{"cumulative_trials":0}' > "$JESSE_MODELS_DIR/_trial_ledger.json"

run_one() {
  local sym="$1" tf="$2" model="$3" events="$4"
  if ! still_time; then
    log "DEADLINE — skip $sym $tf $model $events"
    return 0
  fi
  local candles="$ROOT/storage/candles/${sym}_1m.csv.gz"
  if [[ ! -f "$candles" ]]; then
    log "MISSING candles $candles"
    return 1
  fi
  local out="$ROOT/logs/train_${sym}_${tf}_${events}_${model}.log"
  log "START $sym $tf model=$model events=$events"
  set +e
  "$ROOT/.venv/bin/python" -u "$ROOT/train_gpu.py" \
    --symbol "$sym" --timeframe "$tf" --candles "$candles" \
    --model "$model" --events "$events" \
    --pt-mult 5.5 --sl-mult 1.75 --holding 48 --device cuda \
    >>"$out" 2>&1
  local rc=$?
  set -e
  log "DONE $sym $tf events=$events rc=$rc (see $out)"
  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | tee -a "$LOG"
  df -h / | tail -1 | tee -a "$LOG"
  return 0
}

# Phase 1 — every-bar labels (enough N for B200 LSTM) at live geometry
run_one BTC-USDT 1h both everybar
run_one ETH-USDT 1h both everybar
run_one SOL-USDT 1h both everybar

# Phase 2 — denser bars to keep the B200 busy
run_one BTC-USDT 15m lstm everybar
run_one BTC-USDT 5m lstm everybar
run_one ETH-USDT 15m lstm everybar
run_one BTC-USDT 4h both everybar

# Phase 3 — strategy-event LSTM for live QuantumAI filter comparison
run_one BTC-USDT 1h lstm quantum_ai

# Phase 4 — extra 15m pass if the window remains
if still_time; then
  log "Phase 4 extra BTC 15m lstm"
  run_one BTC-USDT 15m lstm everybar
fi

log "Campaign loop finished. Writing summary."
"$ROOT/.venv/bin/python" - << 'PY'
import json, glob, os
from pathlib import Path
root = Path("/home/ubuntu/jesse-gpu/storage/models")
rows = []
for p in sorted(root.glob("*.json")) + sorted((root / "runs").glob("*/summary.json")):
    try:
        data = json.loads(p.read_text())
    except Exception:
        continue
    rows.append({"path": str(p), "data": data})
print(json.dumps({"summaries_found": len(rows), "files": [r["path"] for r in rows]}, indent=2))
for art in sorted(root.glob("*.joblib")) + sorted(root.glob("*.pt")):
    print(f"ARTIFACT {art.name} {art.stat().st_size}")
PY
log "B200 campaign complete."
nvidia-smi
df -h /
