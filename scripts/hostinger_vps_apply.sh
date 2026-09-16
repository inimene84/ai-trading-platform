#!/usr/bin/env bash
# Canonical production deploy — run ON the Hostinger VPS as root.
# Also invoked by: vps_remote_oneliner.sh, ssh_vps_remote.sh, GitHub Actions.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/root/ai-trading-platform-v3}"
cd "$PROJECT_DIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/vps_ssh_hygiene.sh
source "${SCRIPT_DIR}/lib/vps_ssh_hygiene.sh"

DEPLOY_REF="${DEPLOY_REF:-main}"

COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.prod.yml}"
if [ ! -f "$COMPOSE_FILE" ]; then
  COMPOSE_FILE="docker-compose.yml"
fi

echo "=== Hostinger VPS apply (${DEPLOY_REF:-main}) ==="
echo "Project: $PROJECT_DIR"
echo "Compose: $COMPOSE_FILE"

if [ ! -f "$COMPOSE_FILE" ]; then
  echo "ERROR: no compose file found"
  exit 1
fi

echo ""
echo "=== 0. SSH hygiene ==="
vps_ssh_hygiene

echo ""
echo "=== 1. Git pull ${DEPLOY_REF:-main} ==="
DEPLOY_REF="${DEPLOY_REF:-main}"
git fetch origin "$DEPLOY_REF"
if git show-ref --verify --quiet "refs/heads/${DEPLOY_REF}" || git show-ref --verify --quiet "refs/remotes/origin/${DEPLOY_REF}"; then
  git checkout "$DEPLOY_REF"
  git pull origin "$DEPLOY_REF"
else
  git checkout -B "$DEPLOY_REF" "origin/${DEPLOY_REF}"
fi
git log -1 --oneline

echo ""
echo "=== 2. Env sanity (live trading + P0 gates) ==="
if [ ! -f .env ]; then
  echo "ERROR: .env is required for production deployment"
  exit 1
fi
for required in ADMIN_API_KEY LITELLM_API_KEY QDRANT_API_KEY; do
  if ! grep -qE "^${required}=.+$" .env; then
    echo "ERROR: ${required} must be set in .env"
    exit 1
  fi
done
ADMIN_API_KEY=$(grep '^ADMIN_API_KEY=' .env | cut -d= -f2-)

# Never silently turn a safe/paper config into live trading. A live deploy must
# carry an explicit operator acknowledgement in the server-side .env.
if grep -qE '^PAPER_TRADING=false$' .env \
   && grep -qE '^DRY_RUN_ALL=false$' .env \
   && ! grep -qE '^CONFIRM_LIVE_DEPLOY=true$' .env; then
  echo "ERROR: live trading requested without CONFIRM_LIVE_DEPLOY=true"
  exit 1
fi

# Upsert KEY=VALUE in .env (replace existing line or append).
_upsert_env() {
  local key="$1" value="$2"
  if grep -qE "^${key}=" .env; then
    # Escape sed replacement metacharacters in value (&, \, |)
    local escaped
    escaped=$(printf '%s' "$value" | sed -e 's/[&|\\]/\\&/g')
    sed -i "s|^${key}=.*|${key}=${escaped}|" .env
  else
    echo "${key}=${value}" >> .env
  fi
}

if [ -f .env ]; then
  grep -q '^PYRAMID_MIN_IMPROVEMENT=' .env || echo 'PYRAMID_MIN_IMPROVEMENT=0.005' >> .env
  sed -i 's/^PYRAMID_MIN_IMPROVEMENT=0$/PYRAMID_MIN_IMPROVEMENT=0.005/' .env
  grep -q '^MIN_AVAILABLE_MARGIN_USDT=' .env || echo 'MIN_AVAILABLE_MARGIN_USDT=5' >> .env
  # Kill floor: never below $50 on a ~$128 account
  if grep -q '^TRADING_KILL_FLOOR_USDT=20' .env 2>/dev/null; then
    sed -i 's/^TRADING_KILL_FLOOR_USDT=20/TRADING_KILL_FLOOR_USDT=65/' .env
    echo "  Bumped TRADING_KILL_FLOOR_USDT 20 -> 65"
  fi
  grep -q '^NATIVE_TRAILING_ENABLED=' .env || echo 'NATIVE_TRAILING_ENABLED=false' >> .env
  grep -q '^EQUITY_SIZING_ENABLED=' .env || echo 'EQUITY_SIZING_ENABLED=true' >> .env
  grep -q '^PAPER_TRADING=' .env || echo 'PAPER_TRADING=true' >> .env
  grep -q '^DRY_RUN_ALL=' .env || echo 'DRY_RUN_ALL=true' >> .env
  grep -q '^JSON_LOGS=' .env || echo 'JSON_LOGS=true' >> .env
  if ! grep -q '^GRAFANA_URL=' .env; then
    VPS_IP=$(curl -sf --max-time 3 ifconfig.me 2>/dev/null || hostname -I | awk '{print $1}')
    echo "GRAFANA_URL=http://${VPS_IP:-localhost}:3000" >> .env
  fi

  # ── Weekly-edge fixes (Jul 2026): restore risk halts, tighten trail/pyramid,
  #    drop confirmed loss-leader alts. Force-overwrite unsafe overrides that
  #    previously disabled the guard (drawdown=99 / daily=100 / pyramid=10).
  echo "  Applying weekly-edge risk/trail/pyramid/symbol safeguards..."
  _upsert_env RISK_MAX_DRAWDOWN_PCT 20
  _upsert_env RISK_MAX_DAILY_LOSS_PCT 5
  # 72h peak window: after a realized multi-week drawdown, a 30d peak
  # (~$246 vs ~$170) immediately trips the restored 20% gate and locks the
  # loop forever. Forward protection uses a recent equity anchor instead.
  _upsert_env RISK_PEAK_LOOKBACK_HOURS 72
  _upsert_env PYRAMID_MAX_LAYERS 2
  _upsert_env PYRAMID_BLOCK_UNDERWATER true
  _upsert_env MAX_SAME_DIRECTION_POSITIONS 8
  _upsert_env TRAILING_STOP_ENABLED true
  _upsert_env STEP_TRAIL_ENABLED false
  _upsert_env TRAIL_ACTIVATION_ATR 2.0
  _upsert_env TRAIL_ATR_MULT 0.8
  _upsert_env NATIVE_TRAILING_ENABLED false
  _upsert_env SYMBOL_BLACKLIST "ADAUSDT,ARBUSDT,DOGEUSDT,APTUSDT,ADAUSDC,ARBUSDC,DOGEUSDC,APTUSDC,SIRENUSDT,AIGENSYNUSDT,INUSDT,BUSDT,BRUSDT,SPKUSDT,MAGMAUSDT,VELVETUSDT,HOMEUSDT,BTWUSDT,LABUSDT"

  # Strip weak alts from TRADING_SYMBOLS if present (USDT + USDC forms).
  if grep -qE '^TRADING_SYMBOLS=' .env; then
    CURRENT_SYMS=$(grep '^TRADING_SYMBOLS=' .env | cut -d= -f2-)
    CLEAN_SYMS=$(printf '%s' "$CURRENT_SYMS" | tr ',' '\n' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' | \
      grep -vE '^(ADAUSDT|ARBUSDT|DOGEUSDT|APTUSDT|ADAUSDC|ARBUSDC|DOGEUSDC|APTUSDC)$' | \
      paste -sd, -)
    if [ -n "$CLEAN_SYMS" ] && [ "$CLEAN_SYMS" != "$CURRENT_SYMS" ]; then
      _upsert_env TRADING_SYMBOLS "$CLEAN_SYMS"
      echo "  Stripped ADA/ARB/DOGE/APT from TRADING_SYMBOLS"
    fi
  fi
fi

echo ""
echo "=== 3. Frontend build ==="
if [ -d frontend ]; then
  cd frontend
  export NODE_OPTIONS="${NODE_OPTIONS:---max-old-space-size=2048}"
  npm ci 2>/dev/null || npm install
  npm run build
  cd ..
fi
if [ ! -f frontend/dist/index.html ]; then
  echo "ERROR: frontend/dist/index.html missing — dashboard will be blank"
  exit 1
fi
echo "  frontend/dist OK ($(du -sh frontend/dist | awk '{print $1}'))"

echo ""
echo "=== 4. Docker network ==="
docker network create trading-net 2>/dev/null || true

echo ""
echo "=== 5. Rebuild & start full stack ==="
docker compose -f "$COMPOSE_FILE" up -d --build \
  backend litellm nginx influxdb grafana qdrant mcp-server kronos-infer scrapling

echo ""
echo "=== 5b. InfluxDB buckets ==="
chmod +x scripts/ensure_influx_buckets.sh 2>/dev/null || true
./scripts/ensure_influx_buckets.sh || true

echo ""
echo "=== 6. Wait for backend health ==="
for i in $(seq 1 30); do
  if curl -sf http://127.0.0.1:8001/health >/dev/null 2>&1; then
    echo "Backend healthy on :8001"
    break
  fi
  echo "  waiting backend ($i/30)..."
  sleep 5
done

echo ""
echo "=== 7. Restart nginx (refresh upstream DNS) ==="
docker restart ai-trading-nginx
sleep 3

echo ""
echo "=== 8. Smoke tests ==="
curl -sf http://127.0.0.1:8001/health | head -c 200 || true
echo ""
curl -sf -o /dev/null -w "nginx /api/backend/health: %{http_code}\n" \
  http://127.0.0.1:8081/api/backend/health || true
curl -sf -o /dev/null -w "nginx /api/backend/trading/status: %{http_code}\n" \
  http://127.0.0.1:8081/api/backend/trading/status || true
curl -sf -o /dev/null -w "api/historical: %{http_code}\n" \
  "http://127.0.0.1:8001/api/historical?symbol=BTCUSDT&interval=1h&limit=3" || true
curl -sf -o /dev/null -w "frontend /: %{http_code}\n" \
  http://127.0.0.1:8081/ || true
curl -sf -o /dev/null -w "influxdb :8086: %{http_code}\n" \
  http://127.0.0.1:8086/health || true
curl -sf -o /dev/null -w "scrapling research :8004: %{http_code}\n" \
  http://127.0.0.1:8004/health || true
curl -sf -o /dev/null -w "grafana via nginx: %{http_code}\n" \
  http://127.0.0.1:8081/grafana/api/health || true

echo ""
echo "=== 8b. Grafana datasource + dashboard ==="
GRAFANA_PASS=$(grep '^GRAFANA_PASSWORD=' .env 2>/dev/null | cut -d= -f2- || echo admin)
chmod +x scripts/fix_grafana_influx.sh scripts/deploy_grafana.sh 2>/dev/null || true
./scripts/fix_grafana_influx.sh "http://127.0.0.1:8081/grafana" "admin:${GRAFANA_PASS}" "$PROJECT_DIR" 2>/dev/null || \
  echo "  (Grafana provisioning skipped — set INFLUXDB_TOKEN in .env and re-run fix_grafana_influx.sh)"

echo ""
echo "=== 9. Runtime LLM toggles ==="
curl -sf -X POST http://127.0.0.1:8001/trading/config/update \
  -H 'Content-Type: application/json' \
  -H "X-API-Key: ${ADMIN_API_KEY}" \
  -d '{"use_risk_reviewer_llm": true, "enable_personas": false}' || true

echo ""
echo "=== 10. Ensure trading loop is running with current TRADING_SYMBOLS ==="
SYMS_JSON=$(grep '^TRADING_SYMBOLS=' .env 2>/dev/null | cut -d= -f2- | python3 -c "import sys,json; print(json.dumps([s.strip() for s in sys.stdin.read().split(',') if s.strip()]))" 2>/dev/null || echo '[]')
curl -sf -X POST http://127.0.0.1:8001/trading/loop/stop \
  -H 'Content-Type: application/json' \
  -H "X-API-Key: ${ADMIN_API_KEY}" >/dev/null 2>&1 || true
sleep 1
curl -sf -X POST http://127.0.0.1:8001/trading/loop/start \
  -H 'Content-Type: application/json' \
  -H "X-API-Key: ${ADMIN_API_KEY}" \
  -d "{\"interval_minutes\":15,\"strategy\":\"combined\",\"symbols\":${SYMS_JSON}}" || true
sleep 2
curl -sf http://127.0.0.1:8001/trading/loop/status | python3 -c "
import sys,json
d=json.load(sys.stdin)
print(f\"  loop_running={d.get('running')} symbols={len(d.get('symbols',[]))} cycle_count={d.get('cycle_count')}\")
print(f\"  pairs={d.get('symbols')}\")
" 2>/dev/null || true

echo ""
echo "=== Done ==="
echo "  Dashboard:  http://<vps-ip>:8081/"
echo "  Grafana:    http://<vps-ip>:8081/grafana/  (not :3000 — may be used by other stack)"
echo "  InfluxDB:   internal Docker network / localhost only"
echo "  API health: http://<vps-ip>:8081/health"
