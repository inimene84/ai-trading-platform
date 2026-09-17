#!/usr/bin/env bash
# Create or UPDATE InfluxDB buckets used by the trading platform (idempotent).
# Snapshot Influx before changing retention on a live host.
set -euo pipefail

CONTAINER="${INFLUX_CONTAINER:-vps-influxdb}"
ORG="${INFLUXDB_ORG:-hedge-fund}"

if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  echo "InfluxDB container $CONTAINER is not running — skip bucket setup"
  exit 0
fi

TOKEN="${INFLUXDB_TOKEN:-}"
if [[ -z "$TOKEN" && -f .env ]]; then
  TOKEN=$(grep '^INFLUXDB_TOKEN=' .env | cut -d= -f2- || true)
fi
if [[ -z "$TOKEN" ]]; then
  echo "WARN: INFLUXDB_TOKEN not set — cannot create/update buckets"
  exit 0
fi

BUCKETS=(
  "trading-system:90d"
  "trading-signals:90d"
  "trading-orders:365d"
  "trading-raw:30d"
  "trading-memory:0"
  "news-sentiment:90d"
)

_bucket_id() {
  local name="$1"
  docker exec "$CONTAINER" influx bucket list --org "$ORG" --token "$TOKEN" --name "$name" --json 2>/dev/null \
    | python3 -c "
import json, sys
raw = sys.stdin.read().strip()
if not raw:
    raise SystemExit(1)
data = json.loads(raw)
rows = data if isinstance(data, list) else [data]
if not rows or not rows[0].get('id'):
    raise SystemExit(1)
print(rows[0]['id'])
" 2>/dev/null
}

for spec in "${BUCKETS[@]}"; do
  name="${spec%%:*}"
  retention="${spec##*:}"
  if id=$(_bucket_id "$name"); then
    echo "  updating bucket: $name (retention=${retention} id=${id})"
    docker exec "$CONTAINER" influx bucket update \
      --id "$id" --token "$TOKEN" --retention "${retention}" || true
  else
    echo "  creating bucket: $name (retention=${retention})"
    docker exec "$CONTAINER" influx bucket create \
      --org "$ORG" --token "$TOKEN" \
      --name "$name" --retention "${retention}" || true
  fi
done
echo "Influx buckets OK"
