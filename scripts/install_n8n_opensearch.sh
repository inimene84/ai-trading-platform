#!/usr/bin/env bash
# Install the OpenSearch community nodes into the existing /docker/n8n volume.
#
# Do not type the GitHub path into n8n Settings → Community nodes.
# npm name is n8n-nodes-opensearch (not stevenlafl/n8n-nodes-opensearch).
# The GitHub path 404s on the npm registry and n8n reports
# "Failed to check package version existence".
#
# n8n-nodes-opensearch@0.2.5 also needs @langchain/community (peer). The n8n
# image already ships it; this script symlinks that copy. Never npm-install
# n8n-nodes-base or @n8n/n8n-nodes-langchain into ~/.n8n/nodes — those collide
# with the image and crash n8n ("Node loader n8n-nodes-base is already registered").
#
# Usage (on the VPS):
#   ./scripts/install_n8n_opensearch.sh
set -euo pipefail

N8N_CONTAINER="${N8N_CONTAINER:-n8n}"
N8N_DATA_DIR="${N8N_DATA_DIR:-/var/lib/docker/volumes/n8n_data/_data}"
NODES_DIR="${N8N_DATA_DIR}/nodes"
PKG_VERSION="${N8N_OPENSEARCH_VERSION:-0.2.5}"

if [[ ! -d "${NODES_DIR}" ]]; then
  echo "ERROR: ${NODES_DIR} not found"
  exit 1
fi
if ! docker inspect "${N8N_CONTAINER}" >/dev/null 2>&1; then
  echo "ERROR: container ${N8N_CONTAINER} not found"
  exit 1
fi

echo "=== install n8n-nodes-opensearch@${PKG_VERSION} ==="
docker exec -u node -w /home/node/.n8n/nodes "${N8N_CONTAINER}" npm install \
  "n8n-nodes-opensearch@${PKG_VERSION}" \
  --omit=dev \
  --ignore-scripts \
  --no-audit \
  --no-fund \
  --legacy-peer-deps

echo "=== drop official packages if npm hoisted them ==="
python3 - << PY
import json
from pathlib import Path
p = Path("${NODES_DIR}/package.json")
data = json.loads(p.read_text())
deps = data.get("dependencies") or {}
changed = False
for drop in ("@n8n/n8n-nodes-langchain", "n8n-nodes-base", "n8n-workflow", "n8n-core"):
    if drop in deps:
        del deps[drop]
        changed = True
        print(f"removed {drop} from package.json")
data["dependencies"] = deps
if changed:
    p.write_text(json.dumps(data, indent=2) + "\n")
PY
rm -rf "${NODES_DIR}/node_modules/n8n-nodes-base" "${NODES_DIR}/node_modules/@n8n"

echo "=== symlink image @langchain/community (peer of the vector-store node) ==="
COMM="$(docker exec "${N8N_CONTAINER}" sh -c \
  'find /usr/local/lib/node_modules/n8n/node_modules/.pnpm -path "*/node_modules/@langchain/community/package.json" | head -1')"
if [[ -z "${COMM}" ]]; then
  echo "ERROR: @langchain/community not found inside the n8n image"
  exit 1
fi
COMM_DIR="$(dirname "${COMM}")"
mkdir -p "${NODES_DIR}/node_modules/@langchain"
ln -sfn "${COMM_DIR}" "${NODES_DIR}/node_modules/@langchain/community"
mkdir -p "${NODES_DIR}/node_modules/n8n-nodes-opensearch/node_modules/@langchain"
ln -sfn "${COMM_DIR}" "${NODES_DIR}/node_modules/n8n-nodes-opensearch/node_modules/@langchain/community"

echo "=== register package in n8n sqlite if missing ==="
python3 - << PY
import sqlite3
from datetime import datetime, timezone
p = "${N8N_DATA_DIR}/database.sqlite"
c = sqlite3.connect(p)
now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
pkg = "n8n-nodes-opensearch"
ver = "${PKG_VERSION}"
if not c.execute("SELECT 1 FROM installed_packages WHERE packageName=?", (pkg,)).fetchone():
    c.execute(
        "INSERT INTO installed_packages (packageName, installedVersion, authorName, authorEmail, createdAt, updatedAt) VALUES (?,?,?,?,?,?)",
        (pkg, ver, "Steven Linn", "smlucf@gmail.com", now, now),
    )
    print("inserted installed_packages row")
else:
    c.execute("UPDATE installed_packages SET installedVersion=?, updatedAt=? WHERE packageName=?", (ver, now, pkg))
    print("updated installed_packages version")
nodes = (
    ("OpenSearch", "n8n-nodes-opensearch.opensearch", 1.0, pkg),
    ("OpenSearch Vector Store", "n8n-nodes-opensearch.vectorStoreOpenSearch", 1.0, pkg),
)
for name, typ, latest, package in nodes:
    if not c.execute("SELECT 1 FROM installed_nodes WHERE type=?", (typ,)).fetchone():
        c.execute(
            "INSERT INTO installed_nodes (name, type, latestVersion, package) VALUES (?,?,?,?)",
            (name, typ, latest, package),
        )
        print("inserted", typ)
c.commit()
PY

echo "=== restart n8n ==="
docker restart "${N8N_CONTAINER}"
for i in $(seq 1 20); do
  if docker logs "${N8N_CONTAINER}" --since 30s 2>&1 | grep -q "Editor is now accessible"; then
    echo "n8n is up"
    break
  fi
  if docker logs "${N8N_CONTAINER}" --since 30s 2>&1 | grep -q "already registered"; then
    echo "ERROR: n8n crash-looped on a colliding official package"
    exit 1
  fi
  sleep 3
done
docker logs "${N8N_CONTAINER}" --since 45s 2>&1 | grep -iE "Error loading|already registered|n8n ready|Editor is now" | tail -20
echo "Done. In the editor search for OpenSearch (npm name: n8n-nodes-opensearch)."
