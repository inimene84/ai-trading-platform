#!/usr/bin/env python3
"""Upload family-aware Kie proxy + Agent Zero model catalog to the VPS."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from scripts.vps_ssh_common import scp_cmd, ssh_cmd

ROOT = Path(__file__).resolve().parents[1]
PROXY_SRC = ROOT / "scripts" / "kieai_proxy.py"
CATALOG_SRC = ROOT / "scripts" / "agent0_kie_catalog.py"
INSTALL_SRC = ROOT / "scripts" / "agent0_kie_install.py"


def run(cmd: list[str], label: str) -> int:
    print(f"--- {label} ---")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        print(result.stderr[-2000:])
    return result.returncode


def main() -> int:
    for src, dest, label in [
        (PROXY_SRC, "/docker/kieai-proxy/proxy.py", "upload proxy.py"),
        (CATALOG_SRC, "/tmp/agent0_kie_catalog.py", "upload catalog"),
        (INSTALL_SRC, "/tmp/agent0_kie_install.py", "upload install"),
    ]:
        code = run(scp_cmd(str(src), dest), label)
        if code != 0:
            return code

    remote = r"""
set -euo pipefail
python3 -c 'import yaml' 2>/dev/null || pip3 install pyyaml -q
cd /tmp
python3 agent0_kie_install.py
docker restart kieai-proxy
# Wait until the slim image finishes pip install flask on boot.
ok=0
for i in $(seq 1 40); do
  if curl -sf http://127.0.0.1:11434/health | grep -q kieai-proxy-v8-family; then
    ok=1
    break
  fi
  sleep 3
done
if [ "$ok" != 1 ]; then
  echo "proxy health failed"
  docker logs kieai-proxy --tail 40
  exit 1
fi
echo "=== proxy models (ids only) ==="
python3 - <<'PY'
import json, urllib.request
data = json.load(urllib.request.urlopen("http://127.0.0.1:11434/v1/models", timeout=10))
ids = [m.get("id") for m in data.get("data", [])]
print("count", len(ids))
for mid in ids:
    print(" ", mid)
PY
docker restart a0-instance
echo "a0-instance restarted"
echo "=== probe family routing (no body dump) ==="
python3 - <<'PY'
import json, urllib.error, urllib.request
from pathlib import Path
key = ""
for path in (Path("/root/ai-trading-platform-v3/.env"), Path("/var/lib/docker/volumes/agent-zero_a0-data/_data/.env")):
    if not path.exists():
        continue
    for line in path.read_text().splitlines():
        if line.startswith("KIE_API_KEY=") or line.startswith("API_KEY_KIEAI="):
            key = line.split("=", 1)[1].strip().strip('"')
            break
    if key:
        break
if not key:
    print("NO_KEY")
    raise SystemExit(0)

def probe(model):
    req = urllib.request.Request(
        "http://127.0.0.1:11434/v1/chat/completions",
        data=json.dumps({
            "model": model,
            "stream": False,
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "Reply with the single word PONG."}],
        }).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            payload = json.load(resp)
        text = (((payload.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
        print(f"{model}: HTTP 200 len={len(text)}")
    except urllib.error.HTTPError as exc:
        print(f"{model}: FAIL HTTP {exc.code}")
    except Exception as exc:
        print(f"{model}: FAIL {type(exc).__name__}")

for mid in ("gpt-6-astra", "claude-fable-5-1", "claude-fable-5", "gpt-5-6-luna", "claude-sonnet-5"):
    probe(mid)
PY
"""
    return run(ssh_cmd(remote), "install + restart + probe")


if __name__ == "__main__":
    sys.exit(main())
