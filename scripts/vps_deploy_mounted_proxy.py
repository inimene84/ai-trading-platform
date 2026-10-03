import subprocess
import tempfile
from pathlib import Path
from scripts.vps_ssh_common import ssh_cmd, scp_cmd

DOCKER_COMPOSE = """services:
  kieai-proxy:
    image: python:3.11-slim
    container_name: kieai-proxy
    restart: unless-stopped
    ports:
      - "127.0.0.1:11434:11434"
    networks:
      - n8n_default
    volumes:
      - /docker/kieai-proxy/proxy.py:/app/proxy.py:ro
    environment:
      - KIE_API_KEY=${KIE_API_KEY}
      - OMNIROUTE_API_KEY=<OMNIROUTE_API_KEY>
      - KIEAI_PROXY_PORT=11434
    command: /bin/bash -c "pip install flask requests -q && python /app/proxy.py"

networks:
  n8n_default:
    external: true
"""

PROXY_PY = (Path(__file__).resolve().parent / "kieai_proxy.py").read_text(encoding="utf-8")

# 1. Write proxy.py
with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
    f.write(PROXY_PY)
    p_path = f.name

# 2. Write docker-compose.yml
with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
    f.write(DOCKER_COMPOSE)
    c_path = f.name

try:
    subprocess.run(ssh_cmd("mkdir -p /docker/kieai-proxy"), check=True)
    subprocess.run(scp_cmd(p_path, "/docker/kieai-proxy/proxy.py"), check=True)
    subprocess.run(scp_cmd(c_path, "/docker/kieai-proxy/docker-compose.yml"), check=True)
    res = subprocess.run(ssh_cmd("cd /docker/kieai-proxy && docker compose down && docker compose up -d"), capture_output=True, text=True)
    print("DOCKER COMPOSE RESULT:")
    print(res.stdout)
    if res.stderr:
        print("STDERR:", res.stderr)
finally:
    Path(p_path).unlink(missing_ok=True)
    Path(c_path).unlink(missing_ok=True)
