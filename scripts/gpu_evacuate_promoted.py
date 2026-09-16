#!/usr/bin/env python3
"""Copy promoted GPU artifacts off the ephemeral Hostinger GPU node.

The B200 instance is disposable. This script pulls only production
`*_lstm.pt` / `*_lightgbm.joblib` files (never `*.rejected.*`) plus run
summaries, writes `*_meta.json` sidecars, and optionally pushes them to
the trading VPS Jesse ML bind-mount.

Required env (never commit these):
  GPU_SSH_HOST
  GPU_SSH_PASSWORD or GPU_SSH_PASSWORD_FILE or GPU_SSH_PRIVATE_KEY
  For --push-vps: SSH_HOST and SSH_PRIVATE_KEY (or SSH_KEY_PATH)

Usage:
  python3 scripts/gpu_evacuate_promoted.py
  python3 scripts/gpu_evacuate_promoted.py --push-vps
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import paramiko

DEFAULT_REMOTE = "/home/ubuntu/jesse-gpu"
DEFAULT_VPS_MODELS = "/root/jesse-trading/storage/models"
DEFAULT_VPS_ARCHIVE = "/root/jesse-trading/storage/gpu_evacuate"


def _gpu_password() -> str:
    password = os.environ.get("GPU_SSH_PASSWORD") or ""
    pw_file = os.environ.get("GPU_SSH_PASSWORD_FILE", "/tmp/gpu_ssh.pw")
    if not password and os.path.exists(pw_file):
        password = Path(pw_file).read_text(encoding="utf-8").strip()
    return password


def _normalize_openssh_key(raw: str) -> str:
    begin = "-----BEGIN OPENSSH PRIVATE KEY-----"
    end = "-----END OPENSSH PRIVATE KEY-----"
    if "\n" not in raw and begin in raw:
        body = raw.replace(begin, "").replace(end, "").strip().replace(" ", "\n")
        return f"{begin}\n{body}\n{end}\n"
    return raw if raw.endswith("\n") else raw + "\n"


def _connect(host: str, user: str, port: int, *, password: str = "", key_text: str = "") -> paramiko.SSHClient:
    if not host:
        raise SystemExit("SSH host is required via environment (do not hardcode)")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    kwargs: Dict[str, Any] = {"hostname": host, "port": port, "username": user, "timeout": 25}
    if key_text:
        kwargs["pkey"] = paramiko.Ed25519Key.from_private_key(io.StringIO(_normalize_openssh_key(key_text)))
    elif password:
        kwargs["password"] = password
    else:
        raise SystemExit("password or private key is required")
    client.connect(**kwargs)
    return client


def _run(client: paramiko.SSHClient, cmd: str, timeout: int = 180) -> str:
    _stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    code = stdout.channel.recv_exit_status()
    if code != 0:
        raise RuntimeError(f"remote exit {code}: {err or out}")
    return out


def _sftp_get(sftp: paramiko.SFTPClient, remote: str, local: Path) -> None:
    local.parent.mkdir(parents=True, exist_ok=True)
    sftp.get(remote, str(local))
    print(f"  pulled {local.name} ({local.stat().st_size} bytes)", flush=True)


def _sftp_put(sftp: paramiko.SFTPClient, local: Path, remote: str) -> None:
    sftp.put(str(local), remote)
    print(f"  pushed {local.name} -> {remote}", flush=True)


def pull_from_gpu(dest: Path, remote_dir: str) -> List[Dict[str, Any]]:
    dest.mkdir(parents=True, exist_ok=True)
    print("[*] connecting to GPU node", flush=True)
    client = _connect(
        os.environ.get("GPU_SSH_HOST") or "",
        os.environ.get("GPU_SSH_USER", "ubuntu"),
        int(os.environ.get("GPU_SSH_PORT", "32408")),
        password=_gpu_password(),
        key_text=os.environ.get("GPU_SSH_PRIVATE_KEY") or "",
    )
    try:
        inventory_json = _run(
            client,
            f"{remote_dir}/.venv/bin/python - <<'PY'\n"
            "import glob, json, os, joblib\n"
            f"models = os.path.join('{remote_dir}', 'storage/models')\n"
            "rows = []\n"
            "for p in sorted(glob.glob(os.path.join(models, '*'))):\n"
            "    name = os.path.basename(p)\n"
            "    if not os.path.isfile(p):\n"
            "        continue\n"
            "    rejected = '.rejected.' in name\n"
            "    rec = {'name': name, 'bytes': os.path.getsize(p), 'rejected': rejected}\n"
            "    promoted = (name.endswith('.pt') or name.endswith('.joblib')) and not rejected\n"
            "    if promoted and ('_lstm.pt' in name or name.endswith('_lstm.pt')):\n"
            "        payload = joblib.load(p)\n"
            "        m = payload.get('metrics') or {}\n"
            "        stem = name.rsplit('.', 1)[0]\n"
            "        parts = stem.split('_')\n"
            "        symbol, timeframe, model_type = (parts + ['', '', 'lstm'])[:3]\n"
            "        meta = {\n"
            "            'symbol': symbol, 'timeframe': timeframe, 'model_type': model_type,\n"
            "            'trained_at': payload.get('trained_at'),\n"
            "            'feature_hash': payload.get('feature_hash') or m.get('feature_schema_hash'),\n"
            "            'pt_mult': payload.get('pt_mult', m.get('pt_mult')),\n"
            "            'sl_mult': payload.get('sl_mult', m.get('sl_mult')),\n"
            "            'promotion_ok': bool(payload.get('promotion_ok', m.get('promotion_ok'))),\n"
            "            'metrics': m,\n"
            "        }\n"
            "        meta_name = f'{stem}_meta.json'\n"
            "        with open(os.path.join(models, meta_name), 'w', encoding='utf-8') as fh:\n"
            "            json.dump(meta, fh, indent=2, default=str)\n"
            "        rec['meta'] = meta_name\n"
            "        rec['metrics'] = {\n"
            "            'dsr': m.get('deflated_sharpe_ratio'),\n"
            "            'pbo': m.get('prob_backtest_overfitting'),\n"
            "            'bullish_recall': m.get('bullish_recall'),\n"
            "            'bearish_recall': m.get('bearish_recall'),\n"
            "            'promotion_ok': rec.get('meta') and meta['promotion_ok'],\n"
            "        }\n"
            "    rows.append(rec)\n"
            "print(json.dumps(rows))\n"
            "PY",
        )
        rows = json.loads(inventory_json)
        (dest / "inventory.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        promoted = [
            r
            for r in rows
            if not r["rejected"] and (r["name"].endswith(".pt") or r["name"].endswith("_lstm_meta.json"))
        ]
        print(f"[*] promoted files to copy: {len(promoted)}", flush=True)
        sftp = client.open_sftp()
        try:
            for r in rows:
                if r["rejected"] or not r["name"].endswith(".pt"):
                    continue
                _sftp_get(sftp, f"{remote_dir}/storage/models/{r['name']}", dest / "promoted" / r["name"])
                meta = r.get("meta")
                if meta:
                    _sftp_get(sftp, f"{remote_dir}/storage/models/{meta}", dest / "promoted" / meta)
            summaries = _run(client, f"find {remote_dir}/storage/models/runs -name summary.json 2>/dev/null || true")
            for line in summaries.splitlines():
                remote = line.strip()
                if remote:
                    rel = remote.split("/storage/models/runs/", 1)[-1]
                    _sftp_get(sftp, remote, dest / "runs" / rel)
        finally:
            sftp.close()
    finally:
        client.close()
    return rows


def push_to_vps(dest: Path, models_dir: str, archive_dir: str) -> None:
    key_text = os.environ.get("SSH_PRIVATE_KEY") or ""
    key_path = os.environ.get("SSH_KEY_PATH")
    if not key_text and key_path and os.path.exists(key_path):
        key_text = Path(key_path).read_text(encoding="utf-8")
    print("[*] connecting to trading VPS", flush=True)
    client = _connect(
        os.environ.get("SSH_HOST") or "",
        os.environ.get("SSH_USER", "root"),
        int(os.environ.get("SSH_PORT", "22")),
        key_text=key_text,
    )
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    try:
        _run(client, f"mkdir -p {models_dir} {archive_dir}/{stamp}/runs")
        sftp = client.open_sftp()
        try:
            for path in sorted((dest / "promoted").glob("*")):
                if path.suffix in {".pt", ".json"} and ".rejected." not in path.name:
                    _sftp_put(sftp, path, f"{models_dir}/{path.name}")
                    _sftp_put(sftp, path, f"{archive_dir}/{stamp}/{path.name}")
            if (dest / "inventory.json").exists():
                _sftp_put(sftp, dest / "inventory.json", f"{archive_dir}/{stamp}/inventory.json")
        finally:
            sftp.close()
    finally:
        client.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Evacuate promoted GPU models before the instance is destroyed")
    parser.add_argument("--dest", default="/tmp/gpu_evacuate", help="local staging directory")
    parser.add_argument("--remote-dir", default=os.environ.get("GPU_REMOTE_DIR", DEFAULT_REMOTE))
    parser.add_argument("--push-vps", action="store_true", help="also copy promoted artifacts to the trading VPS")
    parser.add_argument("--skip-pull", action="store_true", help="use existing --dest staging (already pulled)")
    args = parser.parse_args()
    dest = Path(args.dest)
    if not args.skip_pull:
        pull_from_gpu(dest, args.remote_dir)
    if args.push_vps:
        push_to_vps(dest, DEFAULT_VPS_MODELS, DEFAULT_VPS_ARCHIVE)
    print(f"[*] done. staging={dest}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
