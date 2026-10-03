"""Shared SSH target for VPS helper scripts.

Every script in `scripts/vps_*.py` should import from here so connection
configuration comes from environment variables, with the documented
defaults as fallback.

Env vars:
    SSH_HOST              Trading VPS (default target)
    SSH_HOST_TRADING      Alias for SSH_HOST (trading / QuantumTrade)
    SSH_HOST_HERMES       Allikas / OmniRoute / Hermes VPS
    SSH_HOST_ALLIKAS      Alias for SSH_HOST_HERMES
    SSH_HOST_CONSTRUCTION Alias for SSH_HOST_HERMES
    SSH_USER              SSH user (default: root)
    SSH_PORT              SSH port (default: 22)
    SSH_KEY_PATH          Private key file (default: ~/.ssh/id_vps_bot)
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

SSH_HOST_TRADING = os.getenv("SSH_HOST_TRADING") or os.getenv("SSH_HOST") or ""
SSH_HOST = SSH_HOST_TRADING
SSH_HOST_HERMES = (
    os.getenv("SSH_HOST_HERMES")
    or os.getenv("SSH_HOST_ALLIKAS")
    or os.getenv("SSH_HOST_CONSTRUCTION")
    or ""
)
SSH_USER = os.getenv("SSH_USER", "root")
SSH_PORT = os.getenv("SSH_PORT", "22")
SSH_KEY_PATH = os.getenv("SSH_KEY_PATH", "")

TARGET_TRADING = f"{SSH_USER}@{SSH_HOST_TRADING}" if SSH_HOST_TRADING else ""
TARGET_HERMES = f"{SSH_USER}@{SSH_HOST_HERMES}" if SSH_HOST_HERMES else ""
TARGET = TARGET_TRADING


def _host_for_role(role: str | None = None) -> str:
    """Resolve host. Omit role (existing callers) always uses trading SSH_HOST."""
    if role is None or str(role).strip() == "":
        if not SSH_HOST:
            raise ValueError("SSH_HOST environment variable is required")
        return SSH_HOST
    normalized = str(role).strip().lower()
    allikas_roles = {"allikas", "hermes", "construction", "omniroute", "omni"}
    trading_roles = {"trading", "qt", "quantumtrade"}
    if normalized in allikas_roles:
        if not SSH_HOST_HERMES:
            raise ValueError("SSH_HOST_HERMES / SSH_HOST_ALLIKAS is required for the Allikas OmniRoute host")
        return SSH_HOST_HERMES
    if normalized in trading_roles:
        if not SSH_HOST:
            raise ValueError("SSH_HOST environment variable is required")
        return SSH_HOST
    raise ValueError(f"Unknown SSH role {role!r}; use trading or allikas")


def _get_ssh_opts() -> list[str]:
    opts = [
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
    ]
    key_path = SSH_KEY_PATH
    if not key_path:
        priv_key = os.getenv("SSH_PRIVATE_KEY", "")
        if priv_key:
            kf = tempfile.NamedTemporaryFile(delete=False, mode="w")
            begin_marker = "-----BEGIN OPENSSH PRIVATE KEY-----"
            end_marker = "-----END OPENSSH PRIVATE KEY-----"
            if "\n" not in priv_key and begin_marker in priv_key:
                body = priv_key.replace(begin_marker, "").replace(end_marker, "").strip()
                body = body.replace(" ", "\n")
                kf.write(f"{begin_marker}\n{body}\n{end_marker}\n")
            else:
                kf.write(priv_key + "\n")
            kf.close()
            os.chmod(kf.name, 0o600)
            key_path = kf.name
        else:
            default_key = Path.home() / ".ssh" / "id_vps_bot"
            if default_key.is_file():
                key_path = str(default_key)
    if key_path:
        opts = ["-i", key_path, *opts]
    return opts


def ssh_cmd(
    remote_command: str,
    role: str | None = None,
    *,
    host: str | None = None,
) -> list[str]:
    """Build an ssh argv that runs `remote_command` on the selected VPS."""
    if host is not None:
        if not host:
            raise ValueError("SSH target host is required (set SSH_HOST or pass host=)")
        target_host = host
    else:
        target_host = _host_for_role(role)
    return ["ssh", *_get_ssh_opts(), "-p", SSH_PORT, f"{SSH_USER}@{target_host}", remote_command]


def ssh_cmd_trading(remote_command: str) -> list[str]:
    """SSH to the QuantumTrade trading VPS (SSH_HOST / SSH_HOST_TRADING)."""
    return ssh_cmd(remote_command, role="trading")


def ssh_cmd_hermes(remote_command: str) -> list[str]:
    """SSH to the Allikas / Hermes + OmniRoute VPS (SSH_HOST_HERMES)."""
    return ssh_cmd(remote_command, role="allikas")


def scp_cmd(
    local_path: str,
    remote_path: str,
    role: str | None = None,
    *,
    host: str | None = None,
) -> list[str]:
    """Build an scp argv that copies a local file to `remote_path` on the selected VPS."""
    if host is not None:
        if not host:
            raise ValueError("SSH target host is required (set SSH_HOST or pass host=)")
        target_host = host
    else:
        target_host = _host_for_role(role)
    return [
        "scp",
        *_get_ssh_opts(),
        "-P",
        SSH_PORT,
        str(local_path),
        f"{SSH_USER}@{target_host}:{remote_path}",
    ]
