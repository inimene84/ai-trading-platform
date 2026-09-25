"""Shared SSH connection settings for the VPS ops scripts.

Follows the same convention as scripts/ssh_vps_remote.sh and AGENTS.md:
configuration comes from environment variables, with the documented
defaults as fallback.

Env vars:
    SSH_HOST              Trading VPS address (default target)
    SSH_HOST_TRADING      Alias for SSH_HOST (trading / QuantumTrade)
    SSH_HOST_HERMES       Allikas / Hermes + OmniRoute VPS (76.13.78.71)
    SSH_HOST_CONSTRUCTION Alias for SSH_HOST_HERMES
    SSH_USER              SSH user (default: root)
    SSH_PORT              SSH port (default: 22)
    SSH_KEY_PATH          Private key file (default: ~/.ssh/id_vps_bot)
"""

from __future__ import annotations

import os
from pathlib import Path

SSH_USER = os.getenv("SSH_USER", "root")
SSH_PORT = os.getenv("SSH_PORT", "22")
SSH_KEY_PATH = os.getenv("SSH_KEY_PATH", "")

SSH_HOST_TRADING = os.getenv("SSH_HOST_TRADING", os.getenv("SSH_HOST", ""))
SSH_HOST_HERMES = os.getenv(
    "SSH_HOST_HERMES",
    os.getenv("SSH_HOST_CONSTRUCTION", "76.13.78.71"),
)
# Back-compat: SSH_HOST always means the trading VPS.
SSH_HOST = SSH_HOST_TRADING

TARGET_TRADING = f"{SSH_USER}@{SSH_HOST_TRADING}" if SSH_HOST_TRADING else ""
TARGET_HERMES = f"{SSH_USER}@{SSH_HOST_HERMES}" if SSH_HOST_HERMES else ""
# Legacy alias used by older scripts.
TARGET = TARGET_TRADING


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
            import tempfile

            kf = tempfile.NamedTemporaryFile(delete=False, mode="w")
            begin_marker = "-----BEGIN OPENSSH PRIVATE KEY-----"
            end_marker = "-----END OPENSSH PRIVATE KEY-----"
            if "\n" not in priv_key and begin_marker in priv_key:
                body = priv_key.replace(begin_marker, "").replace(end_marker, "").strip()
                body = "\n".join(body.split(" "))
                formatted = f"{begin_marker}\n{body}\n{end_marker}\n"
                kf.write(formatted)
            else:
                kf.write(priv_key)
            kf.close()
            os.chmod(kf.name, 0o600)
            key_path = kf.name
        else:
            default_key = Path.home() / ".ssh" / "id_vps_bot"
            if default_key.exists():
                key_path = str(default_key)

    if key_path:
        opts = ["-i", key_path, *opts]
    return opts


def ssh_cmd(remote_command: str, host: str | None = None) -> list[str]:
    """Build an ssh argv that runs `remote_command` on the target VPS.

    Args:
        remote_command: Shell command to run on the remote host.
        host: Optional override host/IP. Defaults to SSH_HOST (trading VPS).
    """
    target_host = host if host is not None else SSH_HOST_TRADING
    if not target_host:
        raise ValueError("SSH target host is required (set SSH_HOST or pass host=)")
    target = f"{SSH_USER}@{target_host}"
    return ["ssh", *_get_ssh_opts(), "-p", SSH_PORT, target, remote_command]


def ssh_cmd_trading(remote_command: str) -> list[str]:
    """SSH to the QuantumTrade trading VPS (SSH_HOST / SSH_HOST_TRADING)."""
    return ssh_cmd(remote_command, host=SSH_HOST_TRADING)


def ssh_cmd_hermes(remote_command: str) -> list[str]:
    """SSH to the Allikas / Hermes + OmniRoute VPS (SSH_HOST_HERMES)."""
    return ssh_cmd(remote_command, host=SSH_HOST_HERMES)


def scp_cmd(local_path: str, remote_path: str, host: str | None = None) -> list[str]:
    """Build an scp argv that copies a local file to `remote_path` on the VPS."""
    target_host = host or SSH_HOST_TRADING
    if not target_host:
        raise ValueError("SSH_HOST / SSH_HOST_TRADING is required")
    target = f"{SSH_USER}@{target_host}"
    return ["scp", *_get_ssh_opts(), "-P", SSH_PORT, str(local_path), f"{target}:{remote_path}"]
