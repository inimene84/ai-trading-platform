"""Shared SSH connection settings for the VPS ops scripts.

Follows the same convention as scripts/ssh_vps_remote.sh and AGENTS.md:
configuration comes from environment variables, with the documented
defaults as fallback.

Env vars:
    SSH_HOST             Trading VPS (default target)
    SSH_HOST_HERMES      Allikas / OmniRoute / Hermes VPS
    SSH_HOST_ALLIKAS     alias for SSH_HOST_HERMES
    SSH_HOST_CONSTRUCTION alias for SSH_HOST_HERMES
    SSH_USER             SSH user (default: root)
    SSH_PORT             SSH port (default: 22)
    SSH_KEY_PATH         Private key file (default: ~/.ssh/id_vps_bot)
    SSH_TARGET_ROLE      trading | allikas | hermes | construction
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

SSH_HOST = os.getenv("SSH_HOST", "")
SSH_HOST_HERMES = os.getenv("SSH_HOST_HERMES") or os.getenv("SSH_HOST_ALLIKAS") or os.getenv("SSH_HOST_CONSTRUCTION") or ""
SSH_USER = os.getenv("SSH_USER", "root")
SSH_PORT = os.getenv("SSH_PORT", "22")
SSH_KEY_PATH = os.getenv("SSH_KEY_PATH", "")


def _host_for_role(role: str | None = None) -> str:
    role = (role or os.getenv("SSH_TARGET_ROLE") or "trading").strip().lower()
    if role in {"allikas", "hermes", "construction", "omniroute", "omni"}:
        if not SSH_HOST_HERMES:
            raise ValueError("SSH_HOST_HERMES / SSH_HOST_ALLIKAS is required for the Allikas OmniRoute host")
        return SSH_HOST_HERMES
    if not SSH_HOST:
        raise ValueError("SSH_HOST environment variable is required")
    return SSH_HOST


TARGET = f"{SSH_USER}@{SSH_HOST}" if SSH_HOST else ""

def _get_ssh_opts() -> list[str]:
    opts = [
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=accept-new",
    ]
    key_path = SSH_KEY_PATH
    if not key_path:
        # Check SSH_PRIVATE_KEY env var
        priv_key = os.getenv("SSH_PRIVATE_KEY", "")
        if priv_key:
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

def ssh_cmd(remote_command: str, role: str | None = None) -> list[str]:
    """Build an ssh argv that runs `remote_command` on the selected VPS."""
    host = _host_for_role(role)
    return ["ssh", *_get_ssh_opts(), "-p", SSH_PORT, f"{SSH_USER}@{host}", remote_command]


def scp_cmd(local_path: str, remote_path: str, role: str | None = None) -> list[str]:
    """Build an scp argv that copies a local file to `remote_path` on the selected VPS."""
    host = _host_for_role(role)
    return ["scp", *_get_ssh_opts(), "-P", SSH_PORT, str(local_path), f"{SSH_USER}@{host}:{remote_path}"]

