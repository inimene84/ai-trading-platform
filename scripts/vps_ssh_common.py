"""Shared SSH connection settings for the VPS ops scripts.

Follows the same convention as scripts/ssh_vps_remote.sh and AGENTS.md:
configuration comes from environment variables, with the documented
defaults as fallback.

Env vars:
    SSH_HOST      VPS address (must be provided via environment variable)
    SSH_USER      SSH user (default: root)
    SSH_PORT      SSH port (default: 22)
    SSH_KEY_PATH  Private key file (default: ~/.ssh/id_vps_bot)
"""

from __future__ import annotations

import os
from pathlib import Path

SSH_HOST = os.getenv("SSH_HOST", "")
SSH_USER = os.getenv("SSH_USER", "root")
SSH_PORT = os.getenv("SSH_PORT", "22")
SSH_KEY_PATH = os.getenv("SSH_KEY_PATH", "")

TARGET = f"{SSH_USER}@{SSH_HOST}"

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

def ssh_cmd(remote_command: str) -> list[str]:
    """Build an ssh argv that runs `remote_command` on the VPS."""
    return ["ssh", *_get_ssh_opts(), "-p", SSH_PORT, TARGET, remote_command]




def scp_cmd(local_path: str, remote_path: str) -> list[str]:
    """Build an scp argv that copies a local file to `remote_path` on the VPS."""
    return ["scp", *_get_ssh_opts(), "-P", SSH_PORT, str(local_path), f"{TARGET}:{remote_path}"]

