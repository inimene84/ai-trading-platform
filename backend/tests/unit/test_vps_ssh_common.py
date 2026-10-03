"""Unit tests for two-VPS SSH target routing in scripts/vps_ssh_common.py."""

import importlib
import pytest

import scripts.vps_ssh_common as vps_ssh_common


def test_default_ssh_cmd_stays_on_trading_even_if_ssh_target_role_set(monkeypatch):
    monkeypatch.setenv("SSH_HOST", "trading.example.internal")
    monkeypatch.setenv("SSH_HOST_HERMES", "allikas.example.internal")
    monkeypatch.setenv("SSH_TARGET_ROLE", "allikas")
    importlib.reload(vps_ssh_common)

    argv = vps_ssh_common.ssh_cmd("hostname")
    assert "root@trading.example.internal" in argv
    assert "root@allikas.example.internal" not in argv


def test_explicit_allikas_role_and_hermes_helper_use_hermes_host(monkeypatch):
    monkeypatch.setenv("SSH_HOST", "trading.example.internal")
    monkeypatch.setenv("SSH_HOST_HERMES", "allikas.example.internal")
    importlib.reload(vps_ssh_common)

    argv_role = vps_ssh_common.ssh_cmd("hostname", role="allikas")
    argv_helper = vps_ssh_common.ssh_cmd_hermes("hostname")
    assert "root@allikas.example.internal" in argv_role
    assert "root@allikas.example.internal" in argv_helper


def test_hermes_helper_never_falls_back_to_trading_when_unset(monkeypatch):
    monkeypatch.setenv("SSH_HOST", "trading.example.internal")
    monkeypatch.delenv("SSH_HOST_HERMES", raising=False)
    monkeypatch.delenv("SSH_HOST_ALLIKAS", raising=False)
    monkeypatch.delenv("SSH_HOST_CONSTRUCTION", raising=False)
    importlib.reload(vps_ssh_common)

    with pytest.raises(ValueError, match="SSH_HOST_HERMES"):
        vps_ssh_common.ssh_cmd_hermes("hostname")
