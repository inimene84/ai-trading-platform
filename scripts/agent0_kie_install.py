"""Apply Kie family models to Agent Zero volumes. Run on the VPS."""

from __future__ import annotations

import json
import stat
import sys
from pathlib import Path

import yaml

from agent0_kie_catalog import AGENT0_KIE_PRESETS, ALL_KIE_MODELS, build_providers

A0 = Path("/var/lib/docker/volumes/agent-zero_a0-data/_data")
PROVIDERS_PATH = A0 / "plugins" / "kie-ai" / "conf" / "model_providers.yaml"
PRESETS_PATH = A0 / "plugins" / "_model_config" / "presets.yaml"
CONFIG_PATH = A0 / "plugins" / "_model_config" / "config.json"
SELF_HEAL = A0 / "plugins" / "kie-ai" / "extensions" / "python" / "startup_migration" / "_10_self_heal_provider_config.py"
FIX_SH = A0 / "fix-kieai-provider.sh"
SKILL_MD = A0 / "skills" / "restore-llm-providers" / "SKILL.md"
KNOWLEDGE = A0 / "knowledge" / "custom" / "llm_providers_setup.md"


def _read_env_key(env_path: Path, name: str) -> str:
    if not env_path.exists():
        return ""
    prefix = f"{name}="
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if line.startswith(prefix):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def upsert_presets(existing: list, additions: list[dict]) -> list:
    by_name = {p.get("name"): i for i, p in enumerate(existing) if isinstance(p, dict)}
    for preset in additions:
        name = preset["name"]
        if name in by_name:
            current = existing[by_name[name]]
            current["chat"] = preset["chat"]
            current["utility"] = preset["utility"]
        else:
            existing.append(dict(preset))
    return existing


def providers_without_secrets(providers: dict) -> dict:
    out: dict = {}
    for pid, pcfg in providers.items():
        copied = dict(pcfg)
        kwargs = dict(pcfg.get("kwargs") or {})
        kwargs.pop("api_key", None)
        copied["kwargs"] = kwargs
        out[pid] = copied
    return out


def write_self_heal(providers: dict) -> None:
    SELF_HEAL.parent.mkdir(parents=True, exist_ok=True)
    public = providers_without_secrets(providers)
    SELF_HEAL.write_text(
        f'''from __future__ import annotations
import os
from pathlib import Path
import yaml
from helpers.extension import Extension
from helpers.print_style import PrintStyle

PLUGIN_NAME = "kie-ai"
TARGET_PROVIDERS = {public!r}

def _key_from_env(names: list[str]) -> str:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    for path in (Path("/a0/usr/.env"), Path("/a0/.env")):
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" not in line or line.strip().startswith("#"):
                continue
            key, value = line.split("=", 1)
            if key.strip() in names:
                cleaned = value.strip().strip('"').strip("'")
                if cleaned:
                    return cleaned
    return ""

def _with_runtime_keys(providers: dict) -> dict:
    kie = _key_from_env(["API_KEY_KIEAI", "KIE_API_KEY"])
    omni = _key_from_env(["API_KEY_OMNIROUTE", "OMNIROUTE_API_KEY"])
    filled: dict = {{}}
    for pid, pcfg in providers.items():
        item = dict(pcfg)
        kwargs = dict(pcfg.get("kwargs") or {{}})
        if pid == "omniroute":
            if not omni:
                continue
            kwargs["api_key"] = omni
        else:
            kwargs["api_key"] = kie
        item["kwargs"] = kwargs
        filled[pid] = item
    return filled

class SelfHealKieaiProviderConfig(Extension):
    def execute(self, data: dict | None = None, **kwargs):
        try:
            self._repair()
            PrintStyle.hint(f"[{{PLUGIN_NAME}}] Kie.ai providers self-healed at startup.")
        except Exception as exc:
            PrintStyle.error(f"[{{PLUGIN_NAME}}] Provider self-heal failed: {{exc}}")

    def _repair(self):
        ready = _with_runtime_keys(TARGET_PROVIDERS)
        candidate_paths = [
            Path("/a0/conf/model_providers.yaml"),
            Path(__file__).resolve().parents[3] / "conf" / "model_providers.yaml",
        ]
        for conf_path in candidate_paths:
            try:
                data = {{}}
                if conf_path.exists():
                    with conf_path.open("r", encoding="utf-8") as fh:
                        data = yaml.safe_load(fh) or {{}}
                if not isinstance(data, dict):
                    data = {{}}
                chat = data.setdefault("chat", {{}})
                for pid, pcfg in ready.items():
                    chat[pid] = pcfg
                conf_path.parent.mkdir(parents=True, exist_ok=True)
                with conf_path.open("w", encoding="utf-8") as fh:
                    yaml.safe_dump(data, fh, sort_keys=False, allow_unicode=True, default_flow_style=False)
            except Exception as exc:
                PrintStyle.error(f"[{{PLUGIN_NAME}}] Failed repairing {{conf_path}}: {{exc}}")
''',
        encoding="utf-8",
    )


def write_fix_script() -> None:
    FIX_SH.write_text(
        """#!/bin/bash
echo "Repairing Agent Zero LLM Providers..."
python3 - <<'PY'
from pathlib import Path
import yaml
p = Path('/a0/usr/plugins/kie-ai/conf/model_providers.yaml')
print('providers file', p, 'exists', p.exists())
PY
echo "Use the startup self-heal; this stub is kept for the restore skill."
""",
        encoding="utf-8",
    )
    FIX_SH.chmod(FIX_SH.stat().st_mode | stat.S_IEXEC)


def main() -> int:
    a0_env = A0 / ".env"
    host_env = Path("/root/ai-trading-platform-v3/.env")
    kie_key = _read_env_key(a0_env, "API_KEY_KIEAI") or _read_env_key(host_env, "KIE_API_KEY")
    omni_key = _read_env_key(a0_env, "API_KEY_OMNIROUTE") or _read_env_key(host_env, "OMNIROUTE_API_KEY")
    if not kie_key:
        print("ERROR: no Kie API key in Agent Zero or QT .env", file=sys.stderr)
        return 1

    providers = build_providers(kie_key, omni_key)
    if not omni_key:
        providers.pop("omniroute", None)
    PROVIDERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    data: dict = {}
    if PROVIDERS_PATH.exists():
        data = yaml.safe_load(PROVIDERS_PATH.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        data = {}
    chat = data.setdefault("chat", {})
    for pid, pcfg in providers.items():
        chat[pid] = pcfg
    PROVIDERS_PATH.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True, default_flow_style=False),
        encoding="utf-8",
    )
    print(f"Updated {PROVIDERS_PATH}")

    presets: list = []
    if PRESETS_PATH.exists():
        loaded = yaml.safe_load(PRESETS_PATH.read_text(encoding="utf-8")) or []
        if isinstance(loaded, list):
            presets = loaded
    PRESETS_PATH.parent.mkdir(parents=True, exist_ok=True)
    upsert_presets(presets, AGENT0_KIE_PRESETS)
    PRESETS_PATH.write_text(
        yaml.safe_dump(presets, sort_keys=False, allow_unicode=True, default_flow_style=False),
        encoding="utf-8",
    )
    print(f"Updated {PRESETS_PATH} ({len(AGENT0_KIE_PRESETS)} Kie presets upserted)")

    # Keep the operator's current preset; only seed if missing.
    if not CONFIG_PATH.exists():
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(json.dumps({"model_preset": "OmniRoute Auto"}, indent=2), encoding="utf-8")
        print("Seeded config.json with OmniRoute Auto")
    else:
        try:
            current = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            print("Left active preset unchanged:", current.get("model_preset"))
        except json.JSONDecodeError:
            print("Left existing config.json unchanged")

    write_self_heal(providers)
    print(f"Updated {SELF_HEAL}")
    write_fix_script()

    SKILL_MD.parent.mkdir(parents=True, exist_ok=True)
    SKILL_MD.write_text(
        """---
name: "restore-llm-providers"
description: "Verify and restore Kie.ai family providers (Astra, Fable, GPT 5.6, Claude 5) in Agent Zero."
version: "1.1.0"
---

# Restore LLM Providers

Kie.ai families go through `http://kieai-proxy:11434/v1` which maps each model
id onto the documented host path (Claude messages, Codex/Grok responses, Gemini chat).

Models include `gpt-6-astra`, `claude-fable-5-1`, `claude-fable-5`, `gpt-5-6-luna`,
`claude-sonnet-5`, `claude-opus-5`, Codex, Gemini, and Grok.
""",
        encoding="utf-8",
    )
    KNOWLEDGE.parent.mkdir(parents=True, exist_ok=True)
    KNOWLEDGE.write_text(
        "# Agent Zero Kie.ai models\n\n"
        "Proxy: `http://kieai-proxy:11434/v1` (family-aware).\n\n"
        "Models:\n" + "\n".join(f"- `{m}`" for m in ALL_KIE_MODELS) + "\n",
        encoding="utf-8",
    )
    print("Updated skill + knowledge")

    # Mirror API keys without printing them.
    if a0_env.exists():
        lines = a0_env.read_text(encoding="utf-8").splitlines()
        keys_map = {
            "API_KEY_KIEAI": kie_key,
            "API_KEY_KIEAI-CLAUDE": kie_key,
            "API_KEY_KIEAI-GPT-CODEX": kie_key,
            "API_KEY_KIEAI-GPT": kie_key,
            "API_KEY_KIEAI-GEMINI": kie_key,
            "API_KEY_KIEAI-GROK": kie_key,
        }
        if omni_key:
            keys_map["API_KEY_OMNIROUTE"] = omni_key
        seen: set[str] = set()
        out: list[str] = []
        for line in lines:
            if "=" in line and not line.strip().startswith("#"):
                key = line.split("=", 1)[0].strip()
                if key in keys_map:
                    out.append(f"{key}={keys_map[key]}")
                    seen.add(key)
                    continue
            out.append(line)
        for key, value in keys_map.items():
            if key not in seen and value:
                out.append(f"{key}={value}")
        a0_env.write_text("\n".join(out) + "\n", encoding="utf-8")
        print("Updated Agent Zero .env key slots")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
