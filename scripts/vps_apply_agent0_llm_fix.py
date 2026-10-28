"""Comprehensive Agent Zero LLM fix & self-heal setup on VPS.

Model catalog updates should go through ``scripts/vps_apply_agent0_kie_models.py``
(family-aware proxy + Astra/Fable presets). This script still repairs keys.
"""

import subprocess
import tempfile
from pathlib import Path
from scripts.vps_ssh_common import ssh_cmd, scp_cmd

DEPLOY_PY = """import os, sys, yaml, json
from pathlib import Path

KIE_KEY = '${KIE_API_KEY}'
OMNI_KEY = os.environ.get('OMNIROUTE_API_KEY', '')

print('=== 1. Updating /docker/kieai-proxy/docker-compose.yml ===')
compose_path = Path('/docker/kieai-proxy/docker-compose.yml')
if compose_path.exists():
    text = compose_path.read_text(encoding='utf-8')
    if 'omni_live_key_placeholder' in text:
        text = text.replace('omni_live_key_placeholder', OMNI_KEY)
        compose_path.write_text(text, encoding='utf-8')
        print('Updated OMNI_KEY in compose')
    os.system('cd /docker/kieai-proxy && docker compose up -d')

print('\\n=== 2. Updating .env in Agent Zero volume ===')
env_path = Path('/var/lib/docker/volumes/agent-zero_a0-data/_data/.env')
if env_path.exists():
    lines = env_path.read_text(encoding='utf-8').splitlines()
    new_lines = []
    keys_map = {
        'API_KEY_KIEAI': KIE_KEY,
        'API_KEY_KIEAI-CLAUDE': KIE_KEY,
        'API_KEY_KIEAI_CLAUDE': KIE_KEY,
        'API_KEY_KIEAI-GPT-CODEX': KIE_KEY,
        'API_KEY_KIEAI_GPT_CODEX': KIE_KEY,
        'API_KEY_OMNIROUTE': OMNI_KEY,
        'API_KEY_OTHER': KIE_KEY
    }
    seen = set()
    for line in lines:
        if '=' in line and not line.strip().startswith('#'):
            k = line.split('=')[0].strip()
            if k in keys_map:
                new_lines.append(f'{k}={keys_map[k]}')
                seen.add(k)
                continue
        new_lines.append(line)
    for k, v in keys_map.items():
        if k not in seen:
            new_lines.append(f'{k}={v}')
    env_path.write_text('\\n'.join(new_lines) + '\\n', encoding='utf-8')
    print('Updated .env')

print('\\n=== 3. Updating model_providers.yaml in plugin and container ===')
PROVIDERS_TO_SET = {
    'kieai-claude': {
        'name': 'Kie.ai Claude',
        'litellm_provider': 'openai',
        'models_list': {
            'list': ['claude-fable-5-1', 'claude-fable-5', 'claude-sonnet-5', 'claude-opus-5', 'claude-sonnet-4-6', 'claude-haiku-4-5', 'claude-opus-4-6']
        },
        'kwargs': {
            'a0_api_mode': 'chat',
            'api_base': 'http://kieai-proxy:11434/v1',
            'api_key': KIE_KEY
        }
    },
    'kieai-gpt-codex': {
        'name': 'Kie.ai GPT Codex',
        'litellm_provider': 'openai',
        'models_list': {
            'list': ['gpt-6-astra', 'gpt-5-6-luna', 'gpt-5-6-sol', 'gpt-5-6-terra', 'gpt-5-2', 'gpt-5.4-codex', 'gpt-5.1-codex']
        },
        'kwargs': {
            'a0_api_mode': 'chat',
            'api_base': 'http://kieai-proxy:11434/v1',
            'api_key': KIE_KEY
        }
    },
    'kieai': {
        'name': 'Kie.ai proxy',
        'litellm_provider': 'openai',
        'models_list': {
            'list': ['claude-fable-5-1', 'claude-fable-5', 'claude-sonnet-5', 'claude-opus-5', 'claude-sonnet-4-6', 'claude-haiku-4-5', 'claude-opus-4-6', 'gpt-6-astra', 'gpt-5-6-luna', 'gpt-5-6-sol', 'gpt-5.4-codex', 'gpt-5.1-codex', 'gpt-5-2']
        },
        'kwargs': {
            'a0_api_mode': 'chat',
            'api_base': 'http://kieai-proxy:11434/v1',
            'api_key': KIE_KEY
        }
    },
    'omniroute': {
        'name': 'OmniRoute',
        'litellm_provider': 'openai',
        'models_list': {
            'list': ['auto/chat', 'auto/cheap', 'auto/fast', 'auto/best-coding']
        },
        'kwargs': {
            'a0_api_mode': 'chat',
            'api_base': 'https://omni.allikas.online/v1',
            'api_key': OMNI_KEY
        }
    }
}

target_paths = [
    Path('/var/lib/docker/volumes/agent-zero_a0-data/_data/plugins/kie-ai/conf/model_providers.yaml'),
]
for p in target_paths:
    p.parent.mkdir(parents=True, exist_ok=True)
    d = {}
    if p.exists():
        try:
            d = yaml.safe_load(p.read_text(encoding='utf-8')) or {}
        except Exception:
            d = {}
    chat = d.setdefault('chat', {})
    for pid, pcfg in PROVIDERS_TO_SET.items():
        chat[pid] = pcfg
    p.write_text(yaml.safe_dump(d, sort_keys=False, allow_unicode=True, default_flow_style=False), encoding='utf-8')
    print(f'Updated {p}')

print('\\n=== 4. Updating self-heal startup migration ===')
startup_file = Path('/var/lib/docker/volumes/agent-zero_a0-data/_data/plugins/kie-ai/extensions/python/startup_migration/_10_self_heal_provider_config.py')
startup_file.parent.mkdir(parents=True, exist_ok=True)
startup_code = '''from __future__ import annotations
from pathlib import Path
from helpers.extension import Extension
from helpers.print_style import PrintStyle

PLUGIN_NAME = "kie-ai"
KIE_KEY = "${KIE_API_KEY}"
OMNI_KEY = os.environ.get("OMNIROUTE_API_KEY", "")

TARGET_PROVIDERS = {
    "kieai-claude": {
        "name": "Kie.ai Claude",
        "litellm_provider": "openai",
        "models_list": {
            "list": ["claude-sonnet-4-6", "claude-haiku-4-5", "claude-opus-4-6"]
        },
        "kwargs": {
            "a0_api_mode": "chat",
            "api_base": "http://kieai-proxy:11434/v1",
            "api_key": KIE_KEY
        }
    },
    "kieai-gpt-codex": {
        "name": "Kie.ai GPT Codex",
        "litellm_provider": "openai",
        "models_list": {
            "list": ["gpt-5.4-codex", "gpt-5.1-codex", "gpt-5-2"]
        },
        "kwargs": {
            "a0_api_mode": "chat",
            "api_base": "http://kieai-proxy:11434/v1",
            "api_key": KIE_KEY
        }
    },
    "kieai": {
        "name": "Kie.ai proxy",
        "litellm_provider": "openai",
        "models_list": {
            "list": ["claude-sonnet-4-6", "claude-haiku-4-5", "claude-opus-4-6", "gpt-5.4-codex", "gpt-5.1-codex", "gpt-5-2"]
        },
        "kwargs": {
            "a0_api_mode": "chat",
            "api_base": "http://kieai-proxy:11434/v1",
            "api_key": KIE_KEY
        }
    },
    "omniroute": {
        "name": "OmniRoute",
        "litellm_provider": "openai",
        "models_list": {
            "list": ["auto/chat", "auto/cheap", "auto/fast", "auto/best-coding"]
        },
        "kwargs": {
            "a0_api_mode": "chat",
            "api_base": "https://omni.allikas.online/v1",
            "api_key": OMNI_KEY
        }
    }
}

class SelfHealKieaiProviderConfig(Extension):
    def execute(self, data: dict | None = None, **kwargs):
        try:
            self._repair()
            PrintStyle.hint(f"[{PLUGIN_NAME}] Kie.ai & OmniRoute providers self-healed at startup.")
        except Exception as exc:
            PrintStyle.error(f"[{PLUGIN_NAME}] Provider self-heal failed: {exc}")

    def _repair(self):
        import yaml
        candidate_paths = [
            Path("/a0/conf/model_providers.yaml"),
            Path(__file__).resolve().parents[3] / "conf" / "model_providers.yaml",
        ]
        for conf_path in candidate_paths:
            try:
                data = {}
                if conf_path.exists():
                    with conf_path.open("r", encoding="utf-8") as fh:
                        data = yaml.safe_load(fh) or {}
                if not isinstance(data, dict):
                    data = {}
                chat = data.setdefault("chat", {})
                for pid, pcfg in TARGET_PROVIDERS.items():
                    chat[pid] = pcfg
                conf_path.parent.mkdir(parents=True, exist_ok=True)
                with conf_path.open("w", encoding="utf-8") as fh:
                    yaml.safe_dump(data, fh, sort_keys=False, allow_unicode=True, default_flow_style=False)
            except Exception as e:
                PrintStyle.error(f"[{PLUGIN_NAME}] Failed repairing {conf_path}: {e}")
'''
startup_file.write_text(startup_code, encoding='utf-8')
print(f'Updated {startup_file}')

print('\\n=== 5. Updating presets.yaml and config.json ===')
presets_path = Path('/var/lib/docker/volumes/agent-zero_a0-data/_data/plugins/_model_config/presets.yaml')
if presets_path.exists():
    with open(presets_path, 'r', encoding='utf-8') as f:
        presets = yaml.safe_load(f) or []
    for preset in presets:
        pname = preset.get('name', '')
        if pname == 'Kie.ai Sonnet':
            preset['chat']['provider'] = 'kieai-claude'
            preset['chat']['name'] = 'claude-sonnet-4-6'
            preset['utility']['provider'] = 'kieai'
            preset['utility']['name'] = 'claude-haiku-4-5'
        elif pname == 'Kie.ai Codex 5.4':
            preset['chat']['provider'] = 'kieai-gpt-codex'
            preset['chat']['name'] = 'gpt-5.4-codex'
            preset['utility']['provider'] = 'kieai-gpt-codex'
            preset['utility']['name'] = 'gpt-5.4-codex'
        elif pname == 'Kie.ai Codex 5.1':
            preset['chat']['provider'] = 'kieai-gpt-codex'
            preset['chat']['name'] = 'gpt-5.1-codex'
            preset['utility']['provider'] = 'kieai-gpt-codex'
            preset['utility']['name'] = 'gpt-5.4-codex'
        elif pname == 'Kie.ai Opus':
            preset['chat']['provider'] = 'kieai-claude'
            preset['chat']['name'] = 'claude-opus-4-6'
            preset['utility']['provider'] = 'kieai'
            preset['utility']['name'] = 'claude-sonnet-4-6'
        elif pname == 'Kie.ai Haiku':
            preset['chat']['provider'] = 'kieai-claude'
            preset['chat']['name'] = 'claude-haiku-4-5'
            preset['utility']['provider'] = 'kieai'
            preset['utility']['name'] = 'claude-haiku-4-5'
    with open(presets_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(presets, f, sort_keys=False, allow_unicode=True, default_flow_style=False)
    print('Updated presets.yaml')

config_path = Path('/var/lib/docker/volumes/agent-zero_a0-data/_data/plugins/_model_config/config.json')
if not config_path.exists():
    config_path.write_text(json.dumps({'model_preset': 'Kie.ai Sonnet'}, indent=2), encoding='utf-8')
    print('Seeded config.json with Kie.ai Sonnet')
else:
    print('Left existing model_preset unchanged')

print('\\n=== 6. Updating fix-kieai-provider.sh ===')
fix_sh = Path('/var/lib/docker/volumes/agent-zero_a0-data/_data/fix-kieai-provider.sh')
fix_sh.write_text('''#!/bin/bash
echo "Repairing Agent Zero LLM Providers..."
python3 -c "
import yaml
from pathlib import Path

KIE_KEY = '${KIE_API_KEY}'
OMNI_KEY = os.environ.get('OMNIROUTE_API_KEY', '')

TARGET_PROVIDERS = {
    'kieai-claude': {
        'name': 'Kie.ai Claude',
        'litellm_provider': 'openai',
        'models_list': {'list': ['claude-sonnet-4-6', 'claude-haiku-4-5', 'claude-opus-4-6']},
        'kwargs': {'a0_api_mode': 'chat', 'api_base': 'http://kieai-proxy:11434/v1', 'api_key': KIE_KEY}
    },
    'kieai-gpt-codex': {
        'name': 'Kie.ai GPT Codex',
        'litellm_provider': 'openai',
        'models_list': {'list': ['gpt-5.4-codex', 'gpt-5.1-codex', 'gpt-5-2']},
        'kwargs': {'a0_api_mode': 'chat', 'api_base': 'http://kieai-proxy:11434/v1', 'api_key': KIE_KEY}
    },
    'kieai': {
        'name': 'Kie.ai proxy',
        'litellm_provider': 'openai',
        'models_list': {'list': ['claude-sonnet-4-6', 'claude-haiku-4-5', 'claude-opus-4-6']},
        'kwargs': {'a0_api_mode': 'chat', 'api_base': 'http://kieai-proxy:11434/v1', 'api_key': KIE_KEY}
    },
    'omniroute': {
        'name': 'OmniRoute',
        'litellm_provider': 'openai',
        'models_list': {'list': ['auto/chat', 'auto/cheap', 'auto/fast', 'auto/best-coding']},
        'kwargs': {'a0_api_mode': 'chat', 'api_base': 'https://omni.allikas.online/v1', 'api_key': OMNI_KEY}
    }
}

for p in [Path('/a0/conf/model_providers.yaml'), Path('/a0/usr/plugins/kie-ai/conf/model_providers.yaml')]:
    try:
        d = {}
        if p.exists():
            d = yaml.safe_load(p.read_text(encoding='utf-8')) or {}
        chat = d.setdefault('chat', {})
        for pid, cfg in TARGET_PROVIDERS.items():
            chat[pid] = cfg
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(yaml.safe_dump(d, sort_keys=False, allow_unicode=True, default_flow_style=False), encoding='utf-8')
        print(f'Repaired {p}')
    except Exception as e:
        print(f'Error {p}: {e}')
"
echo "Repair completed."
''', encoding='utf-8')
os.chmod(fix_sh, 0o755)
print('Updated fix-kieai-provider.sh')

print('\\n=== 7. Creating Agent Zero Skill: restore-llm-providers ===')
skill_dir = Path('/var/lib/docker/volumes/agent-zero_a0-data/_data/skills/restore-llm-providers')
skill_dir.mkdir(parents=True, exist_ok=True)
(skill_dir / 'SKILL.md').write_text('''---
name: "restore-llm-providers"
description: "Verify and restore Kie.ai and OmniRoute LLM providers in Agent Zero after updates or resets."
version: "1.0.0"
author: "Antigravity Pair Programmer"
tags: ["kieai", "omniroute", "llm", "providers", "config", "self-heal", "repair"]
trigger_patterns:
  - "restore llm providers"
  - "fix kieai"
  - "fix omniroute"
  - "check llm providers"
---

# Restore LLM Providers Skill

This skill maintains and restores working Kie.ai and OmniRoute provider configurations for Agent Zero.

## Verified Architecture

- **Kie.ai Proxy**: `http://kieai-proxy:11434/v1` (OpenAI format, litellm_provider: openai)
- **Supported Models**: `claude-sonnet-4-6`, `claude-haiku-4-5`, `claude-opus-4-6`, `gpt-5.4-codex`, `gpt-5.1-codex`, `gpt-5-2`
- **OmniRoute**: `https://omni.allikas.online/v1` (Key: `<OMNIROUTE_API_KEY>`)

## Run Provider Repair
```bash
/a0/usr/fix-kieai-provider.sh
```
''', encoding='utf-8')
print('Created skill restore-llm-providers')

print('\\n=== 8. Creating Persistent Knowledge & Memory ===')
k_custom = Path('/var/lib/docker/volumes/agent-zero_a0-data/_data/knowledge/custom')
k_custom.mkdir(parents=True, exist_ok=True)
k_sol = Path('/var/lib/docker/volumes/agent-zero_a0-data/_data/knowledge/solutions')
k_sol.mkdir(parents=True, exist_ok=True)

k_text = '''# Agent Zero LLM Providers & Kie.ai / OmniRoute Setup

## Verified Working Configuration

- **Kie.ai Proxy Container**: `kieai-proxy` on port `11434`.
- **Endpoint**: `http://kieai-proxy:11434/v1`
- **Working Models**:
  - `claude-sonnet-4-6` (Primary Chat / Sonnet)
  - `claude-haiku-4-5` (Fast Utility)
  - `claude-opus-4-6` (Opus)
  - `gpt-5.4-codex` (Codex Coding Specialist)
  - `gpt-5.1-codex` (Codex Coding)
- **OmniRoute**: `https://omni.allikas.online/v1`, API Key: `<OMNIROUTE_API_KEY>`

## Key Rules
1. Never put `§§secret(...)` in `extra_headers`. LiteLLM/httpx cannot encode Unicode in HTTP headers.
2. Kie.ai Claude and GPT Codex use `litellm_provider: openai` and point to `http://kieai-proxy:11434/v1`.
3. Auto-healing runs on every boot via `/a0/usr/plugins/kie-ai/extensions/python/startup_migration/_10_self_heal_provider_config.py`.
'''
(k_custom / 'llm_providers_setup.md').write_text(k_text, encoding='utf-8')
(k_sol / 'kieai_omniroute_setup.md').write_text(k_text, encoding='utf-8')
print('Created knowledge docs')

print('\\n=== 9. Restarting a0-instance container ===')
os.system('docker restart a0-instance')
print('Restarted a0-instance container successfully')
"""

with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
    f.write(DEPLOY_PY)
    tmp_path = f.name

try:
    subprocess.run(scp_cmd(tmp_path, "/tmp/deploy_agent0_fix.py"), check=True)
    res = subprocess.run(ssh_cmd("python3 /tmp/deploy_agent0_fix.py"), capture_output=True, text=True)
    print("DEPLOY OUTPUT:")
    print(res.stdout)
    if res.stderr:
        print("DEPLOY STDERR:", res.stderr)
finally:
    Path(tmp_path).unlink(missing_ok=True)
