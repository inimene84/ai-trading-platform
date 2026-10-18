# Hostinger VPS quick fix (<vps-ip>)

## hPanel Node.js env (8192-character cap)

Hostinger **website** Node.js / hPanel "Environment file" is capped at **8192 characters total**. That is not the VPS Docker limit.

- **Do not** paste `/root/ai-trading-platform-v3/.env` or `.env.example` into hPanel. Those files are ~9k–23k characters and the Save/Deploy button will refuse them.
- Live QuantumTrade stays on the VPS. Keep trading secrets, exchange keys, and `JESSE_SYNC_TO_LIVE=false` in the VPS `.env` only.
- If you are deploying a Node/static frontend on shared/cloud hosting (`allikas.online`, `thorspark.cloud`, or a new Node site), paste only `.env.hostinger` / `.env.hostinger.example` (frontend `VITE_*` + `NODE_ENV`). That payload is a few hundred characters.
- Existing Node sites currently have **no** panel env set. Leave them empty unless that site actually needs the Vite keys.

VPS deploy path is unchanged: SSH + `scripts/hostinger_vps_apply.sh` (below). Do not use hPanel Node Save for the Docker stack.

## Canonical deploy (main branch)

Run on the VPS as root (Hostinger browser terminal or SSH):

```bash
cd /root/ai-trading-platform-v3
git pull origin main
chmod +x scripts/hostinger_vps_apply.sh scripts/lib/vps_ssh_hygiene.sh scripts/ensure_influx_buckets.sh
./scripts/hostinger_vps_apply.sh
```

This starts **backend, litellm, nginx, influxdb, grafana, qdrant** and builds **frontend/dist** for the dashboard on `:8081`.

Ensure `.env` has `INFLUXDB_TOKEN` set (required for metrics + Grafana datasources).

One-liner from anywhere:

```bash
curl -fsSL "https://raw.githubusercontent.com/inimene84/ai-trading-platform./main/scripts/vps_bootstrap_and_deploy.sh" | bash
```

From your PC (when SSH works):

```powershell
ssh root@<vps-ip> "cd /root/ai-trading-platform-v3 && git pull origin main && ./scripts/hostinger_vps_apply.sh"
```

## Deprecated scripts

Do **not** use old branch-specific scripts (`cursor/observability-env-vars-f7d8`, etc.).
`vps_deploy_p0_oneliner.sh` and `vps_deploy_risk_reviewer_fix.sh` now forward to `hostinger_vps_apply.sh`.

## Security (recommended)

Set `ADMIN_API_KEY` in VPS `.env` to protect loop start/stop and config writes.
Add the same value in dashboard **Settings** as `ADMIN_API_KEY` so POST requests include `X-API-Key`.
Restrict port **8001** to localhost when nginx proxies the dashboard on **8081**.

## Cloud Agent SSH

Add the cloud-agent public key to `/root/.ssh/authorized_keys`, then run `./scripts/ssh_vps_remote.sh`.
