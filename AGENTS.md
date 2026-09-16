# Cloud Agent Instructions

## VPS Access & Deployment Guidelines

Deployments should be executed via standard CI/CD pipelines or automated deploy hooks.
Direct SSH access requires private keys configured in secure runner environments.

### Required Secrets
- `SSH_HOST`: Target server hostname or IP address (configured in secure environment variables only)
- `SSH_USER`: Deployment user (use a dedicated non-root deploy user with restricted Docker permissions)
- `SSH_PRIVATE_KEY`: Deployment key with pass-phrase protection

Never commit credentials, private keys, or raw IP addresses into git-tracked repositories.

### VPS Services

The VPS runs these Docker containers:
- `ai-trading-backend` - FastAPI backend on port 8001
- `ai-trading-nginx` - Reverse proxy on port 8081
- `ai-trading-litellm` - LLM proxy
- `ai-trading-redis` - Cache
- `vps-influxdb` - Time series DB
- `vps-qdrant` - Vector DB
- `grafana-*` - Monitoring
- `n8n` - Workflows at `/docker/n8n` (`127.0.0.1:5678`). Assistant sandbox overlay: `docker-compose.sandbox.yml` (privileged runner, not published). Search: `http://ai-trading-searxng:8080`.

### Deployment

Run `./scripts/ssh_vps_remote.sh` to deploy latest changes from main branch to VPS.

## Cursor Cloud Specific Instructions

### Key Normalization

The `SSH_PRIVATE_KEY` secret may be stored as a single line with spaces instead of newlines. The `scripts/ssh_vps_remote.sh` script automatically reformats OpenSSH keys before use.

### Testing Backend API

The backend is accessible at `http://<SSH_HOST>:8001`:
```bash
curl -sf "http://${SSH_HOST}:8001/health"
curl -sf "http://${SSH_HOST}:8001/openapi.json" | head -c 500
```
