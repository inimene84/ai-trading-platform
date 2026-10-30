# Secret Remediation & Prevention

GitGuardian flagged hardcoded secrets in this repo (incidents surfaced on PR #29).
The **working tree is now clean** — all live values were replaced with env-var
references. But the old values **remain in git history**, so GitGuardian keeps the
incidents in `Triggered` state until each underlying secret is **rotated** and the
incident is closed on the dashboard.

This doc tracks that cleanup and the prevention now in place.

## 1. Flagged incidents (rotate, then resolve on the dashboard)

| GitGuardian id | Secret | Where it leaked | Rotate action | Status |
| --- | --- | --- | --- | --- |
| 34132442 | **Kie AI API Key** | `scratch/test_kie_endpoints.py` | Revoke + reissue in the Kie.ai dashboard; set `KIE_API_KEY` in `.env` / VPS | ✅ rotated (commit 3982467); close incident |
| 34132439 | InfluxDB token | `docs/ops/fix_influxdb_complete.sh` | Rotate the InfluxDB admin token; set `INFLUXDB_TOKEN` | ☐ rotate + close |
| 34132443 / 34133184 | InfluxDB admin password (`hedgefund123`) | `docker-compose.yml`, ops scripts | Change the InfluxDB admin password; set `INFLUXDB_ADMIN_PASSWORD` | ☐ rotate + close |
| 34132441 | docker-compose.prod.yml user/password | `docker-compose.prod.yml` | Rotate that credential; move to env | ☐ verify + close |

> The "Generic Password" occurrences under 34133184 on commit `22a670f` are the
> **old** `hedgefund123` value still visible in that commit's diff — they are
> resolved by rotating the InfluxDB password above, not by another code change.

### How to close an incident
1. Rotate the underlying secret at its source (Kie.ai, InfluxDB, etc.).
2. Update `.env` locally and the matching env var on the VPS; restart affected
   containers (`docker compose up -d`).
3. On the GitGuardian dashboard, open the incident → mark **Resolved**
   (or **Revoked**) once the old value no longer works.

### Optional: purge from history
Only if you want the values gone from old commits too. **Risky on a shared repo**
(rewrites every contributor's history). Use `git filter-repo` or BFG, then
force-push and have everyone re-clone. Rotation above is the safer, sufficient fix.

## 2. Prevention now in place

- **`.pre-commit-config.yaml`** — `ggshield` + `detect-secrets` run on every
  commit locally and block secrets before they leave your machine.
- **`.secrets.baseline`** — vetted set of known non-secrets (env-var references,
  alembic revision hashes, the `sk-1234` LiteLLM placeholder) so the hook only
  fails on *new* findings.
- **`.github/workflows/secret-scan.yml`** — server-side scan on every push/PR.
- **`.gitignore`** — `scratch/`, `*.pem`, `*.key`, `secrets.*` are ignored.

### One-time developer setup
```bash
pip install pre-commit ggshield detect-secrets
pre-commit install                 # enable the git hook
ggshield auth login                # or export GITGUARDIAN_API_KEY=...
pre-commit run --all-files         # sanity-check the whole repo
```

### CI setup
Add a repo secret **`GITGUARDIAN_API_KEY`** (GitGuardian dashboard → API → tokens)
under **Settings → Secrets and variables → Actions**. The `detect-secrets` job
runs offline and needs no secret.

## 3. Golden rule
Never hardcode a credential. Read it from the environment (`os.getenv(...)`,
`${VAR}` in compose) and keep real values only in `.env` (gitignored) and on the
VPS. `.env.example` should contain placeholders only.
