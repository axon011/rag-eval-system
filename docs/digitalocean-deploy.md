# Deploy to DigitalOcean App Platform

Single-container deploy of `rag-eval-system` on DO App Platform. Ships a live HTTPS URL you can link from your resume/GitHub. Costs ~$5-12/month on the basic plan.

## Why App Platform

- Auto-deploys from GitHub on every push to `master`
- HTTPS + custom domain handled for you
- Handles the ~2GB sentence-transformers image without fuss
- Minimal config — one YAML file vs Azure's multi-step `az` dance

Kubernetes (DOKS) and raw Droplets are also options; Droplets give you a cheaper $4/month bill but more ops work. App Platform is the fastest path to a live URL.

## One-time setup

### 1. Install doctl (one-time)

```bash
# Windows (via scoop)
scoop install doctl

# macOS
brew install doctl

# Linux
snap install doctl
```

Authenticate:

```bash
doctl auth init
# paste the API token from https://cloud.digitalocean.com/account/api/tokens
```

### 2. Add the app spec

The repo already contains `.do/app.yaml` — adjust the `repo` field to point at your GitHub fork if different, then:

```bash
# Create app (first time)
doctl apps create --spec .do/app.yaml

# Grab the app ID for later updates
doctl apps list
```

### 3. Set secrets

```bash
# Grab the app ID from `doctl apps list`
APP_ID=<your-app-id>

doctl apps update $APP_ID --spec .do/app.yaml
# Then set the secret separately via UI or:
doctl apps config set $APP_ID LLM_API_KEY=<your-key> --scope run
```

Or set secrets via the DO UI under App → Settings → App-Level Environment Variables.

## What you get

- Live HTTPS URL like `https://rag-eval-abc123.ondigitalocean.app`
- Auto-deploys on every push to `master`
- Logs streamed to `doctl apps logs $APP_ID --follow`
- Health checks against `/health`

## Cost sanity check

- Basic-XXS instance (512 MB RAM): $5/month
- Basic-S (1 GB RAM, what you probably need for sentence-transformers): $12/month
- Container registry included, no extra cost

If the image exceeds 1 GB RAM at idle (sentence-transformers + torch can spike), bump to `basic-s`. Scale-to-zero isn't supported on App Platform; if you need that, switch to a Droplet with a cron that suspends on idle.

## Updating

Push to `master`. App Platform rebuilds and deploys automatically. No extra commands needed.

## Gotchas

- **First build is slow** (~8 minutes) — sentence-transformers is ~1.5 GB. Subsequent builds use cached layers.
- **Ephemeral storage** — in-memory Qdrant means ingested data is lost on redeploy. For persistent RAG, add a DO Managed Postgres or mount a Volume.
- **RAM matters more than CPU** — sentence-transformers + FastAPI live around 700 MB resident. Start with `basic-s` (1 GB) to avoid OOM kills.
- **Region** — pick `fra1` (Frankfurt) for lowest latency from Germany.
