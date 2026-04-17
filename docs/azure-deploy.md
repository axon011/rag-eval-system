# Deploy to Azure Container Apps

This guide walks through getting `rag-eval-system` running on Azure using Container Apps + Container Registry. The app runs as a single container with Qdrant in in-memory mode and sentence-transformers baked into the image — no external ML services required.

## Why Container Apps

- Handles scale-to-zero (cheaper than App Service for bursty demo traffic)
- Built-in HTTPS + revision management
- Tolerates the 30-second cold start our sentence-transformers model needs on first boot

If you prefer App Service, the image is compatible — just swap the `az containerapp` commands for `az webapp` equivalents.

## One-time setup

```bash
# 1. Variables
RG=rag-eval-rg
LOCATION=westeurope
ACR=ragevalacr$RANDOM           # must be globally unique, alphanumeric
ENV=rag-eval-env
APP=rag-eval-app

# 2. Resource group + ACR
az group create -n $RG -l $LOCATION
az acr create -n $ACR -g $RG --sku Basic --admin-enabled true

# 3. Container Apps environment
az containerapp env create -n $ENV -g $RG -l $LOCATION

# 4. Build and push the image
az acr build -r $ACR -t $ACR.azurecr.io/rag-eval-system:v1 .

# 5. Create the app (first deploy)
az containerapp create \
  -n $APP -g $RG --environment $ENV \
  --image $ACR.azurecr.io/rag-eval-system:v1 \
  --registry-server $ACR.azurecr.io \
  --target-port 8000 --ingress external \
  --min-replicas 0 --max-replicas 2 \
  --cpu 1.0 --memory 2.0Gi \
  --secrets llm-api-key=$LLM_API_KEY \
  --env-vars \
    LLM_PROVIDER=openai \
    LLM_MODEL=gpt-4o-mini \
    LLM_API_KEY=secretref:llm-api-key \
    OPENAI_BASE_URL=https://api.openai.com/v1 \
    EMBED_PROVIDER=sentence-transformers \
    EMBED_MODEL=all-MiniLM-L6-v2 \
    QDRANT_MODE=memory \
    RETRIEVAL_MODE=hybrid \
    CORS_ORIGINS=*
```

The app URL shows up in the `create` output; `/health` should return `200` within a minute.

## Updating

```bash
az acr build -r $ACR -t $ACR.azurecr.io/rag-eval-system:v2 .
az containerapp update -n $APP -g $RG --image $ACR.azurecr.io/rag-eval-system:v2
```

## Automated deploys via GitHub Actions

`.github/workflows/deploy-azure.yml` ships on every push to `master`/`main` that touches app code. You need these GitHub repo secrets:

- `AZURE_CREDENTIALS` — service principal JSON from `az ad sp create-for-rbac --sdk-auth --role contributor --scopes /subscriptions/<sub>/resourceGroups/rag-eval-rg`
- `AZURE_CONTAINER_REGISTRY` — the ACR name (e.g. `ragevalacr1234`)
- `AZURE_RESOURCE_GROUP` — `rag-eval-rg`
- `AZURE_CONTAINER_APP` — `rag-eval-app`

LLM API keys belong in Container Apps secrets (`az containerapp secret set`), not in GitHub.

## Cost sanity check

With scale-to-zero + Basic ACR + ~10 queries/day, the whole setup runs under €10/month. Bump `min-replicas` to 1 only if you need instant responses — that turns cold starts into zero but costs ~€30/month for the always-on instance.

## Azure-specific gotchas

- **Cold start**: first request after idle triggers container spin-up (~15s). Hitting `/health` from a cron keeps it warm if needed.
- **Ephemeral storage**: in-memory Qdrant means ingested data is lost on revision swap or scale-down. For persistent RAG, either mount an Azure File Share for Qdrant's data dir or point at a managed Qdrant cloud instance.
- **MLflow tracking**: `file:///app/mlruns` writes are lost on restart. Use an Azure Storage-backed MLflow server for real experiment tracking.
- **Image size**: sentence-transformers + torch makes the image ~2GB. Container Apps pulls it once per revision, so deploys take a minute.
