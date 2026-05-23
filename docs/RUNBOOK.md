# RUNBOOK — Maintainer's Copilot

## Starting the Stack

```bash
cp .env.example .env          # Add VAULT_ROOT_TOKEN=dev-token-local
docker compose -f docker/docker-compose.yml --env-file .env up -d
```

Services come up in order: `db` → `redis` → `vault` → `vault-init` → `minio` → `migrate` → `api` → `model-server` → `chatbot` → `widget` → hosts.

## Service Ports

| Service | Port | URL |
|---------|------|-----|
| API | 8000 | http://localhost:8000 |
| API docs | 8000 | http://localhost:8000/docs |
| Streamlit UI | 8501 | http://localhost:8501 |
| Widget nginx | 8081 | http://localhost:8081 |
| Model-server | 8001 | http://localhost:8001 |
| Allowed host | 8090 | http://localhost:8090 |
| Disallowed host | 8091 | http://localhost:8091 |
| Postgres | 5432 | postgresql://copilot:copilot@localhost:5432/copilot |
| pgAdmin | 5050 | http://localhost:5050 |
| Redis | 6379 | redis://localhost:6379 |
| MinIO API | 9000 | http://localhost:9000 |
| MinIO Console | 9001 | http://localhost:9001 |
| Vault | 8200 | http://localhost:8200 |
| Langfuse | 3000 | http://localhost:3000 |
| Ollama | 11434 | http://localhost:11434 |

## Where Logs Live

```bash
docker compose -f docker/docker-compose.yml --env-file .env logs api
docker compose -f docker/docker-compose.yml --env-file .env logs model-server
docker compose -f docker/docker-compose.yml --env-file .env logs chatbot
```

All application logs are JSON-structured via structlog. Every line carries `trace_id` and `request_id`.

## Reading a Trace in Langfuse

1. Open http://localhost:3000 (Langfuse)
2. Login with `admin@dev.local` / `admin`
3. Navigate to **Traces** → pick the conversation trace
4. The trace tree shows: root user message → tool calls (classify, rag_search, etc.) → final response
5. Each span shows model, token counts, latency, and redacted inputs/outputs

## Rerunning an Eval

```bash
# Classification eval
cd backend && uv run pytest tests/eval/ -m eval -v

# RAG eval (requires running services)
cd backend && uv run pytest tests/eval/test_rag_thresholds.py -m eval -v
```

Thresholds are in `backend/eval/eval_thresholds.yaml`. Results are written to MinIO: `copilot/eval_reports/`.

## Corpus Ingestion

To (re-)ingest MONAI issues into the RAG corpus:

```bash
docker cp ml/data/raw_issues.jsonl docker-api-1:/app/scripts/raw_issues.jsonl
docker cp ml/data/train.jsonl docker-api-1:/app/scripts/train.jsonl
docker exec -e DATABASE_URL="postgresql+asyncpg://copilot:copilot@db:5432/copilot" \
            -e OLLAMA_HOST="http://ollama:11434" \
            docker-api-1 python scripts/ingest_corpus.py
```

## When Vault Becomes Unreachable Mid-Flight

**Symptoms:** New requests fail with `vault_secret_missing`; running requests continue from the cached settings singleton.

**Mitigation:**
1. `get_settings()` is `lru_cache(maxsize=1)` — the running process keeps functioning with cached secrets until the next process restart.
2. Restart Vault: `docker compose restart vault` → vault-init re-seeds automatically.
3. If secrets changed, restart the api service: `docker compose restart api`.

**Recovery:**
```bash
docker compose -f docker/docker-compose.yml --env-file .env restart vault
# Wait for vault-init to complete
docker compose -f docker/docker-compose.yml --env-file .env logs vault-init
# Restart api to pick up fresh secrets
docker compose -f docker/docker-compose.yml --env-file .env restart api
```

## Uploading Model Artifacts to MinIO

Run once after training a new classifier:

```bash
cd backend && MINIO_ENDPOINT=localhost:9000 \
  CLASSIFIER_ARTIFACT_DIR=../ml/artifacts/classifier/best \
  uv run python scripts/upload_model_to_minio.py
```

The model-server pulls weights from MinIO on every container start (env: `MINIO_ENDPOINT`, `MINIO_BUCKET`, `MINIO_MODEL_PREFIX`).

## Checking Health

```bash
curl http://localhost:8000/healthz   # API
curl http://localhost:8001/healthz   # Model-server
curl http://localhost:8501/_stcore/health  # Streamlit
```

## Picking up Dependency or Bundle Changes (named-volume gotcha)

The dev compose override mounts named volumes for the API venv, model-server venv, and built widget bundle:

| Volume | Container path | Purpose |
|---|---|---|
| `docker_api_dev_venv` | `/app/.venv` on `api` | hot-reload dev convenience |
| `docker_model_server_dev_venv` | `/app/.venv` on `model-server` | same |
| `docker_widget_dist` | `/app/static/widget` on `api`, `/usr/share/nginx/html` on `widget` | shared bundle |

Named volumes **persist across `docker compose up --build`** and **shadow whatever is in the new image**. After:
- `uv add <pkg>` (changes `.venv`)
- editing widget source (changes the bundle)

you must drop the relevant volume before the change appears in the running container. Recipe:

```bash
# Backend dep change
docker compose down api && docker volume rm docker_api_dev_venv && docker compose up -d api

# Model-server change
docker compose down model-server && docker volume rm docker_model_server_dev_venv && docker compose up -d model-server

# Widget bundle change
docker compose down widget api && docker volume rm docker_widget_dist && docker compose up -d api widget
```

Symptom that you forgot: code change is in the image, `docker exec ... cat /app/foo.py` shows the old content. That's the volume mount winning over the image layer.

## Stopping the Stack

```bash
docker compose -f docker/docker-compose.yml --env-file .env down
# To wipe volumes (full reset):
docker compose -f docker/docker-compose.yml --env-file .env down -v
```
