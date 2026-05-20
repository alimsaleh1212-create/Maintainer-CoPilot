# MON Manual Gate — Fresh Clone `docker-compose up` Checklist

**Date:** 2026-05-20  
**Status:** Ready to execute

## Prerequisites

- Docker and Docker Compose installed
- ~5 minutes, stable internet (image pulls)
- Terminal access to this repository

## Fresh Clone Test Steps

```bash
# 1. Clone from scratch (or simulate with a clean state)
git clone https://github.com/your-org/maintainers-copilot project7-fresh
cd project7-fresh

# 2. Copy .env.example → .env
cp .env.example .env

# 3. Paste Vault root token into .env
# Get this from your Vault dev-mode container or existing instance
# VAULT_ROOT_TOKEN=s.xxxxxxxxxxxxxxxx
# Edit .env and fill in VAULT_ROOT_TOKEN

# 4. Start the stack
docker-compose -f docker/docker-compose.yml up -d

# 5. Wait for all services to be healthy (~30-60 seconds)
docker-compose -f docker/docker-compose.yml ps
```

## Expected Output

After `docker-compose up -d`, all 9 services should show as running and healthy:

```
NAME                        COMMAND                  STATE                PORTS
maintainers-copilot-api-1           "uvicorn app.api.main:app..."   Up (healthy)    0.0.0.0:8000→8000/tcp
maintainers-copilot-chatbot-1       ...                             Up              (internal only)
maintainers-copilot-db-1            "postgres"                      Up (healthy)    0.0.0.0:5432→5432/tcp
maintainers-copilot-langfuse-1      "docker-entrypoint.sh"         Up              0.0.0.0:3000→3000/tcp
maintainers-copilot-migrate-1       "sh -c 'alembic upgrade..."     Exited (0)      (runs once, then stops)
maintainers-copilot-minio-1         "/usr/bin/docker-entrypoint.s..." Up (healthy)   0.0.0.0:9000→9000, 9001/tcp
maintainers-copilot-redis-1         "redis-server --save ..."       Up (healthy)    0.0.0.0:6379→6379/tcp
maintainers-copilot-vault-1         "docker-entrypoint.sh server"   Up (healthy)    0.0.0.0:8200→8200/tcp
maintainers-copilot-widget-1        "nginx"                         Up              0.0.0.0:8081→8081/tcp
```

## Verification Steps

Run these after `docker-compose ps` shows all healthy:

### 1. API is responsive
```bash
curl http://localhost:8000/docs
# Expected: HTML page with Swagger UI (200 OK)
```

### 2. PostgreSQL is accessible
```bash
psql -h localhost -U copilot -d copilot -c "SELECT 1;"
# Expected: output "1"
```

### 3. Redis is accessible
```bash
redis-cli -h localhost ping
# Expected: output "PONG"
```

### 4. MinIO is accessible
```bash
curl -I http://localhost:9000/minio/health/live
# Expected: 200 OK
```

### 5. Vault is unsealed and accessible
```bash
curl http://localhost:8200/v1/sys/health
# Expected: 200 or 473 (unsealed or sealed state, both are "alive")
```

### 6. Check logs for errors
```bash
docker-compose -f docker/docker-compose.yml logs api | grep -i error
docker-compose -f docker/docker-compose.yml logs migrate | grep -i error
# Expected: No FATAL or ERROR lines in api or migrate logs
```

## Pass Criteria

✓ All 9 services are running or exited (migrate exits normally)  
✓ All persistent services (api, db, redis, vault, minio) show "healthy"  
✓ API responds to HTTP requests (curl /docs returns 200)  
✓ No error lines in critical service logs (api, migrate, db)  
✓ Can connect to PostgreSQL, Redis, MinIO, Vault from host  

## Cleanup

```bash
# When done, tear down the stack
docker-compose -f docker/docker-compose.yml down -v
```

## If It Fails

**API not starting:**
- Check `docker-compose logs api` — look for Vault connection errors or migration failures
- Verify `VAULT_ROOT_TOKEN` is set correctly in `.env`

**Database not healthy:**
- Check `docker-compose logs db` — volume mount issues or port conflicts
- Ensure port 5432 is not in use by another service

**Vault not healthy:**
- Check `docker-compose logs vault` — token generation or address issues
- Vault dev mode token is logged on first boot; copy from logs if starting fresh

**Migrate service exited with error:**
- Check `docker-compose logs migrate` — Alembic migration failures
- Ensure database is fully healthy before checking migrate logs

---

## MON Gate Status

- ✓ Unit tests (45/45 pass): redaction, settings, label_mapping, classifier_loads, smoke, vault
- ✓ Code quality (Ruff clean, mypy --strict clean)
- ✓ Integration tests (4/4 pass): vault_refusal tests
- ⧗ Manual gate (docker-compose up): Ready to execute

Run the fresh clone test above. If all services boot and respond, MON is complete.
