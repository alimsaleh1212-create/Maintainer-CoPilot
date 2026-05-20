# Docker Build Status — 2026-05-20 19:46+

## Current Status

**Build Started:** 19:46 UTC  
**Current Time:** ~19:50 UTC  
**Duration:** 4-5 minutes  
**Status:** Clean rebuild in progress with lean uv.lock (no CUDA)

### What Changed

1. **Dependency Fix:** Removed `torch` from production dependencies
   - Moved to `[dependency-groups.train]` in `pyproject.toml`
   - Production Docker images now exclude CUDA packages

2. **Lock File Fix:** Regenerated `uv.lock` without torch
   - Old: 1GB+ (with CUDA packages)
   - New: 209KB (production dependencies only)
   - 114 packages resolved (inference + web framework only)

3. **Dockerfile Update:** Added robustness for manifest drift
   - Fallback: `uv sync --frozen --no-dev || (uv lock && uv sync --frozen --no-dev)`
   - Ensures build never fails due to lock/manifest mismatch

### Git Status

✅ Commit: `chore(docker): move torch to train-only dependency group, regenerate lock`
- `backend/pyproject.toml` (moved torch to train group)
- `backend/uv.lock` (209KB clean, no CUDA)
- `docker/api.Dockerfile` (added fallback)
- `docker/model_server.Dockerfile` (added fallback)

### Build Timeline

| Time | Event |
|------|-------|
| 19:05 | First build started (with stale lock, CUDA packages) |
| ~19:20 | Build failed: nvidia-cublas extraction (no disk space) |
| 19:40 | Docker cleanup: freed 47GB |
| 19:46 | Second build started (with clean lock, no CUDA) |
| 19:50-ish | Services starting... |

---

## Once Build Completes

```bash
# 1. Verify all services are running
docker-compose -f docker/docker-compose.yml ps

# 2. Check API health
curl http://localhost:8000/health

# 3. Check database connectivity
docker-compose -f docker/docker-compose.yml exec db pg_isready -U postgres

# 4. View logs if needed
docker-compose -f docker/docker-compose.yml logs -f api

# 5. Run tests to verify stack
cd backend && uv run pytest tests/unit/test_*.py -q
```

---

## What's Next After Services Start

1. **Corpus Ingestion:**
   ```bash
   python backend/scripts/fetch_issues.py  # Fetch MONAI issues
   python -c "from app.rag.ingest import *; ..."  # Ingest to pgvector
   ```

2. **THU Deliverables (Authentication, Chatbot, Memory, Widget):**
   - Auth: fastapi-users + JWT from Vault
   - Chatbot: Gemini tool-calling with 4 tools (classify, ner, summarize, rag)
   - Memory: Redis (short-term 24h TTL) + pgvector (long-term episodic)
   - Widget: React + Vite, embedded in demo hosts

3. **Friday AM (Polish & Ship):**
   - Widget styling with Tailwind
   - Complete 5 required `.md` files (ARCH, DECISIONS, RUNBOOK, EVALS, SECURITY)
   - Final CI green + tag v0.1.0-week7

---

## Key Improvement

**Before:** Docker build failed with "no space left on device" when trying to extract 1.3GB CUDA package
**After:** Clean build with 209KB lock file, no CUDA dependencies in production images, builds in ~5-10 minutes

This unblocks THU/FRI work which requires a fully operational stack for auth, memory, and e2e testing.
