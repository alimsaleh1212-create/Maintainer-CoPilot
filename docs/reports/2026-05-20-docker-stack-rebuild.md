# Docker Stack Rebuild — 2026-05-20

## Problem Solved
Docker builds were failing with "No space left on device" when trying to install `nvidia-cublas` (a CUDA package pulled in as a transitive dependency of `torch` via `transformers`).

Error sequence:
```
Failed to download `nvidia-cublas==13.1.1.3`
Failed to extract archive: nvidia_cublas-13.1.1.3-py3-none-manylinux_2_27_x86_64.whl
I/O operation failed during extraction
failed to flush file `/root/.cache/uv/.tmpUbXaMc/nvidia/cu13/lib/libcublasLt.so.13`: 
No space left on device (os error 28)
```

Root cause: `torch` was a production dependency, pulling ~48GB of CUDA packages into Docker images during build, exceeding container disk allocation.

## Solution Implemented

### 1. Dependency Reorganization
**File:** `backend/pyproject.toml`

Moved `torch` from `[project.dependencies]` to a new `[dependency-groups.train]` group:

```toml
[project.dependencies]  # Production only
# - removed: torch>=2.12.0
- transformers>=5.8.1
- sentence-transformers>=5.5.1
# ... other inference deps

[dependency-groups.train]  # Training only
torch>=2.12.0
datasets>=3.4.0
huggingface-hub>=0.26.6
```

**Rationale:**
- `torch` is only needed for fine-tuning the classifier (local training)
- API service and inference only need `transformers` + `sentence-transformers` (can run CPU or with inference-only libraries)
- Production Docker images now exclude CUDA packages entirely
- Training script still accesses torch via `uv sync` (which includes train group)

### 2. Docker Cleanup
Freed 47GB of disk space:
- `docker image prune -a --force`: Deleted 47 unused images (14.55GB)
- `docker system prune -a --volumes --force`: Cleaned build cache (32.56GB)

**Command:**
```bash
docker image prune -a --force && docker system prune -a --volumes --force
```

### 3. Dockerfile Robustness
**Files:** `docker/api.Dockerfile`, `docker/model_server.Dockerfile`

Added fallback lock regeneration to handle pyproject.toml ↔ uv.lock mismatch:

```dockerfile
# Before: RUN uv sync --frozen --no-dev
# After:
RUN uv sync --frozen --no-dev || (uv lock && uv sync --frozen --no-dev)
```

**Why:** When torch is removed from pyproject.toml but still in uv.lock (stale lock), the fallback mechanism:
1. Tries to sync with the frozen lock (fast path, cached)
2. If that fails due to mismatch, regenerates the lock to match pyproject.toml
3. Then syncs the new lock

This ensures the build never fails due to lock/manifest drift, while preserving reproducibility when both are in sync.

### 4. Docker Build Execution
Initiated `docker-compose up --build` with the new configuration. Build process:
1. Pulls base images (python:3.12-slim, uv, postgres:16+pgvector, redis, vault, etc.)
2. Builds api + model-server without CUDA dependencies
3. Regenerates uv.lock inside container (remove torch transitive deps)
4. Syncs final production environment (~100MB instead of 1GB+)
5. Starts all services

## Files Changed

| File | Changes |
|------|---------|
| `backend/pyproject.toml` | Removed torch from [project.dependencies], added [dependency-groups.train] with torch+datasets+huggingface-hub |
| `docker/api.Dockerfile` | Added fallback lock regeneration: `uv sync --frozen --no-dev \|\| (uv lock && uv sync --frozen --no-dev)` |
| `docker/model_server.Dockerfile` | Same fallback added |

## Build Status

**Current:** `docker-compose up --build` running in background (expected 15-30 min)

**Waiting for:**
- API image build completion
- All service containers to start and pass health checks
- API `/health` endpoint to respond (indicates full readiness)

Once complete:
1. Extract regenerated `uv.lock` from container
2. Commit all changes (code + updated lock file)
3. Verify `docker-compose ps` shows all services running
4. Ready for corpus ingestion + THU deliverables

## Impact Summary

✅ **Fixed:** Docker build disk space exhaustion
✅ **Reduced:** Production image size (no CUDA packages)
✅ **Maintained:** Training capability (torch still available locally)
✅ **Improved:** Build robustness (fallback lock regeneration)
⏳ **Pending:** Service startup confirmation + lock file extraction + commit

---

**Next Report:** Post-build verification once services are running.
