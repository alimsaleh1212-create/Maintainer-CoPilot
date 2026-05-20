# Post-Docker-Build Steps

After `docker-compose up --build` completes successfully, follow these steps:

## 1. Extract Updated uv.lock

The Docker build regenerates `uv.lock` to remove torch from production dependencies. Extract it:

```bash
cd /home/user/workplace/aie_sef_bootcamp/project7

# Create temporary container to extract the lock file
docker run --rm -v "$(pwd)/backend:/mnt" \
  $(docker-compose -f docker/docker-compose.yml images api -q) \
  cp /app/uv.lock /mnt/uv.lock.new

# Verify it was extracted
ls -lh backend/uv.lock.new

# Replace the old one
mv backend/uv.lock.new backend/uv.lock
```

## 2. Verify Services

```bash
# Check all services are running
docker-compose -f docker/docker-compose.yml ps

# Test API health
curl http://localhost:8000/health

# Test database connectivity
docker-compose -f docker/docker-compose.yml exec db psql -U postgres -c "SELECT 1"
```

## 3. Commit Changes

```bash
git add backend/pyproject.toml backend/uv.lock docker/*.Dockerfile
git commit -m "chore: move torch to train-only dependency group for smaller Docker images

- Moved torch, datasets, huggingface-hub to [dependency-groups.train]
- Production Docker images now install only inference dependencies (transformers, sentence-transformers)
- Training script can still use torch via 'uv sync' (includes train group)
- Regenerated uv.lock to remove CUDA packages from production image
- This reduces Docker image size significantly (47GB+ freed after cleanup)
- Updated Dockerfiles with fallback lock regeneration for robustness"
```

## 4. Next Steps

Once committed, you're ready for:
- Corpus ingestion: `python backend/scripts/fetch_issues.py && python backend/app/rag/ingest.py`
- Running tests: `cd backend && uv run pytest -q`
- THU deliverables: Auth, chatbot, memory, widget

---

**Build Details:**
- Docker cleanup freed: 47.11GB (images + cache)
- Lock regeneration: Happened inside Docker build (fallback mechanism)
- Affected files: pyproject.toml, uv.lock, api.Dockerfile, model_server.Dockerfile
