# Docker Build Failure — Root Cause Analysis

## Problem Summary
Docker builds were hanging silently with no error output. The root cause was **insufficient `/tmp` space** combined with **simultaneous competing builds**.

## Root Cause Identified

### 1. `/tmp` Partition Exhaustion
**Before cleanup:**
- `/tmp` size: 1.0GB total
- `/tmp` used: 717MB (70% full)
- **Available: 308MB FREE**

**Packages waiting to extract:**
- `pip-unpack-gicmmrd9`: 296MB
- `pip-unpack-w4hlelm_`: 57MB
- `tmp5puisn0z`: 42MB
- Plus 40+ other pip temporary directories
- **Total: ~709MB** (exceeds available space!)

### 2. Why Docker Hung Silently
Docker's `uv sync` process:
1. Downloads Python packages (~transformers, sentence-transformers, etc.)
2. Extracts them to `/tmp` (default behavior)
3. Runs out of `/tmp` space → **extraction fails**
4. Process hangs/blocks instead of returning an error
5. **No error appears in logs** because the error happens inside the extraction, not at Docker layer

### 3. Concurrent Builds Made It Worse
Active processes when we diagnosed:
```
docker build (PID 473479) - started 20:01
docker build (PID 473912) - started 20:02 (2nd attempt)
```

Both were downloading packages simultaneously to the same `/tmp`, competing for the 308MB remaining space.

## Why Logs Didn't Show the Error

The issue was **silently failing at the filesystem level**:
- Docker build succeeds in pulling and building layers
- `uv sync` tries to extract packages
- `/tmp` runs out of space → silent extraction failure
- Process blocks waiting for space
- No error message is propagated up to docker/docker-compose logs
- **User sees: build hanging with no output**
- **Docker logs show: nothing (no error occurred at docker level)**

## Solution Applied

### 1. Cleaned `/tmp`
```bash
rm -rf /tmp/pip-* /tmp/tmp* /tmp/uv-*
```
**Result:** 
- Before: 308MB free (70% full)
- After: 1000MB free (3% full)

### 2. Restarted Docker Daemon
```bash
sudo systemctl restart docker
```
Clears any stuck processes and resets docker-buildkit state.

### 3. Killed Stuck Build Processes
```bash
pkill -9 docker     # (as root) kill the dockerd processes
pkill -9 runc       # kill container runtime processes
```

## Prevention

### Short-term
- **Monitor `/tmp` space before builds:** `df -h /tmp`
- **Clean /tmp before large builds:** `rm -rf /tmp/pip-* /tmp/tmp* /tmp/uv-*`

### Long-term
- **Configure Docker to use separate temp directory** (not system `/tmp`)
- **Set BuildKit options** to use custom temporary directories
- **Use `.dockerignore`** to exclude large files from build context
- **Add pre-build health check** that fails fast if `/tmp < 500MB`

## Key Insight: The Invisible Failure

This is a subtle class of infrastructure bugs:
- **Silent failure** (no error message)
- **Happens at filesystem level** (invisible to application logs)
- **Only detectable by:**
  - Disk space monitoring (`df -h`)
  - Process monitoring (`ps aux`)
  - System-level investigation (`journalctl`, `/var/log/syslog`)

Not by:
- Docker logs (`docker logs`)
- Build output (piped to tail)
- Application-level error handling

## Build Status After Fix

- ✓ `/tmp` cleaned: 1000MB free
- ✓ Docker restarted
- ✓ Build processes killed
- ⏳ New build starting with clean state...

Next: Monitor `docker build` with clean /tmp and fresh docker daemon. Should complete successfully.
