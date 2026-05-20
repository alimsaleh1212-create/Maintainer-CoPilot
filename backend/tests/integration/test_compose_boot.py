"""Integration test: verify docker-compose stack boots and all services are healthy.

This test runs `docker-compose up` from the repo root and checks that all
services (api, db, redis, vault, etc.) are reachable within 60 seconds.

Run with: pytest tests/integration/test_compose_boot.py -v -s --timeout=120
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import time
from pathlib import Path

import httpx
import pytest


@pytest.fixture(scope="module")
def docker_compose_up() -> None:  # type: ignore[misc]
    """Start docker-compose stack before tests; tear down after."""
    repo_root = Path(__file__).parent.parent.parent.parent
    compose_file = repo_root / "docker" / "docker-compose.yml"

    if not compose_file.exists():
        pytest.skip("docker-compose.yml not found")

    # Start the stack
    print("\n🐳 Starting docker-compose stack...")
    result = subprocess.run(
        ["docker-compose", "-f", str(compose_file), "up", "-d"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        print(f"STDERR: {result.stderr}")
        pytest.fail(f"docker-compose up failed: {result.returncode}")

    yield

    # Tear down
    print("\n🛑 Stopping docker-compose stack...")
    subprocess.run(
        ["docker-compose", "-f", str(compose_file), "down", "-v"],
        cwd=str(repo_root),
        capture_output=True,
        timeout=30,
    )


@pytest.mark.integration
class TestComposeBootHealthy:
    """Test that the docker-compose stack boots successfully."""

    def test_vault_healthy(self, docker_compose_up: None) -> None:
        """Vault service is reachable and healthy."""
        # Arrange: Vault should be at localhost:8200 (per docker-compose.yml)
        vault_url = "http://localhost:8200"
        max_retries = 30
        retry_interval = 2.0

        # Act: Poll Vault health endpoint with retries
        for attempt in range(max_retries):
            try:
                response = httpx.get(f"{vault_url}/v1/sys/health", timeout=5.0)
                # Vault returns 473 (sealed) or 200 (unsealed), both are "healthy"
                if response.status_code in (200, 473):
                    print(f"✓ Vault healthy (attempt {attempt + 1})")
                    return
            except (httpx.ConnectError, httpx.TimeoutException):
                if attempt < max_retries - 1:
                    time.sleep(retry_interval)
                continue

        # Assert
        pytest.fail(f"Vault did not become healthy within {max_retries * retry_interval}s")

    def test_postgres_healthy(self, docker_compose_up: None) -> None:
        """PostgreSQL service is reachable and accepting connections."""
        # Arrange: PostgreSQL at localhost:5432 (per docker-compose.yml)
        import psycopg

        max_retries = 30
        retry_interval = 2.0

        # Act: Try to connect with retries
        for attempt in range(max_retries):
            try:
                conn = psycopg.connect(
                    host="localhost",
                    port=5432,
                    dbname="postgres",
                    user="postgres",
                    password="postgres",
                    connect_timeout=5,
                )
                conn.close()
                print(f"✓ PostgreSQL healthy (attempt {attempt + 1})")
                return
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_interval)
                continue

        # Assert
        pytest.fail(f"PostgreSQL did not become healthy within {max_retries * retry_interval}s")

    def test_redis_healthy(self, docker_compose_up: None) -> None:
        """Redis service is reachable and accepting commands."""
        # Arrange: Redis at localhost:6379 (per docker-compose.yml)
        import redis

        max_retries = 30
        retry_interval = 2.0

        # Act: Try to ping Redis with retries
        for attempt in range(max_retries):
            try:
                r = redis.Redis(host="localhost", port=6379, socket_connect_timeout=5)
                r.ping()
                r.close()
                print(f"✓ Redis healthy (attempt {attempt + 1})")
                return
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_interval)
                continue

        # Assert
        pytest.fail(f"Redis did not become healthy within {max_retries * retry_interval}s")

    def test_minio_healthy(self, docker_compose_up: None) -> None:
        """MinIO service is reachable and accepting connections."""
        # Arrange: MinIO at localhost:9000 (per docker-compose.yml)
        from minio import Minio

        max_retries = 30
        retry_interval = 2.0

        # Act: Try to connect with retries
        for attempt in range(max_retries):
            try:
                client = Minio(
                    "localhost:9000",
                    access_key="minioadmin",
                    secret_key="minioadmin",
                    secure=False,
                )
                client.list_buckets()
                print(f"✓ MinIO healthy (attempt {attempt + 1})")
                return
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_interval)
                continue

        # Assert
        pytest.fail(f"MinIO did not become healthy within {max_retries * retry_interval}s")

    @pytest.mark.slow
    def test_fastapi_api_healthy(self, docker_compose_up: None) -> None:
        """FastAPI /docs endpoint is reachable (indicates app booted)."""
        # Arrange: API at localhost:8000 (per docker-compose.yml)
        api_url = "http://localhost:8000"
        max_retries = 60  # API takes longer; needs DB migrations
        retry_interval = 1.0

        # Act: Poll API /docs with retries
        for attempt in range(max_retries):
            try:
                response = httpx.get(f"{api_url}/docs", timeout=5.0, follow_redirects=True)
                if response.status_code == 200:
                    print(f"✓ FastAPI healthy (attempt {attempt + 1})")
                    return
            except (httpx.ConnectError, httpx.TimeoutException):
                if attempt < max_retries - 1:
                    time.sleep(retry_interval)
                continue

        # Assert
        pytest.fail(f"FastAPI did not become healthy within {max_retries * retry_interval}s")
