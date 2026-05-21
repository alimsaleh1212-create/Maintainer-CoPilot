"""Integration test conftest — skip all integration tests when Docker is unavailable."""

from __future__ import annotations

import subprocess

import pytest


def _docker_available() -> bool:
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# Skip every test in this directory unless Docker is reachable.
if not _docker_available():
    collect_ignore_glob = ["*.py"]


def pytest_collection_modifyitems(items: list, config: pytest.Config) -> None:
    if not _docker_available():
        skip = pytest.mark.skip(reason="Docker not available — skipping integration tests")
        for item in items:
            if "integration" in str(item.fspath):
                item.add_marker(skip)
