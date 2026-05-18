"""Smoke test for the backend package skeleton.

Verifies that `app` and every top-level subpackage import cleanly. If anyone
breaks the package layout, removes an `__init__.py`, or introduces a syntax
error in a module that runs on import, this test catches it before any other
test even runs.
"""

import importlib


def test_backend_package_imports() -> None:
    """Every top-level backend subpackage imports without error."""
    for module_name in (
        "app",
        "app.api",
        "app.api.routes",
        "app.services",
        "app.repositories",
        "app.domain",
        "app.infra",
        "app.infra.llm",
        "app.ml",
        "app.rag",
        "app.tools",
    ):
        importlib.import_module(module_name)
