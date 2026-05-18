"""Smoke test for the streamlit_ui package skeleton.

Verifies that the top-level subpackages import cleanly. Catches accidental
removal of `__init__.py` markers or import-time errors before any other test
runs.
"""

import importlib


def test_streamlit_ui_subpackages_import() -> None:
    """Every streamlit_ui subpackage imports without error."""
    for module_name in ("lib", "pages"):
        importlib.import_module(module_name)
