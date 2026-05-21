"""Widget bundle size test.

The React widget must compile to a bundle small enough to load quickly in the
demo host iframe.  Per CLAUDE.md acceptance criteria:

    ``wc -c frontend/widget/dist/widget.js`` ≤ 200 KB gzipped

If the bundle has not been built yet, the test is skipped rather than failing
so that CI can still run unit tests without the Node build step.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import pytest

# Path from this test file to the project root, then to the widget bundle.
_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
_BUNDLE_PATH = _PROJECT_ROOT / "frontend" / "widget" / "dist" / "widget.js"

_MAX_GZIP_KB = 200
_MAX_GZIP_BYTES = _MAX_GZIP_KB * 1024


# ---------------------------------------------------------------------------
# Bundle size test
# ---------------------------------------------------------------------------


class TestWidgetBundleSize:
    def test_widget_js_under_200kb_gzipped(self) -> None:
        """widget.js must compress to ≤ 200 KB to stay within the iframe load budget."""
        if not _BUNDLE_PATH.exists():
            pytest.skip(
                f"Widget bundle not built. Run `npm run build` in frontend/widget/ "
                f"(looked for {_BUNDLE_PATH})"
            )

        # Arrange
        raw_bytes = _BUNDLE_PATH.read_bytes()

        # Act
        compressed = gzip.compress(raw_bytes, compresslevel=9)
        gzip_kb = len(compressed) / 1024

        # Assert
        assert len(compressed) <= _MAX_GZIP_BYTES, (
            f"Widget bundle too large: {gzip_kb:.1f} KB gzipped "
            f"(limit: {_MAX_GZIP_KB} KB). "
            "Run `npm run build` and check for unintended heavy dependencies."
        )

    def test_widget_bundle_path_is_configured_correctly(self) -> None:
        """Smoke-test that the path resolution logic points to the right location."""
        # The path must end with the expected relative path regardless of OS.
        assert _BUNDLE_PATH.parts[-3:] == ("widget", "dist", "widget.js")

    def test_widget_bundle_exists_or_skips_gracefully(self) -> None:
        """When the bundle is missing the test must skip, not error."""
        if not _BUNDLE_PATH.exists():
            pytest.skip("Bundle not built — skipping size check")

        # If we reach here the file exists; just assert it's readable.
        assert _BUNDLE_PATH.stat().st_size > 0
