"""RAG threshold smoke tests.

CLAUDE.md rule: 'eval_thresholds.yaml values gate merge — zero is refuse-to-boot.'

Validates the RAG golden set structure and threshold config.  When the full
retrieval pipeline is integrated (corpus ingested + RAGAS metrics computed) these
tests will additionally assert actual metric values.

For now the tests guard:
- Threshold config is present and non-zero.
- Golden set has all required fields for each Q/A pair.
- RAG eval runner exits 0 when called directly.

Tagged @pytest.mark.eval so local runs skip by default; CI always runs them.

Run manually:
    uv run pytest tests/eval/test_rag_thresholds.py -v -m eval
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

# ── Paths ─────────────────────────────────────────────────────────────────────
_BACKEND_ROOT = Path(__file__).parent.parent.parent
_EVAL_DIR = _BACKEND_ROOT / "eval"
_THRESHOLDS_PATH = _EVAL_DIR / "eval_thresholds.yaml"
_GOLDEN_SET_PATH = _EVAL_DIR / "golden_rag.jsonl"
_RUNNER_PATH = _EVAL_DIR / "run_rag_eval.py"

_REQUIRED_GOLDEN_FIELDS = {"id", "question", "ideal_answer", "ground_truth_chunks"}


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def rag_thresholds() -> dict:
    """Load committed RAG thresholds."""
    with open(_THRESHOLDS_PATH) as f:
        data: dict = yaml.safe_load(f)
    return data["rag"]


@pytest.fixture(scope="module")
def golden_set() -> list[dict]:
    """Load the RAG golden set from JSONL."""
    items: list[dict] = []
    with open(_GOLDEN_SET_PATH) as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.eval
class TestRagThresholds:
    def test_threshold_config_has_no_zero_values(
        self,
        rag_thresholds: dict,
    ) -> None:
        """Refuse-to-boot rule: no threshold may be zero or disabled."""
        for key, val in rag_thresholds.items():
            if key == "rationale":
                continue
            assert val > 0, (
                f"RAG threshold '{key}' is {val!r}. "
                "Zero thresholds are forbidden — they disable the gate."
            )

    def test_all_required_thresholds_present(
        self,
        rag_thresholds: dict,
    ) -> None:
        """The three mandatory RAGAS + retrieval thresholds must all be defined."""
        required = {"faithfulness", "answer_relevancy", "hit_at_5"}
        missing = required - set(rag_thresholds.keys())
        assert not missing, (
            f"Missing RAG thresholds in eval_thresholds.yaml: {missing}"
        )

    def test_faithfulness_threshold_is_defensible(
        self,
        rag_thresholds: dict,
    ) -> None:
        """Faithfulness threshold must be ≥ 0.70 (grounding requirement)."""
        assert rag_thresholds["faithfulness"] >= 0.70, (
            f"faithfulness threshold {rag_thresholds['faithfulness']} is below 0.70. "
            "Medical-domain RAG must be grounded; raise the threshold or improve retrieval."
        )

    def test_golden_set_is_non_trivial_size(
        self,
        golden_set: list[dict],
    ) -> None:
        """Golden set must have at least 20 Q/A pairs to be statistically meaningful."""
        assert len(golden_set) >= 20, (
            f"Golden RAG set has only {len(golden_set)} items. "
            "Need ≥ 20 for Hit@5 to be meaningful."
        )

    def test_golden_set_all_items_have_required_fields(
        self,
        golden_set: list[dict],
    ) -> None:
        """Every Q/A pair must have id, question, ideal_answer, ground_truth_chunks."""
        bad = [
            qa["id"]
            for qa in golden_set
            if not _REQUIRED_GOLDEN_FIELDS.issubset(qa.keys())
        ]
        assert not bad, (
            f"Golden set items missing required fields: {bad}. "
            f"Required: {_REQUIRED_GOLDEN_FIELDS}"
        )

    def test_golden_set_ids_are_unique(
        self,
        golden_set: list[dict],
    ) -> None:
        """Duplicate IDs in the golden set indicate a data preparation error."""
        ids = [qa["id"] for qa in golden_set]
        assert len(ids) == len(set(ids)), (
            f"Duplicate IDs found in golden RAG set: "
            f"{[x for x in ids if ids.count(x) > 1]}"
        )

    def test_ground_truth_chunks_are_non_empty(
        self,
        golden_set: list[dict],
    ) -> None:
        """Each Q/A pair must have at least one ground-truth chunk to compute Hit@K."""
        bad = [
            qa["id"]
            for qa in golden_set
            if not qa.get("ground_truth_chunks")
        ]
        assert not bad, (
            f"Golden set items with empty ground_truth_chunks: {bad}. "
            "Hit@5 cannot be computed without ground-truth chunk references."
        )

    def test_rag_eval_runner_exits_zero(self) -> None:
        """run_rag_eval.py must exit 0 (structure check passes)."""
        result = subprocess.run(
            [sys.executable, str(_RUNNER_PATH)],
            capture_output=True,
            text=True,
            cwd=str(_BACKEND_ROOT),
        )
        assert result.returncode == 0, (
            f"run_rag_eval.py exited {result.returncode}.\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
