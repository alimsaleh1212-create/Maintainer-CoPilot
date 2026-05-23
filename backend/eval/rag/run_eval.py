#!/usr/bin/env python
"""RAG evaluation runner: RAGAS metrics + retrieval metrics on the golden set.

Two modes
---------
offline (--offline)
    Validates golden set structure and threshold YAML only.
    Safe for CI without live services.  No API calls, no RAGAS invocations.

live (default)
    Calls the live RAG API (/rag/search) to get retrieved contexts.
    Calls the live chat API (/chat, classification service disabled) to get answers.
    Runs RAGAS (faithfulness + answer_relevancy) with Gemini as the LLM judge.
    Computes retrieval metrics: Hit@5, MRR@10.
    Checks all metrics against thresholds.yaml.
    Prints a report and exits non-zero if any threshold is missed.

Usage
-----
    # offline — validate structure only
    python eval/rag/run_eval.py --offline

    # live evaluation (requires running stack + valid JWT)
    python eval/rag/run_eval.py \\
        --api-url http://localhost:8000 \\
        --api-token <jwt> \\
        --gemini-api-key <key>

    # inside the api container (reads VAULT-resolved secrets from env)
    docker exec docker-api-1 uv run --group eval python eval/rag/run_eval.py \\
        --api-url http://localhost:8000 --api-token <jwt>

GEMINI_JUDGMENT_MODEL env var (or --gemini-judgment-model CLI arg) controls which model
acts as the RAGAS judge.  Defaults to gemini-2.5-pro for a stronger evaluation signal;
falls back to the main gemini-model if the judgment model is unavailable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

_EVAL_DIR = Path(__file__).parent
_THRESHOLDS_PATH = _EVAL_DIR / "thresholds.yaml"
_GOLDEN_SET_PATH = _EVAL_DIR / "golden_set.jsonl"
_RESULTS_PATH = _EVAL_DIR / "last_results.json"

_REQUIRED_GOLDEN_FIELDS = {"id", "question", "ideal_answer", "ground_truth_chunks"}


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def load_thresholds() -> dict[str, Any]:
    """Load RAG thresholds from YAML."""
    with open(_THRESHOLDS_PATH) as f:
        return yaml.safe_load(f)  # type: ignore[no-any-return]


def load_golden_set(path: Path | None = None) -> list[dict[str, Any]]:
    """Load golden Q/A pairs from JSONL.

    Args:
        path: Override path; defaults to ``_GOLDEN_SET_PATH``.
    """
    target = path or _GOLDEN_SET_PATH
    rows: list[dict[str, Any]] = []
    with open(target) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def validate_structure(golden_set: list[dict[str, Any]]) -> bool:
    """Return True if every row has the required fields."""
    ok = True
    for qa in golden_set:
        missing = _REQUIRED_GOLDEN_FIELDS - set(qa.keys())
        if missing:
            print(f"  ✗ {qa.get('id')}: missing fields {missing}", flush=True)
            ok = False
    return ok


# ---------------------------------------------------------------------------
# Live evaluation helpers
# ---------------------------------------------------------------------------


def _rag_search(api_url: str, token: str, question: str) -> list[str]:
    """Call /rag/search and return a list of retrieved context strings.

    Falls back to an empty list on any error.
    """
    import httpx

    try:
        resp = httpx.post(
            f"{api_url}/rag/search",
            json={"query": question, "top_k": 5},
            headers={"Authorization": f"Bearer {token}"},
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
        # /rag/search returns {"chunks": [...], "citations": [...], ...}
        # Each chunk has {"text": ..., "source": ..., "score": ...}
        chunks: list[dict[str, Any]] = data.get("chunks", data.get("results", []))
        return [r.get("text", r.get("content", "")) for r in chunks if r]
    except Exception as exc:
        print(f"    ⚠  rag_search failed for '{question[:50]}': {exc}", flush=True)
        return []


def _chat_answer(api_url: str, token: str, question: str, contexts: list[str]) -> str:
    """Ask the chatbot and return its answer string.

    Injects the retrieved contexts into the prompt so the answer is grounded.
    Falls back to the question itself if the chat call fails (ensures
    RAGAS faithfulness score reflects real behaviour, not scaffolding errors).
    """
    import httpx

    context_block = "\n\n".join(contexts[:3])
    augmented_question = (
        f"Context:\n{context_block}\n\nQuestion: {question}"
        if context_block
        else question
    )
    # /chat has maxLength=8192; truncate context so the question always fits
    augmented_question = augmented_question[:8000]
    try:
        resp = httpx.post(
            f"{api_url}/chat",
            json={"message": augmented_question},
            headers={"Authorization": f"Bearer {token}"},
            timeout=60.0,
        )
        resp.raise_for_status()
        data = resp.json()
        return str(data.get("response", data.get("message", "")))
    except Exception as exc:
        print(f"    ⚠  chat answer failed for '{question[:50]}': {exc}", flush=True)
        return ""


def compute_retrieval_metrics(
    golden_set: list[dict[str, Any]], retrieved: list[list[str]]
) -> dict[str, float]:
    """Compute Hit@5 and MRR@10 against ground_truth_chunks keywords.

    ``ground_truth_chunks`` is a list of keyword strings; a retrieved context
    is considered a hit if it contains ANY of the ground-truth keywords
    (case-insensitive substring match).

    Args:
        golden_set: List of golden Q/A dicts.
        retrieved: Parallel list of retrieved context lists.

    Returns:
        Dict with ``hit_at_5`` and ``mrr_at_10``.
    """
    hits = 0
    reciprocal_ranks: list[float] = []

    for qa, contexts in zip(golden_set, retrieved):
        gt_chunks: list[str] = [c.lower() for c in qa.get("ground_truth_chunks", [])]
        if not gt_chunks:
            continue

        found_at: int | None = None
        for rank, ctx in enumerate(contexts[:10], start=1):
            ctx_lower = ctx.lower()
            if any(kw in ctx_lower for kw in gt_chunks):
                found_at = rank
                break

        if found_at is not None and found_at <= 5:
            hits += 1
        if found_at is not None and found_at <= 10:
            reciprocal_ranks.append(1.0 / found_at)
        else:
            reciprocal_ranks.append(0.0)

    n = max(len(golden_set), 1)
    return {
        "hit_at_5": hits / n,
        "mrr_at_10": sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0,
    }


def run_ragas(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
    gemini_api_key: str,
    gemini_judgment_model: str,
) -> dict[str, float]:
    """Run RAGAS faithfulness + answer_relevancy with Gemini as the LLM judge.

    RAGAS uses the LLM to check whether the answer is grounded in the
    retrieved contexts (faithfulness) and whether the answer actually
    addresses the question (answer_relevancy).

    Uses ``gemini_judgment_model`` (e.g. gemini-2.5-pro) as the judge — a
    stronger model than the chat model so evaluation signal is reliable.
    Gemini is exposed via its OpenAI-compatible REST endpoint.

    Args:
        questions: Question strings.
        answers: Generated answer strings (from chatbot).
        contexts: Retrieved context chunks per question.
        ground_truths: Reference answer per question (from golden set).
        gemini_api_key: Gemini API key (from Vault / CLI arg).
        gemini_judgment_model: Gemini model used as RAGAS judge (e.g. 'gemini-2.5-pro').

    Returns:
        Dict with 'faithfulness' and 'answer_relevancy' scores.
    """
    from langchain_openai import ChatOpenAI
    from ragas import evaluate
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.metrics import AnswerRelevancy, Faithfulness

    gemini_base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"

    print(f"    RAGAS judge model: {gemini_judgment_model}", flush=True)
    # gemini-2.5-pro is a "thinking" model that burns reasoning tokens before
    # producing output.  max_tokens must be large enough to include both thinking
    # and response tokens; 8192 covers typical RAGAS judgment lengths.
    llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=gemini_judgment_model,
            api_key=gemini_api_key,  # type: ignore[arg-type]
            base_url=gemini_base_url,
            temperature=1.0,  # Gemini thinking models require temperature=1.0
            max_tokens=8192,
        )
    )

    # Gemini's OpenAI-compat endpoint does NOT support embeddings (501 UNIMPLEMENTED).
    # Use Ollama's nomic-embed-text instead — same model used by the live RAG pipeline.
    try:
        from langchain_ollama import OllamaEmbeddings as _OllamaEmbeddings
        _embed_backend = _OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
    except ImportError:
        from langchain_community.embeddings import OllamaEmbeddings as _OllamaEmbeddings  # type: ignore[no-redef]
        _embed_backend = _OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")  # type: ignore[assignment]
    embeddings = LangchainEmbeddingsWrapper(_embed_backend)

    samples = [
        SingleTurnSample(
            user_input=q,
            response=a,
            retrieved_contexts=c,
            reference=gt,
        )
        for q, a, c, gt in zip(questions, answers, contexts, ground_truths)
    ]
    dataset = EvaluationDataset(samples=samples)

    metrics = [
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=embeddings),
    ]

    result = evaluate(dataset, metrics=metrics)
    df = result.to_pandas()

    return {
        "faithfulness": float(df["faithfulness"].mean()),
        "answer_relevancy": float(df["answer_relevancy"].mean()),
    }


# ---------------------------------------------------------------------------
# Main runners
# ---------------------------------------------------------------------------


def run_offline(golden_set_path: Path | None = None) -> int:
    """Validate golden set structure and threshold config only."""
    resolved = golden_set_path or _GOLDEN_SET_PATH
    print("\nRAG Eval — offline mode (structure check only)")
    print(f"  Thresholds: {_THRESHOLDS_PATH}")
    print(f"  Golden set: {resolved}")

    thresholds = load_thresholds()
    golden_set = load_golden_set(golden_set_path)
    print(f"\n  Golden set size: {len(golden_set)} Q/A pairs", flush=True)

    print("\n  Validating golden set structure …")
    if not validate_structure(golden_set):
        print("  FAIL: golden set has structural errors")
        return 1
    print(f"  PASS: all {len(golden_set)} rows have required fields")

    print("\n  Validating threshold config …")
    required_keys = {"faithfulness", "answer_relevancy", "hit_at_5"}
    missing_keys = required_keys - set(thresholds.keys())
    if missing_keys:
        print(f"  FAIL: missing thresholds: {missing_keys}")
        return 1
    for key in sorted(required_keys):
        val = thresholds[key]
        ok = isinstance(val, (int, float)) and val > 0
        print(f"  {'PASS' if ok else 'FAIL'}  {key}: {val}")
        if not ok:
            return 1

    print("\n  Offline checks passed.")
    return 0


def run_live(
    api_url: str,
    api_token: str,
    gemini_api_key: str,
    gemini_model: str,
    gemini_judgment_model: str,
    skip_ragas: bool,
    golden_set_path: Path | None = None,
) -> int:
    """Run full RAGAS + retrieval metric evaluation against the live API."""
    resolved = golden_set_path or _GOLDEN_SET_PATH
    results_path = resolved.parent / f"{resolved.stem}_results.json" if golden_set_path else _RESULTS_PATH

    print(f"\nRAG Eval — live mode")
    print(f"  API URL:       {api_url}")
    print(f"  Chat model:    {gemini_model}")
    print(f"  RAGAS judge:   {gemini_judgment_model}")
    print(f"  Skip RAGAS:    {skip_ragas}")
    print(f"  Golden set:    {resolved}", flush=True)

    thresholds = load_thresholds()
    golden_set = load_golden_set(golden_set_path)
    n = len(golden_set)
    print(f"\n  Golden set: {n} Q/A pairs")

    print("\n  Collecting RAG search results …", flush=True)
    all_contexts: list[list[str]] = []
    all_answers: list[str] = []
    questions = [qa["question"] for qa in golden_set]
    ground_truths = [qa["ideal_answer"] for qa in golden_set]

    for i, qa in enumerate(golden_set, start=1):
        print(f"  [{i:2d}/{n}] {qa['question'][:60]} …", flush=True)
        contexts = _rag_search(api_url, api_token, qa["question"])
        all_contexts.append(contexts)
        answer = _chat_answer(api_url, api_token, qa["question"], contexts)
        all_answers.append(answer)

    # Retrieval metrics (no LLM needed)
    retrieval_metrics = compute_retrieval_metrics(golden_set, all_contexts)
    print(f"\n  Hit@5:  {retrieval_metrics['hit_at_5']:.3f}")
    print(f"  MRR@10: {retrieval_metrics['mrr_at_10']:.3f}", flush=True)

    # RAGAS metrics
    ragas_metrics: dict[str, float] = {}
    if not skip_ragas:
        print("\n  Running RAGAS evaluation (this may take a few minutes) …", flush=True)
        ragas_metrics = run_ragas(
            questions=questions,
            answers=all_answers,
            contexts=all_contexts,
            ground_truths=ground_truths,
            gemini_api_key=gemini_api_key,
            gemini_judgment_model=gemini_judgment_model,
        )
        print(f"  Faithfulness:     {ragas_metrics['faithfulness']:.3f}")
        print(f"  Answer relevancy: {ragas_metrics['answer_relevancy']:.3f}", flush=True)
    else:
        print("\n  RAGAS skipped (--skip-ragas).")

    # Save results
    all_metrics = {**retrieval_metrics, **ragas_metrics}
    results_payload = {
        "metrics": all_metrics,
        "thresholds": thresholds,
        "n_questions": n,
    }
    results_path.write_text(json.dumps(results_payload, indent=2))
    print(f"\n  Results saved to {results_path}")

    # Threshold checks
    print("\n  Threshold checks:")
    failed = False
    check_keys = ["faithfulness", "answer_relevancy", "hit_at_5"]
    for key in check_keys:
        if key not in all_metrics:
            if not skip_ragas or key not in ("faithfulness", "answer_relevancy"):
                print(f"  SKIP {key} (not computed)")
            continue
        actual = all_metrics[key]
        required = float(thresholds.get(key, 0))
        ok = actual >= required
        print(f"  {'PASS' if ok else 'FAIL'}  {key}: {actual:.3f} (>= {required})")
        if not ok:
            failed = True

    if failed:
        print("\n  One or more thresholds not met.")
        return 1
    print("\n  All thresholds met.")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RAG evaluation runner")
    p.add_argument("--offline", action="store_true", help="Structure check only (no API calls)")
    p.add_argument("--api-url", default="http://localhost:8000", help="Backend API base URL")
    p.add_argument("--api-token", default=os.getenv("EVAL_API_TOKEN", ""), help="JWT bearer token")
    p.add_argument(
        "--gemini-api-key",
        default=os.getenv("GEMINI_API_KEY", ""),
        help="Gemini API key (used both for chat answers and RAGAS judge)",
    )
    p.add_argument(
        "--gemini-model",
        default=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        help="Gemini model used to generate chat answers",
    )
    p.add_argument(
        "--gemini-judgment-model",
        default=os.getenv("GEMINI_JUDGMENT_MODEL", os.getenv("GEMINI_MODEL", "gemini-2.5-pro")),
        help=(
            "Gemini model used as the RAGAS judge (default: GEMINI_JUDGMENT_MODEL env var, "
            "falls back to GEMINI_MODEL). Use a stronger model here for reliable evaluation."
        ),
    )
    p.add_argument(
        "--skip-ragas",
        action="store_true",
        help="Collect retrieval metrics but skip RAGAS LLM evaluation",
    )
    p.add_argument(
        "--golden-set",
        default=None,
        help=(
            "Path to an alternative JSONL golden set (default: eval/rag/golden_set.jsonl). "
            "Results are saved alongside the golden set as <name>_results.json."
        ),
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    golden_path = Path(args.golden_set) if args.golden_set else None

    if args.offline:
        sys.exit(run_offline(golden_set_path=golden_path))
    else:
        sys.exit(
            run_live(
                api_url=args.api_url,
                api_token=args.api_token,
                gemini_api_key=args.gemini_api_key,
                gemini_model=args.gemini_model,
                gemini_judgment_model=args.gemini_judgment_model,
                skip_ragas=args.skip_ragas,
                golden_set_path=golden_path,
            )
        )
