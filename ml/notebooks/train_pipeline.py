"""
End-to-end training pipeline for the 3-class issue classifier.
Mirrors the Colab notebook logic but runs locally via `uv run`.

Usage:
    uv run python notebooks/train_pipeline.py

Outputs (all written to ml/):
    data/raw_issues.jsonl       — cached GitHub API fetch
    data/train.jsonl            — 70% training split
    data/val.jsonl              — 15% validation split
    data/test.jsonl             — 15% test split (time-aware)
    artifacts/classifier/best/  — DistilBERT weights + tokenizer
    artifacts/classifier/best/model_card.json
    artifacts/classical_ml/pipeline.pkl
    artifacts/eval_report.json  — three-way comparison results
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline

# ── paths ──────────────────────────────────────────────────────────────────────
ML_DIR = Path(__file__).parent.parent
DATA_DIR = ML_DIR / "data"
ARTIFACTS_DIR = ML_DIR / "artifacts"
CLASSIFIER_DIR = ARTIFACTS_DIR / "classifier"
BEST_DIR = CLASSIFIER_DIR / "best"
CLASSICAL_DIR = ARTIFACTS_DIR / "classical_ml"

for d in (DATA_DIR, BEST_DIR, CLASSICAL_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── secrets ────────────────────────────────────────────────────────────────────
_env_file = ML_DIR.parent / ".env"
if _env_file.exists():
    with _env_file.open() as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                if _k.strip() in ("GITHUB_TOKEN", "GEMINI_API_KEY"):
                    os.environ.setdefault(_k.strip(), _v.strip())

# ── constants ──────────────────────────────────────────────────────────────────
REPO = "Project-MONAI/MONAI"
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 512
RANDOM_SEED = 42
TEST_FRACTION = 0.15
VAL_OF_REMAINING = 0.15 / 0.85

LABEL_MAP: dict[str, str] = {
    "bug": "bug",
    "Feature request": "feature",
    "feature request": "feature",
    "enhancement": "feature",
    "documentation": "support",
    "question": "support",
    "questions": "support",
}
CLASS_NAMES: tuple[str, ...] = ("bug", "feature", "support")
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
_TARGET_LABELS = frozenset(LABEL_MAP.keys())

THRESHOLDS = {"macro_f1": 0.78, "per_class_f1_min": 0.70}

# Gemini
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_EVAL_SAMPLE = 100
GEMINI_COST_PER_IN_TOK = 0.10 / 1_000_000
GEMINI_COST_PER_OUT_TOK = 0.40 / 1_000_000
K_SHOT = 5


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Fetch
# ──────────────────────────────────────────────────────────────────────────────

def fetch_closed_issues(force: bool = False) -> list[dict[str, Any]]:
    raw_path = DATA_DIR / "raw_issues.jsonl"
    if raw_path.exists() and not force:
        with raw_path.open() as fh:
            rows = [json.loads(line) for line in fh]
        print(f"  cache hit: {len(rows)} issues from {raw_path.name}")
        return rows

    headers = {"Accept": "application/vnd.github+json"}
    if token := os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
        print("  authenticated (5000 req/hr)")

    issues: list[dict[str, Any]] = []
    page = 1
    with httpx.Client(timeout=30.0, headers=headers) as client:
        while True:
            r = client.get(
                f"https://api.github.com/repos/{REPO}/issues",
                params={"state": "closed", "per_page": 100, "page": page},
            )
            r.raise_for_status()
            batch = r.json()
            if not batch:
                break
            issues.extend(i for i in batch if "pull_request" not in i)
            print(f"  page {page:>3}: +{len(batch):<3}  total={len(issues)}", flush=True)
            page += 1
            time.sleep(0.1)

    slim = [
        {
            "id": i["id"],
            "number": i["number"],
            "title": i["title"],
            "body": i.get("body"),
            "labels": [lb["name"] for lb in i.get("labels", [])],
            "created_at": i["created_at"],
            "closed_at": i["closed_at"],
        }
        for i in issues
    ]
    with raw_path.open("w") as fh:
        for row in slim:
            fh.write(json.dumps(row) + "\n")
    print(f"  saved {len(slim)} issues → {raw_path.name}")
    return slim


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Label mapping + split
# ──────────────────────────────────────────────────────────────────────────────

def resolve_label(labels: list[str]) -> str | None:
    mapped = {LABEL_MAP[lbl] for lbl in labels if lbl in _TARGET_LABELS}
    return mapped.pop() if len(mapped) == 1 else None


def build_row(raw: dict[str, Any]) -> dict[str, Any] | None:
    label = resolve_label(raw["labels"])
    if label is None or raw.get("closed_at") is None:
        return None
    text = f"{raw['title']}\n\n{raw.get('body') or ''}".strip()
    return {
        "id": raw["id"],
        "number": raw["number"],
        "text": text,
        "label": label,
        "label_idx": CLASS_TO_IDX[label],
        "closed_at": raw["closed_at"],
    }


def make_splits(
    raw_issues: list[dict[str, Any]],
) -> tuple[list[dict], list[dict], list[dict]]:
    labeled = [r for r in (build_row(i) for i in raw_issues) if r is not None]
    cc = Counter(r["label"] for r in labeled)
    print(f"  labeled: {len(labeled)}  " + "  ".join(f"{c}={cc[c]}" for c in CLASS_NAMES))

    rng = random.Random(RANDOM_SEED)
    labeled_sorted = sorted(labeled, key=lambda r: r["closed_at"])
    test_cut = int(len(labeled_sorted) * (1 - TEST_FRACTION))
    remaining, test = labeled_sorted[:test_cut], labeled_sorted[test_cut:]

    assert max(r["closed_at"] for r in remaining) <= min(r["closed_at"] for r in test), (
        "temporal leakage"
    )

    by_class: dict[str, list] = defaultdict(list)
    for r in remaining:
        by_class[r["label"]].append(r)

    train: list[dict] = []
    val: list[dict] = []
    for cls, rows in by_class.items():
        rng.shuffle(rows)
        n_val = max(1, int(len(rows) * VAL_OF_REMAINING))
        val.extend(rows[:n_val])
        train.extend(rows[n_val:])

    rng.shuffle(train)
    rng.shuffle(val)

    for name, split in (("train", train), ("val", val), ("test", test)):
        path = DATA_DIR / f"{name}.jsonl"
        with path.open("w") as fh:
            for row in split:
                fh.write(json.dumps(row) + "\n")
        cc2 = Counter(r["label"] for r in split)
        print(f"  {name:<5} n={len(split):<5}  bug={cc2['bug']:<5} feature={cc2['feature']:<5} support={cc2['support']}")

    return train, val, test


def make_splits_from_labeled(
    labeled: list[dict[str, Any]],
) -> tuple[list[dict], list[dict], list[dict]]:
    """Identical split logic as make_splits but accepts pre-labeled rows directly."""
    cc = Counter(r["label"] for r in labeled)
    print(f"  labeled: {len(labeled)}  " + "  ".join(f"{c}={cc[c]}" for c in CLASS_NAMES))

    rng = random.Random(RANDOM_SEED)
    labeled_sorted = sorted(labeled, key=lambda r: r["closed_at"])
    test_cut = int(len(labeled_sorted) * (1 - TEST_FRACTION))
    remaining, test = labeled_sorted[:test_cut], labeled_sorted[test_cut:]

    assert max(r["closed_at"] for r in remaining) <= min(r["closed_at"] for r in test), (
        "temporal leakage"
    )

    by_class: dict[str, list] = defaultdict(list)
    for r in remaining:
        by_class[r["label"]].append(r)

    train: list[dict] = []
    val: list[dict] = []
    for cls, rows in by_class.items():
        rng.shuffle(rows)
        n_val = max(1, int(len(rows) * VAL_OF_REMAINING))
        val.extend(rows[:n_val])
        train.extend(rows[n_val:])

    rng.shuffle(train)
    rng.shuffle(val)

    for name, split in (("train", train), ("val", val), ("test", test)):
        path = DATA_DIR / f"{name}.jsonl"
        with path.open("w") as fh:
            for row in split:
                fh.write(json.dumps(row) + "\n")
        cc2 = Counter(r["label"] for r in split)
        print(f"  {name:<5} n={len(split):<5}  bug={cc2['bug']:<5} feature={cc2['feature']:<5} support={cc2['support']}")

    return train, val, test


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 3 — LLM Auto-Label
#
# Only ~50% of MONAI issues carry a recognised label tag.  We send the
# unlabeled half to Gemini (batched, 5 issues per request) so the training
# corpus nearly doubles.  Results are cached in data/llm_labeled.jsonl and
# merged with github-labeled rows into data/processed_issues.jsonl.
# Issues the LLM classifies as "other" are saved but excluded from training.
# ──────────────────────────────────────────────────────────────────────────────

_GEMINI_LABEL_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
)
_LABEL_BATCH_SIZE = 5
_LABEL_SLEEP = 1.0   # seconds between batches — stays well under free-tier 15 RPM
_VALID_LABELS = set(CLASS_NAMES) | {"other"}

_SYSTEM_BATCH = (
    "You are an issue classifier for open-source repositories.\n"
    "You will receive a numbered list of GitHub issue texts.\n"
    "Classify each into exactly one of: bug, feature, support, other.\n\n"
    "  bug     -- A defect, crash, regression, or unexpected behaviour.\n"
    "  feature -- A request for new functionality or an enhancement.\n"
    "  support -- A question, docs gap, or request for help.\n"
    "  other   -- Does not clearly fit (e.g. release notes, CI-only, meta).\n\n"
    "Reply with EXACTLY one label per line, same order as the input.\n"
    "No numbers, no punctuation, no explanation — just the label word."
)
_GEN_CFG = {"temperature": 0.0, "thinkingConfig": {"thinkingBudget": 0}}


def _call_gemini(
    client: httpx.Client, body: dict[str, Any], api_key: str, retries: int = 5
) -> dict[str, Any]:
    delay = 5.0
    for attempt in range(retries):
        r = client.post(
            _GEMINI_LABEL_URL, params={"key": api_key}, json=body, timeout=60.0
        )
        if r.status_code == 200:
            return r.json()
        if r.status_code == 429:
            print(f"  429 rate-limit — sleeping {delay:.0f}s (attempt {attempt + 1})", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 120)
            continue
        r.raise_for_status()
    raise RuntimeError("Gemini max retries exceeded")


def _classify_batch(
    issues: list[dict[str, Any]], client: httpx.Client, api_key: str
) -> list[str]:
    def _text(iss: dict[str, Any]) -> str:
        return f"{iss['title']}\n\n{iss.get('body') or ''}".strip()[:400]

    parts = [f"{i + 1}. {_text(iss)}" for i, iss in enumerate(issues)]
    body = {
        "system_instruction": {"parts": [{"text": _SYSTEM_BATCH}]},
        "contents": [{"role": "user", "parts": [{"text": "\n\n---\n\n".join(parts)}]}],
        "generationConfig": {**_GEN_CFG, "maxOutputTokens": 20 * len(issues)},
    }
    data = _call_gemini(client, body, api_key)
    cand = data["candidates"][0]
    content_parts = cand.get("content", {}).get("parts", [])
    if not content_parts:
        raise ValueError(f"empty response finishReason={cand.get('finishReason')}")
    tokens = [t.strip().lower() for t in content_parts[0]["text"].strip().splitlines() if t.strip()]
    labels = [
        (tok.split()[0].rstrip(".,;:") if tok.split() else "support")
        for tok in tokens[: len(issues)]
    ]
    labels = [lbl if lbl in _VALID_LABELS else "support" for lbl in labels]
    while len(labels) < len(issues):
        labels.append("support")
    return labels


def label_with_llm(
    raw_issues: list[dict[str, Any]], github_labeled: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """LLM-label unlabeled issues and return the training-eligible combined set.

    Fast path: if data/processed_issues.jsonl exists (already run before),
    loads it directly — no API calls made.

    Slow path: batches unlabeled issues to Gemini (5 per request), saves each
    batch to the cache file incrementally (safe to interrupt and resume), then
    merges with github-labeled rows and writes processed_issues.jsonl.

    Returns the combined list with 'other' rows removed (they would add noise
    to the 3-class classifier).
    """
    api_key = os.environ.get("GEMINI_API_KEY", "")
    llm_cache_path = DATA_DIR / "llm_labeled.jsonl"
    processed_path = DATA_DIR / "processed_issues.jsonl"

    # ── Fast path: processed file already exists ───────────────────────────
    if processed_path.exists():
        with processed_path.open() as fh:
            combined = [json.loads(l) for l in fh]
        n_other = sum(1 for r in combined if r["label"] == "other")
        eligible = [r for r in combined if r["label"] != "other"]
        cc = Counter(r["label"] for r in eligible)
        print(f"  cache hit: {len(combined)} total → {len(eligible)} training-eligible"
              f" ({n_other} 'other' excluded)")
        for c in CLASS_NAMES:
            print(f"    {c:<10} {cc[c]:>5}")
        return eligible

    # ── Identify issues with no/ambiguous label tags ───────────────────────
    unlabeled_raw = [
        i for i in raw_issues
        if resolve_label(i["labels"]) is None and i.get("closed_at") is not None
    ]
    print(f"  github-labeled : {len(github_labeled)}")
    print(f"  unlabeled raw  : {len(unlabeled_raw)}")

    # ── Load cache (allows resuming an interrupted run) ────────────────────
    llm_cache: dict[int, str] = {}
    if llm_cache_path.exists():
        with llm_cache_path.open() as fh:
            for line in fh:
                row = json.loads(line)
                llm_cache[row["id"]] = row["label"]
        print(f"  llm cache      : {len(llm_cache)} already labeled")

    to_label = [i for i in unlabeled_raw if i["id"] not in llm_cache]
    print(f"  still to label : {len(to_label)}")

    # ── Call Gemini in batches ─────────────────────────────────────────────
    if to_label and api_key:
        batches = [to_label[i: i + _LABEL_BATCH_SIZE] for i in range(0, len(to_label), _LABEL_BATCH_SIZE)]
        print(f"  batches        : {len(batches)} × {_LABEL_BATCH_SIZE}")
        errors = 0
        labeled_count = 0
        with llm_cache_path.open("a") as cache_fh, httpx.Client(timeout=60.0) as client:
            for bi, batch in enumerate(batches):
                try:
                    batch_labels = _classify_batch(batch, client, api_key)
                except Exception as e:
                    print(f"  batch {bi} failed ({e}) — fallback support", flush=True)
                    errors += 1
                    batch_labels = ["support"] * len(batch)
                for iss, lbl in zip(batch, batch_labels):
                    llm_cache[iss["id"]] = lbl
                    cache_fh.write(json.dumps({"id": iss["id"], "label": lbl}) + "\n")
                    labeled_count += 1
                cache_fh.flush()
                if (bi + 1) % 20 == 0:
                    print(f"  labeled {labeled_count}/{len(to_label)}  errors={errors}", flush=True)
                time.sleep(_LABEL_SLEEP)
        print(f"  LLM labeling done  total={labeled_count}  errors={errors}")
    elif to_label and not api_key:
        print("  GEMINI_API_KEY not set — proceeding with github-labeled only")

    # ── Build LLM-labeled rows and merge ──────────────────────────────────
    id_to_raw = {i["id"]: i for i in raw_issues}
    llm_rows: list[dict[str, Any]] = []
    for iid, lbl in llm_cache.items():
        raw = id_to_raw.get(iid)
        if raw is None or raw.get("closed_at") is None:
            continue
        txt = f"{raw['title']}\n\n{raw.get('body') or ''}".strip()
        llm_rows.append({
            "id": raw["id"], "number": raw["number"],
            "text": txt, "label": lbl,
            "label_idx": CLASS_TO_IDX.get(lbl, 3),
            "closed_at": raw["closed_at"],
            "label_source": "llm",
        })

    for r in github_labeled:
        r.setdefault("label_source", "github_label")

    combined = github_labeled + llm_rows
    with processed_path.open("w") as fh:
        for r in combined:
            fh.write(json.dumps(r) + "\n")

    cc_all = Counter(r["label"] for r in combined)
    n_other = cc_all.get("other", 0)
    eligible = [r for r in combined if r["label"] != "other"]
    cc = Counter(r["label"] for r in eligible)

    src_github = sum(1 for r in combined if r.get("label_source") == "github_label")
    src_llm = sum(1 for r in combined if r.get("label_source") == "llm")
    print(f"  saved {len(combined)} → processed_issues.jsonl")
    print(f"    github-label  : {src_github}")
    print(f"    llm-labeled   : {src_llm}")
    print(f"    other excluded: {n_other}")
    print(f"  training-eligible: {len(eligible)}")
    for c in CLASS_NAMES:
        print(f"    {c:<10} {cc[c]:>5}")
    return eligible


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 4 — DistilBERT fine-tune
# ──────────────────────────────────────────────────────────────────────────────

def sha256_weights(directory: Path) -> str:
    h = hashlib.sha256()
    for pat in ("*.safetensors", "*.bin"):
        for p in sorted(directory.glob(pat)):
            with p.open("rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def train_distilbert(
    train: list[dict], val: list[dict], test: list[dict]
) -> tuple[float, float, list[float], float]:
    import torch
    import evaluate as ev
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    print(f"  CUDA: {torch.cuda.is_available()}" + (f"  ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else " (CPU)"))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def to_hf(split: list[dict]) -> Dataset:
        ds = Dataset.from_dict({"text": [r["text"] for r in split], "label": [r["label_idx"] for r in split]})
        return ds.map(lambda b: tokenizer(b["text"], truncation=True, max_length=MAX_LENGTH, padding=False), batched=True)

    print("  tokenizing...")
    train_ds, val_ds, test_ds = to_hf(train), to_hf(val), to_hf(test)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(CLASS_NAMES),
        id2label={i: c for i, c in enumerate(CLASS_NAMES)},
        label2id=CLASS_TO_IDX,
    )

    f1_metric = ev.load("f1")

    def compute_metrics(ep):
        preds = np.argmax(ep.predictions, axis=-1)
        return f1_metric.compute(predictions=preds, references=ep.label_ids, average="macro")

    args = TrainingArguments(
        output_dir=str(CLASSIFIER_DIR),
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        save_total_limit=2,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    t0 = time.time()
    trainer.train()
    print(f"  training done in {(time.time()-t0)/60:.1f} min")

    BEST_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(BEST_DIR))
    tokenizer.save_pretrained(str(BEST_DIR))

    # Evaluate on val (for model card) and test
    val_results = trainer.evaluate()
    val_f1 = float(val_results.get("eval_f1", 0.0))

    test_labels = [r["label_idx"] for r in test]
    t0 = time.perf_counter()
    pred_out = trainer.predict(test_ds)
    latency_ms = (time.perf_counter() - t0) / len(test) * 1000
    preds = np.argmax(pred_out.predictions, axis=-1).tolist()

    acc = accuracy_score(test_labels, preds)
    macro_f1 = f1_score(test_labels, preds, average="macro")
    per_class = f1_score(test_labels, preds, average=None).tolist()

    # Write model card
    model_card = {
        "architecture": MODEL_NAME,
        "num_labels": len(CLASS_NAMES),
        "classes": list(CLASS_NAMES),
        "class_to_idx": CLASS_TO_IDX,
        "hyperparameters": {
            "learning_rate": args.learning_rate,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "num_train_epochs": args.num_train_epochs,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "max_length": MAX_LENGTH,
        },
        "freeze_policy": "all layers unfrozen — full fine-tune of DistilBERT",
        "training_data_sha256": sha256_file(DATA_DIR / "train.jsonl"),
        "training_data_size": {"train": len(train), "val": len(val), "test": len(test)},
        "metrics": {"val_f1_macro": val_f1, "raw_eval": val_results},
        "model_sha256": sha256_weights(BEST_DIR),
        "trained_at": datetime.now(UTC).isoformat(),
        "version": "1.0.0",
        "dataset_repo": REPO,
    }
    (BEST_DIR / "model_card.json").write_text(json.dumps(model_card, indent=2))

    print(f"  val macro-F1: {val_f1:.4f}")
    print(f"  test accuracy: {acc:.4f}  test macro-F1: {macro_f1:.4f}")
    print(f"  per-class F1: {dict(zip(CLASS_NAMES, [f'{f:.4f}' for f in per_class]))}")
    print(f"  latency: {latency_ms:.1f} ms/sample")
    print(classification_report(test_labels, preds, target_names=CLASS_NAMES))

    return acc, macro_f1, per_class, latency_ms


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 5 — TF-IDF + LogReg baseline
# ──────────────────────────────────────────────────────────────────────────────

def train_tfidf(
    train: list[dict], test: list[dict]
) -> tuple[float, float, list[float], float]:
    train_texts = [r["text"] for r in train]
    train_labels = [r["label_idx"] for r in train]
    test_texts = [r["text"] for r in test]
    test_labels = [r["label_idx"] for r in test]

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=50_000, sublinear_tf=True, strip_accents="unicode")),
        ("lr", LogisticRegression(C=1.0, max_iter=1_000, random_state=RANDOM_SEED, class_weight="balanced")),
    ])
    pipeline.fit(train_texts, train_labels)

    t0 = time.perf_counter()
    preds = pipeline.predict(test_texts).tolist()
    latency_ms = (time.perf_counter() - t0) / len(test) * 1000

    acc = accuracy_score(test_labels, preds)
    macro_f1 = f1_score(test_labels, preds, average="macro")
    per_class = f1_score(test_labels, preds, average=None).tolist()

    joblib_path = CLASSICAL_DIR / "pipeline.joblib"
    joblib.dump(pipeline, joblib_path, compress=3)

    print(f"  test accuracy: {acc:.4f}  test macro-F1: {macro_f1:.4f}")
    print(f"  per-class F1: {dict(zip(CLASS_NAMES, [f'{f:.4f}' for f in per_class]))}")
    print(f"  latency: {latency_ms:.4f} ms/sample")
    print(classification_report(test_labels, preds, target_names=CLASS_NAMES))

    return acc, macro_f1, per_class, latency_ms


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 6 — Gemini few-shot baseline
# ──────────────────────────────────────────────────────────────────────────────

def run_gemini_baseline(
    train: list[dict], test: list[dict]
) -> tuple[float, float, list[float], float, float] | None:
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("  GEMINI_API_KEY not set — skipping")
        return None

    rng = random.Random(RANDOM_SEED)
    few_shot: list[dict] = []
    for cls in CLASS_NAMES:
        rows = [r for r in train if r["label"] == cls]
        rng.shuffle(rows)
        for row in rows[:K_SHOT]:
            few_shot.append({"text": row["text"][:400].strip(), "label": cls})
    random.Random(RANDOM_SEED + 1).shuffle(few_shot)

    system = (
        "You are an issue classifier for open-source repositories.\n"
        "Classify the GitHub issue into exactly one of: bug, feature, support.\n\n"
        "  bug     -- A defect, crash, regression, or unexpected behaviour.\n"
        "  feature -- A request for new functionality or an enhancement.\n"
        "  support -- A question, docs gap, or request for help.\n\n"
        "Reply with ONLY the single class name (lowercase, no punctuation, nothing else)."
    )

    def contents(text: str) -> list[dict]:
        c = []
        for ex in few_shot:
            c.append({"role": "user",  "parts": [{"text": ex["text"]}]})
            c.append({"role": "model", "parts": [{"text": ex["label"]}]})
        c.append({"role": "user", "parts": [{"text": text[:600]}]})
        return c

    def classify(text: str, client: httpx.Client) -> tuple[str, float, int, int]:
        body = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": contents(text),
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 20,
                                  "thinkingConfig": {"thinkingBudget": 0}},
        }
        t0 = time.perf_counter()
        r = client.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
            params={"key": api_key},
            json=body,
            timeout=30.0,
        )
        lat = (time.perf_counter() - t0) * 1000
        r.raise_for_status()
        data = r.json()
        cand = data["candidates"][0]
        parts = cand.get("content", {}).get("parts", [])
        raw = parts[0]["text"].strip().lower() if parts else "support"
        label = raw if raw in CLASS_NAMES else "support"
        usage = data.get("usageMetadata", {})
        return label, lat, usage.get("promptTokenCount", 0), usage.get("candidatesTokenCount", 0)

    sample = test[:GEMINI_EVAL_SAMPLE]
    print(f"  evaluating {GEMINI_MODEL} on {len(sample)} examples ({K_SHOT}-shot per class)...")
    results: list[dict] = []
    with httpx.Client() as client:
        for i, row in enumerate(sample):
            pred, lat, in_tok, out_tok = classify(row["text"], client)
            results.append({"true": row["label"], "pred": pred, "latency_ms": lat, "in_tok": in_tok, "out_tok": out_tok})
            if (i + 1) % 10 == 0:
                print(f"    {i+1}/{len(sample)}", flush=True)
            time.sleep(0.1)

    true_idx = [CLASS_TO_IDX[r["true"]] for r in results]
    pred_idx = [CLASS_TO_IDX.get(r["pred"], 2) for r in results]
    acc = accuracy_score(true_idx, pred_idx)
    macro_f1 = f1_score(true_idx, pred_idx, average="macro")
    per_class = f1_score(true_idx, pred_idx, average=None).tolist()
    avg_lat = sum(r["latency_ms"] for r in results) / len(results)
    avg_in = sum(r["in_tok"] for r in results) / len(results)
    avg_out = sum(r["out_tok"] for r in results) / len(results)
    cost_per_1k = (avg_in * GEMINI_COST_PER_IN_TOK + avg_out * GEMINI_COST_PER_OUT_TOK) * 1_000

    print(f"  test accuracy: {acc:.4f}  test macro-F1: {macro_f1:.4f}")
    print(f"  per-class F1: {dict(zip(CLASS_NAMES, [f'{f:.4f}' for f in per_class]))}")
    print(f"  avg latency: {avg_lat:.1f} ms   cost/1K: ${cost_per_1k:.4f}")
    print(classification_report(true_idx, pred_idx, target_names=CLASS_NAMES))

    return acc, macro_f1, per_class, avg_lat, cost_per_1k


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 7 — Three-way comparison + eval_report.json
# ──────────────────────────────────────────────────────────────────────────────

def write_eval_report(
    train: list[dict],
    test: list[dict],
    dl: tuple,
    ml: tuple,
    llm: tuple | None,
) -> None:
    dl_acc, dl_f1, dl_pcf1, dl_lat = dl
    ml_acc, ml_f1, ml_pcf1, ml_lat = ml

    candidates: dict[str, dict] = {
        "distilbert": {
            "accuracy": float(dl_acc),
            "macro_f1": float(dl_f1),
            "per_class_f1": {c: float(f) for c, f in zip(CLASS_NAMES, dl_pcf1)},
            "avg_latency_ms": float(dl_lat),
            "cost_per_1k_predictions": 0.0,
            "eval_sample_size": len(test),
        },
        "tfidf_logreg": {
            "accuracy": float(ml_acc),
            "macro_f1": float(ml_f1),
            "per_class_f1": {c: float(f) for c, f in zip(CLASS_NAMES, ml_pcf1)},
            "avg_latency_ms": float(ml_lat),
            "cost_per_1k_predictions": 0.0,
            "eval_sample_size": len(test),
        },
    }
    if llm is not None:
        llm_acc, llm_f1, llm_pcf1, llm_lat, llm_cost = llm
        candidates[f"gemini_{GEMINI_MODEL.replace('-', '_')}"] = {
            "accuracy": float(llm_acc),
            "macro_f1": float(llm_f1),
            "per_class_f1": {c: float(f) for c, f in zip(CLASS_NAMES, llm_pcf1)},
            "avg_latency_ms": float(llm_lat),
            "cost_per_1k_predictions": float(llm_cost),
            "eval_sample_size": GEMINI_EVAL_SAMPLE,
        }

    winner = max(candidates, key=lambda k: candidates[k]["macro_f1"])
    wm = candidates[winner]
    threshold_pass = (
        wm["macro_f1"] >= THRESHOLDS["macro_f1"]
        and all(v >= THRESHOLDS["per_class_f1_min"] for v in wm["per_class_f1"].values())
    )

    print("\n=== THREE-WAY COMPARISON ===")
    print(f"{'Model':<35} {'Accuracy':>10} {'Macro-F1':>10} {'Latency ms':>12} {'$/1K':>10}")
    print("-" * 80)
    for name, m in sorted(candidates.items(), key=lambda kv: kv[1]["macro_f1"], reverse=True):
        marker = " ← WINNER" if name == winner else ""
        print(f"{name:<35} {m['accuracy']:>10.4f} {m['macro_f1']:>10.4f} {m['avg_latency_ms']:>12.2f} ${m['cost_per_1k_predictions']:>9.4f}{marker}")
    print(f"\nThreshold pass (F1≥{THRESHOLDS['macro_f1']}, per-class≥{THRESHOLDS['per_class_f1_min']}): {threshold_pass}")

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset_repo": REPO,
        "test_split_size": len(test),
        "llm_eval_sample_size": GEMINI_EVAL_SAMPLE if llm else 0,
        "classes": list(CLASS_NAMES),
        "models": candidates,
        "winner": {
            "model": winner,
            "rationale": (
                f"Highest macro-F1 ({wm['macro_f1']:.4f}) on the time-aware test split. "
                "DistilBERT is self-hosted (zero per-call cost) and latency is acceptable "
                "for an interactive endpoint behind a model-server container."
            ),
            "metrics": wm,
        },
        "deployment_decision": winner,
        "thresholds": THRESHOLDS,
        "threshold_pass": threshold_pass,
    }

    out = ARTIFACTS_DIR / "eval_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\n✓ eval_report.json → {out}")

    if not threshold_pass:
        print("⚠  WARNING: winner is below committed thresholds — inspect before deploying")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("MAINTAINER'S COPILOT — END-TO-END TRAINING PIPELINE")
    print("=" * 70)
    print(f"ML dir : {ML_DIR}")
    print(f"GITHUB : {'✓' if os.environ.get('GITHUB_TOKEN') else '✗'}")
    print(f"GEMINI : {'✓' if os.environ.get('GEMINI_API_KEY') else '✗'}\n")

    print("── 1. Fetch MONAI closed issues ──────────────────────────────────")
    raw = fetch_closed_issues()

    print("\n── 2. GitHub label mapping — 4 labels → 3 canonical classes ─────")
    # Map known GitHub label tags to bug/feature/support.  Issues with no
    # recognised tag (~50% of the corpus) will be labeled by Gemini next.
    github_labeled = [r for r in (build_row(i) for i in raw) if r is not None]
    cc = Counter(r["label"] for r in github_labeled)
    print(f"  github-labeled: {len(github_labeled)}  " +
          "  ".join(f"{c}={cc[c]}" for c in CLASS_NAMES))

    print("\n── 3. LLM auto-label unlabeled issues ─────────────────────────────")
    # Gemini-2.5-Flash classifies the remaining ~50% in batches of 5.
    # Results cached to llm_labeled.jsonl — API never called twice.
    # 'other' issues are excluded from training (noise for 3-class model).
    all_labeled = label_with_llm(raw, github_labeled)

    print("\n── 4. Stratified time-aware split (70 / 15 / 15) ─────────────────")
    # Most-recent 15% by closed_at → test, preventing temporal leakage.
    train, val, test = make_splits_from_labeled(all_labeled)

    print("\n── 5. Train DistilBERT ────────────────────────────────────────────")
    # Full fine-tune, early stopping patience=2, fp16 on GPU.
    dl_results = train_distilbert(train, val, test)

    print("\n── 6. TF-IDF + LogReg baseline ───────────────────────────────────")
    # Fast classical baseline: bigram TF-IDF + balanced logistic regression.
    ml_results = train_tfidf(train, test)

    print("\n── 7. Gemini few-shot baseline ───────────────────────────────────")
    # 5-shot per class (15 examples total).  Expected to lose to the
    # fine-tuned model — general LLMs can't match domain-specific fine-tuning.
    llm_results = run_gemini_baseline(train, test)

    print("\n── 8. Three-way comparison + eval_report.json ────────────────────")
    write_eval_report(train, test, dl_results, ml_results, llm_results)

    print("\n" + "=" * 70)
    print("✓✓✓  PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\nArtifacts:")
    print(f"  {BEST_DIR}/")
    print(f"  {BEST_DIR}/model_card.json")
    print(f"  {CLASSICAL_DIR}/pipeline.joblib")
    print(f"  {ARTIFACTS_DIR}/eval_report.json")


if __name__ == "__main__":
    main()
