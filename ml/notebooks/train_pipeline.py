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
# SECTION 3 — DistilBERT
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
# SECTION 4 — TF-IDF + LogReg baseline
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
# SECTION 5 — Gemini few-shot baseline
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
# SECTION 6 — Three-way comparison + eval_report.json
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

    print("── 1. Fetch MONAI issues ──────────────────────────────────────────")
    raw = fetch_closed_issues()

    print("\n── 2. Load processed issues (github-labeled + LLM-labeled) ───────")
    processed_path = DATA_DIR / "processed_issues.jsonl"
    if processed_path.exists():
        with processed_path.open() as _f:
            all_labeled = [json.loads(l) for l in _f]
        # Exclude "other" — only 3 canonical classes go into training
        all_labeled = [r for r in all_labeled if r["label"] != "other"]
        from collections import Counter as _Counter
        _cc = _Counter(r["label"] for r in all_labeled)
        print(f"  {len(all_labeled)} training-eligible issues  " +
              "  ".join(f"{c}={_cc[c]}" for c in CLASS_NAMES))
    else:
        print("  processed_issues.jsonl not found — falling back to github labels only")
        all_labeled = [r for r in (build_row(i) for i in raw) if r is not None]

    print("\n── 3. Stratified time-aware split ────────────────────────────────")
    train, val, test = make_splits_from_labeled(all_labeled)

    print("\n── 4. Train DistilBERT ────────────────────────────────────────────")
    dl_results = train_distilbert(train, val, test)

    print("\n── 5. TF-IDF + LogReg baseline ───────────────────────────────────")
    ml_results = train_tfidf(train, test)

    print("\n── 6. Gemini few-shot baseline ───────────────────────────────────")
    llm_results = run_gemini_baseline(train, test)

    print("\n── 7. Three-way comparison + eval_report.json ────────────────────")
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
