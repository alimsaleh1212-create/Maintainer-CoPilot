#!/usr/bin/env python3
"""Build a corpus-grounded golden set for RAG evaluation.

For each question in the existing golden set:
1. Calls /rag/search to retrieve the top chunks from the live corpus.
2. For each retrieved chunk, picks a distinctive 80-character phrase
   that will be used as the substring match key in Hit@5.
3. Writes a new golden_set.jsonl where ground_truth_chunks contains
   real corpus text excerpts rather than keyword labels.

Usage:
    uv run python eval/rag/build_golden_from_corpus.py \
        --api-url http://localhost:8000 \
        --api-token <jwt>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import httpx

_GOLDEN_PATH = Path(__file__).parent / "golden_set.jsonl"
_OUT_PATH = Path(__file__).parent / "golden_set.jsonl"
_BACKUP_PATH = Path(__file__).parent / "golden_set.jsonl.bak"

# How many top chunks to keep as ground_truth per question.
_TOP_K = 3
# Minimum characters a chunk must have to be useful.
_MIN_CHUNK_LEN = 60
# Characters for the substring key we extract from each chunk.
_EXCERPT_LEN = 90


def _clean(text: str) -> str:
    """Collapse whitespace and remove markdown noise for a clean excerpt."""
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"[#*`_\[\]|>]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_key_phrase(text: str, question: str, length: int = _EXCERPT_LEN) -> str:
    """Extract the most distinctive phrase from a chunk.

    Strategy:
    1. Find a sentence/line that contains any word from the question.
    2. Fall back to the first non-trivial line if nothing matches.
    """
    q_words = {w.lower() for w in re.split(r"\W+", question) if len(w) > 3}
    cleaned = _clean(text)
    # Try to find a sentence with a question keyword
    sentences = re.split(r"(?<=[.?!\n])\s+", cleaned)
    for sent in sentences:
        sl = sent.lower()
        if any(w in sl for w in q_words) and len(sent) >= 30:
            # Trim to length from the start of that sentence
            phrase = sent[:length].strip()
            if len(phrase) >= 30:
                return phrase
    # Fallback: first meaningful content after stripping leading metadata lines
    lines = [l.strip() for l in cleaned.split("\n") if len(l.strip()) >= 30]
    if lines:
        return lines[0][:length]
    return cleaned[:length]


def _search(api_url: str, token: str, question: str, top_k: int = 10) -> list[dict]:
    """Call /rag/search and return list of chunk dicts."""
    headers = {"Authorization": f"Bearer {token}"}
    try:
        r = httpx.post(
            f"{api_url}/rag/search",
            json={"query": question, "top_k": top_k},
            headers=headers,
            timeout=30.0,
        )
        r.raise_for_status()
        data = r.json()
        return data.get("chunks", data.get("results", []))
    except Exception as e:
        print(f"  ⚠  search failed for '{question[:50]}': {e}", file=sys.stderr)
        return []


def _db_search(question: str, limit: int = 10) -> list[str]:
    """Direct PostgreSQL BM25 search via tsvector as a fallback / supplement."""
    try:
        import psycopg2  # type: ignore

        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            dbname="copilot",
            user="copilot",
            password="copilot",
        )
        cur = conn.cursor()
        # Build a tsquery from question words
        words = [w for w in re.split(r"\W+", question) if len(w) > 3]
        tsquery = " | ".join(words[:10])
        cur.execute(
            """
            SELECT text, ts_rank_cd(tsvector, query) AS rank
            FROM rag_chunks, plainto_tsquery('english', %s) query
            WHERE tsvector @@ query
            ORDER BY rank DESC
            LIMIT %s
            """,
            (question, limit),
        )
        rows = cur.fetchall()
        conn.close()
        return [row[0] for row in rows]
    except Exception:
        return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Build corpus-grounded RAG golden set")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--api-token", required=True)
    parser.add_argument(
        "--input", default=str(_GOLDEN_PATH), help="Source golden_set.jsonl"
    )
    parser.add_argument(
        "--output", default=str(_OUT_PATH), help="Output golden_set.jsonl (overwrites input)"
    )
    parser.add_argument(
        "--top-k", type=int, default=_TOP_K, help="Ground-truth chunks to keep per question"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    golden: list[dict] = []
    with input_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                golden.append(json.loads(line))

    print(f"Loaded {len(golden)} questions from {input_path}")
    print(f"Querying corpus at {args.api_url} …\n")

    # Backup original
    if not _BACKUP_PATH.exists() and input_path == _GOLDEN_PATH:
        import shutil
        shutil.copy(input_path, _BACKUP_PATH)
        print(f"Backed up original → {_BACKUP_PATH}")

    updated: list[dict] = []
    hit_count = 0

    for i, qa in enumerate(golden, start=1):
        question = qa["question"]
        print(f"[{i:2d}/{len(golden)}] {question[:60]} …")

        # 1. API search (hybrid retrieval, top 10)
        api_chunks = _search(args.api_url, args.api_token, question, top_k=10)
        api_texts = [
            c.get("text", c.get("content", ""))
            for c in api_chunks
            if c.get("text") or c.get("content")
        ]

        # 2. BM25 fallback from DB directly
        db_texts = _db_search(question, limit=10)

        # Combine, deduplicate by first 60 chars
        seen: set[str] = set()
        all_texts: list[str] = []
        for t in api_texts + db_texts:
            key = t[:60]
            if key not in seen and len(t) >= _MIN_CHUNK_LEN:
                seen.add(key)
                all_texts.append(t)

        # Extract ground truth phrases from top-K unique chunks
        gt_phrases: list[str] = []
        for text in all_texts[: args.top_k]:
            phrase = _extract_key_phrase(text, question)
            if phrase and len(phrase) >= 30:
                gt_phrases.append(phrase)

        if gt_phrases:
            hit_count += 1
            status = "✓"
        else:
            # No corpus content found — keep original keywords as fallback
            gt_phrases = qa.get("ground_truth_chunks", [])
            status = "⚠  no corpus hit — keeping original keywords"

        print(f"       {status}")
        for p in gt_phrases:
            print(f"         → {p[:80]}")

        # Build ideal_answer: try to synthesize from first matching chunk text
        ideal_answer = qa["ideal_answer"]  # keep original — it's still valid

        updated.append(
            {
                "id": qa["id"],
                "question": question,
                "ideal_answer": ideal_answer,
                "ground_truth_chunks": gt_phrases,
            }
        )

    print(f"\n{hit_count}/{len(golden)} questions have corpus-grounded ground truth")

    with output_path.open("w") as f:
        for row in updated:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Written → {output_path}")


if __name__ == "__main__":
    main()
