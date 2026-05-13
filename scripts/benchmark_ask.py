"""
Time a single /api/ask round-trip. Run API first: python backend/run.py

  python scripts/benchmark_ask.py
  python scripts/benchmark_ask.py --url http://127.0.0.1:8000 --question "What is RAG?"
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def main() -> int:
    p = argparse.ArgumentParser(description="Benchmark POST /api/ask")
    p.add_argument("--url", default="http://127.0.0.1:8000", help="API base URL")
    p.add_argument("--question", default="What is in the knowledge base?", help="Test question")
    p.add_argument("-k", type=int, default=5, help="Retrieval k")
    p.add_argument("--model", default="llama3.1:8b", help="Ollama model name")
    p.add_argument(
        "--similarity-min",
        type=float,
        default=None,
        help="Optional cosine similarity floor (same as API similarity_min)",
    )
    args = p.parse_args()
    base = args.url.rstrip("/")
    body = {"question": args.question, "k": args.k, "model": args.model}
    if args.similarity_min is not None:
        body["similarity_min"] = args.similarity_min
    payload = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{base}/api/ask",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            body = resp.read()
    except urllib.error.HTTPError as e:
        sys.stderr.write(f"HTTP {e.code}: {e.read().decode('utf-8', errors='replace')}\n")
        return 1
    except urllib.error.URLError as e:
        sys.stderr.write(f"Request failed: {e}\n")
        return 1
    dt = time.perf_counter() - t0
    data = json.loads(body.decode("utf-8"))
    print(f"total_s={dt:.3f} answer_chars={len(data.get('answer', ''))} sources={len(data.get('sources', []))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
