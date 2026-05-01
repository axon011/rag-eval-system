"""Run the golden Q&A set against a live FastAPI server and compute honest metrics.

This is the no-LLM-judge path — we measure things that don't need a model
to score. RAGAs faithfulness/answer-relevancy require an external LLM judge
(default OpenAI, optionally Ollama via LiteLLM); a small local judge like
llama3.2:3b is unreliable for those metrics, so we leave the LLM-judge
metrics for a separate run with a stronger judge.

What we measure here:
  - retrieval_hit@k:  for each question, does any retrieved chunk contain
                      a substantial prefix of the ground-truth context?
                      Substring match, threshold = 60 chars.
  - citation_present: does the answer contain a `[Source` or `Source N`
                      reference? Tells us whether the model honored the
                      "cite sources" instruction.
  - latency:          end-to-end /query/ latency, in ms. Reports mean,
                      median, p95.
  - error_rate:       fraction of questions that returned an HTTP error.

Output: `eval/results/run_<YYYY-MM-DD>.json` in the same shape that
`eval/check_regression.py` expects, so the CI gate can compare runs.

Usage:
  # Make sure the FastAPI server is up and the corpus is ingested.
  python eval/build_corpus.py
  # (restart server here if you want a clean corpus state)
  curl -X POST -F "file=@eval/corpus.md" http://127.0.0.1:8002/ingest/
  python eval/run_eval_local.py
"""

from __future__ import annotations

import json
import os
import statistics
import sys
from datetime import datetime
from pathlib import Path

import httpx

ROOT = Path(__file__).parent
DATASET_PATH = ROOT / "dataset.json"
RESULTS_DIR = ROOT / "results"

SERVER_URL = os.getenv("EVAL_SERVER_URL", "http://127.0.0.1:8002")
HIT_PREFIX_CHARS = 60  # how many chars of the ground-truth context must
                       # appear in a retrieved chunk for it to count as a hit
TOP_K = int(os.getenv("EVAL_TOP_K", "5"))
TIMEOUT_S = float(os.getenv("EVAL_TIMEOUT_S", "60"))


def _normalize(text: str) -> str:
    """Lowercase + collapse whitespace so substring match isn't fooled by
    incidental formatting differences between dataset and chunk."""
    return " ".join((text or "").lower().split())


def _is_hit(retrieved_chunks: list[dict], ground_truth_context: str) -> bool:
    needle = _normalize(ground_truth_context)[:HIT_PREFIX_CHARS]
    if not needle:
        return False
    for chunk in retrieved_chunks:
        haystack = _normalize(chunk.get("text", ""))
        if needle in haystack:
            return True
    return False


def _has_citation(answer: str) -> bool:
    if not answer:
        return False
    lo = answer.lower()
    return "[source" in lo or "source 1" in lo or "source 2" in lo or "source 3" in lo


def main() -> int:
    if not DATASET_PATH.exists():
        print(f"ERROR: dataset not found at {DATASET_PATH}", file=sys.stderr)
        return 2

    with DATASET_PATH.open("r", encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"Server:  {SERVER_URL}")
    print(f"Dataset: {len(dataset)} questions")
    print(f"top_k:   {TOP_K}")
    print()

    # Probe server config so the saved results know what was running.
    try:
        cfg = httpx.get(f"{SERVER_URL}/query/config", timeout=5.0).json()
    except Exception as e:
        print(f"ERROR: couldn't reach {SERVER_URL}/query/config: {e}", file=sys.stderr)
        return 3

    print(f"Server config: {cfg}")
    print()

    per_question: list[dict] = []
    latencies_ms: list[float] = []
    hits = 0
    citations = 0
    errors = 0

    with httpx.Client(timeout=TIMEOUT_S) as client:
        for i, item in enumerate(dataset, 1):
            question = item["question"]
            ground_truth_context = item["context"]

            try:
                resp = client.post(
                    f"{SERVER_URL}/query/",
                    json={
                        "question": question,
                        "rewrite_query": False,
                        "max_chunks": TOP_K,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                errors += 1
                print(f"  [{i:>2}/{len(dataset)}]  ERROR  {question[:60]}  ({e})")
                per_question.append({
                    "question": question,
                    "error": str(e),
                })
                continue

            answer = data.get("answer", "")
            sources = data.get("sources", [])
            latency_ms = float(data.get("latency_ms", 0))

            hit = _is_hit(sources, ground_truth_context)
            cited = _has_citation(answer)
            if hit:
                hits += 1
            if cited:
                citations += 1
            latencies_ms.append(latency_ms)

            mark = "OK " if hit else "MISS"
            cite_mark = "[c]" if cited else "[ ]"
            print(
                f"  [{i:>2}/{len(dataset)}]  {mark} {cite_mark}  "
                f"{latency_ms:>7.0f}ms  {question[:60]}"
            )

            per_question.append({
                "question": question,
                "answer_chars": len(answer),
                "retrieved": data.get("retrieved_chunks", 0),
                "latency_ms": round(latency_ms, 1),
                "hit": hit,
                "cited": cited,
            })

    n = len(dataset)
    n_with_latency = len(latencies_ms)
    metrics = {
        "retrieval_hit_at_k": round(hits / n, 4) if n else 0.0,
        "citation_presence": round(citations / n, 4) if n else 0.0,
        "error_rate": round(errors / n, 4) if n else 0.0,
        "latency_ms_mean": round(statistics.mean(latencies_ms), 1) if latencies_ms else 0.0,
        "latency_ms_median": round(statistics.median(latencies_ms), 1) if latencies_ms else 0.0,
        "latency_ms_p95": round(
            statistics.quantiles(latencies_ms, n=20)[-1], 1
        ) if n_with_latency >= 20 else (max(latencies_ms) if latencies_ms else 0.0),
    }

    config = {
        "server": SERVER_URL,
        "top_k": TOP_K,
        "hit_prefix_chars": HIT_PREFIX_CHARS,
        "embed_model": cfg.get("embed_model"),
        "llm_model": cfg.get("llm_model"),
        "retrieval_mode": cfg.get("retrieval_mode"),
        "eval_date": datetime.now().isoformat(timespec="seconds"),
    }

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  retrieval_hit@{TOP_K}:    {metrics['retrieval_hit_at_k']:.1%}  ({hits}/{n})")
    print(f"  citation_presence:    {metrics['citation_presence']:.1%}  ({citations}/{n})")
    print(f"  error_rate:           {metrics['error_rate']:.1%}  ({errors}/{n})")
    print(f"  latency mean:         {metrics['latency_ms_mean']:>7.0f} ms")
    print(f"  latency median:       {metrics['latency_ms_median']:>7.0f} ms")
    print(f"  latency p95:          {metrics['latency_ms_p95']:>7.0f} ms")

    # Save results JSON in the shape eval/check_regression.py expects.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"run_{datetime.now().strftime('%Y-%m-%d')}.json"
    out_path.write_text(
        json.dumps(
            {
                "config": config,
                "metrics": metrics,
                "per_question": per_question,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print()
    print(f"saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
