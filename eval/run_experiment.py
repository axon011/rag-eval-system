"""Register the eval set as a first-class Langfuse Dataset and run an Experiment.

langfuse_tracking.log_experiment records only RUN-LEVEL RAGAs numbers as scores
on a single trace — nothing shows up under Evaluation -> Datasets / Experiments.
This module fills that gap: it turns eval/dataset.json into a real Langfuse
Dataset (one row per question, with reference answer + context) and runs a
Dataset Experiment where every row gets its own trace and per-row scores, so
runs are comparable side by side in the Experiments view.

Runs on the local Ollama stack. Generation is slow on CPU, so by default it
evaluates a SUBSET of the dataset (--limit, default 6) — the full 50-item set is
still registered; only the experiment run is subset. The row count actually run
is printed and attached to the run metadata, never silently truncated.

Usage (from repo root):
    python -m eval.run_experiment --limit 6
    python -m eval.run_experiment --limit 0        # run all items
"""
import argparse
import json
import os
import re
from datetime import datetime
from typing import Any, List

from langfuse import Evaluation, get_client

from app.core.pipeline import RAGPipeline
from app.core import tracing  # noqa: F401  importing sets the LANGFUSE_TIMEOUT default

DATASET_NAME = "rag-eval-set"
DATASET_FILE = os.path.join(os.path.dirname(__file__), "dataset.json")
CORPUS_FILE = os.path.join(os.path.dirname(__file__), "corpus.md")


# --------------------------------------------------------------------------- #
# Dataset registration (idempotent)
# --------------------------------------------------------------------------- #
def _load_dataset_json() -> List[dict]:
    with open(DATASET_FILE, encoding="utf-8") as f:
        return json.load(f)


def ensure_dataset(client) -> int:
    """Create the Dataset and its items idempotently. Returns the item count.

    create_dataset is NOT idempotent (a second call errors on the duplicate
    name), so guard on existence. create_dataset_item upserts on a deterministic
    id, so re-running only refreshes rows instead of duplicating them.
    """
    items = _load_dataset_json()

    try:
        client.get_dataset(DATASET_NAME)
        exists = True
    except Exception:
        exists = False

    if not exists:
        client.create_dataset(
            name=DATASET_NAME,
            description=(
                "RAG QA evaluation set: LangGraph / RAG questions with reference "
                "answers and reference context, sourced from eval/dataset.json."
            ),
        )
        print(f"[dataset] created '{DATASET_NAME}'")
    else:
        print(f"[dataset] '{DATASET_NAME}' already exists — upserting items")

    for i, it in enumerate(items):
        # The experiment run propagates item metadata onto every span as an OTel
        # attribute, and values over 200 chars are dropped there with a warning
        # per span. Keep a short reference-context pointer so the metadata stays
        # under that cap and the run output stays clean; the full context lives
        # in eval/dataset.json.
        ref = (it.get("context") or "").strip()
        client.create_dataset_item(
            dataset_name=DATASET_NAME,
            id=f"{DATASET_NAME}-{i:03d}",  # deterministic -> upsert, no dupes
            input=it["question"],
            expected_output=it["answer"],
            metadata={"reference_context_excerpt": ref[:180]},
        )
    print(f"[dataset] {len(items)} items registered")
    return len(items)


# --------------------------------------------------------------------------- #
# Task + evaluators
# --------------------------------------------------------------------------- #
def _chunk(text: str, size: int = 512) -> List[str]:
    return [text[i:i + size] for i in range(0, len(text), size)]


def _ingest_corpus(pipe: RAGPipeline) -> None:
    with open(CORPUS_FILE, encoding="utf-8") as f:
        chunks = _chunk(f.read(), 512)
    info = pipe.ingest_documents(chunks)
    print(f"[ingest] {info}")


def make_task(pipe: RAGPipeline):
    def task(*, item, **kwargs) -> dict:
        # item is a Langfuse DatasetItem: .input / .expected_output / .metadata
        res = pipe.query(item.input, rewrite_query=True)
        return {
            "answer": res["answer"],
            "contexts": [s["text"] for s in res["sources"]],
            "retrieved_chunks": res["retrieved_chunks"],
        }
    return task


def _tokens(s: Any) -> set:
    return set(re.findall(r"[a-z0-9]+", str(s or "").lower()))


def _answer_of(output: Any) -> str:
    return output.get("answer", "") if isinstance(output, dict) else str(output or "")


def eval_answer_non_empty(*, input, output, expected_output=None, metadata=None, **kwargs):
    ans = _answer_of(output).strip()
    return Evaluation(
        name="answer_non_empty",
        value=1.0 if len(ans) >= 20 else 0.0,
        comment=f"answer is {len(ans)} chars",
    )


def eval_reference_token_recall(*, input, output, expected_output=None, metadata=None, **kwargs):
    """Fraction of reference-answer tokens that appear in the generated answer.

    A cheap, deterministic proxy for answer correctness — no judge LLM, so it
    stays fast and reproducible on the local CPU stack.
    """
    ans_tok, ref_tok = _tokens(_answer_of(output)), _tokens(expected_output)
    recall = len(ans_tok & ref_tok) / len(ref_tok) if ref_tok else 0.0
    return Evaluation(
        name="reference_token_recall",
        value=round(recall, 3),
        comment=f"{len(ans_tok & ref_tok)}/{len(ref_tok)} reference tokens present",
    )


def eval_context_reference_overlap(*, input, output, expected_output=None, metadata=None, **kwargs):
    """Retrieval-quality proxy: reference tokens covered by retrieved contexts."""
    ctxs = output.get("contexts", []) if isinstance(output, dict) else []
    ctx_tok = set().union(*[_tokens(c) for c in ctxs]) if ctxs else set()
    ref_tok = _tokens(expected_output)
    overlap = len(ctx_tok & ref_tok) / len(ref_tok) if ref_tok else 0.0
    return Evaluation(
        name="context_reference_overlap",
        value=round(overlap, 3),
        comment=f"{len(ctxs)} contexts retrieved",
    )


def run_mean_recall(*, item_results, **kwargs):
    """Run-level aggregate: mean reference_token_recall across all rows."""
    vals = [
        ev.value
        for r in item_results
        for ev in r.evaluations
        if ev.name == "reference_token_recall"
    ]
    mean = sum(vals) / len(vals) if vals else 0.0
    return Evaluation(
        name="mean_reference_token_recall",
        value=round(mean, 3),
        comment=f"mean over {len(vals)} items",
    )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Langfuse Dataset Experiment for the RAG eval set.")
    parser.add_argument("--limit", type=int, default=6,
                        help="Number of dataset items to run (0 = all). Default 6 (CPU-friendly).")
    parser.add_argument("--run-name", default=None, help="Explicit dataset run name.")
    args = parser.parse_args()

    client = get_client()
    if not client.auth_check():
        raise SystemExit("[experiment] Langfuse auth failed — check keys / LANGFUSE_HOST")

    total = ensure_dataset(client)

    dataset = client.get_dataset(DATASET_NAME)
    items = list(dataset.items)
    run_items = items if args.limit in (0, None) else items[:args.limit]
    print(f"[experiment] running {len(run_items)} of {total} dataset items "
          f"({'FULL SET' if len(run_items) == total else 'SUBSET'})")

    pipe = RAGPipeline()
    _ingest_corpus(pipe)

    run_name = args.run_name or f"rag-eval_{datetime.now().strftime('%Y-%m-%d_%H%M')}"

    result = client.run_experiment(
        name="RAG eval-set experiment",
        run_name=run_name,
        description=f"RAG pipeline over {len(run_items)}/{total} eval items on the local Ollama stack.",
        data=run_items,
        task=make_task(pipe),
        evaluators=[
            eval_answer_non_empty,
            eval_reference_token_recall,
            eval_context_reference_overlap,
        ],
        run_evaluators=[run_mean_recall],
        max_concurrency=1,  # tinyllama on CPU — serialize, don't OOM
        metadata={"items_run": str(len(run_items)), "items_total": str(total)},
    )

    tracing.flush()
    client.flush()

    print("\n" + "=" * 60)
    print(f"[experiment] run_name       : {result.run_name}")
    print(f"[experiment] dataset_run_id : {getattr(result, 'dataset_run_id', None)}")
    print(f"[experiment] dataset_run_url: {getattr(result, 'dataset_run_url', None)}")
    print(f"[experiment] items run      : {len(result.item_results)} / {total}")
    for ev in result.run_evaluations:
        print(f"[experiment] run score      : {ev.name} = {ev.value} ({ev.comment})")
    print("=" * 60)


if __name__ == "__main__":
    main()
