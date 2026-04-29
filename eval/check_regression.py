"""Compare the latest eval run against a baseline and exit non-zero on regression.

Used by CI after `python eval/run_eval.py` produces a fresh
`eval/results/run_<date>.json`. Reads the two most recent result files in
chronological order, computes per-metric percentage drops, and exits with
status 1 if any metric drops by more than the configured threshold.

Configuration via env:
    REGRESSION_THRESHOLD  fractional drop that fails CI (default 0.05 = 5%)
    METRICS_TO_CHECK      comma-separated metric names (default: all four
                          RAGAs metrics)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List


DEFAULT_METRICS = [
    "faithfulness",
    "answer_relevancy",
    "context_recall",
    "context_precision",
]
DEFAULT_THRESHOLD = 0.05


def _load_results_dir(results_dir: Path) -> List[Path]:
    if not results_dir.exists():
        return []
    return sorted(p for p in results_dir.glob("run_*.json"))


def _read(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    results_dir = Path(__file__).parent / "results"
    runs = _load_results_dir(results_dir)

    if len(runs) < 2:
        print(
            f"Not enough runs to compare ({len(runs)} found). "
            "Need at least a baseline + latest."
        )
        return 0

    baseline_path, latest_path = runs[-2], runs[-1]
    baseline = _read(baseline_path)
    latest = _read(latest_path)

    threshold = float(os.getenv("REGRESSION_THRESHOLD", DEFAULT_THRESHOLD))
    metrics = os.getenv("METRICS_TO_CHECK", ",".join(DEFAULT_METRICS)).split(",")
    metrics = [m.strip() for m in metrics if m.strip()]

    print(f"Baseline: {baseline_path.name}")
    print(f"Latest:   {latest_path.name}")
    print(f"Threshold: {threshold:.0%} drop fails")
    print()

    regressions: List[str] = []
    for metric in metrics:
        b = baseline.get("metrics", {}).get(metric)
        l = latest.get("metrics", {}).get(metric)
        if b is None or l is None:
            print(f"  {metric}: missing in one of the runs, skipped")
            continue
        # Avoid division by zero and meaningless ratios on tiny baselines.
        if b <= 0:
            print(f"  {metric}: baseline non-positive ({b}), skipped")
            continue
        drop = (b - l) / b
        marker = "FAIL" if drop > threshold else "ok"
        print(f"  {metric}: {b:.3f} -> {l:.3f}  ({drop:+.1%})  [{marker}]")
        if drop > threshold:
            regressions.append(f"{metric} dropped {drop:.1%}")

    print()
    if regressions:
        print("REGRESSION DETECTED:")
        for r in regressions:
            print(f"  - {r}")
        return 1

    print("No regressions detected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
