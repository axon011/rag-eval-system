"""Build a single Markdown corpus from `dataset.json` contexts.

The eval dataset's `context` field is the ground-truth supporting text for
each question. Concatenating those into one document gives us a corpus the
50 golden questions can actually retrieve against — which is what we need
to produce honest eval results without depending on Perinet's private
corpus.

Output: `eval/corpus.md`. Each context becomes a `## Topic N` section so
chunking has natural boundaries.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parent
DATASET_PATH = ROOT / "dataset.json"
CORPUS_PATH = ROOT / "corpus.md"


def main() -> int:
    with DATASET_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    sections = ["# Eval Corpus\n"]
    sections.append(
        "_Generated from `dataset.json` contexts. Each section is the "
        "ground-truth context for one question in the golden set._\n"
    )
    for i, item in enumerate(data, 1):
        question = item.get("question", "").strip()
        context = item.get("context", "").strip()
        if not context:
            continue
        sections.append(f"\n## Topic {i}: {question}\n")
        sections.append(context + "\n")

    CORPUS_PATH.write_text("\n".join(sections), encoding="utf-8")
    n = sum(1 for d in data if d.get("context"))
    chars = CORPUS_PATH.stat().st_size
    print(f"wrote {CORPUS_PATH} — {n} sections, {chars:,} chars")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
