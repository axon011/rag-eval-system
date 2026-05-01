# Eval Results

Two evaluation paths live here, producing JSON files in this directory:

## 1. Local-server eval (no LLM judge required)

`python eval/run_eval_local.py` — runs the 50-question golden set against
a live FastAPI server at `EVAL_SERVER_URL` (default `http://127.0.0.1:8002`)
and computes metrics that don't need an LLM-as-judge:

- `retrieval_hit_at_k` — for each question, did any of the top-k retrieved
  chunks contain a 60-char prefix of the ground-truth context?
- `citation_presence` — does the answer reference `[Source N]` per the
  prompt's citation instruction?
- `error_rate` — fraction of HTTP errors
- `latency_ms_*` — mean / median / p95 of end-to-end query latency

Run order (assumes server is up):

```bash
python eval/build_corpus.py         # corpus.md from dataset.json contexts
curl -X POST -F "file=@eval/corpus.md" http://127.0.0.1:8002/ingest/
python eval/run_eval_local.py
python eval/check_regression.py     # compares last two committed runs
```

The most recent run (`run_<YYYY-MM-DD>.json`) is committed so that
`check_regression.py` always has a baseline. Subsequent runs in CI replace
it on green builds.

## 2. RAGAs LLM-judge eval

`python eval/run_eval.py` — uses RAGAs to compute LLM-judged metrics:

- `faithfulness` — claims in the answer that are grounded in retrieved context
- `answer_relevancy` — does the answer address the question
- `context_recall` — did retrieval surface all the ground-truth supporting text
- `context_precision` — are the top-ranked chunks the relevant ones

Requires an LLM judge configured (defaults to OpenAI; can be pointed at
Ollama via LiteLLM, but small judge models are unreliable on these
metrics — use a strong judge for results worth comparing).

## Output schema

Both runners write to `run_<YYYY-MM-DD>.json` with this top-level shape:

```json
{
  "config": { "embed_model": "...", "llm_model": "...", "...": "..." },
  "metrics": { "metric_name": 0.0 },
  "timestamp": "..."
}
```

`check_regression.py` reads `metrics.*` from the two most recent files and
exits non-zero if any value drops more than `REGRESSION_THRESHOLD`
(default 5%). The local-eval runner additionally writes a `per_question`
array with hit/cited/latency_ms for each entry, useful for drilling into
which specific questions regressed.
