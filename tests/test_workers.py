"""Tests for the embedding worker and the regression-check script."""

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


class TestEmbeddingWorker:
    def test_precompute_returns_vectors(self):
        from app.workers.embedding_worker import EmbeddingWorker

        worker = EmbeddingWorker(max_workers=1)

        def fake_embed(chunks):
            return [[float(i)] * 4 for i, _ in enumerate(chunks)]

        chunks = ["alpha", "beta", "gamma"]
        vectors = asyncio.run(worker.precompute(chunks, fake_embed))

        assert len(vectors) == 3
        assert vectors[0] == [0.0, 0.0, 0.0, 0.0]
        assert vectors[2] == [2.0, 2.0, 2.0, 2.0]
        worker.shutdown()

    def test_precompute_empty_chunks(self):
        from app.workers.embedding_worker import EmbeddingWorker

        worker = EmbeddingWorker(max_workers=1)

        def fake_embed(chunks):
            raise AssertionError("embed_fn should not be called for empty input")

        vectors = asyncio.run(worker.precompute([], fake_embed))
        assert vectors == []
        worker.shutdown()

    def test_precompute_runs_off_event_loop(self):
        """The embed call must not block the running loop."""
        from app.workers.embedding_worker import EmbeddingWorker

        worker = EmbeddingWorker(max_workers=1)

        def slow_embed(chunks):
            import time
            time.sleep(0.05)
            return [[1.0]] * len(chunks)

        async def driver():
            # Schedule a quick coroutine and the embedding concurrently.
            embed_task = asyncio.create_task(
                worker.precompute(["a", "b"], slow_embed)
            )
            await asyncio.sleep(0)  # yield once so the executor starts
            tick = await asyncio.wait_for(asyncio.sleep(0, result=42), timeout=0.2)
            vectors = await embed_task
            return tick, vectors

        tick, vectors = asyncio.run(driver())
        assert tick == 42
        assert len(vectors) == 2
        worker.shutdown()

    def test_get_worker_returns_singleton(self):
        from app.workers.embedding_worker import get_worker

        a = get_worker()
        b = get_worker()
        assert a is b

    def test_async_ingest_flag_default(self, monkeypatch):
        from app.workers.embedding_worker import is_async_ingest_enabled

        monkeypatch.delenv("INGEST_ASYNC_EMBED", raising=False)
        assert is_async_ingest_enabled() is True

        monkeypatch.setenv("INGEST_ASYNC_EMBED", "false")
        assert is_async_ingest_enabled() is False

        monkeypatch.setenv("INGEST_ASYNC_EMBED", "TRUE")
        assert is_async_ingest_enabled() is True


class TestRegressionCheck:
    def _write_run(self, path: Path, metrics: dict):
        path.write_text(
            json.dumps({"config": {}, "metrics": metrics, "timestamp": "x"}),
            encoding="utf-8",
        )

    def test_no_regression_returns_zero(self, tmp_path, monkeypatch):
        results = tmp_path / "results"
        results.mkdir()
        self._write_run(
            results / "run_2026-04-01.json",
            {
                "faithfulness": 0.80,
                "answer_relevancy": 0.85,
                "context_recall": 0.78,
                "context_precision": 0.82,
            },
        )
        self._write_run(
            results / "run_2026-04-15.json",
            {
                "faithfulness": 0.82,
                "answer_relevancy": 0.89,
                "context_recall": 0.80,
                "context_precision": 0.84,
            },
        )

        from eval import check_regression

        with patch.object(check_regression, "Path") as MockPath:
            MockPath.return_value = results
            MockPath.side_effect = lambda *args, **kwargs: results if not args else Path(*args, **kwargs)
            # Easier: monkeypatch the module-level results path resolution
            monkeypatch.setattr(
                check_regression,
                "_load_results_dir",
                lambda _: sorted(results.glob("run_*.json")),
            )
            assert check_regression.main() == 0

    def test_regression_returns_one(self, tmp_path, monkeypatch):
        results = tmp_path / "results"
        results.mkdir()
        self._write_run(
            results / "run_2026-04-01.json",
            {
                "faithfulness": 0.82,
                "answer_relevancy": 0.89,
                "context_recall": 0.80,
                "context_precision": 0.84,
            },
        )
        self._write_run(
            results / "run_2026-04-15.json",
            {
                "faithfulness": 0.50,  # 39% drop
                "answer_relevancy": 0.85,
                "context_recall": 0.78,
                "context_precision": 0.82,
            },
        )

        from eval import check_regression

        monkeypatch.setattr(
            check_regression,
            "_load_results_dir",
            lambda _: sorted(results.glob("run_*.json")),
        )
        monkeypatch.setenv("REGRESSION_THRESHOLD", "0.05")
        assert check_regression.main() == 1

    def test_single_run_is_not_a_regression(self, tmp_path, monkeypatch):
        results = tmp_path / "results"
        results.mkdir()
        self._write_run(
            results / "run_2026-04-01.json",
            {"faithfulness": 0.82},
        )

        from eval import check_regression

        monkeypatch.setattr(
            check_regression,
            "_load_results_dir",
            lambda _: sorted(results.glob("run_*.json")),
        )
        assert check_regression.main() == 0
