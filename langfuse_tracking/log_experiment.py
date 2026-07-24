"""Langfuse experiment tracking — the Langfuse counterpart to MLflowTracker.

Deliberately mirrors mlflow_tracking.log_experiment.MLflowTracker's interface
(log_experiment(config, metrics, run_name)) so it drops in alongside it. Both
run together: MLflow stays the experiment ledger, Langfuse adds the traces the
numbers came from.

WHY BOTH: MLflow answers "which config scored best?". Langfuse answers "show me
the 12 questions where faithfulness was worst, and what was retrieved for them".
The second question is the one you act on, and MLflow structurally cannot answer
it — a run-level metric has no link back to individual interactions.
"""

import os
from datetime import datetime
from typing import Any, Dict, Optional


class LangfuseTracker:
    def __init__(self, experiment_name: str = "rag-evaluation"):
        self.experiment_name = experiment_name
        self.client = None

        if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
            print("[langfuse] keys not set — tracking disabled")
            return

        try:
            from langfuse import get_client
            client = get_client()
            if client.auth_check():
                self.client = client
            else:
                print("[langfuse] auth failed — tracking disabled")
        except Exception as e:
            print(f"[langfuse] unavailable ({type(e).__name__}) — tracking disabled")

    @property
    def enabled(self) -> bool:
        return self.client is not None

    def log_experiment(
        self,
        config: Dict[str, Any],
        metrics: Dict[str, float],
        run_name: Optional[str] = None,
    ) -> Optional[str]:
        """Record an eval run as a trace carrying the RAGAs metrics as scores.

        Returns the trace id, or None when tracking is disabled.
        """
        if not self.enabled:
            return None

        run_name = run_name or f"eval_{datetime.now().strftime('%Y-%m-%d_%H%M')}"

        try:
            with self.client.start_as_current_observation(
                as_type="span",
                name=run_name,
                input=config,
                metadata={"experiment": self.experiment_name, **config},
            ):
                # RAGAs metrics become SCORES, not params. This is the whole
                # migration in one loop: a number that lived on an MLflow run
                # now lives on a trace you can click into.
                for key, value in metrics.items():
                    try:
                        self.client.score_current_trace(
                            name=key, value=float(value), data_type="NUMERIC"
                        )
                    except (TypeError, ValueError):
                        # Non-numeric metric — record it categorically rather
                        # than dropping it.
                        self.client.score_current_trace(
                            name=key, value=str(value), data_type="CATEGORICAL"
                        )

                trace_id = self.client.get_current_trace_id()
                url = self.client.get_trace_url()

            self.client.flush()  # batched delivery — flush or lose it
            if url:
                print(f"Logged to Langfuse: {url}")
            return trace_id

        except Exception as e:
            print(f"[langfuse] logging failed ({type(e).__name__}: {e})")
            return None

    def flush(self) -> None:
        if self.enabled:
            try:
                self.client.flush()
            except Exception:
                pass


if __name__ == "__main__":
    tracker = LangfuseTracker()
    print("enabled:", tracker.enabled)
    tracker.log_experiment(
        config={"chunk_size": 512, "top_k": 5, "retrieval_mode": "hybrid"},
        metrics={"faithfulness": 0.82, "answer_relevancy": 0.89,
                 "context_recall": 0.78, "context_precision": 0.80},
        run_name="smoke-test",
    )
