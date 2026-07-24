"""Optional Langfuse tracing.

Design rule: tracing must NEVER change pipeline behaviour. If Langfuse is not
configured (no keys), unreachable, or raises for any reason, every helper here
degrades to a no-op and the pipeline runs exactly as before. Observability that
can take down the thing it observes is worse than no observability.

Enable by setting in .env:
    LANGFUSE_PUBLIC_KEY=pk-lf-...
    LANGFUSE_SECRET_KEY=sk-lf-...
    LANGFUSE_HOST=http://localhost:3000
"""

import os
from contextlib import contextmanager
from typing import Any, Dict, Optional

_client = None
_checked = False
_last_trace_url = None
_last_trace_id = None


def get_client():
    """Return a Langfuse client, or None if unconfigured/unavailable.

    Cached after the first call so a missing install or bad host costs one
    import attempt, not one per query.
    """
    global _client, _checked
    if _checked:
        return _client
    _checked = True

    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        return None

    try:
        from langfuse import get_client as _get
        client = _get()
        # auth_check() is a network call; do it once so a misconfigured host
        # fails loudly here at startup rather than silently per-request.
        if not client.auth_check():
            print("[tracing] Langfuse auth failed — tracing disabled")
            return None
        _client = client
    except Exception as e:  # ImportError, network, anything
        print(f"[tracing] Langfuse unavailable ({type(e).__name__}) — tracing disabled")
        _client = None
    return _client


class _NullSpan:
    """Stand-in with the same surface as a Langfuse observation."""

    def update(self, **kwargs):
        return self

    def score(self, **kwargs):
        return self

    def update_trace(self, **kwargs):
        return self


@contextmanager
def observe(as_type: str = "span", name: str = "unnamed", **kwargs):
    """Open a Langfuse observation, or yield a no-op if tracing is off.

    as_type: "span" | "generation" | "retriever" | "event"
    Nesting is automatic — it follows Python scope, not passed parent IDs.
    """
    client = get_client()
    if client is None:
        yield _NullSpan()
        return

    global _last_trace_url, _last_trace_id

    # Guard ONLY the SDK setup. If Langfuse itself can't open the observation,
    # fall back to a no-op and run the body untraced.
    try:
        cm = client.start_as_current_observation(as_type=as_type, name=name, **kwargs)
    except Exception as e:
        print(f"[tracing] could not start '{name}' ({type(e).__name__}) — continuing untraced")
        yield _NullSpan()
        return

    # From here the body runs INSIDE the span. A body exception must propagate
    # unchanged — the observability layer must never swallow or mask the app's
    # real error (and yielding a second time here would itself raise
    # "generator didn't stop after throw()"). The SDK records the error on the
    # span as the exception unwinds.
    with cm as span:
        try:
            _last_trace_url = client.get_trace_url()
            _last_trace_id = client.get_current_trace_id()
        except Exception:
            pass
        yield span


def score_trace(name: str, value: Any, data_type: Optional[str] = None,
                comment: Optional[str] = None,
                target_trace_id: Optional[str] = None) -> None:
    """Attach a score to a trace. Silent no-op if tracing is off.

    IMPORTANT: score_current_trace() only works INSIDE an active span. Outside
    one it logs "No active span in current context" and drops the score
    silently. That is the normal case for RAG evaluation, where RAGAs computes
    metrics only after every question has run — so when no span is active we
    fall back to create_score(trace_id=...), which attaches retroactively.

    Pass target_trace_id to score a specific earlier trace (e.g. per-question
    scores after a batch eval).
    """
    client = get_client()
    if client is None:
        return

    payload: Dict[str, Any] = {"name": name, "value": value}
    if data_type:
        payload["data_type"] = data_type
    if comment:
        payload["comment"] = comment

    tid = target_trace_id or _last_trace_id
    try:
        # Prefer the explicit, context-free path — it works both inside and
        # outside a span, so there is no silent-drop failure mode.
        if tid:
            client.create_score(trace_id=tid, **payload)
        else:
            client.score_current_trace(**payload)
    except Exception as e:
        print(f"[tracing] score '{name}' failed ({type(e).__name__})")


def trace_url() -> Optional[str]:
    """URL of the current trace, or the most recent one if none is active."""
    client = get_client()
    if client is None:
        return None
    try:
        return client.get_trace_url() or _last_trace_url
    except Exception:
        return _last_trace_url


def trace_id() -> Optional[str]:
    """Id of the most recent trace — useful for linking eval rows to traces."""
    return _last_trace_id


def flush() -> None:
    """Force delivery. MUST be called before a short-lived process exits —
    the SDK batches asynchronously, so a script that returns immediately
    loses its traces silently."""
    client = get_client()
    if client is not None:
        try:
            client.flush()
        except Exception:
            pass
