"""Opt-in local re-ranking of search hits via an embeddings endpoint.

Second-stage retrieval: the palace's first stage (minilm vectors + BM25
hybrid) over-fetches candidates; when enabled, this module re-scores the
deduped pool against the query with a stronger *local* embedding model
(e.g. nomic-embed via LM Studio at :1234) and reorders before the final
cut. Cross-encoder-style precision without any cloud call — the endpoint
must be one the user runs themselves.

Off by default; enable by setting both:

    MEMPALACE_RERANK_URL    e.g. http://127.0.0.1:1234/v1/embeddings
    MEMPALACE_RERANK_MODEL  e.g. text-embedding-nomic-embed-text-v1.5@f32

Fail-soft by contract: any error — endpoint down, timeout, bad payload —
returns the hits in their original order. Search never degrades because
the reranker is unavailable; it only ever improves ordering.

Nomic-style task prefixes ("search_query: " / "search_document: ") are
applied; models that don't use them tolerate them harmlessly.
"""

import json
import logging
import math
import os
import urllib.request

logger = logging.getLogger("mempalace_mcp")

RERANK_URL_ENV = "MEMPALACE_RERANK_URL"
RERANK_MODEL_ENV = "MEMPALACE_RERANK_MODEL"

# Candidates beyond this are left in first-stage order (they can only enter
# the final top-N if the pool is smaller than the cut, which it never is in
# practice). Bounds latency on the local embedder.
MAX_RERANK_POOL = 30
# Per-candidate text cap. Embedding models truncate long inputs anyway;
# capping keeps the request body small and latency predictable.
MAX_TEXT_CHARS = 2000
# Real cost observed ~0.5s for a 30-candidate pool (nomic f32, M4); a hung
# endpoint must not stall search longer than this before fail-soft.
TIMEOUT_SECONDS = 5.0


def rerank_enabled() -> bool:
    return bool(os.environ.get(RERANK_URL_ENV)) and bool(os.environ.get(RERANK_MODEL_ENV))


def _embed(texts: list) -> list:
    """Embed texts via the configured OpenAI-compatible endpoint."""
    body = json.dumps(
        {"model": os.environ[RERANK_MODEL_ENV], "input": texts},
    ).encode("utf-8")
    req = urllib.request.Request(
        os.environ[RERANK_URL_ENV],
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp:
        payload = json.load(resp)
    data = sorted(payload["data"], key=lambda d: d["index"])
    return [d["embedding"] for d in data]


def _cosine(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def maybe_rerank(query: str, hits: list) -> list:
    """Reorder ``hits`` by local-embedder relevance to ``query``.

    No-op unless enabled. Only the first MAX_RERANK_POOL hits are
    re-scored; the tail keeps its first-stage order behind them. Each
    re-scored hit gains a ``rerank_score`` field for transparency.
    """
    if not rerank_enabled() or not hits or not query:
        return hits
    pool = hits[:MAX_RERANK_POOL]
    tail = hits[MAX_RERANK_POOL:]
    try:
        texts = ["search_query: " + query] + [
            "search_document: " + (h.get("text") or "")[:MAX_TEXT_CHARS] for h in pool
        ]
        vectors = _embed(texts)
        qvec, dvecs = vectors[0], vectors[1:]
        if len(dvecs) != len(pool):
            raise ValueError(f"embedding count mismatch: {len(dvecs)} != {len(pool)}")
        for h, dvec in zip(pool, dvecs):
            h["rerank_score"] = round(_cosine(qvec, dvec), 4)
        # Stable sort: equal scores keep first-stage order.
        pool = sorted(pool, key=lambda h: -h["rerank_score"])
        return pool + tail
    except Exception:
        logger.debug("rerank failed (non-fatal); keeping first-stage order", exc_info=True)
        return hits
