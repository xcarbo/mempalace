"""Opt-in local re-scoring of search hits via an embeddings endpoint.

Second-stage retrieval: the palace's first stage (minilm vectors + BM25
hybrid) over-fetches candidates; when enabled, this module re-scores the
deduped pool against the query with a stronger *local* embedding model
(e.g. nomic-embed via LM Studio at :1234). The scores do NOT replace the
first-stage ranking — the searcher BLENDS them with the fused
vector+BM25 score and re-applies the archive demotion afterwards
(:func:`mempalace.searcher._blend_rerank`).

That blend is a measured correction, not a style choice: the original
implementation sorted the pool by raw bi-encoder cosine alone, discarding
both the fused score and the archive demotion, and made recall WORSE on
the 300-case eval (R@5 0.907→0.847, MRR 0.809→0.687, p50 227→902 ms;
hurt 20 cases, rescued 2 — measured 2026-08-08). A second-stage signal
may only ever be one voice among the lanes, never the whole ranking.

Off by default; enable by setting both:

    MEMPALACE_RERANK_URL    e.g. http://127.0.0.1:1234/v1/embeddings
    MEMPALACE_RERANK_MODEL  e.g. text-embedding-nomic-embed-text-v1.5@f32

Setting ``MEMPALACE_RERANK_MODE=cross`` swaps the scorer for the cross-encoder
in :mod:`mempalace.cross_rerank` (URL then points at a completion endpoint).
Same contract, same blend, stronger signal, one request per candidate instead
of one for the pool.

Fail-soft by contract: any error — endpoint down, timeout, bad payload —
leaves the hits unscored, and the searcher keeps first-stage order.
Search never breaks because the reranker is unavailable.

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
# "embedding" (default) = bi-encoder, this module. "cross" = cross-encoder,
# mempalace.cross_rerank. Both satisfy the same annotate_rerank_scores
# contract, so the searcher's blend is identical either way.
RERANK_MODE_ENV = "MEMPALACE_RERANK_MODE"

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


def annotate_rerank_scores(query: str, pool: list) -> bool:
    """Score each hit in ``pool`` against ``query`` with the local embedder.

    Adds a ``rerank_score`` field (cosine in the rerank model's space) to
    every hit; does NOT reorder anything — ordering is the searcher's job,
    which blends this signal with the first-stage fused score. Returns
    ``True`` when every hit was scored, ``False`` on any failure (in which
    case no hit ordering may rely on ``rerank_score`` being present).
    """
    if not rerank_enabled() or not pool or not query:
        return False

    if os.environ.get(RERANK_MODE_ENV, "embedding").strip().lower() == "cross":
        # Cross-encoder: reads query and candidate together and returns a
        # probability, where this module returns a cosine. The searcher
        # min-max normalises whatever it finds in `rerank_score`, so the two
        # scales never meet and no caller has to know which ran.
        from .cross_rerank import score_pairs

        scores = score_pairs(query, [(h.get("text") or "") for h in pool])
        if scores is None or len(scores) != len(pool):
            logger.debug("cross-rerank returned nothing usable; keeping first-stage order")
            return False
        for h, sc in zip(pool, scores):
            h["rerank_score"] = round(sc, 4)
        return True

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
        return True
    except Exception:
        logger.debug("rerank scoring failed (non-fatal); keeping first-stage order", exc_info=True)
        return False
