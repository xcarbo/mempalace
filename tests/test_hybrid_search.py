"""Tests for the hybrid closet+drawer retrieval in search_memories.

The hybrid path queries drawers directly (the floor) AND closets, applying a
rank-based boost to drawers whose source_file appears in top closet hits.
This avoids the "weak-closets regression" where low-signal closets (from
regex extraction on narrative content) could hide drawers that direct
search would have found.
"""

from mempalace.palace import (
    get_closets_collection,
    get_collection,
    upsert_closet_lines,
)
from mempalace.searcher import _hybrid_rank, search_memories


def _seed_drawers(palace_path):
    """Insert 4 short drawers with deterministic content."""
    col = get_collection(palace_path, create=True)
    col.upsert(
        ids=["D1", "D2", "D3", "D4"],
        documents=[
            "We switched the auth service to use JWT tokens with a 24h expiry.",
            "Database migration to PostgreSQL 15 completed last Tuesday.",
            "The frontend team is debating whether to adopt TanStack Query.",
            "Kafka consumer rebalance timeout set to 45 seconds after incident.",
        ],
        metadatas=[
            {"wing": "backend", "room": "auth", "source_file": "fixture_D1.md"},
            {"wing": "backend", "room": "db", "source_file": "fixture_D2.md"},
            {"wing": "frontend", "room": "state", "source_file": "fixture_D3.md"},
            {"wing": "backend", "room": "queue", "source_file": "fixture_D4.md"},
        ],
    )


def _seed_strong_closet_for(palace_path, drawer_id, source_file, topics):
    """Insert a closet whose content strongly overlaps the query keywords."""
    col = get_closets_collection(palace_path)
    lines = [f"{t}||→{drawer_id}" for t in topics]
    upsert_closet_lines(
        col,
        closet_id_base=f"closet_{drawer_id}",
        lines=lines,
        metadata={
            "wing": "backend",
            "room": "auth",
            "source_file": source_file,
            "generated_by": "test",
        },
    )


# ── core invariant: closets can only HELP, never HIDE ─────────────────────


class TestHybridInvariant:
    def test_no_closets_degrades_to_direct_drawer_search(self, tmp_path):
        palace = str(tmp_path / "palace")
        _seed_drawers(palace)
        # No closets created.
        result = search_memories("Kafka rebalance timeout", palace, n_results=3)
        ids = [h["source_file"] for h in result["results"]]
        assert ids, "should return results"
        assert "fixture_D4.md" in ids, "direct drawer search alone should surface the Kafka drawer"

    def test_weak_closets_do_not_hide_direct_drawer_hits(self, tmp_path):
        """A closet that points at a wrong drawer must NOT suppress the
        drawer that direct search would have ranked first."""
        palace = str(tmp_path / "palace")
        _seed_drawers(palace)
        # Seed a misleading closet: it matches a generic phrase but points at D3.
        _seed_strong_closet_for(
            palace,
            drawer_id="D3",
            source_file="fixture_D3.md",
            topics=["Kafka queue tuning", "consumer rebalance config"],
        )
        result = search_memories("Kafka consumer rebalance timeout", palace, n_results=5)
        ids = [h["source_file"] for h in result["results"]]
        assert "fixture_D4.md" in ids, (
            "D4 must appear — direct drawer search alone would rank it first. "
            "Closet pointing to D3 should only boost D3, never hide D4."
        )

    def test_closet_boost_lifts_matching_drawer(self, tmp_path):
        """When a closet agrees with direct search, the matching drawer
        should be boosted to rank 1."""
        palace = str(tmp_path / "palace")
        _seed_drawers(palace)
        _seed_strong_closet_for(
            palace,
            drawer_id="D1",
            source_file="fixture_D1.md",
            topics=["JWT auth tokens", "session expiry", "authentication service"],
        )
        result = search_memories("JWT auth tokens expiry", palace, n_results=3)
        ids = [h["source_file"] for h in result["results"]]
        assert ids[0] == "fixture_D1.md"
        top = result["results"][0]
        assert top["matched_via"] == "drawer+closet"
        assert top["closet_boost"] > 0


# ── closet_boost metadata ────────────────────────────────────────────────


class TestClosetMetadata:
    def test_closet_preview_exposed_when_boosted(self, tmp_path):
        palace = str(tmp_path / "palace")
        _seed_drawers(palace)
        _seed_strong_closet_for(
            palace,
            drawer_id="D1",
            source_file="fixture_D1.md",
            topics=["JWT auth tokens", "session expiry", "authentication service"],
        )
        result = search_memories("JWT auth tokens expiry", palace, n_results=2)
        top = result["results"][0]
        assert top["source_file"] == "fixture_D1.md"
        assert top["matched_via"] == "drawer+closet"
        assert top["closet_boost"] > 0
        assert "closet_preview" in top

    def test_drawer_only_hits_have_no_closet_preview(self, tmp_path):
        palace = str(tmp_path / "palace")
        _seed_drawers(palace)
        # No closets
        result = search_memories("TanStack Query", palace, n_results=2)
        assert result["results"]
        for h in result["results"]:
            assert h["matched_via"] == "drawer"
            assert "closet_preview" not in h
            assert h["closet_boost"] == 0.0


# ── source_file filter scopes both drawer and closet queries (#1815) ──────


class TestSourceFileFilter:
    def test_source_file_filter_excludes_other_sources(self, tmp_path):
        palace = str(tmp_path / "palace")
        _seed_drawers(palace)
        result = search_memories(
            "Kafka consumer rebalance timeout",
            palace,
            n_results=5,
            source_file="fixture_D4.md",
        )
        ids = [h["source_file"] for h in result["results"]]
        assert ids, "the matching source_file drawer should be returned"
        assert set(ids) == {"fixture_D4.md"}

    def test_source_file_filter_overrides_closet_boost_for_other_source(self, tmp_path):
        # A strong closet pointing at D1 must NOT leak D1 in when the search
        # is scoped to a different source_file — the where clause is applied
        # to the closet query too, not just the drawer query.
        palace = str(tmp_path / "palace")
        _seed_drawers(palace)
        _seed_strong_closet_for(
            palace,
            drawer_id="D1",
            source_file="fixture_D1.md",
            topics=["Kafka queue tuning", "consumer rebalance config"],
        )
        result = search_memories(
            "Kafka consumer rebalance",
            palace,
            n_results=5,
            source_file="fixture_D4.md",
        )
        ids = [h["source_file"] for h in result["results"]]
        assert "fixture_D1.md" not in ids
        assert set(ids) <= {"fixture_D4.md"}


def test_hybrid_rank_breaks_score_ties_by_authored_at():
    """Identical-content hits get identical vector + BM25 scores; the tie must break
    toward the more recently authored drawer, not arbitrary backend order."""
    older = {
        "text": "alpha beta gamma",
        "distance": 0.2,
        "metadata": {"authored_at": "2026-06-21T10:00:00.000Z"},
    }
    newer = {
        "text": "alpha beta gamma",
        "distance": 0.2,
        "metadata": {"authored_at": "2026-06-27T10:00:00.000Z"},
    }
    # Input order puts the older drawer first; the tiebreak should reorder it.
    results = [older, newer]
    _hybrid_rank(results, "alpha beta gamma")
    assert results[0]["metadata"]["authored_at"] == "2026-06-27T10:00:00.000Z"
    assert results[1]["metadata"]["authored_at"] == "2026-06-21T10:00:00.000Z"


def test_hybrid_rank_tiebreak_handles_top_level_authored_at():
    """The search_memories path puts authored_at at the top level (no `metadata`
    nesting); the tie-break must read it there too."""
    older = {"text": "alpha beta gamma", "distance": 0.2, "authored_at": "2026-06-21T10:00:00.000Z"}
    newer = {"text": "alpha beta gamma", "distance": 0.2, "authored_at": "2026-06-27T10:00:00.000Z"}
    results = [older, newer]
    _hybrid_rank(results, "alpha beta gamma")
    assert results[0]["authored_at"] == "2026-06-27T10:00:00.000Z"
    assert results[1]["authored_at"] == "2026-06-21T10:00:00.000Z"


# ── the hybrid rank must SEE more than it returns ─────────────────────────
#
# Regression for the recall failure found on 2026-08-02: the candidate pool
# was cut to n_results by vector distance alone *before* `_finalize_candidate_
# hits` ran the 0.6*vector + 0.4*BM25 hybrid re-rank. The BM25 half could
# therefore only reorder drawers that vector had already picked, so a drawer
# whose text literally contained the query words never came back if its
# embedding distance fell one slot below the cut. On the live palace that lost
# the `wings-registry` drawer (13th by distance among 158k drawers) for a query
# whose words it contains verbatim — every night since 2026-07-26.


def _seed_many_drawers(palace_path, count=20):
    col = get_collection(palace_path, create=True)
    col.upsert(
        ids=[f"M{i}" for i in range(count)],
        documents=[
            f"Deployment note {i}: the service rollout used a canary batch of {i} pods."
            for i in range(count)
        ],
        metadatas=[
            {"wing": "ops", "room": "deploys", "source_file": f"fixture_M{i}.md"}
            for i in range(count)
        ],
    )


def _captured_finalize_pool(monkeypatch):
    """Record how many candidates reach the ranking stage."""
    from mempalace import searcher as searcher_mod

    seen = {}
    original = searcher_mod._finalize_candidate_hits

    def _spy(**kwargs):
        seen["pool"] = len(kwargs["hits"])
        return original(**kwargs)

    monkeypatch.setattr(searcher_mod, "_finalize_candidate_hits", _spy)
    return seen


def test_ranking_stage_sees_more_candidates_than_it_returns(tmp_path, monkeypatch):
    from mempalace import searcher as searcher_mod

    # The bug only showed with the second-stage reranker off — its wider pool
    # was accidentally masking it.
    monkeypatch.setattr(searcher_mod, "rerank_enabled", lambda: False)
    palace = str(tmp_path / "palace")
    _seed_many_drawers(palace)
    seen = _captured_finalize_pool(monkeypatch)

    result = search_memories("canary rollout deployment", palace_path=palace, n_results=3)

    assert seen["pool"] > 3, "hybrid rank must rank a wide pool, not the vector top-N"
    assert len(result["results"]) <= 3, "the caller's result count must not change"


def test_ranking_pool_is_not_widened_past_available_candidates(tmp_path, monkeypatch):
    """A tiny palace must not grow phantom candidates — the pool is a cap, not a quota."""
    from mempalace import searcher as searcher_mod

    monkeypatch.setattr(searcher_mod, "rerank_enabled", lambda: False)
    palace = str(tmp_path / "palace")
    _seed_drawers(palace)  # 4 drawers
    seen = _captured_finalize_pool(monkeypatch)

    search_memories("auth JWT tokens", palace_path=palace, n_results=2)

    assert seen["pool"] <= 4
