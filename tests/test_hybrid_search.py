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


# ── archive demotion (the "dim" state of decision f6639c96) ───────────────
#
# The palace holds two things in one index: drawers written deliberately, and
# whole transcripts mined automatically as a backup for the ~half of sessions
# that never get a deliberate save. Both are verbatim and neither is ever
# destroyed. They are not equally relevant, and on 2026-08-08 the archive was
# 62% of the palace and 70% of every result set — it had crowded the curated
# layer out of its own search. Demotion attenuates, it never excludes.


def _seed_archive_and_curated(palace_path):
    """One archive drawer and one curated drawer, both matching the query."""
    col = get_collection(palace_path, create=True)
    col.upsert(
        ids=["ARCH", "CUR"],
        documents=[
            "We talked about the deploy rollback policy in passing during the session.",
            "Decision: the deploy rollback policy is two-stage with a manual gate.",
        ],
        metadatas=[
            {"wing": "sessions", "room": "technical", "source_file": "transcript.jsonl"},
            {"wing": "ops", "room": "decisions", "source_file": "decisions.md"},
        ],
    )


def test_archive_drawer_is_demoted_below_a_curated_one(tmp_path, monkeypatch):
    monkeypatch.delenv("MEMPALACE_ARCHIVE_WINGS", raising=False)
    monkeypatch.delenv("MEMPALACE_ARCHIVE_RANK_PENALTY", raising=False)
    palace = str(tmp_path / "palace")
    _seed_archive_and_curated(palace)

    hits = search_memories("deploy rollback policy", palace_path=palace, n_results=5)["results"]
    ids = [h["drawer_id"] for h in hits]

    assert "ARCH" in ids, "demotion must never drop the archive from the results"
    assert ids.index("CUR") < ids.index("ARCH")
    assert next(h for h in hits if h["drawer_id"] == "ARCH")["archive_demoted"] is True


def test_archive_still_wins_when_it_is_clearly_the_better_answer(tmp_path, monkeypatch):
    """The check the golden set cannot make: no golden case has a transcript answer.

    A demotion that buried the archive outright would pass every golden query
    and still be wrong — the archive is the only record for half the sessions.
    """
    monkeypatch.delenv("MEMPALACE_ARCHIVE_WINGS", raising=False)
    monkeypatch.delenv("MEMPALACE_ARCHIVE_RANK_PENALTY", raising=False)
    palace = str(tmp_path / "palace")
    col = get_collection(palace, create=True)
    col.upsert(
        ids=["ARCH", "CUR"],
        documents=[
            "The kafka consumer rebalance timeout was raised to 45 seconds after the incident.",
            "Unrelated note about frontend state management libraries.",
        ],
        metadatas=[
            {"wing": "sessions", "room": "technical", "source_file": "transcript.jsonl"},
            {"wing": "ops", "room": "decisions", "source_file": "decisions.md"},
        ],
    )

    hits = search_memories("kafka consumer rebalance timeout", palace_path=palace, n_results=5)
    assert hits["results"][0]["drawer_id"] == "ARCH"


def test_archive_demotion_is_fully_reversible(tmp_path, monkeypatch):
    """One env var restores the flat corpus — nothing is written to any drawer."""
    monkeypatch.setenv("MEMPALACE_ARCHIVE_WINGS", "")
    palace = str(tmp_path / "palace")
    _seed_archive_and_curated(palace)

    hits = search_memories("deploy rollback policy", palace_path=palace, n_results=5)["results"]
    assert not any(h.get("archive_demoted") for h in hits)


def test_archive_penalty_env_rejects_nonsense(monkeypatch):
    from mempalace.searcher import ARCHIVE_PENALTY_DEFAULT, archive_rank_penalty

    for bad in ("", "banana", "-1", "0", "1.5"):
        monkeypatch.setenv("MEMPALACE_ARCHIVE_RANK_PENALTY", bad)
        assert archive_rank_penalty() == ARCHIVE_PENALTY_DEFAULT
    monkeypatch.setenv("MEMPALACE_ARCHIVE_RANK_PENALTY", "0.5")
    assert archive_rank_penalty() == 0.5


def test_archive_wings_env_accepts_a_list(monkeypatch):
    from mempalace.searcher import archive_wings

    monkeypatch.setenv("MEMPALACE_ARCHIVE_WINGS", "sessions, diary ,")
    assert archive_wings() == {"sessions", "diary"}


# ── fusion mode: RRF vs weighted (MEMPALACE_FUSION) ──────────────────────────


def test_fusion_mode_defaults_to_weighted_and_rejects_nonsense(monkeypatch):
    from mempalace.searcher import fusion_mode

    monkeypatch.delenv("MEMPALACE_FUSION", raising=False)
    assert fusion_mode() == "weighted"
    monkeypatch.setenv("MEMPALACE_FUSION", "banana")
    assert fusion_mode() == "weighted"
    monkeypatch.setenv("MEMPALACE_FUSION", "rrf")
    assert fusion_mode() == "rrf"


def test_rrf_k_env_parses_and_rejects_nonsense(monkeypatch):
    from mempalace.searcher import RRF_K_DEFAULT, _rrf_k

    monkeypatch.delenv("MEMPALACE_RRF_K", raising=False)
    assert _rrf_k() == RRF_K_DEFAULT
    for bad in ("banana", "0", "-5"):
        monkeypatch.setenv("MEMPALACE_RRF_K", bad)
        assert _rrf_k() == RRF_K_DEFAULT
    monkeypatch.setenv("MEMPALACE_RRF_K", "10")
    assert _rrf_k() == 10


def test_rrf_ranks_dual_lane_candidate_above_single_lane_ones(monkeypatch):
    """Under RRF a candidate present in BOTH lanes must beat candidates that
    only one lane found, even when each of those leads its own lane."""
    from mempalace.searcher import _hybrid_rank

    monkeypatch.setenv("MEMPALACE_FUSION", "rrf")
    results = [
        # A: vector rank 2 AND bm25 rank 1.
        {"text": "alpha beta alpha beta", "distance": 0.5, "wing": "w"},
        # B: vector rank 1 only (no lexical overlap with the query).
        {"text": "unrelated words entirely", "distance": 0.2, "wing": "w"},
        # C: lexical-only (union lane), bm25 rank 2, no vector distance.
        {"text": "alpha beta", "distance": None, "wing": "w"},
    ]
    ranked = _hybrid_rank(results, "alpha beta")
    assert [r["text"] for r in ranked] == [
        "alpha beta alpha beta",
        "unrelated words entirely",
        "alpha beta",
    ]


def test_rrf_applies_archive_demotion_as_multiplier(monkeypatch):
    """The archive demotion must survive the fusion swap: a sessions-wing
    candidate that narrowly leads both lanes loses to a close curated
    second under the x0.75 multiplier."""
    from mempalace.searcher import _hybrid_rank

    monkeypatch.setenv("MEMPALACE_FUSION", "rrf")
    results = [
        {"text": "alpha beta alpha beta", "distance": 0.2, "wing": "sessions"},
        {"text": "alpha beta alpha", "distance": 0.3, "wing": "mempalace"},
    ]
    ranked = _hybrid_rank(results, "alpha beta")
    assert ranked[0]["wing"] == "mempalace"
    assert ranked[1].get("archive_demoted") is True


def test_weighted_mode_is_unchanged_by_default(monkeypatch):
    """With MEMPALACE_FUSION unset the historical convex combination must
    keep ranking exactly as before (guard against RRF becoming a silent
    default)."""
    from mempalace.searcher import _hybrid_rank

    monkeypatch.delenv("MEMPALACE_FUSION", raising=False)
    results = [
        {"text": "unrelated words entirely", "distance": 0.1, "wing": "w"},
        {"text": "alpha beta alpha beta", "distance": 1.4, "wing": "w"},
    ]
    ranked = _hybrid_rank(results, "alpha beta")
    # weighted: doc1 = 0.6*0.9 = 0.54; doc2 = 0.6*(1-1.4=0 -> 0) + 0.4*1.0 = 0.4
    assert ranked[0]["text"] == "unrelated words entirely"


# ── second-stage rerank blend (_blend_rerank) ────────────────────────────────


def _annotate_with(scores):
    """Patchable annotate_rerank_scores stand-in assigning fixed scores."""

    def _annotate(query, pool):
        for h, s in zip(pool, scores):
            h["rerank_score"] = s
        return True

    return _annotate


def test_blend_keeps_fused_winner_when_rerank_disagrees_mildly(monkeypatch):
    """The rerank signal is one voice, not the whole ranking: sorting by raw
    rerank cosine alone would flip H1/H2 here; the 50/50 blend must not."""
    from mempalace import searcher as searcher_mod

    hits = [
        {"text": "H1", "_fused_raw": 1.0, "wing": "w"},
        {"text": "H2", "_fused_raw": 0.5, "wing": "w"},
        {"text": "H3", "_fused_raw": 0.0, "wing": "w"},
    ]
    monkeypatch.setattr(searcher_mod, "annotate_rerank_scores", _annotate_with([0.80, 0.85, 0.0]))
    out = searcher_mod._blend_rerank("q", hits)
    assert [h["text"] for h in out] == ["H1", "H2", "H3"]


def test_blend_reapplies_archive_demotion_after_reranking(monkeypatch):
    """The old reranker discarded the archive demotion inside the reranked
    head (measured: R@5 0.907->0.847). The blend must re-apply it: an
    archive drawer leading both signals loses to a near-tied curated one."""
    from mempalace import searcher as searcher_mod

    hits = [
        {"text": "ARCH", "_fused_raw": 1.0, "wing": "sessions"},
        {"text": "CUR", "_fused_raw": 0.98, "wing": "mempalace"},
        {"text": "PAD", "_fused_raw": 0.0, "wing": "mempalace"},
    ]
    monkeypatch.setattr(searcher_mod, "annotate_rerank_scores", _annotate_with([1.0, 0.98, 0.0]))
    out = searcher_mod._blend_rerank("q", hits)
    assert [h["text"] for h in out] == ["CUR", "ARCH", "PAD"]


def test_blend_fails_soft_to_first_stage_order(monkeypatch):
    from mempalace import searcher as searcher_mod

    hits = [
        {"text": "A", "_fused_raw": 1.0, "wing": "w"},
        {"text": "B", "_fused_raw": 0.5, "wing": "w"},
    ]
    monkeypatch.setattr(searcher_mod, "annotate_rerank_scores", lambda q, p: False)
    out = searcher_mod._blend_rerank("q", hits)
    assert [h["text"] for h in out] == ["A", "B"]


def test_blend_weight_env_parses_and_rejects_nonsense(monkeypatch):
    from mempalace.searcher import RERANK_BLEND_DEFAULT, _rerank_blend_weight

    monkeypatch.delenv("MEMPALACE_RERANK_BLEND", raising=False)
    assert _rerank_blend_weight() == RERANK_BLEND_DEFAULT
    for bad in ("banana", "-0.1", "1.5"):
        monkeypatch.setenv("MEMPALACE_RERANK_BLEND", bad)
        assert _rerank_blend_weight() == RERANK_BLEND_DEFAULT
    monkeypatch.setenv("MEMPALACE_RERANK_BLEND", "0.3")
    assert _rerank_blend_weight() == 0.3


# ── vector-lane candidate floor (demotion needs curated material) ────────────


def test_vector_candidate_floor_applies_only_when_archive_can_crowd(monkeypatch):
    """The archive demotion runs on the candidate pool; with a fetch of only
    n*3 chunks the pool was ~90% sessions rows and a curated drawer at
    vector rank 16+ was cut before demotion could promote it (agent-03,
    2026-08-08). The floor mirrors _LEXICAL_CANDIDATE_FLOOR: unscoped
    searches over an archive-bearing palace fetch at least
    _VECTOR_CANDIDATE_FLOOR chunks; wing-scoped searches and palaces with
    the mechanism disabled keep the cheap proportional fetch."""
    from mempalace import searcher as searcher_mod
    from mempalace.searcher import _VECTOR_CANDIDATE_FLOOR, _vector_candidate_count

    monkeypatch.delenv("MEMPALACE_ARCHIVE_WINGS", raising=False)
    monkeypatch.setattr(searcher_mod, "rerank_enabled", lambda: False)
    assert _vector_candidate_count(5, None) == _VECTOR_CANDIDATE_FLOOR
    assert _vector_candidate_count(30, None) == 90  # proportional once past the floor
    assert _vector_candidate_count(5, "mempalace") == 15

    monkeypatch.setenv("MEMPALACE_ARCHIVE_WINGS", "")
    assert _vector_candidate_count(5, None) == 15

    monkeypatch.delenv("MEMPALACE_ARCHIVE_WINGS", raising=False)
    monkeypatch.setattr(searcher_mod, "rerank_enabled", lambda: True)
    assert _vector_candidate_count(5, None) == max(
        _VECTOR_CANDIDATE_FLOOR, searcher_mod.MAX_RERANK_POOL * 2
    )
