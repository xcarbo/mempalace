"""Tests for ``candidate_strategy="union"`` in ``search_memories``.

The default ``"vector"`` strategy gathers candidates from the vector index
only. Docs with strong BM25 signal but vector embeddings far from the query
get skipped — terminology guides looked up by narrative-shaped queries are
the canonical case.

The ``"union"`` strategy also pulls top-K BM25-only candidates from sqlite
FTS5 and merges them into the rerank pool. Both signal sources contribute
candidates; the hybrid rerank picks the best from a richer pool.

Since 2026-08-08 "union" is also the DEFAULT: vector-only retrieval cannot
find a drawer whose embedding is poor even when its text matches the query
word for word, and BM25 was only ever re-ranking what vector had already
chosen. Passing ``candidate_strategy="vector"`` still selects the vector-only
lane; passing nothing now gets both.
"""

from mempalace.palace import get_collection
from mempalace.searcher import search_memories


def _seed_drawers(palace_path):
    """Seed a corpus where the right doc for one query is BM25-strong but
    vector-distant.

    D1-D3 are short narrative tickets that semantically cluster around
    "customer support / order / shipped" vocabulary. D4 is a meta-document
    of bullet rules ("brand voice") that contains rare keywords like
    "Absolutely" and "apologize" the query repeats verbatim — strong BM25
    signal but stylistically far from the narrative tickets.
    """
    col = get_collection(palace_path, create=True)
    col.upsert(
        ids=["D1", "D2", "D3", "D4"],
        documents=[
            "Customer wrote in asking why their order shipped without "
            "the promo sticker. Standard reply explaining the threshold.",
            "Order delivery delayed three days; customer requested a "
            "refund. Support agent processed return via ticket queue.",
            "Customer asked about the missing freebie; the reply "
            "explained the campaign mechanics and shipped status.",
            "Brand voice rules: dry, sturdy, never effusive. "
            "Never 'Absolutely!' Never apologize for policy — explain it. "
            "Avoid premium / curated / elevated vocabulary.",
        ],
        metadatas=[
            {"wing": "shop", "room": "support", "source_file": "ticket_D1.md"},
            {"wing": "shop", "room": "support", "source_file": "ticket_D2.md"},
            {"wing": "shop", "room": "support", "source_file": "ticket_D3.md"},
            {"wing": "shop", "room": "guides", "source_file": "brand_voice_D4.md"},
        ],
    )


_NARRATIVE_QUERY = (
    "A support agent is drafting a reply to a customer asking why their "
    "order shipped without a free sticker. Draft the reply, but never say "
    "'Absolutely!' and do not apologize for policy."
)


class TestCandidateUnion:
    def test_the_default_now_includes_the_lexical_lane(self, tmp_path):
        """Omitting the parameter must get the union lane, not vector-only.

        The BM25-strong, vector-distant doc is the whole point: a caller that
        does not choose a strategy should still find it.
        """
        palace = str(tmp_path / "palace")
        _seed_drawers(palace)
        default = search_memories(_NARRATIVE_QUERY, palace, n_results=5)
        union = search_memories(_NARRATIVE_QUERY, palace, n_results=5, candidate_strategy="union")
        assert "brand_voice_D4.md" in {h["source_file"] for h in default["results"]}
        assert [h["source_file"] for h in default["results"]] == [
            h["source_file"] for h in union["results"]
        ], "the default must behave as union"

    def test_explicit_vector_strategy_still_means_vector_only(self, tmp_path):
        """The opt-out has to keep working, or the change is not reversible."""
        palace = str(tmp_path / "palace")
        _seed_drawers(palace)
        vector = search_memories(_NARRATIVE_QUERY, palace, n_results=2, candidate_strategy="vector")
        assert all(h.get("distance") is not None for h in vector["results"])

    def test_default_degrades_when_the_backend_has_no_lexical_search(self, tmp_path, monkeypatch):
        """Taking the default on a lexical-less backend must not fail the search.

        An explicit ``union`` still raises — the caller asked for a capability
        that is not there — but nobody should lose search by not choosing.
        """
        from mempalace.backends import UnsupportedCapabilityError
        import mempalace.searcher as searcher_mod

        palace = str(tmp_path / "palace")
        _seed_drawers(palace)

        def _no_lexical(*a, **k):
            raise UnsupportedCapabilityError("supports_lexical_search")

        monkeypatch.setitem(searcher_mod._CANDIDATE_MERGERS, "union", _no_lexical)

        default = search_memories(_NARRATIVE_QUERY, palace, n_results=5)
        assert default.get("results"), "default must degrade to vector candidates, not fail"
        assert "error" not in default

        explicit = search_memories(
            _NARRATIVE_QUERY, palace, n_results=5, candidate_strategy="union"
        )
        assert explicit.get("unsupported_capability") == "supports_lexical_search"

    def test_union_surfaces_bm25_strong_vector_distant_doc(self, tmp_path):
        """The brand-voice doc has strong BM25 signal for the query but is
        stylistically far from the narrative tickets. Union mode must
        retrieve it; vector-only mode is allowed to miss it."""
        palace = str(tmp_path / "palace")
        _seed_drawers(palace)
        result = search_memories(_NARRATIVE_QUERY, palace, n_results=5, candidate_strategy="union")
        ids = [h["source_file"] for h in result["results"]]
        assert "brand_voice_D4.md" in ids, (
            f"union mode must surface BM25-strong docs even when vector signal is weak; got {ids}"
        )

    def test_union_preserves_vector_hits(self, tmp_path):
        """Union mode must not drop docs that vector-only mode finds —
        the rerank pool grows, it doesn't shrink."""
        palace = str(tmp_path / "palace")
        _seed_drawers(palace)
        vector = search_memories(_NARRATIVE_QUERY, palace, n_results=5, candidate_strategy="vector")
        union = search_memories(_NARRATIVE_QUERY, palace, n_results=5, candidate_strategy="union")
        vec_ids = {h["source_file"] for h in vector["results"]}
        union_ids = {h["source_file"] for h in union["results"]}
        # In a 4-doc corpus with n_results=5, both should return all 4.
        # The invariant is: union should not lose anything vector found.
        missing = vec_ids - union_ids
        assert not missing, f"union dropped docs that vector found: {missing}"

    def test_union_handles_empty_palace(self, tmp_path):
        """No drawers — union mode should return empty results, not crash."""
        palace = str(tmp_path / "palace")
        get_collection(palace, create=True)  # create empty collection
        result = search_memories("anything", palace, n_results=5, candidate_strategy="union")
        assert result.get("results", []) == []

    def test_invalid_candidate_strategy_raises(self, tmp_path):
        """Bad arg should raise rather than silently fall back."""
        palace = str(tmp_path / "palace")
        _seed_drawers(palace)
        import pytest

        with pytest.raises(ValueError, match="candidate_strategy"):
            search_memories("anything", palace, n_results=5, candidate_strategy="bogus")

    def test_invalid_strategy_raises_even_when_vector_disabled(self, tmp_path):
        """Validation must happen before the ``vector_disabled`` early return —
        invalid values must fail consistently regardless of routing."""
        palace = str(tmp_path / "palace")
        _seed_drawers(palace)
        import pytest

        with pytest.raises(ValueError, match="candidate_strategy"):
            search_memories(
                "anything",
                palace,
                n_results=5,
                vector_disabled=True,
                candidate_strategy="bogus",
            )

    def test_union_respects_n_results_limit(self, tmp_path):
        """When the merged candidate set is larger than ``n_results``, the
        result must be trimmed back to the requested size — the MCP
        ``limit`` contract depends on this invariant."""
        palace = str(tmp_path / "palace")
        _seed_drawers(palace)
        # 4-doc corpus, n_results=2 → union pool can grow to ~8 candidates,
        # rerank reorders them, but final list must respect the cap.
        result = search_memories(_NARRATIVE_QUERY, palace, n_results=2, candidate_strategy="union")
        assert len(result["results"]) <= 2, (
            f"union must trim to n_results=2; got {len(result['results'])} results"
        )

    def test_max_distance_bounds_the_vector_lane_only(self, tmp_path):
        """``max_distance`` is a VECTOR-distance threshold (changed 2026-08-08).

        It used to skip the lexical lane entirely whenever a threshold was
        set, which meant the lane never ran for any default caller — the CLI
        and tool paths both pass 1.5. A lexical hit has no vector distance, so
        it can neither satisfy nor violate a vector bound; it is admitted on
        lexical relevance and marked ``matched_via="bm25_backend"``. The
        threshold still does its real job: every hit that HAS a distance
        respects it."""
        palace = str(tmp_path / "palace")
        _seed_drawers(palace)
        # Sanity: without max_distance, union surfaces the BM25-strong doc.
        unfiltered = search_memories(
            _NARRATIVE_QUERY, palace, n_results=5, candidate_strategy="union"
        )
        assert "brand_voice_D4.md" in {h["source_file"] for h in unfiltered["results"]}

        # With a tight max_distance the lexical lane still runs, and every
        # hit carrying a distance still respects the bound.
        filtered = search_memories(
            _NARRATIVE_QUERY,
            palace,
            n_results=5,
            candidate_strategy="union",
            max_distance=0.5,
        )
        for h in filtered["results"]:
            if h.get("distance") is None:
                assert h["matched_via"] == "bm25_backend", (
                    f"a hit without a vector distance must be identifiable as "
                    f"lexical-lane; offending hit: {h}"
                )
                continue
            assert h["distance"] <= 0.5, f"hit violates max_distance=0.5: distance={h['distance']}"
        assert "brand_voice_D4.md" in {h["source_file"] for h in filtered["results"]}, (
            "the lexical lane must still contribute under a distance threshold"
        )

    def test_union_dedup_is_chunk_precise_not_basename(self, tmp_path):
        """Two files with the same basename in different directories must
        not collide — union must dedup on full path (or chunk-level key),
        not on basename alone. Otherwise a BM25-strong README from one
        directory silently shadows a BM25-strong README from another.
        """
        palace = str(tmp_path / "palace")
        col = get_collection(palace, create=True)
        col.upsert(
            ids=["A_README", "B_README", "narrative"],
            documents=[
                # Both README files share the basename README.md but live
                # in different directories. Each contains distinctive
                # terminology a query might surface via BM25.
                "PROJECT ALPHA: configuration for the Frobnitz subsystem. "
                "Set FROBNITZ_TIMEOUT=30 to enable widget rotation.",
                "PROJECT BETA: configuration for the Wibble subsystem. "
                "Set WIBBLE_THRESHOLD=0.5 to enable signal smoothing.",
                "Engineers occasionally chat about how the legacy "
                "subsystems all need their config knobs tweaked.",
            ],
            metadatas=[
                {"wing": "code", "room": "docs", "source_file": "alpha/README.md"},
                {"wing": "code", "room": "docs", "source_file": "beta/README.md"},
                {"wing": "code", "room": "docs", "source_file": "chat.md"},
            ],
        )
        # Query that hits BM25 for BOTH READMEs (distinct vocab from each).
        # Vector-only might pick the chat doc as semantically "closest";
        # union must surface both READMEs without basename collision.
        result = search_memories(
            "FROBNITZ_TIMEOUT WIBBLE_THRESHOLD configuration",
            palace,
            n_results=5,
            candidate_strategy="union",
        )
        sources = [h["source_file"] for h in result["results"]]
        readme_count = sum(1 for s in sources if s == "README.md")
        assert readme_count >= 2, (
            f"union must surface both README.md files from different dirs "
            f"(basename collision would drop one); got sources={sources}"
        )

    def test_union_respects_source_file_filter(self, tmp_path):
        """Union pulls BM25 candidates from sqlite FTS5 directly; the
        source_file filter must constrain that pool too, not just the vector
        path — otherwise union silently re-injects other sources (#1815)."""
        palace = str(tmp_path / "palace")
        _seed_drawers(palace)
        result = search_memories(
            _NARRATIVE_QUERY,
            palace,
            n_results=5,
            candidate_strategy="union",
            source_file="ticket_D2.md",
        )
        sources = {h["source_file"] for h in result["results"]}
        assert sources <= {"ticket_D2.md"}, (
            f"union must honor source_file on the BM25 pool; got {sources}"
        )
        # The BM25-strong brand-voice doc must NOT leak past the filter.
        assert "brand_voice_D4.md" not in sources


class TestHybridRankTolerantOfMissingDistance:
    """``_hybrid_rank`` accepts ``distance=None`` — required for BM25-only
    candidates injected by union mode."""

    def test_distance_none_scored_as_zero_vector_sim(self):
        from mempalace.searcher import _hybrid_rank

        results = [
            {"text": "alpha beta gamma", "distance": 0.2},  # close vector match
            {"text": "alpha alpha alpha", "distance": None},  # BM25-only — heavy term repetition
        ]
        # Query matches "alpha" heavily; the BM25-only candidate with no
        # vector signal should still rank competitively on BM25 alone.
        ranked = _hybrid_rank(results, "alpha")
        assert all("bm25_score" in r for r in ranked), "rerank should add bm25_score"
        # Both must survive — neither should crash on distance=None.
        assert len(ranked) == 2
