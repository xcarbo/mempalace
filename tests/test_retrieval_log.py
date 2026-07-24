"""Tests for the retrieval log (flight recorder) and drawer_id exposure.

The log is the raw signal for recall regression checks and future
salience-based curation; drawer_id in search hits is what makes the log
(and mempalace_get_drawer follow-ups) actionable.
"""

import json

import pytest

from mempalace import retrieval_log
from mempalace.retrieval_log import log_retrieval, retrieval_log_path
from mempalace.searcher import _drawer_id_from, search_memories


@pytest.fixture
def scratch_home(tmp_path, monkeypatch):
    """Point the log at a scratch HOME so tests never touch the real one."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv(retrieval_log.DISABLE_ENV, raising=False)
    return tmp_path


def _read_records(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ── log_retrieval ──────────────────────────────────────────────────────


class TestLogRetrieval:
    def test_appends_jsonl_record(self, scratch_home):
        log_retrieval("search", query="jwt", drawer_ids=["d1", "d2"], returned=2)
        records = _read_records(retrieval_log_path())
        assert len(records) == 1
        rec = records[0]
        assert rec["tool"] == "search"
        assert rec["query"] == "jwt"
        assert rec["drawer_ids"] == ["d1", "d2"]
        assert rec["returned"] == 2
        assert "ts" in rec and "pid" in rec

    def test_appends_across_calls(self, scratch_home):
        log_retrieval("search", query="a")
        log_retrieval("get_drawer", drawer_id="d1", found=True)
        records = _read_records(retrieval_log_path())
        assert [r["tool"] for r in records] == ["search", "get_drawer"]

    def test_month_stamped_filename(self, scratch_home):
        log_retrieval("search", query="a")
        path = retrieval_log_path()
        assert path.endswith(".jsonl")
        assert "retrieval-" in path
        assert str(scratch_home) in path

    def test_opt_out_env(self, scratch_home, monkeypatch):
        monkeypatch.setenv(retrieval_log.DISABLE_ENV, "0")
        log_retrieval("search", query="a")
        import os

        assert not os.path.exists(retrieval_log_path())

    @pytest.mark.parametrize("value", ["0", "false", "off", "no", " FALSE "])
    def test_opt_out_values(self, monkeypatch, value):
        monkeypatch.setenv(retrieval_log.DISABLE_ENV, value)
        assert retrieval_log.retrieval_log_enabled() is False

    @pytest.mark.parametrize("value", ["1", "true", "", "on"])
    def test_enabled_values(self, monkeypatch, value):
        monkeypatch.setenv(retrieval_log.DISABLE_ENV, value)
        assert retrieval_log.retrieval_log_enabled() is True

    def test_fail_soft_on_unwritable_dir(self, tmp_path, monkeypatch):
        # HOME points at a *file*, so makedirs must fail — and the call
        # must swallow it: a logging failure never breaks a read path.
        blocker = tmp_path / "not-a-dir"
        blocker.write_text("x")
        monkeypatch.setenv("HOME", str(blocker))
        monkeypatch.delenv(retrieval_log.DISABLE_ENV, raising=False)
        log_retrieval("search", query="a")  # must not raise

    def test_non_serializable_values_stringified(self, scratch_home):
        class Odd:
            def __str__(self):
                return "odd-thing"

        log_retrieval("search", query="a", extra=Odd())
        records = _read_records(retrieval_log_path())
        assert records[0]["extra"] == "odd-thing"


# ── _drawer_id_from ────────────────────────────────────────────────────


class TestDrawerIdFrom:
    def test_parent_drawer_id_wins(self):
        meta = {"parent_drawer_id": "drawer_w_r_abc"}
        assert _drawer_id_from(meta, "drawer_w_r_abc_chunk_000003") == "drawer_w_r_abc"

    def test_chunk_suffix_stripped_from_record_id(self):
        assert _drawer_id_from({}, "drawer_w_r_abc_chunk_000001") == "drawer_w_r_abc"

    def test_plain_record_id_passthrough(self):
        assert _drawer_id_from({}, "drawer_w_r_abc") == "drawer_w_r_abc"

    def test_none_when_no_identity(self):
        assert _drawer_id_from({}, None) is None
        assert _drawer_id_from(None, None) is None


# ── drawer_id in search results ────────────────────────────────────────


class TestSearchResultsCarryDrawerId:
    def test_vector_hits_have_drawer_id(self, palace_path, seeded_collection):
        result = search_memories("JWT authentication", palace_path)
        assert result["results"]
        for hit in result["results"]:
            assert "drawer_id" in hit
            assert hit["drawer_id"]


# ── handler instrumentation ────────────────────────────────────────────


class TestToolSearchLogs:
    def test_tool_search_writes_log_line(self, scratch_home, monkeypatch):
        from mempalace import mcp_server

        canned = {
            "query": "jwt",
            "filters": {"wing": None, "room": None, "source_file": None},
            "total_before_filter": 1,
            "results": [{"drawer_id": "d1", "effective_distance": 0.3, "text": "x"}],
        }
        monkeypatch.setattr(mcp_server, "search_memories", lambda *a, **k: dict(canned))
        monkeypatch.setattr(mcp_server, "_refresh_vector_disabled_flag", lambda: None)
        monkeypatch.setattr(mcp_server, "_vector_disabled", False)

        mcp_server.tool_search("jwt")

        records = _read_records(retrieval_log_path())
        assert len(records) == 1
        rec = records[0]
        assert rec["tool"] == "search"
        assert rec["drawer_ids"] == ["d1"]
        assert rec["distances"] == [0.3]


# ── chunk-hit dedup ────────────────────────────────────────────────────


class TestChunkDedup:
    def test_chunked_drawer_surfaces_once(self, palace_path, collection):
        """Chunks of one logical drawer must collapse to a single hit."""
        collection.add(
            ids=[f"drawer_w_r_multi_chunk_{i:06d}" for i in range(3)] + ["drawer_w_r_other"],
            documents=[
                "kubernetes ingress routing rules for the staging cluster",
                "kubernetes ingress certificate renewal via cert-manager",
                "kubernetes ingress canary weights and rollback procedure",
                "postgres backup rotation schedule",
            ],
            metadatas=[
                {
                    "wing": "w",
                    "room": "r",
                    "source_file": "k8s.md",
                    "chunk_index": i,
                    "parent_drawer_id": "drawer_w_r_multi",
                }
                for i in range(3)
            ]
            + [{"wing": "w", "room": "r", "source_file": "pg.md", "chunk_index": 0}],
        )
        result = search_memories("kubernetes ingress", palace_path, n_results=5)
        ids = [h["drawer_id"] for h in result["results"]]
        assert ids.count("drawer_w_r_multi") == 1
        assert len(ids) == len(set(ids))

    def test_dedupe_keeps_hits_without_identity(self):
        from mempalace.searcher import _dedupe_by_drawer_id

        hits = [{"text": "a"}, {"text": "b"}, {"drawer_id": "d1"}, {"drawer_id": "d1"}]
        out = _dedupe_by_drawer_id(hits)
        assert len(out) == 3
        assert [h.get("drawer_id") for h in out] == [None, None, "d1"]
