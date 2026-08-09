"""Tests for the opt-in local rerank scoring stage (mempalace/rerank.py).

Transport is mocked — no test talks to a real endpoint. The contract
under test: disabled by default, ANNOTATES scores when enabled (ordering
is the searcher's job — see the _blend_rerank tests in
test_hybrid_search.py), and fails soft (no scores, False) on every error
path.
"""

import json

import pytest

from mempalace import rerank
from mempalace.rerank import annotate_rerank_scores, rerank_enabled


@pytest.fixture
def enabled_env(monkeypatch):
    monkeypatch.setenv(rerank.RERANK_URL_ENV, "http://127.0.0.1:1234/v1/embeddings")
    monkeypatch.setenv(rerank.RERANK_MODEL_ENV, "test-embed")


def _fake_urlopen_factory(vectors, calls=None):
    """urlopen stub returning an OpenAI-style embeddings payload."""

    class _Resp:
        def __init__(self, body):
            self._body = body

        def read(self):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def _fake_urlopen(req, timeout=None):
        if calls is not None:
            calls.append(json.loads(req.data.decode("utf-8")))
        payload = {
            "data": [{"index": i, "embedding": v} for i, v in enumerate(vectors)],
        }
        return _Resp(json.dumps(payload).encode("utf-8"))

    return _fake_urlopen


class TestGating:
    def test_suite_runs_with_rerank_off_regardless_of_ambient_env(self):
        """Guard the autouse ``_no_local_rerank`` fixture in conftest.

        Deliberately touches no env vars: this machine's zshrc used to export
        MEMPALACE_RERANK_URL/_MODEL into every shell and agent, and when they
        leaked into the suite LM Studio silently re-scored ranking assertions
        (two tests failed locally, passed in CI). Unlike
        ``test_disabled_by_default``, which clears the vars itself, this fails
        if the fixture is removed.
        """
        assert rerank_enabled() is False

    def test_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv(rerank.RERANK_URL_ENV, raising=False)
        monkeypatch.delenv(rerank.RERANK_MODEL_ENV, raising=False)
        assert rerank_enabled() is False
        hits = [{"text": "a"}, {"text": "b"}]
        assert annotate_rerank_scores("q", hits) is False
        assert "rerank_score" not in hits[0]

    def test_url_alone_not_enough(self, monkeypatch):
        monkeypatch.setenv(rerank.RERANK_URL_ENV, "http://x")
        monkeypatch.delenv(rerank.RERANK_MODEL_ENV, raising=False)
        assert rerank_enabled() is False

    def test_empty_hits_passthrough(self, enabled_env):
        assert annotate_rerank_scores("q", []) is False


class TestScoring:
    def test_annotates_cosine_without_reordering(self, enabled_env, monkeypatch):
        # query aligns with hit B; first-stage order has A first. The scorer
        # must record that WITHOUT touching the order — ordering belongs to
        # the searcher's blend, which also re-applies the archive demotion.
        vectors = [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]  # query, A, B
        monkeypatch.setattr(rerank.urllib.request, "urlopen", _fake_urlopen_factory(vectors))
        hits = [{"text": "A"}, {"text": "B"}]
        assert annotate_rerank_scores("q", hits) is True
        assert [h["text"] for h in hits] == ["A", "B"]
        assert hits[0]["rerank_score"] == 0.0
        assert hits[1]["rerank_score"] == 1.0

    def test_applies_task_prefixes_and_truncation(self, enabled_env, monkeypatch):
        calls = []
        vectors = [[1.0], [1.0]]
        monkeypatch.setattr(rerank.urllib.request, "urlopen", _fake_urlopen_factory(vectors, calls))
        annotate_rerank_scores("find it", [{"text": "x" * (rerank.MAX_TEXT_CHARS + 500)}])
        sent = calls[0]["input"]
        assert sent[0] == "search_query: find it"
        assert sent[1].startswith("search_document: ")
        assert len(sent[1]) <= len("search_document: ") + rerank.MAX_TEXT_CHARS


class TestFailSoft:
    def test_endpoint_error_scores_nothing(self, enabled_env, monkeypatch):
        def _boom(req, timeout=None):
            raise OSError("connection refused")

        monkeypatch.setattr(rerank.urllib.request, "urlopen", _boom)
        hits = [{"text": "A"}, {"text": "B"}]
        assert annotate_rerank_scores("q", hits) is False
        assert all("rerank_score" not in h for h in hits)

    def test_count_mismatch_scores_nothing(self, enabled_env, monkeypatch):
        vectors = [[1.0]]  # only the query vector comes back
        monkeypatch.setattr(rerank.urllib.request, "urlopen", _fake_urlopen_factory(vectors))
        hits = [{"text": "A"}, {"text": "B"}]
        assert annotate_rerank_scores("q", hits) is False
