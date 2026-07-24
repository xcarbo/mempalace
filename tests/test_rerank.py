"""Tests for the opt-in local rerank stage (mempalace/rerank.py).

Transport is mocked — no test talks to a real endpoint. The contract
under test: disabled by default, reorders by embedding cosine when
enabled, and fails soft (original order) on every error path.
"""

import json

import pytest

from mempalace import rerank
from mempalace.rerank import maybe_rerank, rerank_enabled


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
    def test_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv(rerank.RERANK_URL_ENV, raising=False)
        monkeypatch.delenv(rerank.RERANK_MODEL_ENV, raising=False)
        assert rerank_enabled() is False
        hits = [{"text": "a"}, {"text": "b"}]
        assert maybe_rerank("q", hits) is hits

    def test_url_alone_not_enough(self, monkeypatch):
        monkeypatch.setenv(rerank.RERANK_URL_ENV, "http://x")
        monkeypatch.delenv(rerank.RERANK_MODEL_ENV, raising=False)
        assert rerank_enabled() is False

    def test_empty_hits_passthrough(self, enabled_env):
        assert maybe_rerank("q", []) == []


class TestReordering:
    def test_reorders_by_cosine(self, enabled_env, monkeypatch):
        # query aligns with hit B; first-stage order has A first.
        vectors = [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]  # query, A, B
        monkeypatch.setattr(rerank.urllib.request, "urlopen", _fake_urlopen_factory(vectors))
        hits = [{"text": "A"}, {"text": "B"}]
        out = maybe_rerank("q", hits)
        assert [h["text"] for h in out] == ["B", "A"]
        assert out[0]["rerank_score"] == 1.0
        assert out[1]["rerank_score"] == 0.0

    def test_applies_task_prefixes_and_truncation(self, enabled_env, monkeypatch):
        calls = []
        vectors = [[1.0], [1.0]]
        monkeypatch.setattr(rerank.urllib.request, "urlopen", _fake_urlopen_factory(vectors, calls))
        maybe_rerank("find it", [{"text": "x" * (rerank.MAX_TEXT_CHARS + 500)}])
        sent = calls[0]["input"]
        assert sent[0] == "search_query: find it"
        assert sent[1].startswith("search_document: ")
        assert len(sent[1]) <= len("search_document: ") + rerank.MAX_TEXT_CHARS

    def test_tail_beyond_pool_keeps_order(self, enabled_env, monkeypatch):
        monkeypatch.setattr(rerank, "MAX_RERANK_POOL", 2)
        vectors = [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
        monkeypatch.setattr(rerank.urllib.request, "urlopen", _fake_urlopen_factory(vectors))
        hits = [{"text": "A"}, {"text": "B"}, {"text": "tail1"}, {"text": "tail2"}]
        out = maybe_rerank("q", hits)
        assert [h["text"] for h in out] == ["B", "A", "tail1", "tail2"]


class TestFailSoft:
    def test_endpoint_error_keeps_order(self, enabled_env, monkeypatch):
        def _boom(req, timeout=None):
            raise OSError("connection refused")

        monkeypatch.setattr(rerank.urllib.request, "urlopen", _boom)
        hits = [{"text": "A"}, {"text": "B"}]
        assert maybe_rerank("q", hits) is hits

    def test_count_mismatch_keeps_order(self, enabled_env, monkeypatch):
        vectors = [[1.0]]  # only the query vector comes back
        monkeypatch.setattr(rerank.urllib.request, "urlopen", _fake_urlopen_factory(vectors))
        hits = [{"text": "A"}, {"text": "B"}]
        assert maybe_rerank("q", hits) is hits
