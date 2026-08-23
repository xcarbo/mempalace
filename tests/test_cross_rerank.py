"""Cross-encoder scorer: parsing, fail-soft, and the dispatch into rerank."""

from __future__ import annotations

import io
import json
import os
import urllib.error
from unittest import mock

import pytest

from mempalace import cross_rerank
from mempalace.rerank import annotate_rerank_scores


def _payload(yes: float | None, no: float | None, extra: dict | None = None) -> bytes:
    tops = []
    if yes is not None:
        tops.append({"token": "yes", "logprob": yes})
    if no is not None:
        tops.append({"token": "no", "logprob": no})
    for tok, lp in (extra or {}).items():
        tops.append({"token": tok, "logprob": lp})
    return json.dumps(
        {"choices": [{"text": "yes", "logprobs": {"content": [{"top_logprobs": tops}]}}]}
    ).encode()


class _Resp:
    def __init__(self, body: bytes):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setenv("MEMPALACE_RERANK_URL", "http://x/v1/completions")
    monkeypatch.setenv("MEMPALACE_RERANK_MODEL", "some-reranker")
    monkeypatch.delenv("MEMPALACE_RERANK_KEY_FILE", raising=False)


def test_score_is_softmax_of_yes_against_no(configured):
    # yes == no must land exactly on 0.5; that is the only value that pins the
    # formula rather than merely "something monotonic".
    with mock.patch("urllib.request.urlopen", return_value=_Resp(_payload(-1.0, -1.0))):
        assert cross_rerank.score_pairs("q", ["d"]) == pytest.approx([0.5])

    with mock.patch("urllib.request.urlopen", return_value=_Resp(_payload(-0.15, -5.91))):
        (score,) = cross_rerank.score_pairs("q", ["d"])
    assert score > 0.99


def test_case_variants_fold_into_one_verdict(configured):
    # The model spreads probability across yes/Yes/YES; counting only the exact
    # lowercase token throws that away and understates a confident hit.
    with mock.patch(
        "urllib.request.urlopen",
        return_value=_Resp(_payload(None, -0.5, extra={"YES": -0.5})),
    ):
        (score,) = cross_rerank.score_pairs("q", ["d"])
    assert score == pytest.approx(0.5)


def test_missing_verdict_token_does_not_explode(configured):
    # Neither yes nor no in the window: both fall to the same floor, so the
    # score is 0.5 ("no information"), not a crash and not a silent 0.0.
    with mock.patch("urllib.request.urlopen", return_value=_Resp(_payload(None, None))):
        assert cross_rerank.score_pairs("q", ["d"]) == pytest.approx([0.5])


def test_returns_none_when_unconfigured(monkeypatch):
    monkeypatch.delenv("MEMPALACE_RERANK_URL", raising=False)
    monkeypatch.delenv("MEMPALACE_RERANK_MODEL", raising=False)
    assert cross_rerank.score_pairs("q", ["d"]) is None


def test_returns_none_on_transport_failure(configured):
    # Fail-soft is the contract: an unreachable or asleep endpoint must leave
    # the caller in first-stage order, never raise into a search.
    with mock.patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
        assert cross_rerank.score_pairs("q", ["d"]) is None


def test_key_file_is_read_not_the_env_value(configured, monkeypatch, tmp_path):
    # A FILE, because an env var set in a shell profile never reaches cron or
    # launchd — the trap the outbox secret fell into.
    key = tmp_path / "api.key"
    key.write_text("sk-test-123\n")
    monkeypatch.setenv("MEMPALACE_RERANK_KEY_FILE", str(key))
    seen = {}

    def _capture(req, timeout=None):
        seen["auth"] = req.headers.get("Authorization")
        return _Resp(_payload(-1.0, -1.0))

    with mock.patch("urllib.request.urlopen", side_effect=_capture):
        cross_rerank.score_pairs("q", ["d"])
    assert seen["auth"] == "Bearer sk-test-123"


def test_unreadable_key_file_sends_no_header(configured, monkeypatch, tmp_path):
    monkeypatch.setenv("MEMPALACE_RERANK_KEY_FILE", str(tmp_path / "missing.key"))
    seen = {}

    def _capture(req, timeout=None):
        seen["auth"] = req.headers.get("Authorization")
        return _Resp(_payload(-1.0, -1.0))

    with mock.patch("urllib.request.urlopen", side_effect=_capture):
        cross_rerank.score_pairs("q", ["d"])
    assert seen["auth"] is None


def test_prompt_uses_the_raw_template_not_a_chat_shape():
    # Driving this family through a chat endpoint puts the verdict in
    # reasoning_content and returns empty content. The raw template is the fix,
    # so the closed <think> block is load-bearing, not decoration.
    p = cross_rerank._prompt("the query", "the document")
    assert "<|im_start|>system" in p
    assert "<think>\n\n</think>" in p
    assert "<Query>: the query" in p
    assert "<Document>: the document" in p


def test_document_text_is_capped(configured):
    seen = {}

    def _capture(req, timeout=None):
        seen["body"] = json.loads(req.data)
        return _Resp(_payload(-1.0, -1.0))

    with mock.patch("urllib.request.urlopen", side_effect=_capture):
        cross_rerank.score_pairs("q", ["x" * 10_000])
    assert seen["body"]["prompt"].count("x") == cross_rerank.MAX_TEXT_CHARS
    assert seen["body"]["max_tokens"] == 1


class TestDispatch:
    """`MEMPALACE_RERANK_MODE` picks the scorer; the contract is unchanged."""

    def test_cross_mode_annotates_via_the_cross_encoder(self, configured, monkeypatch):
        monkeypatch.setenv("MEMPALACE_RERANK_MODE", "cross")
        pool = [{"text": "a"}, {"text": "b"}]
        with mock.patch.object(cross_rerank, "score_pairs", return_value=[0.9, 0.1]) as sp:
            assert annotate_rerank_scores("q", pool) is True
        assert [h["rerank_score"] for h in pool] == [0.9, 0.1]
        assert sp.call_args[0][1] == ["a", "b"]

    def test_cross_mode_failure_keeps_first_stage_order(self, configured, monkeypatch):
        monkeypatch.setenv("MEMPALACE_RERANK_MODE", "cross")
        pool = [{"text": "a"}]
        with mock.patch.object(cross_rerank, "score_pairs", return_value=None):
            assert annotate_rerank_scores("q", pool) is False
        assert "rerank_score" not in pool[0]

    def test_length_mismatch_is_refused_wholesale(self, configured, monkeypatch):
        # A short result would leave some hits unscored, and the searcher's
        # min-max would read a missing score as "least relevant" rather than
        # "unknown". Refuse the whole annotation instead.
        monkeypatch.setenv("MEMPALACE_RERANK_MODE", "cross")
        pool = [{"text": "a"}, {"text": "b"}]
        with mock.patch.object(cross_rerank, "score_pairs", return_value=[0.9]):
            assert annotate_rerank_scores("q", pool) is False
        assert not any("rerank_score" in h for h in pool)

    def test_default_mode_is_still_the_bi_encoder(self, configured, monkeypatch):
        monkeypatch.delenv("MEMPALACE_RERANK_MODE", raising=False)
        with (
            mock.patch.object(cross_rerank, "score_pairs") as sp,
            mock.patch("mempalace.rerank._embed", return_value=[[1.0, 0.0], [1.0, 0.0]]),
        ):
            assert annotate_rerank_scores("q", [{"text": "a"}]) is True
        sp.assert_not_called()

    def test_mode_is_case_and_whitespace_tolerant(self, configured, monkeypatch):
        monkeypatch.setenv("MEMPALACE_RERANK_MODE", "  CROSS  ")
        with mock.patch.object(cross_rerank, "score_pairs", return_value=[0.5]) as sp:
            annotate_rerank_scores("q", [{"text": "a"}])
        sp.assert_called_once()


def test_worker_pool_is_bounded_and_not_unbounded():
    # The value is a measured default for the current endpoint (16 beat 8 and
    # 24 on 32 real drawers at an 8192-token context) and will move if the
    # server is reloaded differently. What must NOT change is that there is a
    # ceiling at all: one request per candidate with no bound would open a
    # socket per hit and make a deep pool slower, not faster.
    assert 1 <= cross_rerank.MAX_WORKERS <= 32
    assert os.environ.get("MEMPALACE_RERANK_MODE") != "cross"


def test_workers_never_exceed_the_candidate_count(configured):
    # Otherwise a 2-hit pool spins up a 16-thread executor for nothing.
    seen = {}
    real = cross_rerank.futures.ThreadPoolExecutor

    def _spy(n, *a, **kw):
        seen["n"] = n
        return real(n, *a, **kw)

    with (
        mock.patch.object(cross_rerank.futures, "ThreadPoolExecutor", _spy),
        mock.patch.object(
            cross_rerank, "_post", return_value=json.loads(_payload(-1.0, -1.0).decode())
        ),
    ):
        cross_rerank.score_pairs("q", ["a", "b"])
    assert seen["n"] == 2


class TestContextShrink:
    """A too-small server context window must cost one candidate's fidelity,
    not the entire second stage."""

    @staticmethod
    def _ctx_error():
        return urllib.error.HTTPError(
            "http://x",
            400,
            "Bad Request",
            {},
            io.BytesIO(
                b'{"error":{"message":"Message too long: 654 tokens exceeds the '
                b'512-token context window.","code":"context_length_exceeded"}}'
            ),
        )

    def test_retries_with_a_shorter_document(self, configured):
        lengths = []

        def _post(query, document, url, model, key):
            lengths.append(len(document))
            if len(document) > 500:
                raise TestContextShrink._ctx_error()
            return json.loads(_payload(-0.1, -4.0).decode())

        with mock.patch.object(cross_rerank, "_post", side_effect=_post):
            (score,) = cross_rerank.score_pairs("q", ["x" * 5000])
        assert score > 0.9
        # Capped to MAX_TEXT_CHARS, then halved until it fits.
        assert lengths == [2000, 1000, 500]

    def test_gives_up_at_the_floor_rather_than_looping(self, configured):
        with mock.patch.object(
            cross_rerank,
            "_post",
            side_effect=lambda *a: (_ for _ in ()).throw(TestContextShrink._ctx_error()),
        ):
            assert cross_rerank.score_pairs("q", ["x" * 5000]) is None

    def test_a_non_context_400_is_not_retried(self, configured):
        calls = []

        def _post(*a):
            calls.append(1)
            raise urllib.error.HTTPError(
                "http://x", 400, "Bad Request", {}, io.BytesIO(b'{"error":{"message":"bad model"}}')
            )

        with mock.patch.object(cross_rerank, "_post", side_effect=_post):
            assert cross_rerank.score_pairs("q", ["x" * 5000]) is None
        assert len(calls) == 1, "a malformed request must fail fast, not shrink and retry"
