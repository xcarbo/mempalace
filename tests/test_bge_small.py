"""Offline tests for BgeSmallONNX.

The real ONNX model is ~130 MB and pulled from HuggingFace on first use, so
these tests mock huggingface_hub.hf_hub_download, tokenizers.Tokenizer, and
onnxruntime.InferenceSession to keep CI fast and network-free (same pattern
as tests/test_embeddinggemma.py).
"""

import sys
import threading
import time

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("huggingface_hub")
pytest.importorskip("tokenizers")

import mempalace.embedding as embedding  # noqa: E402  (after importorskip)


@pytest.fixture(autouse=True)
def isolate_embedding_state(monkeypatch):
    monkeypatch.setattr(embedding, "_EF_CACHE", {})
    monkeypatch.setattr(embedding, "_WARNED", set())


def _make_fake_session(hidden_dim=384, with_token_type_ids=True):
    """Fake onnxruntime InferenceSession returning a last_hidden_state tensor.

    Shape: (batch, seq, hidden_dim). Values are deterministic and vary by
    position so CLS pooling (position 0) is distinguishable from mean pooling.
    """

    class _Named:
        def __init__(self, name):
            self.name = name

    class _Session:
        def __init__(self, *args, **kwargs):
            self.feeds_seen = []

        def get_inputs(self):
            names = ["input_ids", "attention_mask"]
            if with_token_type_ids:
                names.append("token_type_ids")
            return [_Named(n) for n in names]

        def get_outputs(self):
            return [_Named("last_hidden_state")]

        def run(self, _output_names, feeds):
            self.feeds_seen.append(feeds)
            batch, seq = feeds["input_ids"].shape
            hidden = np.zeros((batch, seq, hidden_dim), dtype=np.float32)
            # CLS row (position 0) gets a distinct non-trivial pattern.
            hidden[:, 0, :] = (
                np.arange(batch * hidden_dim, dtype=np.float32).reshape(batch, hidden_dim) + 1.0
            )
            # Other positions get garbage that would corrupt a mean pool.
            if seq > 1:
                hidden[:, 1:, :] = 999.0
            return [hidden]

    return _Session


class _FakeTokenizer:
    """Stand-in for tokenizers.Tokenizer with the methods _lazy_load uses."""

    def __init__(self):
        self._truncation_max = None

    def enable_padding(self):
        pass

    def enable_truncation(self, max_length):
        self._truncation_max = max_length

    def encode_batch(self, texts):
        class _Enc:
            def __init__(self, n):
                self.ids = [0] * n
                self.attention_mask = [1] * n

        max_len = max(len(t.split()) for t in texts)
        return [_Enc(max_len) for _ in texts]


@pytest.fixture
def patched_lazy_load(monkeypatch):
    """Patch the third-party deps imported inside BgeSmallONNX._lazy_load."""
    calls = {"hf_hub_download": 0, "InferenceSession": 0, "Tokenizer.from_file": 0}
    sessions = []

    def fake_download(repo, filename=None, subfolder=None, **kwargs):
        calls["hf_hub_download"] += 1
        return f"/tmp/fake/{subfolder or ''}/{filename}"

    fake_session_cls = _make_fake_session()

    def fake_session_ctor(*args, **kwargs):
        calls["InferenceSession"] += 1
        session = fake_session_cls()
        sessions.append(session)
        return session

    def fake_tokenizer_from_file(_path):
        calls["Tokenizer.from_file"] += 1
        return _FakeTokenizer()

    import huggingface_hub
    import onnxruntime
    import tokenizers

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    monkeypatch.setattr(onnxruntime, "InferenceSession", fake_session_ctor)
    monkeypatch.setattr(tokenizers.Tokenizer, "from_file", staticmethod(fake_tokenizer_from_file))

    calls["sessions"] = sessions
    return calls


def test_name_is_stable():
    """ChromaDB persists this on the collection — changing it breaks reads."""
    assert embedding.BgeSmallONNX.name() == "bge_small_en_v15"


def test_lazy_load_runs_once(patched_lazy_load):
    ef = embedding.BgeSmallONNX()
    ef(["one"])
    ef(["two"])
    assert patched_lazy_load["hf_hub_download"] == 2  # model + tokenizer, once
    assert patched_lazy_load["InferenceSession"] == 1
    assert patched_lazy_load["Tokenizer.from_file"] == 1


def test_output_shape_is_native_384(patched_lazy_load):
    ef = embedding.BgeSmallONNX()
    out = ef(["one", "two", "three"])
    assert np.asarray(out).shape == (3, 384)


def test_output_is_l2_normalized(patched_lazy_load):
    ef = embedding.BgeSmallONNX()
    out = ef(["hello world", "another sentence"])
    norms = np.linalg.norm(np.asarray(out), axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), f"vectors not unit-norm: {norms}"


def test_pooling_is_cls_not_mean(patched_lazy_load):
    """BGE pools the [CLS] token. The fake session poisons every non-CLS
    position with 999.0, so a mean pool would produce a near-constant vector
    while the CLS row varies by dimension."""
    ef = embedding.BgeSmallONNX()
    out = np.asarray(ef(["a few words here"]))
    # CLS row is arange-based: strictly increasing after normalization.
    assert (np.diff(out[0]) > 0).all(), "expected the CLS row's increasing pattern"


def test_documents_are_not_prefixed(patched_lazy_load, monkeypatch):
    captured = []
    original = _FakeTokenizer.encode_batch

    def recording(self, texts):
        captured.extend(texts)
        return original(self, texts)

    monkeypatch.setattr(_FakeTokenizer, "encode_batch", recording)
    ef = embedding.BgeSmallONNX()
    ef(["raw document text"])
    assert captured == ["raw document text"]


def test_queries_are_instruction_prefixed(patched_lazy_load, monkeypatch):
    captured = []
    original = _FakeTokenizer.encode_batch

    def recording(self, texts):
        captured.extend(texts)
        return original(self, texts)

    monkeypatch.setattr(_FakeTokenizer, "encode_batch", recording)
    ef = embedding.BgeSmallONNX()
    ef.embed_query(["find my notes"])
    assert captured == [embedding._BGE_QUERY_PREFIX + "find my notes"]


def test_token_type_ids_fed_when_session_requires(patched_lazy_load):
    ef = embedding.BgeSmallONNX()
    ef(["doc"])
    feeds = patched_lazy_load["sessions"][0].feeds_seen[0]
    assert "token_type_ids" in feeds
    assert (feeds["token_type_ids"] == 0).all()


def test_token_type_ids_omitted_when_session_lacks_input(monkeypatch):
    import huggingface_hub
    import onnxruntime
    import tokenizers

    session_cls = _make_fake_session(with_token_type_ids=False)
    sessions = []

    def ctor(*args, **kwargs):
        s = session_cls()
        sessions.append(s)
        return s

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: "/tmp/fake")
    monkeypatch.setattr(onnxruntime, "InferenceSession", ctor)
    monkeypatch.setattr(
        tokenizers.Tokenizer, "from_file", staticmethod(lambda _p: _FakeTokenizer())
    )

    ef = embedding.BgeSmallONNX()
    ef(["doc"])
    assert "token_type_ids" not in sessions[0].feeds_seen[0]


def test_call_chunks_large_batches(patched_lazy_load, monkeypatch):
    """__call__ may never see more than _BGE_BATCH_SIZE docs per forward pass
    (#1770 — attention buffers grow with batch x len^2)."""
    batch_sizes = []
    original = _FakeTokenizer.encode_batch

    def recording(self, texts):
        batch_sizes.append(len(texts))
        return original(self, texts)

    monkeypatch.setattr(_FakeTokenizer, "encode_batch", recording)
    ef = embedding.BgeSmallONNX()
    n = embedding._BGE_BATCH_SIZE * 2 + 6
    out = ef([f"doc {i}" for i in range(n)])
    assert batch_sizes == [embedding._BGE_BATCH_SIZE, embedding._BGE_BATCH_SIZE, 6]
    assert np.asarray(out).shape == (n, 384)


def test_custom_batch_size_is_honored(patched_lazy_load, monkeypatch):
    batch_sizes = []
    original = _FakeTokenizer.encode_batch

    def recording(self, texts):
        batch_sizes.append(len(texts))
        return original(self, texts)

    monkeypatch.setattr(_FakeTokenizer, "encode_batch", recording)
    ef = embedding.BgeSmallONNX(batch_size=10)
    out = ef([f"doc {i}" for i in range(24)])
    assert batch_sizes == [10, 10, 4]
    assert len(out) == 24


def test_batch_size_below_one_is_rejected():
    with pytest.raises(ValueError, match="batch_size"):
        embedding.BgeSmallONNX(batch_size=0)
    with pytest.raises(ValueError, match="batch_size"):
        embedding.BgeSmallONNX(batch_size=-3)


def test_call_empty_input_returns_empty(patched_lazy_load):
    """Zero docs must yield zero embeddings without loading the model."""
    ef = embedding.BgeSmallONNX()
    assert ef([]) == []
    assert ef(None) == []
    assert ef.embed_query([]) == []
    assert patched_lazy_load["hf_hub_download"] == 0, "empty input must not trigger the download"


def test_call_bare_string_is_wrapped(patched_lazy_load):
    """A single string is one document, not a sequence of characters."""
    ef = embedding.BgeSmallONNX()
    assert np.asarray(ef("standalone document")).shape == (1, 384)


def test_concurrent_first_calls_load_model_once(patched_lazy_load, monkeypatch):
    """Cold concurrent calls must build exactly one session."""
    import huggingface_hub

    fixture_download = huggingface_hub.hf_hub_download

    def slow_download(*args, **kwargs):
        time.sleep(0.05)  # widen the race window the lock must close
        return fixture_download(*args, **kwargs)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", slow_download)

    ef = embedding.BgeSmallONNX()
    barrier = threading.Barrier(2)
    results = [None, None]

    def worker(slot):
        barrier.wait(timeout=5)
        results[slot] = ef([f"doc {slot}"])

    threads = [threading.Thread(target=worker, args=(slot,)) for slot in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert patched_lazy_load["InferenceSession"] == 1
    assert all(r is not None and len(r) == 1 for r in results)


def test_get_embedding_function_dispatches_to_bge_small(monkeypatch):
    """model='bge-small' must build BgeSmallONNX, not the MiniLM EF."""
    monkeypatch.setattr(
        embedding, "_resolve_providers", lambda device: (["CPUExecutionProvider"], "cpu")
    )
    ef = embedding.get_embedding_function(device="cpu", model="bge-small")
    assert isinstance(ef, embedding.BgeSmallONNX)
    assert ef.name() == "bge_small_en_v15"


def test_get_embedding_function_threads_cap_passed_to_bge_small(monkeypatch):
    captured = {}

    class DummyBge:
        def __init__(self, preferred_providers=None, intra_op_num_threads=0):
            captured["threads"] = intra_op_num_threads

    monkeypatch.setattr(embedding, "BgeSmallONNX", DummyBge)
    monkeypatch.setattr(
        embedding, "_resolve_providers", lambda device: (["CPUExecutionProvider"], "cpu")
    )
    monkeypatch.setattr(embedding, "_resolve_intra_op_threads", lambda: 3)

    embedding.get_embedding_function("cpu", "bge-small")

    assert captured["threads"] == 3


def test_cache_key_separates_bge_from_minilm(monkeypatch):
    """Switching model must not return the cached EF for the other model."""

    class DummyMiniLM:
        def __init__(self, preferred_providers=None, intra_op_num_threads=0):
            self.kind = "minilm"

    monkeypatch.setattr(embedding, "_build_ef_class", lambda: DummyMiniLM)
    monkeypatch.setattr(
        embedding, "_resolve_providers", lambda device: (["CPUExecutionProvider"], "cpu")
    )

    ml = embedding.get_embedding_function(device="cpu", model="minilm")
    bge = embedding.get_embedding_function(device="cpu", model="bge-small")
    assert isinstance(bge, embedding.BgeSmallONNX)
    assert ml is not bge


def test_missing_deps_raise_helpful_error(monkeypatch):
    """A broken install should say how to recover, not spill a bare ImportError."""
    monkeypatch.setitem(sys.modules, "tokenizers", None)

    ef = embedding.BgeSmallONNX()
    with pytest.raises(ImportError, match=r"pip install.*mempalace"):
        ef(["anything"])
