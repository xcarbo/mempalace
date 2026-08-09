"""Offline tests for NomicEmbedONNX.

The real ONNX model is ~550 MB and pulled from HuggingFace on first use, so
these tests mock huggingface_hub.hf_hub_download, tokenizers.Tokenizer, and
onnxruntime.InferenceSession to keep CI fast and network-free (same pattern
as tests/test_embeddinggemma.py / tests/test_bge_small.py).
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


def _make_fake_session(hidden_dim=768):
    """Fake InferenceSession returning last_hidden_state (batch, seq, 768).

    Token vectors are constant across positions so mean pooling is exact and
    padding-invariant in the fake; per-dim values vary so layer-norm and
    truncation act on real structure.
    """

    class _Named:
        def __init__(self, name):
            self.name = name

    class _Session:
        def __init__(self, *args, **kwargs):
            self.feeds_seen = []

        def get_inputs(self):
            return [_Named(n) for n in ("input_ids", "attention_mask", "token_type_ids")]

        def get_outputs(self):
            return [_Named("last_hidden_state")]

        def run(self, _output_names, feeds):
            self.feeds_seen.append(feeds)
            batch, seq = feeds["input_ids"].shape
            per_doc = (
                np.arange(batch * hidden_dim, dtype=np.float32).reshape(batch, hidden_dim) + 1.0
            )
            hidden = np.repeat(per_doc[:, None, :], seq, axis=1)
            return [hidden]

    return _Session


class _FakeTokenizer:
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
    """Patch the third-party deps imported inside NomicEmbedONNX._lazy_load."""
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

    import huggingface_hub
    import onnxruntime
    import tokenizers

    def fake_tokenizer_from_file(_path):
        calls["Tokenizer.from_file"] += 1
        return _FakeTokenizer()

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    monkeypatch.setattr(onnxruntime, "InferenceSession", fake_session_ctor)
    monkeypatch.setattr(tokenizers.Tokenizer, "from_file", staticmethod(fake_tokenizer_from_file))

    calls["sessions"] = sessions
    return calls


def test_name_carries_the_truncation_width():
    """384-d and 768-d truncations are different vector spaces; the persisted
    EF name must differ so ChromaDB forces a rebuild when dim changes."""
    assert embedding.NomicEmbedONNX().name() == "nomic_embed_text_v15_384d"
    assert embedding.NomicEmbedONNX(dim=768).name() == "nomic_embed_text_v15_768d"


def test_lazy_load_runs_once(patched_lazy_load):
    ef = embedding.NomicEmbedONNX()
    ef(["one"])
    ef(["two"])
    assert patched_lazy_load["hf_hub_download"] == 2  # model + tokenizer, once
    assert patched_lazy_load["InferenceSession"] == 1


def test_output_shape_is_truncated_to_384(patched_lazy_load):
    ef = embedding.NomicEmbedONNX()
    out = ef(["one", "two", "three"])
    assert np.asarray(out).shape == (3, 384)


def test_dim_768_yields_full_width(patched_lazy_load):
    ef = embedding.NomicEmbedONNX(dim=768)
    out = ef(["one"])
    assert np.asarray(out).shape == (1, 768)


def test_output_is_l2_normalized(patched_lazy_load):
    ef = embedding.NomicEmbedONNX()
    out = ef(["hello world", "another sentence"])
    norms = np.linalg.norm(np.asarray(out), axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), f"vectors not unit-norm: {norms}"


def test_layer_norm_applied_before_truncation(patched_lazy_load):
    """The MRL recipe layer-norms the pooled 768-vector, THEN truncates.

    The fake's pooled vector is arange-based (mean far from 0), so if
    truncation happened first the leading 384 dims would all be negative
    after a later layer-norm over 384 — while the correct order leaves the
    first 384 of the 768-wide layer-norm, i.e. all strictly below the 768-d
    mean: strictly negative AND strictly increasing. Check the signature the
    correct order produces."""
    ef = embedding.NomicEmbedONNX()
    out = np.asarray(ef(["doc"]))
    # arange row: after 768-wide layer-norm, first half is the negative,
    # strictly increasing half.
    assert (out[0] < 0).all()
    assert (np.diff(out[0]) > 0).all()


def test_documents_get_search_document_prefix(patched_lazy_load, monkeypatch):
    captured = []
    original = _FakeTokenizer.encode_batch

    def recording(self, texts):
        captured.extend(texts)
        return original(self, texts)

    monkeypatch.setattr(_FakeTokenizer, "encode_batch", recording)
    ef = embedding.NomicEmbedONNX()
    ef(["raw document text"])
    assert captured == ["search_document: raw document text"]


def test_queries_get_search_query_prefix(patched_lazy_load, monkeypatch):
    captured = []
    original = _FakeTokenizer.encode_batch

    def recording(self, texts):
        captured.extend(texts)
        return original(self, texts)

    monkeypatch.setattr(_FakeTokenizer, "encode_batch", recording)
    ef = embedding.NomicEmbedONNX()
    ef.embed_query(["find my notes"])
    assert captured == ["search_query: find my notes"]


def test_call_chunks_large_batches(patched_lazy_load, monkeypatch):
    """__call__ may never see more than _NOMIC_BATCH_SIZE docs per forward
    pass (#1770 — attention buffers grow with batch x len^2)."""
    batch_sizes = []
    original = _FakeTokenizer.encode_batch

    def recording(self, texts):
        batch_sizes.append(len(texts))
        return original(self, texts)

    monkeypatch.setattr(_FakeTokenizer, "encode_batch", recording)
    ef = embedding.NomicEmbedONNX()
    n = embedding._NOMIC_BATCH_SIZE * 2 + 3
    out = ef([f"doc {i}" for i in range(n)])
    assert batch_sizes == [embedding._NOMIC_BATCH_SIZE, embedding._NOMIC_BATCH_SIZE, 3]
    assert np.asarray(out).shape == (n, 384)


def test_batch_size_below_one_is_rejected():
    with pytest.raises(ValueError, match="batch_size"):
        embedding.NomicEmbedONNX(batch_size=0)


def test_dim_out_of_range_is_rejected():
    with pytest.raises(ValueError, match="dim"):
        embedding.NomicEmbedONNX(dim=0)
    with pytest.raises(ValueError, match="dim"):
        embedding.NomicEmbedONNX(dim=1024)


def test_call_empty_input_returns_empty(patched_lazy_load):
    """Zero docs must yield zero embeddings without loading the model."""
    ef = embedding.NomicEmbedONNX()
    assert ef([]) == []
    assert ef(None) == []
    assert ef.embed_query([]) == []
    assert patched_lazy_load["hf_hub_download"] == 0, "empty input must not trigger the download"


def test_call_bare_string_is_wrapped(patched_lazy_load):
    """A single string is one document, not a sequence of characters."""
    ef = embedding.NomicEmbedONNX()
    assert np.asarray(ef("standalone document")).shape == (1, 384)


def test_token_type_ids_fed_when_session_requires(patched_lazy_load):
    ef = embedding.NomicEmbedONNX()
    ef(["doc"])
    feeds = patched_lazy_load["sessions"][0].feeds_seen[0]
    assert "token_type_ids" in feeds
    assert (feeds["token_type_ids"] == 0).all()


def test_concurrent_first_calls_load_model_once(patched_lazy_load, monkeypatch):
    """Cold concurrent calls must build exactly one session."""
    import huggingface_hub

    fixture_download = huggingface_hub.hf_hub_download

    def slow_download(*args, **kwargs):
        time.sleep(0.05)  # widen the race window the lock must close
        return fixture_download(*args, **kwargs)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", slow_download)

    ef = embedding.NomicEmbedONNX()
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


def test_get_embedding_function_dispatches_to_nomic(monkeypatch):
    """model='nomic' must build NomicEmbedONNX at the 384-d default."""
    monkeypatch.setattr(
        embedding, "_resolve_providers", lambda device: (["CPUExecutionProvider"], "cpu")
    )
    ef = embedding.get_embedding_function(device="cpu", model="nomic")
    assert isinstance(ef, embedding.NomicEmbedONNX)
    assert ef.name() == "nomic_embed_text_v15_384d"


def test_get_embedding_function_threads_cap_passed_to_nomic(monkeypatch):
    captured = {}

    class DummyNomic:
        def __init__(self, preferred_providers=None, intra_op_num_threads=0):
            captured["threads"] = intra_op_num_threads

    monkeypatch.setattr(embedding, "NomicEmbedONNX", DummyNomic)
    monkeypatch.setattr(
        embedding, "_resolve_providers", lambda device: (["CPUExecutionProvider"], "cpu")
    )
    monkeypatch.setattr(embedding, "_resolve_intra_op_threads", lambda: 5)

    embedding.get_embedding_function("cpu", "nomic")

    assert captured["threads"] == 5


def test_missing_deps_raise_helpful_error(monkeypatch):
    """A broken install should say how to recover, not spill a bare ImportError."""
    monkeypatch.setitem(sys.modules, "tokenizers", None)

    ef = embedding.NomicEmbedONNX()
    with pytest.raises(ImportError, match=r"pip install.*mempalace"):
        ef(["anything"])
