"""Regression lock: a read handout must not write to the palace.

The bug (found 2026-07-31): ``ChromaBackend.get_collection`` called
``_pin_hnsw_threads`` on **every** handout, and that helper's
``collection.modify(configuration=...)`` is a write — an UPDATE against the
sysdb ``collections`` row. mempalace-api calls ``get_collection`` once per HTTP
request and then reads outside its handout lock, so the service wrote to the
palace on every request while other threads read from the same client.
chromadb's Rust core does not tolerate that. It panics:

    PanicException: ColumnDecode { index: "7", source: Utf8Error { .. } }
    PanicException: SqliteError { code: 522, message: "disk I/O error" }

522 is ``SQLITE_IOERR_SHORT_READ``; the ``Utf8Error`` variant is the same torn
read surfacing as a string column instead of a short one. The palace file was
never corrupt — only the reading process's view of it.

Two things made it lethal rather than merely noisy:

* ``pyo3_runtime.PanicException`` derives from ``BaseException``, so
  ``_pin_hnsw_threads``' own ``except Exception`` guard — the one that made it
  "best-effort" — was blind to the only failure it ever hit.
* Nothing invalidated the client afterwards, so a single panic wedged the
  service until it was restarted.

The fix has two halves. The retrofit runs once per ``PersistentClient`` rather
than once per handout, which is all its contract ever required; and it is
skipped entirely for a collection created with ``hnsw:num_threads=1``, which is
every palace mempalace has made. That second half is load-bearing, not an
optimisation: client rebuilds are driven by external writes, so a retrofit that
fires per rebuild still writes while readers are live. Gating it on the
effective in-memory config instead of the creation record was tried and brought
the panic back within 90 seconds on the live service.
"""

import threading

import pytest

from mempalace.backends import chroma as chroma_mod
from mempalace.backends.chroma import ChromaBackend, is_rust_panic


@pytest.fixture
def pin_spy(monkeypatch):
    """Count how many times the hnsw pin (a sysdb write) is issued."""
    calls = []
    real = chroma_mod._pin_hnsw_threads

    def spy(collection):
        calls.append(getattr(collection, "name", "?"))
        return real(collection)

    monkeypatch.setattr(chroma_mod, "_pin_hnsw_threads", spy)
    return calls


def test_pin_runs_once_per_client_not_once_per_handout(palace_path, pin_spy):
    """Steady state — an existing palace, which is all a service ever opens.

    A brand-new palace legitimately pins twice: chromadb creates
    ``chroma.sqlite3`` lazily during ``create_collection``, so the file
    appearing counts as a real disk change and rebuilds the client once. Build
    the palace first, then measure.
    """
    warmup = ChromaBackend()
    warmup.get_collection(palace_path, collection_name="pin_test", create=True)
    warmup.close()
    pin_spy.clear()

    backend = ChromaBackend()
    try:
        for _ in range(10):
            backend.get_collection(palace_path, collection_name="pin_test", create=True)
        assert len(pin_spy) == 1, (
            f"hnsw pin issued {len(pin_spy)} writes across 10 read handouts; "
            "it must run once per client"
        )
    finally:
        backend.close()


def test_pin_reruns_when_the_client_is_rebuilt(palace_path, pin_spy):
    """The chromadb 1.5.x in-memory hnsw config does not survive a reopen, so a
    genuinely new client must be pinned again — the gate is per client, not
    per process."""
    backend = ChromaBackend()
    try:
        backend.get_collection(palace_path, collection_name="pin_test", create=True)
        assert len(pin_spy) == 1

        # Model an external writer moving chroma.sqlite3 (a miner folding the
        # WAL back in), which is what drives _client() to rebuild.
        backend._freshness[palace_path] = (999999, 1.0)
        backend.get_collection(palace_path, collection_name="pin_test", create=True)
        assert len(pin_spy) == 2, "a rebuilt client must be re-pinned"
    finally:
        backend.close()


def test_client_rebuild_does_not_close_the_client_it_retires(palace_path, monkeypatch):
    """A rebuild must swap the cache entry WITHOUT closing the old client.

    The backend hands a collection out and the caller reads through it after
    the handout returns. ``close()`` releases the rust-side SQLite handles and
    unmaps the segments immediately, whoever still holds a Python reference, so
    closing here turns a concurrent read into a SIGBUS inside
    ``chromadb_rust_bindings``. Refcounting frees the client at the only moment
    that is safe: when nothing references it any more.
    """
    closed = []
    real_close = chroma_mod._close_client
    monkeypatch.setattr(
        chroma_mod,
        "_close_client",
        lambda c: (closed.append(c), real_close(c))[1],
    )

    backend = ChromaBackend()
    try:
        backend.get_collection(palace_path, collection_name="pin_test", create=True)
        first = backend._clients[palace_path]

        backend._freshness[palace_path] = (999999, 1.0)
        backend.get_collection(palace_path, collection_name="pin_test", create=True)

        assert first not in closed, "a rebuild closed a client a caller may still be reading"
        assert backend._clients[palace_path] is not first
    finally:
        backend.close()


def test_concurrent_handouts_and_reads_do_not_panic(palace_path, pin_spy):
    """The shape mempalace-api actually runs: the handout is serialised, the
    read is not. Pre-fix this panicked inside chromadb's Rust core within a
    few dozen reads.

    The crash itself is a poor assertion — whether chromadb panics depends on
    timing, and running another test in this file first was enough to mask it,
    so this test passed on the broken code in a whole-file run. It therefore
    asserts the *behaviour* that causes the crash as well: zero writes issued
    during the concurrent phase. That fails deterministically on reverted code
    regardless of ordering.
    """
    backend = ChromaBackend()
    collection = backend.get_collection(palace_path, collection_name="pin_test", create=True)
    ids = [f"drawer_{i:04d}" for i in range(20)]
    collection.add(
        ids=ids,
        documents=[f"document {i} " + ("filler " * 30) for i in range(20)],
        metadatas=[{"wing": "w", "room": "r"} for _ in ids],
    )

    handout_lock = threading.Lock()
    stop = threading.Event()
    failures = []
    reads = []

    def worker(seed):
        try:
            for i in range(60):
                if stop.is_set():
                    return
                with handout_lock:  # mirrors mempalace_api._collection_lock
                    col = backend.get_collection(
                        palace_path, collection_name="pin_test", create=True
                    )
                col.get(ids=[ids[(seed + i) % len(ids)]])  # read outside the lock
                reads.append(1)
        except BaseException as exc:  # PanicException is not an Exception
            failures.append(f"{'panic' if is_rust_panic(exc) else type(exc).__name__}: {exc}")
            stop.set()

    pin_spy.clear()  # count only what the concurrent phase issues
    threads = [threading.Thread(target=worker, args=(s,)) for s in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)

    try:
        assert not failures, failures[0]
        assert len(reads) == 360
        # The order-independent half: 360 concurrent handouts, zero writes.
        assert pin_spy == [], (
            f"{len(pin_spy)} palace writes issued across 360 concurrent read "
            "handouts; a read handout must never write"
        )
    finally:
        backend.close()


@pytest.mark.parametrize(
    "metadata,needs_retrofit",
    [
        # Created with the safe value — no retrofit, so no write on the read path.
        ({"hnsw:num_threads": 1, "hnsw:space": "cosine"}, False),
        # Legacy palace: predates the change, still needs pinning.
        ({"hnsw:space": "cosine"}, True),
        ({"hnsw:num_threads": 4}, True),
        ({}, True),
        (None, True),
        ("not-a-dict", True),
    ],
)
def test_retrofit_gate_reads_the_creation_record(metadata, needs_retrofit):
    """The gate asks how the collection was CREATED, not what a client holds.

    Reading the effective in-memory config instead makes this return True on
    every freshly opened client, which fires a write on every client rebuild
    and reintroduces the panic. See _hnsw_threads_needs_retrofit.
    """

    class FakeCollection:
        metadata = None

    fake = FakeCollection()
    fake.metadata = metadata
    assert chroma_mod._hnsw_threads_needs_retrofit(fake) is needs_retrofit


def test_legacy_collection_still_gets_retrofitted():
    """A palace created before num_threads=1 must still be pinned."""
    modified = []

    class LegacyCollection:
        metadata = {"hnsw:space": "cosine"}

        def modify(self, *a, **kw):
            modified.append((a, kw))

    chroma_mod._pin_hnsw_threads(LegacyCollection())
    assert len(modified) == 1, "legacy collection was not retrofitted"


def test_modern_collection_is_never_written_to(palace_path):
    """End-to-end: a collection mempalace created takes no write, ever."""
    backend = ChromaBackend()
    try:
        wrapper = backend.get_collection(palace_path, collection_name="pin_test", create=True)
        inner = getattr(wrapper, "_collection", wrapper)
        assert inner.metadata.get("hnsw:num_threads") == 1
        assert chroma_mod._hnsw_threads_needs_retrofit(inner) is False

        modified = []
        inner.modify = lambda *a, **kw: modified.append((a, kw))
        chroma_mod._pin_hnsw_threads(inner)
        assert modified == [], "issued a write against a collection that never needed one"
    finally:
        backend.close()


def test_is_rust_panic_matches_by_name_without_importing_pyo3():
    """``pyo3_runtime`` is synthesised by pyo3 and only appears in sys.modules
    once a panic has been raised, so it cannot be imported up front."""

    class PanicException(BaseException):
        pass

    assert is_rust_panic(PanicException("boom"))
    assert not is_rust_panic(ValueError("boom"))
    assert not is_rust_panic(KeyboardInterrupt())
