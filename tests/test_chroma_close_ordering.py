"""Regression lock: the WAL checkpoint must happen BEFORE the client is released.

The bug (found 2026-07-29): ``close()`` stopped every cached chromadb client and
only then ran ``checkpoint_wal`` on each path. Once chromadb's Rust core has
dropped a palace, a statement issued by Python's ``sqlite3`` against that file
can land on a mapping the core left behind. If the file was unlinked and
recreated in between — a rebuild, a repair, a test's ``rmtree`` — that access is
a **SIGBUS**.

Two things make it nasty:

* ``checkpoint_wal``'s ``except sqlite3.Error`` cannot catch a signal. The
  interpreter dies mid-close with no traceback and no failed test.
* Whether it fires depends on the SQLite build — 3.51.0 survives, 3.50.4 and
  3.53.3 die — so it hid behind whichever interpreter happened to be installed.

These tests assert the ordering directly rather than trying to provoke a crash:
a test that reproduces the bug would take the whole suite down with it, and
would only do so on some SQLite builds.
"""

import sqlite3

import pytest

from mempalace.backends import chroma as chroma_mod
from mempalace.backends.base import PalaceRef
from mempalace.backends.chroma import ChromaBackend


@pytest.fixture
def order_log(monkeypatch):
    """Record the order of checkpoint / client-release calls."""
    events = []

    real_checkpoint = chroma_mod.checkpoint_wal
    real_close = chroma_mod._close_client

    def spy_checkpoint(path):
        events.append(("checkpoint", str(path)))
        return real_checkpoint(path)

    def spy_close(client):
        events.append(("release", "client"))
        return real_close(client)

    monkeypatch.setattr(chroma_mod, "checkpoint_wal", spy_checkpoint)
    monkeypatch.setattr(chroma_mod, "_close_client", spy_close)
    return events


def _seed(backend, path):
    ref = PalaceRef(id=str(path), local_path=str(path))
    backend.get_collection(palace=ref, collection_name="mempalace_drawers", create=True).upsert(
        documents=["x"], ids=["x"], metadatas=[{"k": "v"}]
    )
    return ref


def test_close_checkpoints_before_releasing_any_client(tmp_path, order_log):
    """Every checkpoint must precede every client release.

    Reversing these two loops is the bug. It looks harmless in review — the
    same work, the other way round — and costs nothing until a palace path is
    recreated, at which point the process dies without a traceback.
    """
    backend = ChromaBackend()
    _seed(backend, tmp_path / "palace-a")
    _seed(backend, tmp_path / "palace-b")
    order_log.clear()

    backend.close()

    kinds = [kind for kind, _ in order_log]
    assert "checkpoint" in kinds, "close() must checkpoint the WAL"
    assert "release" in kinds, "close() must release its clients"
    assert kinds.index("release") > max(i for i, k in enumerate(kinds) if k == "checkpoint"), (
        f"every checkpoint must precede every client release, got {kinds}"
    )


def test_close_palace_checkpoints_before_releasing_its_client(tmp_path, order_log):
    """Same ordering contract on the single-palace path."""
    backend = ChromaBackend()
    ref = _seed(backend, tmp_path / "palace-a")
    order_log.clear()

    backend.close_palace(ref)

    kinds = [kind for kind, _ in order_log]
    assert kinds[0] == "checkpoint", f"checkpoint must come first, got {kinds}"


def test_close_still_checkpoints_every_cached_palace(tmp_path, order_log):
    """Reordering must not drop a palace — all cached paths still get folded."""
    backend = ChromaBackend()
    _seed(backend, tmp_path / "palace-a")
    _seed(backend, tmp_path / "palace-b")
    order_log.clear()

    backend.close()

    checkpointed = {path for kind, path in order_log if kind == "checkpoint"}
    assert len(checkpointed) == 2, f"both palaces must be checkpointed, got {checkpointed}"


def test_wal_is_actually_truncated_with_the_client_still_attached(tmp_path):
    """The reorder is only safe if it still does the job.

    A checkpoint issued while the writer is attached could plausibly return
    SQLITE_BUSY and leave the -wal in place. Measured: it does not.
    """
    palace = tmp_path / "palace"
    backend = ChromaBackend()
    ref = PalaceRef(id=str(palace), local_path=str(palace))
    col = backend.get_collection(palace=ref, collection_name="mempalace_drawers", create=True)
    for i in range(200):
        col.upsert(documents=[f"doc {i}" * 40], ids=[f"id{i}"], metadatas=[{"k": "v"}])

    wal = palace / "chroma.sqlite3-wal"
    assert wal.exists() and wal.stat().st_size > 0, "expected a populated -wal to fold"

    chroma_mod.checkpoint_wal(str(palace))

    # "Folded" has two legal shapes and they differ by SQLite version: 3.51.0
    # truncates the -wal to 0 bytes and leaves it in place, 3.53.3 removes the
    # file outright. Asserting size == 0 passed on one and raised
    # FileNotFoundError on the other — a version difference dressed up as a
    # product bug. Accept either; what matters is that no WAL content survives.
    folded = not wal.exists() or wal.stat().st_size == 0
    assert folded, (
        f"checkpoint must fold the WAL even while attached; "
        f"-wal still holds {wal.stat().st_size:,} bytes"
    )

    backend.close()


def test_reopening_a_recreated_palace_path_survives(tmp_path):
    """The scenario the ordering bug died on, as a behaviour test.

    Cache two palaces, close, delete each directory, reopen a fresh backend at
    the same path. On a bad SQLite build the old order SIGBUSes here.
    """
    import shutil

    backend = ChromaBackend()
    for name in ("palace-a", "palace-b"):
        _seed(backend, tmp_path / name)
    backend.close()

    for name in ("palace-a", "palace-b"):
        path = tmp_path / name
        shutil.rmtree(path)
        fresh = ChromaBackend()
        ref = PalaceRef(id=str(path), local_path=str(path))
        col = fresh.get_collection(palace=ref, collection_name="mempalace_drawers", create=True)
        col.upsert(documents=["y"], ids=["y"], metadatas=[{"k": "v2"}])
        assert col.count() == 1
        fresh.close()


def test_checkpoint_wal_is_a_noop_on_a_missing_database(tmp_path):
    """Unconditional call site — a path with no database must not raise."""
    chroma_mod.checkpoint_wal(str(tmp_path / "does-not-exist"))


def test_checkpoint_wal_swallows_sqlite_errors(tmp_path, monkeypatch):
    """It is best-effort by contract; a locked or corrupt file must not
    propagate out of close()."""
    palace = tmp_path / "palace"
    palace.mkdir()
    (palace / "chroma.sqlite3").write_bytes(b"not a database")

    def boom(*a, **kw):
        raise sqlite3.DatabaseError("file is not a database")

    monkeypatch.setattr(sqlite3, "connect", boom)
    chroma_mod.checkpoint_wal(str(palace))  # must not raise
