"""WAL journal-mode enablement for ChromaDB's backing SQLite database.

Motivating incident: a long-running ``mempalace mine`` held the exclusive
SQLite write lock for ~5 hours in the default rollback (``delete``) journal
mode, blocking every ``memp`` read across all projects. In WAL mode a single
writer and readers proceed concurrently, so the outage could not happen.

These tests pin two guarantees:

1. A freshly-created palace comes up in ``journal_mode=wal``.
2. An existing ``delete``-mode palace is migrated to WAL on next open.

plus the concurrency property that motivates the change: with WAL a reader is
not blocked while a writer holds an OPEN exclusive write transaction (and, as a
control, that ``delete`` mode *does* block under the same conditions).
"""

import os
import sqlite3
from contextlib import closing

import chromadb

from mempalace.backends.chroma import ChromaBackend, enable_wal_journal


def _journal_mode(db_path: str) -> str:
    with closing(sqlite3.connect(db_path)) as conn:
        return conn.execute("PRAGMA journal_mode").fetchone()[0].lower()


def _db(palace_path) -> str:
    return os.path.join(str(palace_path), "chroma.sqlite3")


def test_fresh_palace_comes_up_in_wal(tmp_path):
    """A palace created through ChromaBackend reports journal_mode == wal."""
    backend = ChromaBackend()
    palace = tmp_path / "palace"
    col = backend.get_collection(str(palace), "mempalace_drawers", create=True)
    col.upsert(documents=["hello world"], ids=["a"], metadatas=[{"wing": "w"}])
    backend.close()

    assert _journal_mode(_db(palace)) == "wal"


def test_existing_delete_palace_migrated_to_wal_on_open(tmp_path):
    """A pre-existing rollback-journal palace is flipped to WAL when reopened."""
    palace = tmp_path / "palace"
    palace.mkdir()

    # Build the palace the way chromadb 1.5.x does natively: delete-journal.
    client = chromadb.PersistentClient(path=str(palace))
    client.get_or_create_collection("mempalace_drawers", metadata={"hnsw:space": "cosine"})
    del client
    assert _journal_mode(_db(palace)) == "delete", "precondition: starts in rollback mode"

    # Opening through the backend must migrate it in place, without data loss.
    backend = ChromaBackend()
    col = backend.get_collection(str(palace), "mempalace_drawers", create=False)
    col.upsert(documents=["after migration"], ids=["m1"], metadatas=[{"wing": "w"}])
    backend.close()

    assert _journal_mode(_db(palace)) == "wal"


def test_enable_wal_journal_is_idempotent(tmp_path):
    """Re-running the enabler on an already-WAL palace is a harmless no-op."""
    palace = tmp_path / "palace"
    palace.mkdir()
    client = chromadb.PersistentClient(path=str(palace))
    client.get_or_create_collection("mempalace_drawers", metadata={"hnsw:space": "cosine"})
    del client

    enable_wal_journal(str(palace))
    assert _journal_mode(_db(palace)) == "wal"
    # Second call: still WAL, no error.
    enable_wal_journal(str(palace))
    assert _journal_mode(_db(palace)) == "wal"


def test_enable_wal_journal_missing_db_is_noop(tmp_path):
    """No chroma.sqlite3 yet (brand-new palace dir) -> silent no-op, no crash."""
    palace = tmp_path / "palace"
    palace.mkdir()
    # Must not raise even though there is no database file.
    enable_wal_journal(str(palace))
    assert not os.path.exists(_db(palace))


def test_mode_ro_read_works_after_backend_close_on_wal_palace(tmp_path):
    """Closing the backend must checkpoint the WAL so read-only (mode=ro)
    consumers can reopen the palace.

    Without a truncating checkpoint on close, a WAL palace leaves an orphaned
    ``-wal`` with no live connection, and ``sqlite3.connect(?mode=ro)`` fails
    with "unable to open database file" — which would break every mode=ro
    reader (repair preflight, status, list fast-paths, extract_via_sqlite).
    """
    palace = tmp_path / "palace"
    backend = ChromaBackend()
    col = backend.get_collection(str(palace), "mempalace_drawers", create=True)
    col.upsert(documents=["hello world"], ids=["a"], metadatas=[{"wing": "w"}])
    backend.close()

    db = _db(palace)
    # -wal, if present, must be folded back into the main db (empty).
    if os.path.exists(db + "-wal"):
        assert os.path.getsize(db + "-wal") == 0
    with closing(sqlite3.connect(f"file:{db}?mode=ro", uri=True)) as conn:
        assert conn.execute("PRAGMA quick_check").fetchone()[0].lower() == "ok"


def test_status_reports_journal_mode_after_migration(tmp_path):
    """repair.status() surfaces the palace journal mode, which reads WAL once a
    palace has been opened through the backend."""
    from mempalace import repair

    palace = tmp_path / "palace"
    backend = ChromaBackend()
    col = backend.get_collection(str(palace), "mempalace_drawers", create=True)
    col.upsert(documents=["hello"], ids=["a"], metadatas=[{"wing": "w"}])
    backend.close()

    assert repair.sqlite_journal_mode(str(palace)) == "wal"
    result = repair.status(palace_path=str(palace))
    assert result.get("journal_mode") == "wal"


def _seed(mode: str, db_path: str) -> None:
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute(f"PRAGMA journal_mode={mode}")
        conn.execute("CREATE TABLE t(x INTEGER)")
        conn.execute("INSERT INTO t VALUES (1)")
        conn.commit()


def test_wal_reader_not_blocked_by_open_exclusive_writer(tmp_path):
    """The concurrency win: under WAL a reader reads while a writer holds an
    OPEN exclusive write transaction — it does not block."""
    db_path = str(tmp_path / "wal.sqlite3")
    _seed("wal", db_path)

    writer = sqlite3.connect(db_path, timeout=0.1)
    writer.execute("BEGIN EXCLUSIVE")
    writer.execute("INSERT INTO t VALUES (2)")  # write lock held, uncommitted
    try:
        reader = sqlite3.connect(db_path, timeout=0.5)
        # Bounded busy_timeout: if WAL blocked us this would raise "database
        # is locked" after 0.5s instead of returning promptly.
        with closing(reader):
            count = reader.execute("SELECT COUNT(*) FROM t").fetchone()[0]
        assert count == 1  # sees the last committed snapshot, not blocked
    finally:
        writer.rollback()
        writer.close()


def test_delete_mode_reader_blocked_by_open_exclusive_writer(tmp_path):
    """Control: rollback (delete) journal mode DOES block a reader under an
    open exclusive writer — the exact failure the WAL switch prevents."""
    db_path = str(tmp_path / "delete.sqlite3")
    _seed("delete", db_path)

    writer = sqlite3.connect(db_path, timeout=0.1)
    writer.execute("BEGIN EXCLUSIVE")
    writer.execute("INSERT INTO t VALUES (2)")
    try:
        reader = sqlite3.connect(db_path, timeout=0.5)
        with closing(reader):
            blocked = False
            try:
                reader.execute("SELECT COUNT(*) FROM t").fetchone()
            except sqlite3.OperationalError as exc:
                blocked = "locked" in str(exc).lower()
        assert blocked, "expected rollback-mode reader to be blocked by exclusive writer"
    finally:
        writer.rollback()
        writer.close()
