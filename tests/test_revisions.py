"""drawer_revisions.jsonl: the text update_drawer replaces stays recoverable.

Ported from upstream #2767 and #2866, plus the paths that exist only here: the
``if_unchanged`` check-and-set, daemon batch steps and ``on_conflict: park``.
"""

import errno
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from mempalace import mcp_server, revisions, service
from mempalace.revisions import (
    append_drawer_revision,
    load_drawer_revisions,
    purge_drawer_revisions,
    revisions_path,
)

AUTH_DRAWER = "drawer_proj_backend_aaa"
AUTH_TEXT = (
    "The authentication module uses JWT tokens for session management. "
    "Tokens expire after 24 hours. Refresh tokens are stored in HttpOnly cookies."
)


@pytest.fixture
def revision_palace(monkeypatch, config, palace_path, kg, seeded_collection):
    monkeypatch.setattr(mcp_server, "_config", config)
    monkeypatch.setattr(mcp_server, "_get_kg", lambda *a, **kw: kg)
    return palace_path


def _contents(palace_path, drawer_id=None):
    return [row["content"] for row in load_drawer_revisions(palace_path, drawer_id)]


# ── the file ─────────────────────────────────────────────────────────────


def test_append_creates_a_missing_palace_dir_and_writes_lf(tmp_path):
    palace = tmp_path / "remote-palace"

    rid = append_drawer_revision(str(palace), "d1", "line one\r\nline two", {"wing": "w"})

    raw = revisions_path(str(palace)).read_bytes()
    assert raw.endswith(b"}\n") and b"}\r\n" not in raw
    [row] = load_drawer_revisions(str(palace), "d1")
    assert row["revision_id"] == rid
    assert row["content"] == "line one\r\nline two"
    assert row["metadata"] == {"wing": "w"}


def test_the_file_is_private_to_its_owner(tmp_path):
    append_drawer_revision(str(tmp_path), "d1", "secret")
    assert revisions_path(str(tmp_path)).stat().st_mode & 0o777 == 0o600


def test_load_skips_a_torn_line(tmp_path):
    append_drawer_revision(str(tmp_path), "d1", "first")
    with revisions_path(str(tmp_path)).open("ab") as handle:
        handle.write(b'{"revision_id": "torn", "drawer_')
    append_drawer_revision(str(tmp_path), "d1", "second")

    assert _contents(str(tmp_path), "d1") == ["first", "second"]


def test_purge_drops_matching_drawers_and_torn_lines(tmp_path):
    palace = str(tmp_path)
    append_drawer_revision(palace, "d1", "gone", {"source_file": "a.md"})
    with revisions_path(palace).open("ab") as handle:
        handle.write(b'{"revision_id": "torn", "drawer_')
    append_drawer_revision(palace, "d2", "kept", {"source_file": "b.md"})
    append_drawer_revision(palace, "d3", "gone by source", {"source_file": "c.md"})

    assert purge_drawer_revisions(palace, ["d1"], source_file="c.md") == 2

    raw = revisions_path(palace).read_bytes()
    assert b"torn" not in raw and b"\r" not in raw
    assert _contents(palace) == ["kept"]
    assert list(tmp_path.iterdir()) == [revisions_path(palace)]


def test_purge_without_a_match_leaves_the_file_alone(tmp_path):
    palace = str(tmp_path)
    append_drawer_revision(palace, "d1", "kept")
    before = revisions_path(palace).read_bytes()

    assert purge_drawer_revisions(palace, ["other"]) == 0
    assert revisions_path(palace).read_bytes() == before
    assert purge_drawer_revisions(str(tmp_path / "missing"), ["d1"]) == 0


def test_purge_removes_the_file_when_nothing_is_left(tmp_path):
    palace = str(tmp_path)
    append_drawer_revision(palace, "d1", "only")

    assert purge_drawer_revisions(palace, ["d1"]) == 1
    assert not revisions_path(palace).exists()


def test_an_append_during_a_purge_survives(tmp_path, monkeypatch):
    """An update that appends while a purge is rewriting must not be lost."""
    palace = str(tmp_path)
    append_drawer_revision(palace, "gone", "old")
    append_drawer_revision(palace, "kept", "kept")
    path = revisions_path(palace)
    real_read = Path.read_bytes
    racer = {}

    def read_then_race(self):
        data = real_read(self)
        if self == path and "thread" not in racer:
            racer["thread"] = threading.Thread(
                target=append_drawer_revision, args=(palace, "late", "appended mid-purge")
            )
            racer["thread"].start()
            racer["thread"].join(timeout=0.5)  # the purge holds the lock, so it waits
        return data

    monkeypatch.setattr(Path, "read_bytes", read_then_race)

    assert purge_drawer_revisions(palace, ["gone"]) == 1
    racer["thread"].join(timeout=15)
    assert not racer["thread"].is_alive()
    assert [r["drawer_id"] for r in load_drawer_revisions(palace)] == ["kept", "late"]


def test_appends_from_many_threads_are_all_kept(tmp_path):
    palace = str(tmp_path)
    threads = [
        threading.Thread(target=append_drawer_revision, args=(palace, f"d{i}", "x" * 200_000))
        for i in range(12)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert sorted(r["drawer_id"] for r in load_drawer_revisions(palace)) == sorted(
        f"d{i}" for i in range(12)
    )


# ── a failed write is never acknowledged ─────────────────────────────────


@pytest.mark.skipif(os.name != "posix", reason="requires a POSIX file-size limit")
def test_real_short_write_does_not_acknowledge_a_torn_revision(tmp_path):
    script = """
import json, resource, signal, sys
from mempalace.revisions import append_drawer_revision, revisions_path
import mempalace.palace
signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
_, hard = resource.getrlimit(resource.RLIMIT_FSIZE)
resource.setrlimit(resource.RLIMIT_FSIZE, (64, hard))
try:
    append_drawer_revision(sys.argv[1], "d1", "prior verbatim text " * 1000)
    result = {"acknowledged": True}
except OSError:
    result = {"acknowledged": False}
result["bytes_written"] = revisions_path(sys.argv[1]).stat().st_size
print(json.dumps(result))
"""
    child = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "palace")],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert json.loads(child.stdout.splitlines()[-1]) == {
        "acknowledged": False,
        "bytes_written": 64,
    }


@pytest.mark.parametrize("failure", ["zero", "after_prefix", "fsync"])
def test_failed_revision_keeps_old_drawer_and_can_be_retried(
    monkeypatch, revision_palace, seeded_collection, failure
):
    real_open, real_write, real_fsync = os.open, os.write, os.fsync
    revision_fd = None
    wrote_prefix = False

    def remember_open(path, flags, *args, **kwargs):
        nonlocal revision_fd
        fd = real_open(path, flags, *args, **kwargs)
        if str(path) == str(revisions_path(revision_palace)):
            revision_fd = fd
        return fd

    def fail_write(fd, data):
        nonlocal wrote_prefix
        if fd == revision_fd and failure != "fsync":
            if failure == "zero":
                return 0
            if not wrote_prefix:
                wrote_prefix = True
                return real_write(fd, data[:13])
            raise OSError(errno.ENOSPC, "injected full revision device")
        return real_write(fd, data)

    def fail_fsync(fd):
        if fd == revision_fd and failure == "fsync":
            raise OSError(errno.EIO, "injected revision fsync failure")
        return real_fsync(fd)

    with monkeypatch.context() as patch:
        patch.setattr(os, "open", remember_open)
        patch.setattr(os, "write", fail_write)
        patch.setattr(os, "fsync", fail_fsync)
        result = mcp_server.tool_update_drawer(AUTH_DRAWER, content="Replacement.")

    assert result["success"] is False
    assert "could not record prior drawer text" in result["error"]
    stored = seeded_collection.get(ids=[AUTH_DRAWER], include=["documents"])
    assert stored["documents"] == [AUTH_TEXT]

    retried = mcp_server.tool_update_drawer(AUTH_DRAWER, content="Replacement.")
    assert retried["success"] is True
    saved = load_drawer_revisions(revision_palace, AUTH_DRAWER)[-1]
    assert saved["content"] == AUTH_TEXT
    assert saved["revision_id"] == retried["revision_id"]


@pytest.mark.parametrize("write_size", [1, 7, 64])
def test_successive_short_writes_preserve_complete_unicode_revision(
    tmp_path, monkeypatch, write_size
):
    real_write = os.write
    monkeypatch.setattr(os, "write", lambda fd, data: real_write(fd, data[:write_size]))
    original = "café 🚀\r\nzero\x00 and stray \udcff"

    revision_id = append_drawer_revision(str(tmp_path), "d1", original)

    [saved] = load_drawer_revisions(str(tmp_path))
    assert saved["revision_id"] == revision_id
    assert saved["content"] == original


# ── update_drawer ────────────────────────────────────────────────────────


def test_update_keeps_the_verbatim_prior_text_and_its_metadata(revision_palace):
    result = mcp_server.tool_update_drawer(AUTH_DRAWER, content="replacement text")

    assert result["success"] is True and result["revision_id"]
    [row] = load_drawer_revisions(revision_palace, AUTH_DRAWER)
    assert row["revision_id"] == result["revision_id"]
    assert row["content"] == AUTH_TEXT
    assert row["metadata"]["wing"] == "project" and row["metadata"]["room"] == "backend"
    assert mcp_server.tool_get_drawer(AUTH_DRAWER)["content"] == "replacement text"


def test_a_long_drawer_keeps_all_of_its_chunks_as_one_revision(revision_palace):
    original = "\n\n".join(f"Paragraph {i}: " + "words of the roadmap " * 30 for i in range(40))
    added = mcp_server.tool_add_drawer("proj", "roadmap", original)
    assert added["chunks"] > 1
    digest = mcp_server.tool_get_drawer(added["drawer_id"])["content_sha256"]

    result = mcp_server.tool_update_drawer(
        added["drawer_id"], content="short now", if_unchanged=digest
    )

    assert result["success"] is True, result
    assert _contents(revision_palace, added["drawer_id"]) == [original]
    assert mcp_server.tool_get_drawer(added["drawer_id"])["content"] == "short now"


def test_a_chain_of_rewrites_keeps_every_earlier_version(revision_palace):
    for text in ("second", "third", "fourth"):
        assert mcp_server.tool_update_drawer(AUTH_DRAWER, content=text)["success"]
    assert _contents(revision_palace, AUTH_DRAWER) == [AUTH_TEXT, "second", "third"]


def test_a_failed_write_still_leaves_the_prior_text(
    revision_palace, monkeypatch, seeded_collection
):
    real_get = mcp_server._get_collection

    def failing_collection(create=False):
        col = real_get(create=create)

        def boom(**_kwargs):
            raise RuntimeError("write failed")

        col.update = boom
        col.upsert = boom
        return col

    monkeypatch.setattr(mcp_server, "_get_collection", failing_collection)

    result = mcp_server.tool_update_drawer(AUTH_DRAWER, content="must not stick")

    assert result["success"] is False
    stored = seeded_collection.get(ids=[AUTH_DRAWER], include=["documents"])
    assert stored["documents"] == [AUTH_TEXT]
    assert _contents(revision_palace, AUTH_DRAWER) == [AUTH_TEXT]


def test_a_metadata_only_update_of_a_short_drawer_saves_nothing(revision_palace):
    result = mcp_server.tool_update_drawer(AUTH_DRAWER, room="moved")

    assert result["success"] is True and "revision_id" not in result
    assert not revisions_path(revision_palace).exists()


def test_wal_points_at_the_revision_and_stays_redacted(revision_palace, monkeypatch):
    logged = []
    monkeypatch.setattr(mcp_server, "_wal_log", lambda op, entry: logged.append(entry))

    result = mcp_server.tool_update_drawer(AUTH_DRAWER, content="replacement")

    assert logged[0]["revision_id"] == result["revision_id"]
    assert AUTH_TEXT not in json.dumps(logged)


def test_update_runs_on_sqlite_exact(monkeypatch, tmp_path, kg):
    """Every fleet palace is sqlite_exact; the default test backend is chroma."""
    from mempalace.config import MempalaceConfig

    monkeypatch.setenv("MEMPALACE_BACKEND", "sqlite_exact")
    palace = tmp_path / "palace"
    palace.mkdir()
    monkeypatch.setenv("MEMPALACE_PALACE_PATH", str(palace))
    config = MempalaceConfig()
    assert config.palace_path == str(palace)
    monkeypatch.setattr(mcp_server, "_config", config)
    monkeypatch.setattr(mcp_server, "_get_kg", lambda *a, **kw: kg)
    monkeypatch.setattr(mcp_server, "_collection_cache", None, raising=False)
    monkeypatch.setattr(mcp_server, "_palace_db_inode", 0, raising=False)
    monkeypatch.setattr(mcp_server, "_palace_db_mtime", 0.0, raising=False)

    added = mcp_server.tool_add_drawer("proj", "roadmap", "the first plan")
    digest = mcp_server.tool_get_drawer(added["drawer_id"])["content_sha256"]
    result = mcp_server.tool_update_drawer(
        added["drawer_id"], content="the second plan", if_unchanged=digest
    )

    assert result["success"] is True, result
    assert (palace / "sqlite_exact.sqlite3").exists()
    assert _contents(str(palace), added["drawer_id"]) == ["the first plan"]


# ── check-and-set, batch and park ────────────────────────────────────────


def test_a_refused_check_and_set_writes_no_revision(revision_palace):
    result = mcp_server.tool_update_drawer(AUTH_DRAWER, content="stale edit", if_unchanged="0" * 64)

    assert result["conflict"] is True
    assert not revisions_path(revision_palace).exists()
    assert mcp_server.tool_get_drawer(AUTH_DRAWER)["content"] == AUTH_TEXT


def test_a_batch_update_keeps_the_prior_text(revision_palace):
    digest = mcp_server.tool_get_drawer(AUTH_DRAWER)["content_sha256"]

    result = service.run_batch(
        {
            "palace_path": revision_palace,
            "wing": "project",
            "steps": [
                {
                    "tool": "update-drawer",
                    "args": {
                        "drawer_id": AUTH_DRAWER,
                        "content": "written by a batch",
                        "if_unchanged": digest,
                    },
                    "on_conflict": "park",
                }
            ],
        }
    )

    assert result["success"] is True, result
    assert result["parked"] == []
    assert _contents(revision_palace, AUTH_DRAWER) == [AUTH_TEXT]


def test_a_parked_batch_update_writes_no_revision_and_keeps_the_proposal(revision_palace):
    proposed = "# Roadmap\n\nThe new plan, exactly as written."

    result = service.run_batch(
        {
            "palace_path": revision_palace,
            "wing": "project",
            "steps": [
                {
                    "tool": "update-drawer",
                    "args": {
                        "drawer_id": AUTH_DRAWER,
                        "content": proposed,
                        "if_unchanged": "0" * 64,
                    },
                    "on_conflict": "park",
                }
            ],
        }
    )

    assert result["success"] is True, result
    assert len(result["parked"]) == 1
    assert not revisions_path(revision_palace).exists()
    parked = mcp_server.tool_get_drawer(result["parked"][0]["parked_as"])
    assert parked["content"].endswith(proposed)


def test_a_batch_stops_when_the_prior_text_cannot_be_saved(revision_palace, monkeypatch):
    digest = mcp_server.tool_get_drawer(AUTH_DRAWER)["content_sha256"]

    def full_disk(*_args, **_kwargs):
        raise OSError(errno.ENOSPC, "no space")

    monkeypatch.setattr(revisions, "append_drawer_revision", full_disk)

    result = service.run_batch(
        {
            "palace_path": revision_palace,
            "wing": "project",
            "steps": [
                {
                    "tool": "update-drawer",
                    "args": {"drawer_id": AUTH_DRAWER, "content": "x", "if_unchanged": digest},
                    "on_conflict": "park",
                }
            ],
        }
    )

    assert result["success"] is False
    assert "could not record prior drawer text" in result["error"]
    assert mcp_server.tool_get_drawer(AUTH_DRAWER)["content"] == AUTH_TEXT


# ── delete ───────────────────────────────────────────────────────────────


def _revise(*drawer_ids):
    for drawer_id in drawer_ids:
        assert mcp_server.tool_update_drawer(drawer_id, content=f"new {drawer_id}")["success"]


def _revised_ids(palace_path):
    return {row["drawer_id"] for row in load_drawer_revisions(palace_path)}


def test_delete_removes_only_that_drawers_revisions(revision_palace):
    _revise(AUTH_DRAWER, "drawer_proj_backend_bbb")

    result = mcp_server.tool_delete_drawer(AUTH_DRAWER)

    assert result["success"] is True
    assert result["revisions_deleted"] == 1
    assert _revised_ids(revision_palace) == {"drawer_proj_backend_bbb"}


def test_delete_by_source_removes_that_sources_revisions(revision_palace):
    _revise(AUTH_DRAWER, "drawer_proj_backend_bbb")

    result = mcp_server.tool_delete_by_source("auth.py", dry_run=False)

    assert result["deleted"] == 1
    assert result["revisions_deleted"] == 1
    assert _revised_ids(revision_palace) == {"drawer_proj_backend_bbb"}


def test_a_failed_delete_keeps_the_revisions(revision_palace, monkeypatch):
    _revise(AUTH_DRAWER)
    real_get = mcp_server._get_collection

    def failing_collection(create=False):
        col = real_get(create=create)

        def boom(**_kwargs):
            raise RuntimeError("delete failed")

        col.delete = boom
        return col

    monkeypatch.setattr(mcp_server, "_get_collection", failing_collection)

    assert mcp_server.tool_delete_drawer(AUTH_DRAWER)["success"] is False
    assert mcp_server.tool_delete_by_source("auth.py", dry_run=False)["success"] is False
    assert _revised_ids(revision_palace) == {AUTH_DRAWER}


def test_a_failed_purge_keeps_the_delete_and_says_so(revision_palace, monkeypatch):
    _revise(AUTH_DRAWER)

    def locked(*_args, **_kwargs):
        raise PermissionError("revisions file is held open")

    monkeypatch.setattr(revisions, "purge_drawer_revisions", locked)

    result = mcp_server.tool_delete_drawer(AUTH_DRAWER)

    assert result["success"] is True
    assert result["revisions_deleted"] is None
    assert result["revisions_error"] == "revision purge failed: PermissionError"
    assert _revised_ids(revision_palace) == {AUTH_DRAWER}


# ── found in review ──────────────────────────────────────────────────────


def test_rewriting_the_same_text_saves_nothing(revision_palace):
    result = mcp_server.tool_update_drawer(AUTH_DRAWER, content=AUTH_TEXT)

    assert result["success"] is True and "revision_id" not in result
    assert not revisions_path(revision_palace).exists()


def test_a_no_change_update_of_a_chunked_drawer_saves_nothing(revision_palace):
    added = mcp_server.tool_add_drawer("proj", "roadmap", "words of the roadmap " * 200)
    assert added["chunks"] > 1

    assert mcp_server.tool_update_drawer(added["drawer_id"], wing="proj")["success"]
    assert not revisions_path(revision_palace).exists()

    assert mcp_server.tool_update_drawer(added["drawer_id"], room="moved")["revision_id"]


def test_a_purge_that_failed_can_be_retried_after_the_delete(revision_palace, monkeypatch):
    _revise(AUTH_DRAWER, "drawer_proj_backend_bbb")

    def locked(*_args, **_kwargs):
        raise PermissionError("revisions file is held open")

    with monkeypatch.context() as patch:
        patch.setattr(revisions, "purge_drawer_revisions", locked)
        assert mcp_server.tool_delete_drawer(AUTH_DRAWER)["revisions_deleted"] is None
        assert mcp_server.tool_delete_by_source("db.py", dry_run=False)["success"] is True

    again = mcp_server.tool_delete_drawer(AUTH_DRAWER)
    assert again["success"] is False and again["revisions_deleted"] == 1
    by_source = mcp_server.tool_delete_by_source("db.py", dry_run=False)
    assert by_source["deleted"] == 0 and by_source["revisions_deleted"] == 1
    assert not revisions_path(revision_palace).exists()


def test_the_palace_lock_comes_before_the_drawer_lock(revision_palace, monkeypatch):
    """Same order as a batch. The other order waits on the palace while
    holding the drawer, and a batch then blocks on the drawer with the palace
    held, so every other writer in the palace stalls behind the two."""
    import contextlib

    from mempalace import palace

    order = []
    real_palace, real_drawer = palace.short_writer_palace_lock, palace.mine_lock

    @contextlib.contextmanager
    def spy_palace(path):
        order.append("palace")
        with real_palace(path):
            yield

    @contextlib.contextmanager
    def spy_drawer(key):
        if key.startswith("update_drawer:"):
            order.append("drawer")
        with real_drawer(key):
            yield

    monkeypatch.setattr(palace, "short_writer_palace_lock", spy_palace)
    monkeypatch.setattr(palace, "mine_lock", spy_drawer)

    assert mcp_server.tool_update_drawer(AUTH_DRAWER, content="new")["success"]
    assert mcp_server.tool_delete_drawer(AUTH_DRAWER)["success"]
    assert order[:2] == ["palace", "drawer"]
    assert order.count("drawer") == 2


def test_a_busy_palace_refuses_the_update_with_nothing_saved(revision_palace, monkeypatch):
    from mempalace import palace

    def busy(_path):
        raise palace.MineAlreadyRunning("palace is held by PID 999")

    monkeypatch.setattr(palace, "short_writer_palace_lock", busy)

    result = mcp_server.tool_update_drawer(AUTH_DRAWER, content="new")

    assert result["success"] is False and "PID 999" in result["error"]
    assert not revisions_path(revision_palace).exists()


def test_a_delete_waits_for_an_update_of_the_same_drawer(revision_palace, monkeypatch):
    """Otherwise the update recreates the deleted drawer, or its revision
    lands after the purge and the deleted text stays on disk."""
    updating, finish_update = threading.Event(), threading.Event()
    real_wal = mcp_server._wal_log

    def pause_update(operation, entry):
        if operation == "update_drawer":
            updating.set()
            finish_update.wait(timeout=5)
        return real_wal(operation, entry)

    monkeypatch.setattr(mcp_server, "_wal_log", pause_update)
    update = threading.Thread(
        target=mcp_server.tool_update_drawer, args=(AUTH_DRAWER,), kwargs={"content": "new"}
    )
    delete = threading.Thread(target=mcp_server.tool_delete_drawer, args=(AUTH_DRAWER,))
    update.start()
    assert updating.wait(timeout=5)
    delete.start()
    delete.join(timeout=0.3)
    assert delete.is_alive(), "the delete ran while the update held the drawer"
    finish_update.set()
    update.join(timeout=10)
    delete.join(timeout=10)

    assert not revisions_path(revision_palace).exists()
    assert "error" in mcp_server.tool_get_drawer(AUTH_DRAWER)


def test_a_purge_and_a_first_append_fsync_the_directory(tmp_path, monkeypatch):
    """Without it a crash after the rename can bring the deleted text back."""
    palace = str(tmp_path)
    synced = []
    real_fsync = revisions._fsync_directory

    def spy(path):
        synced.append(str(path))
        return real_fsync(path)

    monkeypatch.setattr(revisions, "_fsync_directory", spy)

    append_drawer_revision(palace, "d1", "gone")
    assert synced == [palace]
    append_drawer_revision(palace, "d2", "kept")
    assert synced == [palace]
    purge_drawer_revisions(palace, ["d1"])
    assert synced == [palace, palace]
    purge_drawer_revisions(palace, ["d2"])
    assert synced == [palace, palace, palace]


def test_copy_drawer_revisions_keeps_what_the_target_has(tmp_path):
    source, dest, empty = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    append_drawer_revision(str(source), "d1", "old palace")

    assert revisions.copy_drawer_revisions(str(source), str(dest)) is True
    assert _contents(str(dest)) == ["old palace"]

    append_drawer_revision(str(dest), "d2", "new palace")
    assert revisions.copy_drawer_revisions(str(source), str(dest)) is False
    assert _contents(str(dest)) == ["old palace", "new palace"]
    assert revisions.copy_drawer_revisions(str(empty), str(dest)) is False
