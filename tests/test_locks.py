"""Tests for mempalace/locks.py — holder records, contention diagnostics,
residue GC, and the `memp locks` CLI subcommand.

Hardening from the 2026-07-10→12 write-path outage: an orphaned process held
the palace mine lock idle for ~4h with only a terse "held by PID N" message,
and ~2,200 residual 0-byte lock files accumulated in ~/.mempalace/locks/.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import threading

import pytest

from mempalace import locks as locks_mod
from mempalace.locks import (
    ORPHAN_GUIDANCE,
    gc_stale_locks,
    maybe_gc_stale_locks,
    parse_holder_record,
    write_holder_record,
)
from mempalace.palace import MineAlreadyRunning, mine_palace_lock


def _set_home(monkeypatch, tmp_path) -> str:
    """Redirect HOME/USERPROFILE so ~/.mempalace/locks lands under tmp_path."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    lock_dir = os.path.join(str(tmp_path), ".mempalace", "locks")
    os.makedirs(lock_dir, exist_ok=True)
    return lock_dir


def _reset_gc_throttle(monkeypatch) -> None:
    """The acquire-time GC runs once per process; reset it per test."""
    monkeypatch.setattr(locks_mod, "_gc_done_for_pid", None)


def _write_record_file(path: str, record: dict) -> None:
    """Create a lock file with a sentinel byte + JSON holder record."""
    with open(path, "wb") as fh:
        fh.write(b"\0" + json.dumps(record).encode("utf-8"))


def _dead_pid() -> int:
    """Spawn a trivial child and reap it — its PID is provably dead."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


@contextlib.contextmanager
def _hold_palace_lock(palace: str):
    """Hold ``mine_palace_lock`` from a helper thread for the block's duration.

    flock via an independent open file description conflicts even within the
    same process, so the main thread contending on the same palace observes a
    genuine held lock (the re-entrant pass-through is per-thread). Called
    AFTER the test redirects HOME / patches argv, so the holder record is
    written where and how the test expects.
    """
    acquired = threading.Event()
    release = threading.Event()
    errors: list = []

    def _hold():
        try:
            with mine_palace_lock(palace):
                acquired.set()
                release.wait(timeout=30)
        except Exception as exc:  # pragma: no cover - surfaced via assert below
            errors.append(exc)
            acquired.set()

    thread = threading.Thread(target=_hold)
    thread.start()
    try:
        assert acquired.wait(timeout=10), "holder thread failed to acquire in time"
        assert not errors, f"holder thread failed: {errors}"
        yield
    finally:
        release.set()
        thread.join(timeout=10)


# ---------------------------------------------------------------------------
# 1. Rich contention diagnostics
# ---------------------------------------------------------------------------


def test_contention_error_includes_pid_alive_age_argv(tmp_path, monkeypatch):
    _set_home(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "argv", ["memp", "mine", "/some/dir"])
    palace = str(tmp_path / "palace")

    with _hold_palace_lock(palace):
        with pytest.raises(MineAlreadyRunning) as excinfo:
            with mine_palace_lock(palace):
                pytest.fail("second acquire of a held palace lock must raise")

    msg = str(excinfo.value)
    assert f"PID {os.getpid()}" in msg
    assert "alive" in msg
    assert "memp mine /some/dir" in msg  # recorded argv
    assert "parent PID" in msg
    assert "lock held for" in msg
    assert "acquired 20" in msg  # ISO acquire timestamp
    assert msg.endswith(ORPHAN_GUIDANCE), f"message must end with orphan guidance; got: {msg!r}"


def test_contention_error_reports_dead_recorded_holder(tmp_path, monkeypatch):
    """A stale record naming a dead PID must be called out, not shown as live."""
    _set_home(monkeypatch, tmp_path)
    dead = _dead_pid()
    # palace.py binds the name at import time — patch it where it's used.
    import mempalace.palace as palace_module

    monkeypatch.setattr(
        palace_module,
        "read_holder_record",
        lambda _lf: {"pid": dead, "argv": ["memp", "mine"], "ppid": 1},
    )
    palace = str(tmp_path / "palace")

    with _hold_palace_lock(palace):
        with pytest.raises(MineAlreadyRunning) as excinfo:
            with mine_palace_lock(palace):
                pytest.fail("second acquire of a held palace lock must raise")

    msg = str(excinfo.value)
    assert f"PID {dead}" in msg
    assert "NOT alive" in msg


def test_old_format_lock_body_falls_back_to_mtime_age(tmp_path):
    """Legacy 'PID argv' text bodies still parse; age comes from file mtime."""
    record = parse_holder_record("12345 mempalace mine ~/code")
    assert record == {"pid": 12345, "argv": ["mempalace", "mine", "~/code"]}

    lock_path = tmp_path / "legacy.lock"
    lock_path.write_bytes(b"\x0012345 mempalace mine")
    old = 3_700  # ~1h 1m ago
    os.utime(lock_path, (os.stat(lock_path).st_atime, os.stat(lock_path).st_mtime - old))
    duration = locks_mod.held_duration_seconds(str(lock_path), record)
    assert duration is not None and duration >= old - 5

    described = locks_mod.describe_holder(str(lock_path), record)
    assert "PID 12345" in described
    assert "lock held for 1h" in described


# ---------------------------------------------------------------------------
# 2. Residue GC
# ---------------------------------------------------------------------------


def test_gc_removes_zero_byte_unheld_residue(tmp_path, monkeypatch):
    lock_dir = _set_home(monkeypatch, tmp_path)
    residue = os.path.join(lock_dir, "0123456789abcdef.lock")
    open(residue, "wb").close()

    assert gc_stale_locks() == 1
    assert not os.path.exists(residue)


def test_gc_removes_dead_pid_residue(tmp_path, monkeypatch):
    lock_dir = _set_home(monkeypatch, tmp_path)
    residue = os.path.join(lock_dir, "deadbeefdeadbeef.lock")
    _write_record_file(
        residue,
        {"pid": _dead_pid(), "argv": ["memp", "mine"], "acquired_at": "2026-07-10T18:30:00+00:00"},
    )

    assert gc_stale_locks() == 1
    assert not os.path.exists(residue)


def test_gc_keeps_unheld_lock_with_live_recorded_pid(tmp_path, monkeypatch):
    lock_dir = _set_home(monkeypatch, tmp_path)
    keeper = os.path.join(lock_dir, "livepid.lock")
    _write_record_file(keeper, {"pid": os.getpid(), "argv": ["pytest"]})

    assert gc_stale_locks() == 0
    assert os.path.exists(keeper)


@pytest.mark.skipif(os.name == "nt", reason="fcntl.flock is POSIX-only")
def test_gc_never_removes_live_held_lock(tmp_path, monkeypatch):
    """A held lock must survive GC even when it is a 0-byte file.

    A same-process flock on an independent fd conflicts with the GC's probe,
    so this simulates a live holder without spawning a child process.
    """
    import fcntl

    lock_dir = _set_home(monkeypatch, tmp_path)
    held_path = os.path.join(lock_dir, "held-zero-byte.lock")
    fh = open(held_path, "wb")
    fcntl.flock(fh, fcntl.LOCK_EX)
    try:
        assert gc_stale_locks() == 0
        assert os.path.exists(held_path)
    finally:
        fcntl.flock(fh, fcntl.LOCK_UN)
        fh.close()


def test_gc_ignores_non_lock_files(tmp_path, monkeypatch):
    lock_dir = _set_home(monkeypatch, tmp_path)
    stray = os.path.join(lock_dir, "README.txt")
    with open(stray, "w") as fh:
        fh.write("not a lock")

    assert gc_stale_locks() == 0
    assert os.path.exists(stray)


def test_maybe_gc_runs_once_per_process(tmp_path, monkeypatch):
    lock_dir = _set_home(monkeypatch, tmp_path)
    _reset_gc_throttle(monkeypatch)
    first = os.path.join(lock_dir, "first.lock")
    open(first, "wb").close()

    assert maybe_gc_stale_locks() == 1

    second = os.path.join(lock_dir, "second.lock")
    open(second, "wb").close()
    assert maybe_gc_stale_locks() == 0
    assert os.path.exists(second), "throttled second call must not sweep again"


def test_palace_lock_acquire_sweeps_residues(tmp_path, monkeypatch):
    """Acquiring mine_palace_lock GCs residues left by crashed processes."""
    lock_dir = _set_home(monkeypatch, tmp_path)
    _reset_gc_throttle(monkeypatch)
    residue = os.path.join(lock_dir, "crashed-mine.lock")
    open(residue, "wb").close()

    with mine_palace_lock(str(tmp_path / "palace")):
        pass

    assert not os.path.exists(residue)


# ---------------------------------------------------------------------------
# Holder records
# ---------------------------------------------------------------------------


def test_mine_palace_lock_records_pid_argv_and_acquire_time(tmp_path, monkeypatch):
    _set_home(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "argv", ["memp", "mine", "~/code"])
    with mine_palace_lock(str(tmp_path / "palace")):
        lock_dir = tmp_path / ".mempalace" / "locks"
        lock_files = list(lock_dir.glob("mine_palace_*.lock"))
        assert lock_files, "palace lock file must exist while held"
        record = parse_holder_record(lock_files[0].read_bytes()[1:].decode("utf-8"))
        assert record["pid"] == os.getpid()
        assert record["ppid"] == os.getppid()
        assert record["argv"] == ["memp", "mine", "~/code"]
        assert record["acquired_at"].startswith("20")


def test_mine_lock_records_holder_identity(tmp_path, monkeypatch):
    """Per-file mine locks are no longer anonymous 0-byte files while held."""
    from mempalace.palace import _mine_lock_path, mine_lock

    _set_home(monkeypatch, tmp_path)
    _reset_gc_throttle(monkeypatch)
    source = str(tmp_path / "some-transcript.jsonl")
    with mine_lock(source):
        lock_path = _mine_lock_path(source)
        with open(lock_path, "rb") as fh:
            record = parse_holder_record(fh.read()[1:].decode("utf-8"))
        assert record is not None and record["pid"] == os.getpid()


def test_write_holder_record_survives_broken_file():
    """Holder-record failures must never block lock acquisition."""

    class BrokenLock:
        def seek(self, _offset):
            raise OSError("disk says no")

    write_holder_record(BrokenLock())  # must not raise


# ---------------------------------------------------------------------------
# 3. `memp locks` CLI
# ---------------------------------------------------------------------------


def _run_cmd_locks(json_flag: bool = False, gc_flag: bool = False) -> None:
    from mempalace.cli import cmd_locks

    cmd_locks(argparse.Namespace(json=json_flag, gc=gc_flag))


def test_cmd_locks_human_table_lists_holder_facts(tmp_path, monkeypatch, capsys):
    lock_dir = _set_home(monkeypatch, tmp_path)
    _write_record_file(
        os.path.join(lock_dir, "aaaa.lock"),
        {"pid": os.getpid(), "argv": ["memp", "mine"], "acquired_at": "2026-07-12T07:00:00+00:00"},
    )
    open(os.path.join(lock_dir, "bbbb.lock"), "wb").close()

    _run_cmd_locks()
    out = capsys.readouterr().out
    assert "NAME" in out and "HELD" in out and "PID" in out and "ALIVE" in out
    assert "aaaa.lock" in out and "bbbb.lock" in out
    assert str(os.getpid()) in out
    assert "memp mine" in out
    assert "2 lock file(s), 0 held" in out


def test_cmd_locks_json_output(tmp_path, monkeypatch, capsys):
    lock_dir = _set_home(monkeypatch, tmp_path)
    dead = _dead_pid()
    _write_record_file(os.path.join(lock_dir, "dead.lock"), {"pid": dead, "argv": ["x"]})

    _run_cmd_locks(json_flag=True)
    payload = json.loads(capsys.readouterr().out)
    assert payload["locks_dir"] == lock_dir
    assert payload["gc_removed"] is None
    (entry,) = payload["locks"]
    assert entry["name"] == "dead.lock"
    assert entry["pid"] == dead
    assert entry["pid_alive"] is False
    assert entry["held"] is False


def test_cmd_locks_is_read_only_without_gc(tmp_path, monkeypatch):
    lock_dir = _set_home(monkeypatch, tmp_path)
    residue = os.path.join(lock_dir, "residue.lock")
    open(residue, "wb").close()

    _run_cmd_locks()
    assert os.path.exists(residue), "listing must not GC without --gc"


def test_cmd_locks_gc_sweeps_and_reports(tmp_path, monkeypatch, capsys):
    lock_dir = _set_home(monkeypatch, tmp_path)
    open(os.path.join(lock_dir, "residue.lock"), "wb").close()
    _write_record_file(os.path.join(lock_dir, "live.lock"), {"pid": os.getpid()})

    _run_cmd_locks(gc_flag=True)
    out = capsys.readouterr().out
    assert "GC'd 1 residual lock file(s)" in out
    assert not os.path.exists(os.path.join(lock_dir, "residue.lock"))
    assert os.path.exists(os.path.join(lock_dir, "live.lock"))


def test_cmd_locks_json_gc_reports_removed_count(tmp_path, monkeypatch, capsys):
    lock_dir = _set_home(monkeypatch, tmp_path)
    open(os.path.join(lock_dir, "residue.lock"), "wb").close()

    _run_cmd_locks(json_flag=True, gc_flag=True)
    payload = json.loads(capsys.readouterr().out)
    assert payload["gc_removed"] == 1
    assert payload["locks"] == []


def test_memp_locks_registered_in_cli_dispatch():
    """`memp locks --help` must be reachable as a first-class subcommand."""
    result = subprocess.run(
        [sys.executable, "-m", "mempalace.cli", "locks", "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0
    assert "--gc" in result.stdout
    assert "--json" in result.stdout
