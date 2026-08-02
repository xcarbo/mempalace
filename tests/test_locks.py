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
import time
from datetime import datetime, timezone

from unittest import mock

import pytest

from mempalace import locks as locks_mod
from mempalace import palace as palace_mod
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
    genuine held lock. Since the v3.6.0 merge the re-entrant pass-through is
    process-wide (#1859), so the contending acquire inside the block runs with
    ``_held_by_this_process`` patched to False — simulating the cross-process
    contender (orphaned mine, second CLI) these diagnostics exist for. Called
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
        with mock.patch.object(palace_mod, "_held_by_this_process", return_value=False):
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
# Dead vs live holders on the real palace-lock path (2026-08-02)
#
# The failure under investigation was "orphaned lock blocks every writer". The
# asymmetry that matters: a lock whose holder is dead must be reclaimed, and a
# lock whose holder is alive must NEVER be broken — breaking a live one
# corrupts concurrent writes into a 158k-drawer palace, which is far worse than
# leaving a stale file lying around.
# ---------------------------------------------------------------------------


def _palace_lock_path(lock_dir: str, palace: str) -> str:
    """The lock file ``mine_palace_lock(palace)`` will use — same key formula."""
    import hashlib

    resolved = os.path.realpath(os.path.expanduser(palace))
    key = hashlib.sha256(os.path.normcase(resolved).encode()).hexdigest()[:16]
    return os.path.join(lock_dir, f"mine_palace_{key}.lock")


def test_dead_holder_palace_lock_is_broken_and_reclaimed(tmp_path, monkeypatch):
    """A lock file naming a dead PID must not keep the next writer out."""
    lock_dir = _set_home(monkeypatch, tmp_path)
    _reset_gc_throttle(monkeypatch)
    palace = str(tmp_path / "palace")
    lock_path = _palace_lock_path(lock_dir, palace)
    _write_record_file(
        lock_path,
        {
            "pid": _dead_pid(),
            "ppid": 1,
            "argv": ["/repo/mempalace/__main__.py", "mine", "/transcripts", "--mode", "convos"],
            "acquired_at": "2026-08-02T16:31:00+00:00",
        },
    )

    with mine_palace_lock(palace):
        # Reclaimed: the record now names us, not the dead miner.
        with open(lock_path, "rb") as fh:
            record = parse_holder_record(fh.read()[1:].decode("utf-8"))
        assert record["pid"] == os.getpid()


def test_live_holder_palace_lock_is_still_respected(tmp_path, monkeypatch):
    """The liveness check must not overcorrect into breaking real locks."""
    lock_dir = _set_home(monkeypatch, tmp_path)
    _reset_gc_throttle(monkeypatch)
    palace = str(tmp_path / "palace")
    lock_path = _palace_lock_path(lock_dir, palace)

    with _hold_palace_lock(palace):
        with pytest.raises(MineAlreadyRunning):
            with mine_palace_lock(palace):
                pytest.fail("a live holder's lock must never be broken")
        assert os.path.exists(lock_path), "a held lock file must survive contention"
        with open(lock_path, "rb") as fh:
            record = parse_holder_record(fh.read()[1:].decode("utf-8"))
        assert record["pid"] == os.getpid(), "holder record must still name the live holder"


def test_gc_removes_recycled_pid_residue(tmp_path, monkeypatch):
    """A live PID younger than the lock it supposedly holds is a recycled PID."""
    lock_dir = _set_home(monkeypatch, tmp_path)
    residue = os.path.join(lock_dir, "recycled.lock")
    # This process is alive but was certainly not running in 2020.
    _write_record_file(
        residue,
        {"pid": os.getpid(), "argv": ["memp", "mine"], "acquired_at": "2020-01-01T00:00:00+00:00"},
    )

    assert gc_stale_locks() == 1
    assert not os.path.exists(residue)


def test_gc_keeps_live_holder_that_predates_the_lock(tmp_path, monkeypatch):
    """The recycled-PID test must only fire on a positive contradiction."""
    lock_dir = _set_home(monkeypatch, tmp_path)
    keeper = os.path.join(lock_dir, "genuinely-live.lock")
    _write_record_file(
        keeper,
        {
            "pid": os.getpid(),
            "argv": ["memp", "mine"],
            # Acquired a moment ago: consistent with this process being the holder.
            "acquired_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    )

    assert gc_stale_locks() == 0
    assert os.path.exists(keeper)


def test_gc_keeps_live_holder_with_no_acquire_timestamp(tmp_path, monkeypatch):
    """Legacy records carry no acquire time — unknown must mean 'leave it alone'."""
    lock_dir = _set_home(monkeypatch, tmp_path)
    keeper = os.path.join(lock_dir, "legacy-format.lock")
    with open(keeper, "wb") as fh:
        fh.write(b"\0" + f"{os.getpid()} memp mine /some/dir".encode())

    assert gc_stale_locks() == 0
    assert os.path.exists(keeper)


# ---------------------------------------------------------------------------
# Bounded wait for short writers
# ---------------------------------------------------------------------------


def test_wait_seconds_acquires_once_the_holder_releases(tmp_path, monkeypatch):
    """A short writer queues behind a live mine instead of failing outright."""
    _set_home(monkeypatch, tmp_path)
    _reset_gc_throttle(monkeypatch)
    palace = str(tmp_path / "palace")
    acquired = threading.Event()
    holder_done = threading.Event()

    def _hold():
        with mine_palace_lock(palace):
            acquired.set()
            time.sleep(0.6)
        holder_done.set()

    thread = threading.Thread(target=_hold)
    thread.start()
    try:
        assert acquired.wait(timeout=10)
        with mock.patch.object(palace_mod, "_held_by_this_process", return_value=False):
            started = time.monotonic()
            with mine_palace_lock(palace, wait_seconds=10):
                waited = time.monotonic() - started
        assert holder_done.is_set(), "waiter must not acquire before the holder released"
        assert waited >= 0.3, "the waiter should have actually waited, not raced through"
    finally:
        thread.join(timeout=10)


def test_wait_seconds_zero_still_refuses_immediately(tmp_path, monkeypatch):
    """The default stays non-blocking: mines must not queue behind each other."""
    _set_home(monkeypatch, tmp_path)
    palace = str(tmp_path / "palace")

    with _hold_palace_lock(palace):
        started = time.monotonic()
        with pytest.raises(MineAlreadyRunning):
            with mine_palace_lock(palace):
                pytest.fail("wait_seconds=0 must refuse a contended lock")
        assert time.monotonic() - started < 5


def test_wait_seconds_gives_up_at_the_deadline(tmp_path, monkeypatch):
    """A bounded wait is bounded — it must raise, not hang, when the deadline passes."""
    _set_home(monkeypatch, tmp_path)
    palace = str(tmp_path / "palace")

    with _hold_palace_lock(palace):
        started = time.monotonic()
        with pytest.raises(MineAlreadyRunning):
            with mine_palace_lock(palace, wait_seconds=1):
                pytest.fail("the wait must end in the usual contention error")
        elapsed = time.monotonic() - started
        assert 0.9 <= elapsed < 10


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


# ---------------------------------------------------------------------------
# 5. The contention event, and the sweep's safety margins
# ---------------------------------------------------------------------------


def _write_log_records(home) -> list[dict]:
    """Every record the write log collected under ``home`` this test."""
    from mempalace.write_log import write_log_path

    path = write_log_path()
    assert path.startswith(str(home)), "write log must be inside the scratch HOME"
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_contention_emits_a_structured_lock_acquire_failure(tmp_path, monkeypatch):
    """A blocked write must be answerable later, not just printed once.

    ``_palace_contention_error`` is the single place a contended palace lock
    becomes an error, so it is the only place the event can be recorded — which
    is exactly why the recorder was put there. Every other contention test in
    this file asserts the English sentence and would stay green if the
    ``log_write`` call were deleted, leaving "did writes recover after the
    outage?" unanswerable again.
    """
    _set_home(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "argv", ["memp", "mine", "/some/dir"])
    palace = str(tmp_path / "palace")

    with _hold_palace_lock(palace):
        with pytest.raises(MineAlreadyRunning):
            with mine_palace_lock(palace):
                pytest.fail("second acquire of a held palace lock must raise")

    events = [r for r in _write_log_records(tmp_path) if r["op"] == "lock_acquire"]
    assert len(events) == 1
    event = events[0]
    assert event["ok"] is False
    assert event["error_class"] == "lock_contention"
    assert event["palace"] == os.path.realpath(palace)
    assert event["holder_pid"] == os.getpid()
    assert "memp mine /some/dir" in event["holder_argv"]
    assert event["held_seconds"] >= 0


def test_an_uncontended_acquire_records_nothing(tmp_path, monkeypatch):
    """Only failures are events. A recorder that logged every acquire would
    grow a line per write and drown the failures it exists to surface."""
    _set_home(monkeypatch, tmp_path)
    palace = str(tmp_path / "palace")

    with mine_palace_lock(palace):
        pass

    assert _write_log_records(tmp_path) == []


def test_gc_leaves_a_lock_file_that_was_replaced_mid_sweep(tmp_path, monkeypatch):
    """The sweep must never unlink an inode it did not prove stale.

    ``_gc_one_lock_file`` opens the path, proves the file unheld and stale, then
    unlinks *the path*. Between the open and the unlink another process can
    acquire the lock — creating a fresh file at the same name — and the sweep
    would delete a live holder's lock, letting a second writer into the same
    palace and corrupting the HNSW graph. The guard is a re-check that the open
    fd still names the same inode.

    The monkeypatch is a scheduling seam only: it makes the replacement happen
    at the one instant the race needs. What is asserted is the observable
    outcome on disk — the replacement file survives.
    """
    lock_dir = _set_home(monkeypatch, tmp_path)
    path = os.path.join(lock_dir, "mine_palace_deadbeef.lock")
    open(path, "wb").close()  # 0-byte residue: provably stale, would be swept

    real_probe = locks_mod._probe_lock
    swapped = {"done": False}

    def _probe_then_replace(lock_file):
        result = real_probe(lock_file)
        if not swapped["done"]:
            swapped["done"] = True
            # A contender wins the race: fresh file, same name, new inode.
            os.remove(path)
            with open(path, "wb") as fh:
                fh.write(b"\0" + json.dumps({"pid": os.getpid()}).encode())
        return result

    monkeypatch.setattr(locks_mod, "_probe_lock", _probe_then_replace)

    gc_stale_locks(lock_dir)

    assert os.path.exists(path), "sweep deleted a lock file it never proved stale"
    assert swapped["done"], "the race was never triggered; test proves nothing"


def test_maybe_gc_never_raises_when_the_sweep_fails(tmp_path, monkeypatch):
    """A broken locks directory must not block every write in the system.

    ``maybe_gc_stale_locks`` runs on the acquire path of every palace write. If
    it could propagate an OSError — an unreadable locks dir, a permissions
    change — a housekeeping problem would become a total write outage.
    """
    _set_home(monkeypatch, tmp_path)
    _reset_gc_throttle(monkeypatch)

    def boom(*_a, **_kw):
        raise OSError("locks dir is unreadable")

    monkeypatch.setattr(locks_mod, "gc_stale_locks", boom)

    assert maybe_gc_stale_locks() == 0


def test_list_locks_reports_hold_duration_for_a_live_holder(tmp_path, monkeypatch):
    """`memp locks` is what a human reads during a write outage.

    The duration field is the one that answers "has this been stuck for hours?",
    and it is only populated for locks that are actually held — the case no
    existing test covers, because they all inspect residue.
    """
    _set_home(monkeypatch, tmp_path)
    palace = str(tmp_path / "palace")

    with _hold_palace_lock(palace):
        entries = locks_mod.list_locks()

    held = [e for e in entries if e["held"]]
    assert len(held) == 1, f"expected exactly one held lock, got {entries}"
    assert held[0]["pid"] == os.getpid()
    assert held[0]["pid_alive"] is True
    assert isinstance(held[0]["held_for_seconds"], int)
    assert held[0]["held_for_seconds"] >= 0
