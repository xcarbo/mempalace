"""Tests for ChromaCollection's write-watchdog (blast-radius bound).

Motivation — real incident: a ``mempalace mine --mode convos`` process hung
for 4h45m *inside* a native ChromaDB ``upsert`` (Rust core parked on a
condition variable) while holding ``mine_palace_lock`` — which is also
ChromaDB's exclusive SQLite writer. Every ``memp`` read across every project
was jammed for the duration.

The watchdog hard-bounds that blast radius: no single palace-mutating backend
call may hold the lock longer than a configurable timeout. On expiry it logs
loudly and force-exits the process (``os._exit``), which the OS turns into an
immediate release of both the flock and the SQLite lock. Because ingest is
append-only, the in-flight (uncommitted) batch is simply discarded and the
existing palace is untouched.

Properties tested:

* A write that exceeds ``MEMPALACE_WRITE_WATCHDOG_SECONDS`` trips the watchdog
  *while the write is still in flight* (i.e. while the lock is held).
* A fast write never trips it.
* Setting the timeout to 0 disables the watchdog entirely.
* ``ChromaCollection`` built without ``palace_path`` (legacy no-lock path) is
  never watched.
* End-to-end: a genuinely hung write in a child process is force-killed with
  the watchdog exit code, within seconds (not hours); afterward the palace
  lock is free again and the palace directory is byte-for-byte intact.
"""

from __future__ import annotations

import multiprocessing
import os
import sys
import textwrap
import time

import pytest

from mempalace.backends import chroma as chroma_mod
from mempalace.backends.chroma import ChromaCollection

WATCHDOG_ENV = "MEMPALACE_WRITE_WATCHDOG_SECONDS"


def _get_mp_context():
    """``spawn`` everywhere — matches test_chroma_collection_lock.py."""
    return multiprocessing.get_context("spawn")


class _SlowColl:
    """Fake chromadb collection whose upsert blocks for ``hold`` seconds.

    Records when the underlying write actually completed so a test can prove
    the watchdog fired *before* the write returned (i.e. mid-lock).
    """

    def __init__(self, hold: float):
        self._hold = hold
        self.finished_at: float | None = None
        self.upserts: list[dict] = []

    def upsert(self, **kwargs):
        time.sleep(self._hold)
        self.finished_at = time.monotonic()
        self.upserts.append(kwargs)


@pytest.fixture
def recorded_abort(monkeypatch):
    """Replace the real (``os._exit``) abort with a recorder.

    Returns the list the recorder appends ``(op, palace_path, elapsed, when)``
    to. Because we monkeypatch the module-level name, the watchdog's own
    ``os._exit`` never runs, so the test process survives.
    """
    calls: list[tuple] = []

    def _fake_abort(op, palace_path, elapsed):
        calls.append((op, palace_path, elapsed, time.monotonic()))

    monkeypatch.setattr(chroma_mod, "_watchdog_abort", _fake_abort)
    return calls


def test_slow_write_trips_watchdog_mid_flight(tmp_path, monkeypatch, recorded_abort):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv(WATCHDOG_ENV, "0.3")
    palace = str(tmp_path / "palace")

    slow = _SlowColl(hold=0.8)
    col = ChromaCollection(slow, palace_path=palace)
    col.upsert(documents=["d"], ids=["i"])

    assert len(recorded_abort) == 1, "watchdog must fire exactly once"
    op, path, elapsed, when = recorded_abort[0]
    assert op == "upsert"
    assert path == palace
    assert elapsed >= 0.3
    # Fired before the underlying write returned -> lock was still held.
    assert slow.finished_at is not None
    assert when <= slow.finished_at + 0.05


def test_fast_write_does_not_trip_watchdog(tmp_path, monkeypatch, recorded_abort):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv(WATCHDOG_ENV, "5")
    palace = str(tmp_path / "palace")

    fast = _SlowColl(hold=0.0)
    col = ChromaCollection(fast, palace_path=palace)
    col.upsert(documents=["d"], ids=["i"])

    assert recorded_abort == []
    assert len(fast.upserts) == 1


def test_zero_timeout_disables_watchdog(tmp_path, monkeypatch, recorded_abort):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv(WATCHDOG_ENV, "0")
    palace = str(tmp_path / "palace")

    slow = _SlowColl(hold=0.3)
    col = ChromaCollection(slow, palace_path=palace)
    col.upsert(documents=["d"], ids=["i"])

    assert recorded_abort == []


def test_no_palace_path_is_not_watched(tmp_path, monkeypatch, recorded_abort):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv(WATCHDOG_ENV, "0.2")

    slow = _SlowColl(hold=0.5)
    col = ChromaCollection(slow)  # no palace_path -> legacy path, no watchdog
    col.upsert(documents=["d"], ids=["i"])

    assert recorded_abort == []


# ---------------------------------------------------------------------------
# End-to-end: real os._exit abort releases the lock and leaves palace intact
# ---------------------------------------------------------------------------

_CHILD_SCRIPT = textwrap.dedent(
    """
    import os, sys, time

    os.environ["HOME"] = sys.argv[1]
    os.environ["MEMPALACE_WRITE_WATCHDOG_SECONDS"] = "1"

    from mempalace.backends.chroma import ChromaCollection

    class Hang:
        def upsert(self, **kwargs):
            time.sleep(600)  # wedged native call stand-in

    palace = sys.argv[2]
    col = ChromaCollection(Hang(), palace_path=palace)
    col.upsert(documents=["d"], ids=["i"])
    # If we ever get here the watchdog failed to abort.
    sys.exit(0)
    """
)

_PROBE_SCRIPT = textwrap.dedent(
    """
    import os, sys
    os.environ["HOME"] = sys.argv[1]
    from mempalace.palace import mine_palace_lock, MineAlreadyRunning
    try:
        with mine_palace_lock(sys.argv[2]):
            print("ACQUIRED")
    except MineAlreadyRunning:
        print("BUSY")
    """
)


@pytest.mark.skipif(os.name == "nt", reason="flock release-on-death semantics are POSIX")
def test_watchdog_kills_hung_write_and_releases_lock(tmp_path):
    import subprocess

    home = tmp_path / "home"
    home.mkdir()
    palace = tmp_path / "palace"
    palace.mkdir()
    sentinel = palace / "existing.txt"
    sentinel.write_text("untouched")
    before = sorted((p.name, p.read_bytes()) for p in palace.iterdir())

    child_py = tmp_path / "child.py"
    child_py.write_text(_CHILD_SCRIPT)
    probe_py = tmp_path / "probe.py"
    probe_py.write_text(_PROBE_SCRIPT)

    started = time.monotonic()
    proc = subprocess.run(
        [sys.executable, str(child_py), str(home), str(palace)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    elapsed = time.monotonic() - started

    # Force-killed with the watchdog exit code, and fast (seconds, not hours).
    assert proc.returncode == chroma_mod._WATCHDOG_EXIT_CODE, proc.stderr
    assert elapsed < 30, f"watchdog took too long: {elapsed:.1f}s"
    assert "watchdog" in proc.stderr.lower()

    # Lock released by process death -> a fresh acquirer succeeds.
    probe = subprocess.run(
        [sys.executable, str(probe_py), str(home), str(palace)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert probe.stdout.strip() == "ACQUIRED", (probe.stdout, probe.stderr)

    # Append-only: the crash left the existing palace untouched.
    after = sorted((p.name, p.read_bytes()) for p in palace.iterdir())
    assert after == before
