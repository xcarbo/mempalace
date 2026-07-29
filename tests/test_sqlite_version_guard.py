"""Tests for the SQLite compatibility guard.

The bug it exists for: chromadb's Rust core and Python's ``sqlite3`` both touch
the palace database, and on some SQLite builds any statement Python runs against
a database the Rust side just closed kills the process with **SIGBUS** — from
the second palace in a process onward. No exception, no traceback, no failed
test; the interpreter simply dies mid-write.

Measured 2026-07-29: 3.50.4 crashes, 3.51.0 is clean, 3.53.3 crashes. Both older
and newer than the good version break, so "stay on latest" is actively wrong
here and the guard has to name versions rather than compare them.
"""

import pytest

from mempalace import palace


@pytest.fixture(autouse=True)
def _reset_warn_once():
    """The guard warns once per process; tests need a clean slate each time."""
    palace._sqlite_version_warned = False
    yield
    palace._sqlite_version_warned = False


def _pin_sqlite(monkeypatch, version):
    import sqlite3

    monkeypatch.setattr(sqlite3, "sqlite_version", version)


def test_known_good_version_is_silent(monkeypatch):
    _pin_sqlite(monkeypatch, "3.51.0")
    assert palace.warn_if_sqlite_untested() is None


def test_known_bad_version_is_named_as_known_to_crash(monkeypatch):
    _pin_sqlite(monkeypatch, "3.53.3")
    msg = palace.warn_if_sqlite_untested()

    assert msg is not None
    assert "3.53.3" in msg
    assert "KNOWN" in msg
    # The actionable half: a user who hits this needs to know the cause is
    # their interpreter build, not their palace.
    assert "Homebrew" in msg


def test_the_older_bad_version_is_also_caught(monkeypatch):
    """3.50.4 was the original SIGBUS; a guard that only knows about newer
    versions would miss the uv-managed interpreter that started all this."""
    _pin_sqlite(monkeypatch, "3.50.4")
    msg = palace.warn_if_sqlite_untested()

    assert msg is not None
    assert "3.50.4" in msg


def test_unknown_version_warns_without_claiming_breakage(monkeypatch):
    """An untested version is not proof of breakage — say so honestly."""
    _pin_sqlite(monkeypatch, "3.99.0")
    msg = palace.warn_if_sqlite_untested()

    assert msg is not None
    assert "untested" in msg
    assert "KNOWN to crash" not in msg


def test_warning_fires_once_per_process(monkeypatch):
    """Every palace open calls this; warning on each would bury the signal."""
    _pin_sqlite(monkeypatch, "3.53.3")

    assert palace.warn_if_sqlite_untested() is not None
    assert palace.warn_if_sqlite_untested() is None


def test_guard_never_raises_and_never_blocks(monkeypatch):
    """Refusing to open the palace would be worse than the risk it warns about.

    A user on an untested build must still be able to read their memory.
    """
    _pin_sqlite(monkeypatch, "0.0.0-nonsense")
    palace.warn_if_sqlite_untested()  # must not raise


def test_get_collection_invokes_the_guard(monkeypatch, tmp_path):
    """The guard is worthless if nothing calls it.

    Catches the call in get_collection being dropped — every other test here
    would stay green while the warning never reached a user again.
    """
    called = []
    monkeypatch.setattr(palace, "warn_if_sqlite_untested", lambda: called.append(1))

    with pytest.raises(Exception):
        # Any failure after the guard is fine; we only assert it ran first.
        palace.get_collection(str(tmp_path / "nonexistent"), backend="not-a-backend")

    assert called, "get_collection must call warn_if_sqlite_untested()"
