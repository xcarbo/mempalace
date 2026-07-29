"""Tests for the write log — the flight recorder for the WRITE path.

Reads were already observable via the retrieval log; writes were not. Save
failures landed as free text in hook.log, whose timestamps carry no date, so
"did anything fail to save yesterday?" was unanswerable. These tests pin the
two properties that make it answerable: every record carries a real UTC
timestamp, and every failure carries a stable error_class.
"""

import json

import pytest

from mempalace import write_log
from mempalace.write_log import (
    ERR_BACKEND,
    ERR_CAS_CONFLICT,
    ERR_LOCK_CONTENTION,
    ERR_NOT_FOUND,
    ERR_UNKNOWN,
    ERR_VALIDATION,
    classify_error,
    log_write,
    log_write_failure,
    write_log_path,
)


@pytest.fixture
def scratch_home(tmp_path, monkeypatch):
    """Point the log at a scratch HOME so tests never touch the real one."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv(write_log.DISABLE_ENV, raising=False)
    return tmp_path


def _read_records(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ── log_write ──────────────────────────────────────────────────────────


def test_log_write_appends_record_with_required_fields(scratch_home):
    log_write("add_drawer", ok=True, wing="w", room="r", drawer_id="d1")

    records = _read_records(write_log_path())
    assert len(records) == 1
    rec = records[0]
    assert rec["op"] == "add_drawer"
    assert rec["ok"] is True
    assert rec["wing"] == "w"
    assert rec["drawer_id"] == "d1"
    assert isinstance(rec["pid"], int)
    # A real, parseable UTC timestamp is the whole point — hook.log's
    # time-only stamps are what made a 24h window impossible.
    assert rec["ts"].endswith("+00:00")


def test_log_write_appends_rather_than_truncates(scratch_home):
    log_write("add_drawer", ok=True, drawer_id="a")
    log_write("update_drawer", ok=True, drawer_id="b")

    records = _read_records(write_log_path())
    assert [r["op"] for r in records] == ["add_drawer", "update_drawer"]


def test_opt_out_writes_nothing(scratch_home, monkeypatch):
    monkeypatch.setenv(write_log.DISABLE_ENV, "0")
    log_write("add_drawer", ok=True)

    assert not scratch_home.joinpath(".mempalace/telemetry").exists()


@pytest.mark.parametrize("falsey", ["0", "false", "off", "no", "FALSE"])
def test_opt_out_accepts_the_documented_falsey_values(scratch_home, monkeypatch, falsey):
    monkeypatch.setenv(write_log.DISABLE_ENV, falsey)
    assert write_log.write_log_enabled() is False


def test_logging_never_raises_even_when_the_dir_is_unwritable(scratch_home, monkeypatch):
    """Fail-soft by contract: a logging failure must not break a write."""

    def boom(*a, **kw):
        raise OSError("disk full")

    monkeypatch.setattr(write_log.os, "makedirs", boom)
    log_write("add_drawer", ok=True)  # must not raise


def test_non_serializable_values_are_stringified_not_dropped(scratch_home):
    log_write("add_drawer", ok=True, weird=object())

    rec = _read_records(write_log_path())[0]
    assert "weird" in rec
    assert isinstance(rec["weird"], str)


# ── classify_error ─────────────────────────────────────────────────────


def test_lock_contention_is_classified_from_the_holder_message():
    exc = RuntimeError("palace /p is held by PID 123 (mine), alive")
    assert classify_error(exc) == ERR_LOCK_CONTENTION


def test_cas_conflict_is_classified():
    assert classify_error(RuntimeError("drawer changed, conflict")) == ERR_CAS_CONFLICT


def test_not_found_is_classified():
    assert classify_error(RuntimeError("Drawer d1 not found")) == ERR_NOT_FOUND


def test_validation_errors_are_classified():
    assert classify_error(ValueError("invalid wing")) == ERR_VALIDATION


def test_backend_errors_are_classified():
    exc = RuntimeError("malformed inverted index for FTS5 table")
    assert classify_error(exc) == ERR_BACKEND


def test_unrecognised_errors_fall_back_to_unknown_not_a_neighbour():
    """Conservative by design — never silently fold into a nearby class."""
    assert classify_error(RuntimeError("something entirely new")) == ERR_UNKNOWN


# ── log_write_failure ──────────────────────────────────────────────────


def test_log_write_failure_records_class_and_truncated_message(scratch_home):
    log_write_failure("update_drawer", ValueError("invalid room"), drawer_id="d9")

    rec = _read_records(write_log_path())[0]
    assert rec["ok"] is False
    assert rec["op"] == "update_drawer"
    assert rec["error_class"] == ERR_VALIDATION
    assert rec["error"] == "invalid room"
    assert rec["drawer_id"] == "d9"


def test_log_write_failure_truncates_a_huge_message(scratch_home):
    log_write_failure("add_drawer", RuntimeError("x" * 5000))

    rec = _read_records(write_log_path())[0]
    assert len(rec["error"]) <= 500


def test_month_stamped_path_gives_natural_rotation(scratch_home):
    from datetime import datetime, timezone

    jan = datetime(2026, 1, 5, tzinfo=timezone.utc)
    feb = datetime(2026, 2, 5, tzinfo=timezone.utc)
    assert write_log_path(jan) != write_log_path(feb)
    assert write_log_path(jan).endswith("writes-2026-01.jsonl")
