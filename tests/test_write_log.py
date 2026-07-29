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


# ── Wiring ─────────────────────────────────────────────────────────────
#
# Everything above calls log_write() directly, so all of it stays green if the
# recorder is never called by anything. The recorder's whole value is that the
# real write path calls it: three log_write() sites in mcp_server.tool_add_drawer
# / tool_update_drawer and one in palace._palace_contention_error. Delete any of
# them and the module keeps its 100% coverage while the flight recorder goes
# blind. These tests drive the public tools and assert the record that lands.


@pytest.fixture
def wired(monkeypatch, tmp_path, config, palace_path, seeded_collection, kg):
    """MCP tools pointed at a scratch palace, write log in a scratch HOME.

    Returns the log HOME so a test can assert on the absence of the file as
    well as its contents.
    """
    from mempalace import mcp_server

    monkeypatch.setattr(mcp_server, "_config", config)
    monkeypatch.setattr(mcp_server, "_get_kg", lambda *a, **kw: kg)

    log_home = tmp_path / "log_home"
    log_home.mkdir()
    monkeypatch.setenv("HOME", str(log_home))
    monkeypatch.delenv(write_log.DISABLE_ENV, raising=False)
    return log_home


def _records_for(op):
    return [r for r in _read_records(write_log_path()) if r["op"] == op]


def test_add_drawer_records_a_successful_write(wired):
    """A filed drawer must leave a record naming it.

    Catches the log_write() call in tool_add_drawer being dropped — which no
    test in this file would otherwise notice.
    """
    from mempalace.mcp_server import tool_add_drawer

    result = tool_add_drawer(wing="project", room="backend", content="verbatim words")
    assert result["success"] is True

    records = _records_for("add_drawer")
    assert len(records) == 1
    assert records[0]["ok"] is True
    assert records[0]["drawer_id"] == result["drawer_id"]
    assert records[0]["wing"] == "project"
    assert records[0]["room"] == "backend"


def test_update_drawer_records_a_successful_write(wired):
    """An updated drawer must leave a record, flagged as unchecked.

    ``checked`` is how the health agent tells a blind write from a check-and-set
    one; a wiring change that stopped passing it would hide every unguarded
    singleton write.
    """
    from mempalace.mcp_server import tool_update_drawer

    assert tool_update_drawer("drawer_proj_backend_aaa", content="rewritten")["success"] is True

    records = _records_for("update_drawer")
    assert len(records) == 1
    assert records[0]["ok"] is True
    assert records[0]["drawer_id"] == "drawer_proj_backend_aaa"
    assert records[0]["checked"] is False


def test_cas_conflict_records_the_refused_write_and_keeps_the_first_writer(wired):
    """A refused check-and-set is the one failure that raises no exception.

    Nothing else in the system would ever record it: the tool returns a plain
    dict. This pins both halves — the event lands with both digests, and the
    earlier writer's content survives.
    """
    from mempalace.mcp_server import tool_get_drawer, tool_update_drawer

    stale = tool_get_drawer("drawer_proj_backend_aaa")["content_sha256"]
    assert tool_update_drawer("drawer_proj_backend_aaa", content="agent A wrote this")["success"]

    refused = tool_update_drawer(
        "drawer_proj_backend_aaa",
        content="agent B clobbering from a stale read",
        if_unchanged=stale,
    )
    assert refused["conflict"] is True

    conflicts = [r for r in _records_for("update_drawer") if r["ok"] is False]
    assert len(conflicts) == 1
    assert conflicts[0]["error_class"] == ERR_CAS_CONFLICT
    assert conflicts[0]["expected_sha256"] == stale[:16]
    assert conflicts[0]["actual_sha256"] != stale[:16]
    assert conflicts[0]["drawer_id"] == "drawer_proj_backend_aaa"

    assert tool_get_drawer("drawer_proj_backend_aaa")["content"] == "agent A wrote this"


def test_a_checked_write_that_succeeds_is_flagged_checked(wired):
    from mempalace.mcp_server import tool_get_drawer, tool_update_drawer

    digest = tool_get_drawer("drawer_proj_backend_aaa")["content_sha256"]
    assert tool_update_drawer(
        "drawer_proj_backend_aaa", content="guarded rewrite", if_unchanged=digest
    )["success"]

    assert _records_for("update_drawer")[0]["checked"] is True


def test_opting_out_silences_the_real_write_path(wired, monkeypatch):
    """MEMPALACE_WRITE_LOG=0 must reach through the tools, not just log_write."""
    from mempalace.mcp_server import tool_add_drawer

    monkeypatch.setenv(write_log.DISABLE_ENV, "0")
    assert tool_add_drawer(wing="project", room="backend", content="quiet please")["success"]

    assert not wired.joinpath(".mempalace/telemetry").exists()


def test_a_broken_write_log_never_breaks_the_write_itself(wired, monkeypatch):
    """Fail-soft at the integration level, not just in isolation.

    The unit test proves log_write() swallows an OSError, but it fakes the
    failure by replacing ``os.makedirs`` — which is the real, process-wide
    ``os`` module, so it would break the storage backend too. Here the failure
    is made at the filesystem instead: a plain file sits where the telemetry
    directory needs to be, so only the log's own makedirs fails. The drawer must
    still land and be readable, because telemetry may never cost a memory.
    """
    from mempalace.mcp_server import tool_add_drawer, tool_get_drawer

    blocker = wired / ".mempalace"
    blocker.mkdir()
    blocker.joinpath("telemetry").write_text("not a directory")

    result = tool_add_drawer(wing="project", room="backend", content="survives a dead recorder")
    assert result["success"] is True
    assert tool_get_drawer(result["drawer_id"])["content"] == "survives a dead recorder"
