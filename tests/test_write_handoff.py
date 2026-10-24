"""Hook hand-off to the write queue, and failure visibility (fork).

Found by the 2026-10-24 audits (the batch half lives in tests/test_batch.py):

- A hook diary save waited up to 30 s for its daemon job, the whole hook
  budget, so a diary queued behind a running mine froze the agent. When the
  wait ran out the hook reported nothing filed, the Stop hook kept its save
  marker, and the next Stop filed a second diary for the same messages while
  the daemon still wrote the first one.
- A batch step that raised, or a ``${id.field}`` that named a missing field,
  failed the job with no record of what had already been written, and a
  parked update whose follow-up could not be filed still counted as parked.
- None of those failures reached a human.
"""

import json
import os
import threading
import time
from unittest.mock import patch

import pytest

from mempalace import daemon, hooks_cli, mcp_server, service
from mempalace.hooks_cli import SAVE_INTERVAL, hook_session_start, hook_stop


def _write_transcript(path, n):
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps({"message": {"role": "user", "content": f"message {i}"}}) + "\n")


def _run_hook(hook_fn, data, state_dir):
    out = []
    with (
        patch("mempalace.hooks_cli._output", side_effect=out.append),
        patch("mempalace.hooks_cli.STATE_DIR", state_dir),
        patch("mempalace.hooks_cli._ingest_transcript"),
        patch("mempalace.hooks_cli._maybe_auto_ingest"),
        patch("mempalace.hooks_cli._palace_root_exists", return_value=True),
    ):
        hook_fn(data, "claude-code")
    return out[-1] if out else None


@pytest.fixture
def daemon_routing(tmp_path, monkeypatch):
    palace_dir = tmp_path / "palace"
    palace_dir.mkdir(exist_ok=True)
    monkeypatch.setenv("MEMPALACE_HOOKS_DAEMON", "yes")
    monkeypatch.setenv("MEMPALACE_PALACE_PATH", str(palace_dir))
    with patch("mempalace.hooks_cli._daemon_available", return_value=True):
        yield palace_dir


def _queued(job_id="job-1", state="queued"):
    return {"id": job_id, "state": state, "result": None, "error": None}


# ── diary hand-off ───────────────────────────────────────────────────────


def test_diary_wait_is_short_and_well_under_the_hook_budget():
    # The harness kills Stop / SessionEnd / PreCompact hooks at 30 s.
    assert 0 < hooks_cli._DIARY_HANDOFF_WAIT_SECONDS <= 5
    assert 0 < hooks_cli._SYNC_MINE_HANDOFF_WAIT_SECONDS <= 5


@pytest.mark.parametrize("state", ["queued", "running"])
def test_diary_still_pending_after_the_wait_is_handed_off(daemon_routing, tmp_path, state):
    transcript = tmp_path / "t.jsonl"
    _write_transcript(transcript, 3)
    with (
        patch("mempalace.hooks_cli.STATE_DIR", tmp_path),
        patch("mempalace.daemon.submit_job", return_value=_queued(state=state)) as submit,
        patch("mempalace.hooks_cli._log") as log,
    ):
        result = hooks_cli._save_diary_direct(str(transcript), "s1", agent_name="claude")

    kwargs = submit.call_args.kwargs
    assert kwargs["timeout"] == hooks_cli._DIARY_HANDOFF_WAIT_SECONDS
    assert kwargs["return_on_timeout"] is True
    assert result["drawers_filed"] == 0  # truthful: nothing is in the palace yet
    assert result["handed_off"] == "job-1"
    logged = " ".join(str(c.args[0]) for c in log.call_args_list)
    assert "Diary checkpoint queued: job-1" in logged
    assert "failed" not in logged


def _stop(tmp_path, transcript, submit_return):
    data = {"session_id": "s1", "stop_hook_active": False, "transcript_path": str(transcript)}
    with patch("mempalace.daemon.submit_job", return_value=submit_return) as submit:
        out = _run_hook(hook_stop, data, tmp_path)
    return out, submit


@pytest.mark.parametrize(
    "pending",
    [
        _queued(),
        _queued(state="running"),
        {
            "id": "job-1",
            "state": "queued",
            "result": None,
            "error": {"error_class": "LockHeldByOtherProcess", "message": "held by PID 9"},
        },
    ],
    ids=["queued", "running", "lock-deferred"],
)
def test_stop_hook_advances_its_marker_on_a_hand_off_and_does_not_refile(
    daemon_routing, tmp_path, pending
):
    transcript = tmp_path / "t.jsonl"
    _write_transcript(transcript, SAVE_INTERVAL)

    out, submit = _stop(tmp_path, transcript, pending)
    assert submit.call_count == 1
    assert (tmp_path / "s1_last_save").read_text() == str(SAVE_INTERVAL)
    assert "queued" in out["systemMessage"]

    # The next Stop on the same messages must not queue a second diary.
    out, submit = _stop(tmp_path, transcript, pending)
    submit.assert_not_called()
    assert out == {}


def test_a_failed_diary_job_leaves_the_marker_alone(daemon_routing, tmp_path):
    transcript = tmp_path / "t.jsonl"
    _write_transcript(transcript, SAVE_INTERVAL)
    failed = {
        "id": "job-1",
        "state": "failed",
        "result": {"success": False, "error": "disk full"},
        "error": {"message": "disk full"},
    }

    out, submit = _stop(tmp_path, transcript, failed)
    assert submit.call_count == 1
    assert not (tmp_path / "s1_last_save").exists()
    assert out == {}

    # A real failure is retried by the next Stop.
    _out, submit = _stop(tmp_path, transcript, failed)
    assert submit.call_count == 1


def test_sync_mine_still_pending_is_logged_as_queued(tmp_path, monkeypatch):
    mempal_dir = tmp_path / "project"
    mempal_dir.mkdir()
    monkeypatch.setenv("MEMPAL_DIR", str(mempal_dir))
    monkeypatch.setenv("MEMPALACE_HOOKS_DAEMON", "yes")
    with (
        patch("mempalace.hooks_cli.STATE_DIR", tmp_path),
        patch("mempalace.hooks_cli._daemon_available", return_value=True),
        patch("mempalace.daemon.submit_job", return_value=_queued(state="running")) as submit,
        patch("mempalace.hooks_cli._log") as log,
        patch("mempalace.hooks_cli.subprocess.run") as run,
    ):
        hooks_cli._mine_sync()

    assert submit.call_args.kwargs["timeout"] == hooks_cli._SYNC_MINE_HANDOFF_WAIT_SECONDS
    assert submit.call_args.kwargs["return_on_timeout"] is True
    run.assert_not_called()
    logged = " ".join(str(c.args[0]) for c in log.call_args_list)
    assert "Daemon sync mine queued: job-1" in logged
    assert "failed" not in logged


def test_client_wait_can_return_the_pending_job_instead_of_raising():
    client = daemon.DaemonClient.__new__(daemon.DaemonClient)
    with patch.object(client, "get_job", return_value=_queued(state="running")):
        job = client.wait("job-1", timeout=0.05, return_on_timeout=True)
        assert job["state"] == "running"
        with pytest.raises(daemon.DaemonError, match="timed out"):
            client.wait("job-1", timeout=0.05)


# ── end to end: a diary queued behind a running mine ─────────────────────


@pytest.fixture
def live_palace(monkeypatch, config, palace_path, kg, tmp_path):
    monkeypatch.setattr(mcp_server, "_config", config)
    monkeypatch.setattr(mcp_server, "_get_kg", lambda *a, **kw: kg)
    monkeypatch.setenv(daemon.STATE_ROOT_ENV, str(tmp_path / "state"))
    monkeypatch.setenv("MEMPALACE_PALACE_PATH", palace_path)
    monkeypatch.setattr(daemon, "SHUTDOWN_DRAIN_SECONDS", 1.0)
    return palace_path


def test_diary_behind_a_running_mine_is_filed_once(live_palace, tmp_path, monkeypatch):
    gate = threading.Event()
    real_execute = service.execute_job

    def slow_mine(kind, payload):
        if kind == "mine":
            gate.wait(30)
            return {"success": True}
        return real_execute(kind, payload)

    monkeypatch.setattr(service, "execute_job", slow_mine)
    monkeypatch.setattr(hooks_cli, "_DIARY_HANDOFF_WAIT_SECONDS", 0.5)
    monkeypatch.setenv("MEMPALACE_HOOKS_DAEMON", "yes")

    server = threading.Thread(
        target=daemon.run_server, kwargs={"palace_path": live_palace, "port": 0}, daemon=True
    )
    server.start()
    client = None
    for _ in range(600):
        client = daemon.get_client_if_running(live_palace)
        if client is not None:
            break
        time.sleep(0.05)
    assert client is not None, "daemon never came up"
    try:
        mine = client.submit("mine", {"source": "/nowhere", "mode": "convos"})
        for _ in range(200):
            if client.get_job(mine["id"])["state"] == "running":
                break
            time.sleep(0.05)
        assert client.get_job(mine["id"])["state"] == "running"

        transcript = tmp_path / "t.jsonl"
        _write_transcript(transcript, SAVE_INTERVAL)
        data = {"session_id": "s1", "stop_hook_active": False, "transcript_path": str(transcript)}
        state = tmp_path / "hook_state"
        state.mkdir()

        started = time.monotonic()
        with patch("mempalace.hooks_cli._daemon_available", return_value=True):
            first = _run_hook(hook_stop, data, state)
            elapsed = time.monotonic() - started
            second = _run_hook(hook_stop, data, state)

        assert elapsed < 5, f"the Stop hook waited {elapsed:.1f}s behind the mine"
        assert "queued" in first["systemMessage"]
        assert second == {}

        gate.set()
        client.wait(mine["id"], timeout=20)
        diaries = [j for j in client.list_jobs(limit=50) if j["kind"] == "diary_write"]
        assert len(diaries) == 1, diaries
        assert client.wait(diaries[0]["id"], timeout=20)["state"] == "succeeded"
        assert mcp_server.tool_diary_read(agent_name="claude")["total"] == 1
    finally:
        gate.set()
        client.shutdown()
        server.join(timeout=10)


# ── failures reach a human ───────────────────────────────────────────────


def _run_one(runtime, kind, payload):
    job = runtime.store.enqueue(kind, payload)
    runtime.start_worker()
    try:
        for _ in range(400):
            got = runtime.store.get(job.id)
            if got.state in daemon.TERMINAL_STATES:
                return got
            time.sleep(0.025)
    finally:
        runtime.shutdown_event.set()
        runtime.worker_thread.join(timeout=5)
    raise AssertionError("job never finished")


def _days_ago(days):
    from datetime import datetime, timedelta, timezone

    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()


@pytest.fixture
def hook_log(tmp_path, monkeypatch):
    state = tmp_path / "hook_state"
    state.mkdir()
    monkeypatch.setattr(hooks_cli, "STATE_DIR", state)
    monkeypatch.setattr(hooks_cli, "_palace_root_exists", lambda: True)
    return state / "hook.log"


def test_a_failed_job_is_logged_counted_and_kept_until_acknowledged(
    tmp_path, monkeypatch, hook_log
):
    monkeypatch.setenv(daemon.STATE_ROOT_ENV, str(tmp_path / "state"))
    palace = tmp_path / "palace"
    palace.mkdir()
    monkeypatch.setattr(
        service,
        "execute_job",
        lambda kind, payload: {"success": False, "error": "step 2 (kg_add) failed: boom"},
    )

    runtime = daemon.DaemonRuntime(str(palace))
    job = _run_one(runtime, "batch", {"steps": []})
    assert job.state == "failed"

    log = hook_log.read_text()
    assert f"Daemon job failed: batch {job.id}" in log
    assert "step 2 (kg_add) failed: boom" in log

    assert daemon.unacknowledged_failures(str(palace)) == 1

    # Retention must not prune what nobody has seen...
    with runtime.store._connect() as conn:
        conn.execute("UPDATE jobs SET finished_at = ?", (_days_ago(10),))
    assert runtime.store.prune_terminal(older_than_days=7) == 0

    assert daemon.acknowledge_failures(str(palace)) == 1
    assert daemon.unacknowledged_failures(str(palace)) == 0
    assert runtime.store.prune_terminal(older_than_days=7) == 1


def test_unacknowledged_failures_are_still_bounded(tmp_path, monkeypatch):
    monkeypatch.setenv(daemon.STATE_ROOT_ENV, str(tmp_path / "state"))
    store = daemon.QueueStore(daemon.queue_path(str(tmp_path / "palace")))
    job = store.enqueue("batch", {"steps": []})
    store.finish(job.id, state="failed", error={"message": "x"})
    with store._connect() as conn:
        conn.execute(
            "UPDATE jobs SET finished_at = ?", (_days_ago(daemon.FAILED_RETENTION_DAYS + 1),)
        )
    assert store.prune_terminal(older_than_days=7) == 1


def test_a_job_cancelled_by_shutdown_is_not_reported_as_failed(tmp_path, monkeypatch, hook_log):
    monkeypatch.setenv(daemon.STATE_ROOT_ENV, str(tmp_path / "state"))
    runtime = daemon.DaemonRuntime(str(tmp_path / "palace"))
    job = runtime.store.enqueue("batch", {"steps": []})
    runtime.store.finish(job.id, state="cancelled")
    runtime._finish_failed(job, result={}, error={"message": "late"})
    assert runtime.store.get(job.id).state == "cancelled"
    assert not hook_log.exists()


def test_a_dead_lettered_job_is_logged(tmp_path, monkeypatch, hook_log):
    monkeypatch.setenv(daemon.STATE_ROOT_ENV, str(tmp_path / "state"))
    store = daemon.QueueStore(daemon.queue_path(str(tmp_path / "palace")))
    job = store.enqueue("diary_write", {"entry": "x"})
    with store._connect() as conn:
        conn.execute(
            "UPDATE jobs SET state = 'running', attempts = ? WHERE id = ?",
            (daemon.MAX_ATTEMPTS, job.id),
        )
    store.recover_running()
    assert store.get(job.id).state == "failed"
    assert f"Daemon job failed: diary_write {job.id}: max attempts exceeded" in hook_log.read_text()


def test_daemon_status_shows_unacknowledged_failures(tmp_path, monkeypatch, capsys):
    import argparse

    from mempalace import cli

    monkeypatch.setenv(daemon.STATE_ROOT_ENV, str(tmp_path / "state"))
    palace = tmp_path / "palace"
    palace.mkdir()
    store = daemon.QueueStore(daemon.queue_path(daemon.canonical_palace_path(str(palace))))
    job = store.enqueue("batch", {"steps": []})
    store.finish(job.id, state="failed", error={"message": "x"})

    args = argparse.Namespace(palace=str(palace), daemon_action="status", backend=None)
    with patch("mempalace.daemon.get_client_if_running", return_value=None):
        with pytest.raises(SystemExit):
            cli.cmd_daemon(args)
    out = capsys.readouterr().out
    assert "1 failed job" in out
    assert "memp daemon ack" in out

    cli.cmd_daemon(argparse.Namespace(palace=str(palace), daemon_action="ack", backend=None))
    assert "Acknowledged 1" in capsys.readouterr().out
    assert daemon.unacknowledged_failures(str(palace)) == 0


def test_session_start_reports_unacknowledged_failures(tmp_path, monkeypatch):
    monkeypatch.setenv(daemon.STATE_ROOT_ENV, str(tmp_path / "state"))
    palace = tmp_path / "palace"
    palace.mkdir()
    monkeypatch.setenv("MEMPALACE_PALACE_PATH", str(palace))
    store = daemon.QueueStore(daemon.queue_path(daemon.canonical_palace_path(str(palace))))
    job = store.enqueue("diary_write", {"entry": "x"})
    store.finish(job.id, state="failed", error={"message": "x"})

    out = _run_hook(hook_session_start, {"session_id": "s1"}, tmp_path)
    assert "1 failed" in out["systemMessage"]

    daemon.acknowledge_failures(str(palace))
    assert _run_hook(hook_session_start, {"session_id": "s1"}, tmp_path) == {}


def test_failure_counter_is_cheap_when_there_is_no_queue(tmp_path, monkeypatch):
    monkeypatch.setenv(daemon.STATE_ROOT_ENV, str(tmp_path / "state"))
    assert daemon.unacknowledged_failures(str(tmp_path / "nowhere")) == 0
    assert not daemon.queue_path(str(tmp_path / "nowhere")).exists()


def test_hook_log_line_is_skipped_when_the_palace_root_is_gone(tmp_path, monkeypatch):
    monkeypatch.setattr(hooks_cli, "STATE_DIR", tmp_path / "gone" / "hook_state")
    monkeypatch.setattr(hooks_cli, "_palace_root_exists", lambda: False)
    daemon._log_failed_job("batch", "j1", {"message": "x"})
    assert not os.path.exists(tmp_path / "gone")


def test_stop_hook_checkpoint_line_carries_the_failure_notice(daemon_routing, tmp_path):
    transcript = tmp_path / "t.jsonl"
    _write_transcript(transcript, SAVE_INTERVAL)
    with patch("mempalace.hooks_cli._unacknowledged_daemon_failures", return_value=2):
        out, _submit = _stop(tmp_path, transcript, _queued())
    assert "2 failed write-queue jobs" in out["systemMessage"]


def test_jobs_failed_lists_only_unacknowledged_failures(tmp_path, monkeypatch, capsys):
    import argparse

    from mempalace import cli

    monkeypatch.setenv(daemon.STATE_ROOT_ENV, str(tmp_path / "state"))
    palace = tmp_path / "palace"
    palace.mkdir()
    store = daemon.QueueStore(daemon.queue_path(daemon.canonical_palace_path(str(palace))))
    old = store.enqueue("batch", {"steps": []})
    store.finish(old.id, state="failed", error={"message": "seen already"})
    daemon.acknowledge_failures(str(palace))
    ok = store.enqueue("batch", {"steps": [1]})
    store.finish(ok.id, state="succeeded")
    new = store.enqueue("batch", {"steps": [2]})
    store.finish(new.id, state="failed", error={"message": "step 2 failed"})

    args = argparse.Namespace(
        palace=str(palace), daemon_action="jobs", backend=None, limit=20, failed=True
    )
    with patch("mempalace.daemon.get_client_if_running", return_value=None):
        cli.cmd_daemon(args)
    out = capsys.readouterr().out
    assert new.id in out and "step 2 failed" in out
    assert old.id not in out and ok.id not in out


# ── a submit that timed out after the daemon stored the job ──────────────
#
# POST /jobs has a 5 s client timeout. If the daemon stores the diary and
# the reply is lost, the hook sees an error, keeps its marker, and the next
# Stop used to queue a second diary for the same messages. Each unconfirmed
# save window now carries one dedupe key, and the daemon hands back the job
# already stored under it (queued, running or succeeded) instead of a new one.


def test_a_window_key_reuses_a_succeeded_job_but_not_a_failed_one(tmp_path, monkeypatch):
    monkeypatch.setenv(daemon.STATE_ROOT_ENV, str(tmp_path / "state"))
    store = daemon.QueueStore(daemon.queue_path(str(tmp_path / "palace")))

    first = store.enqueue("diary_write", {"entry": "a"}, dedupe_key="k", reuse_succeeded=True)
    store.finish(first.id, state="succeeded", result={"success": True})
    again = store.enqueue("diary_write", {"entry": "b"}, dedupe_key="k", reuse_succeeded=True)
    assert again.id == first.id and again.state == "succeeded"

    # Without the flag a finished job drops out (mines re-run on purpose).
    assert store.enqueue("diary_write", {"entry": "c"}, dedupe_key="k").id != first.id

    store2 = daemon.QueueStore(daemon.queue_path(str(tmp_path / "palace2")))
    failed = store2.enqueue("diary_write", {"entry": "a"}, dedupe_key="k", reuse_succeeded=True)
    store2.finish(failed.id, state="failed", error={"message": "x"})
    retry = store2.enqueue("diary_write", {"entry": "b"}, dedupe_key="k", reuse_succeeded=True)
    assert retry.id != failed.id and retry.state == "queued"


def _start_daemon(palace):
    server = threading.Thread(
        target=daemon.run_server, kwargs={"palace_path": palace, "port": 0}, daemon=True
    )
    server.start()
    for _ in range(600):
        client = daemon.get_client_if_running(palace)
        if client is not None:
            return server, client
        time.sleep(0.05)
    raise AssertionError("daemon never came up")


def test_a_submit_that_timed_out_after_the_job_was_stored_files_one_diary(
    live_palace, tmp_path, monkeypatch
):
    monkeypatch.setenv("MEMPALACE_HOOKS_DAEMON", "yes")
    server, client = _start_daemon(live_palace)
    real_submit = daemon.DaemonClient.submit

    def stored_then_timed_out(self, *args, **kwargs):
        real_submit(self, *args, **kwargs)
        raise daemon.DaemonError("timed out")

    try:
        transcript = tmp_path / "t.jsonl"
        _write_transcript(transcript, SAVE_INTERVAL)
        data = {"session_id": "s1", "stop_hook_active": False, "transcript_path": str(transcript)}
        state = tmp_path / "hook_state"
        state.mkdir()

        with patch("mempalace.hooks_cli._daemon_available", return_value=True):
            with patch.object(daemon.DaemonClient, "submit", stored_then_timed_out):
                assert _run_hook(hook_stop, data, state) == {}
            assert not (state / "s1_last_save").exists()

            first = [j for j in client.list_jobs(limit=50) if j["kind"] == "diary_write"]
            assert len(first) == 1
            assert client.wait(first[0]["id"], timeout=20)["state"] == "succeeded"

            # One more message, then the next Stop.
            _write_transcript(transcript, SAVE_INTERVAL + 1)
            second = _run_hook(hook_stop, data, state)

        assert "already in the palace" in second["systemMessage"]
        assert (state / "s1_last_save").read_text() == str(SAVE_INTERVAL + 1)
        assert not (state / "s1_diary_key").exists()
        diaries = [j for j in client.list_jobs(limit=50) if j["kind"] == "diary_write"]
        assert len(diaries) == 1, diaries
        assert mcp_server.tool_diary_read(agent_name="claude")["total"] == 1
    finally:
        client.shutdown()
        server.join(timeout=10)


def test_a_retry_with_the_daemon_gone_does_not_file_directly(live_palace, tmp_path, monkeypatch):
    # Stop 1's submit timed out after the daemon stored the diary; by Stop 2
    # the daemon no longer answers, so the hook routes direct. It must find
    # the stored job and not write a second diary itself.
    monkeypatch.setenv(daemon.STATE_ROOT_ENV, str(tmp_path / "state"))
    state = tmp_path / "hook_state"
    state.mkdir()
    monkeypatch.setattr(hooks_cli, "STATE_DIR", state)
    key = hooks_cli._diary_window_key("s1")
    store = daemon.QueueStore(daemon.queue_path(daemon.canonical_palace_path(live_palace)))
    job = store.enqueue("diary_write", {"entry": "x"}, dedupe_key=key, reuse_succeeded=True)

    transcript = tmp_path / "t.jsonl"
    _write_transcript(transcript, SAVE_INTERVAL)
    data = {"session_id": "s1", "stop_hook_active": False, "transcript_path": str(transcript)}
    with patch("mempalace.hooks_cli._daemon_available", return_value=False):
        out = _run_hook(hook_stop, data, state)
        assert "queued" in out["systemMessage"]
        assert mcp_server.tool_diary_read(agent_name="claude").get("total", 0) == 0

        store.finish(job.id, state="succeeded", result={"success": True, "entry_id": "e1"})
        _write_transcript(transcript, 2 * SAVE_INTERVAL)
        (state / "s1_diary_key").write_text(key)  # the same window, retried
        out = _run_hook(hook_stop, data, state)
        assert "already in the palace" in out["systemMessage"]
        assert mcp_server.tool_diary_read(agent_name="claude").get("total", 0) == 0
    assert not (state / "s1_diary_key").exists()


def test_an_old_window_key_is_not_reused(tmp_path):
    with patch("mempalace.hooks_cli.STATE_DIR", tmp_path):
        key = hooks_cli._diary_window_key("s1")
        assert hooks_cli._diary_window_key("s1") == key
        old = time.time() - hooks_cli._DIARY_WINDOW_KEY_TTL_SECONDS - 60
        os.utime(tmp_path / "s1_diary_key", (old, old))
        assert hooks_cli._diary_window_key("s1") != key


def test_a_window_key_that_cannot_be_saved_is_logged(tmp_path):
    with (
        patch("mempalace.hooks_cli.STATE_DIR", tmp_path / "missing"),
        patch("mempalace.hooks_cli._log") as log,
    ):
        assert hooks_cli._diary_window_key("s1").startswith("diary:s1:")
    assert "may file twice" in log.call_args.args[0]


def test_the_direct_path_writes_no_window_key(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMPALACE_HOOKS_DAEMON", "no")
    transcript = tmp_path / "t.jsonl"
    _write_transcript(transcript, 3)
    with (
        patch("mempalace.hooks_cli.STATE_DIR", tmp_path),
        patch("mempalace.hooks_cli._daemon_available", return_value=False),
        patch("mempalace.mcp_server.tool_diary_write", return_value={"success": True}),
    ):
        result = hooks_cli._save_diary_direct(
            str(transcript), "s1", agent_name="claude", dedupe_window=True
        )
    assert result["drawers_filed"] == 1
    assert not (tmp_path / "s1_diary_key").exists()


def test_each_save_window_gets_its_own_key(daemon_routing, tmp_path):
    transcript = tmp_path / "t.jsonl"
    _write_transcript(transcript, SAVE_INTERVAL)
    keys = []

    def submit(kind, payload, **kwargs):
        keys.append(kwargs.get("dedupe_key"))
        assert kwargs.get("reuse_succeeded") is True
        if len(keys) == 1:
            raise daemon.DaemonError("timed out")
        return {"id": f"job-{len(keys)}", "state": "succeeded", "result": {"success": True}}

    data = {"session_id": "s1", "stop_hook_active": False, "transcript_path": str(transcript)}
    with patch("mempalace.daemon.submit_job", side_effect=submit):
        _run_hook(hook_stop, data, tmp_path)  # timed out: same window
        _run_hook(hook_stop, data, tmp_path)  # filed: window closes
        _write_transcript(transcript, 2 * SAVE_INTERVAL)
        _run_hook(hook_stop, data, tmp_path)  # a new window
    assert keys[0] == keys[1] and keys[0].startswith("diary:s1:")
    assert keys[2] != keys[1]


def test_session_end_drops_the_window_key(tmp_path):
    (tmp_path / "s1_last_save").write_text("15")
    (tmp_path / "s1_diary_key").write_text("diary:s1:x")
    with patch("mempalace.hooks_cli.STATE_DIR", tmp_path):
        hooks_cli._clear_session_last_save("s1")
    assert not (tmp_path / "s1_diary_key").exists()
    assert not (tmp_path / "s1_last_save").exists()


# ── memp daemon wait --timeout ───────────────────────────────────────────


def _wait_args(palace, job_id, timeout):
    import argparse

    return argparse.Namespace(
        palace=str(palace), daemon_action="wait", backend=None, job_id=job_id, timeout=timeout
    )


@pytest.fixture
def queue(tmp_path, monkeypatch):
    monkeypatch.setenv(daemon.STATE_ROOT_ENV, str(tmp_path / "state"))
    palace = tmp_path / "palace"
    palace.mkdir()
    store = daemon.QueueStore(daemon.queue_path(daemon.canonical_palace_path(str(palace))))
    return palace, store


def test_daemon_wait_timeout_prints_the_state_and_exits_3(queue, capsys):
    from mempalace import cli

    palace, store = queue
    job = store.enqueue("batch", {"steps": []})
    client = daemon.DaemonClient.__new__(daemon.DaemonClient)
    with (
        patch("mempalace.daemon.get_client_if_running", return_value=client),
        patch.object(client, "get_job", return_value=daemon.job_to_dict(job)),
        pytest.raises(SystemExit) as exc,
    ):
        cli.cmd_daemon(_wait_args(palace, job.id, 0.1))
    assert exc.value.code == daemon.WAIT_TIMEOUT_EXIT == 3
    captured = capsys.readouterr()
    assert f"{job.id}  batch  queued" in captured.out
    assert "still queued" in captured.err


def test_daemon_wait_timeout_with_the_daemon_stopped_exits_3(queue, capsys):
    from mempalace import cli

    palace, store = queue
    job = store.enqueue("batch", {"steps": []})
    with (
        patch("mempalace.daemon.get_client_if_running", return_value=None),
        pytest.raises(SystemExit) as exc,
    ):
        cli.cmd_daemon(_wait_args(palace, job.id, 30))
    assert exc.value.code == 3
    assert "daemon is not running" in capsys.readouterr().err


def test_daemon_wait_timeout_still_reports_succeeded_and_failed(queue, capsys):
    from mempalace import cli

    palace, store = queue
    ok = store.enqueue("batch", {"steps": [1]})
    store.finish(ok.id, state="succeeded", result={"success": True})
    bad = store.enqueue("batch", {"steps": [2]})
    store.finish(bad.id, state="failed", result={"success": False, "error": "boom"})
    with patch("mempalace.daemon.get_client_if_running", return_value=None):
        cli.cmd_daemon(_wait_args(palace, ok.id, 30))  # exit 0: no SystemExit
        with pytest.raises(SystemExit) as exc:
            cli.cmd_daemon(_wait_args(palace, bad.id, 30))
    assert exc.value.code == 1
    assert "boom" in capsys.readouterr().err


def test_daemon_wait_keeps_waiting_through_a_failed_poll(queue, capsys):
    from mempalace import cli

    palace, store = queue
    job = store.enqueue("batch", {"steps": []})
    done = dict(daemon.job_to_dict(job), state="succeeded", result={"success": True})
    client = daemon.DaemonClient.__new__(daemon.DaemonClient)
    polls = [daemon.DaemonError("connection reset"), daemon.job_to_dict(job), done]

    def get_job(_job_id):
        item = polls.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    with (
        patch("mempalace.daemon.get_client_if_running", return_value=client),
        patch.object(client, "get_job", side_effect=get_job),
    ):
        cli.cmd_daemon(_wait_args(palace, job.id, 30))  # exit 0: no SystemExit
    assert polls == []


def test_daemon_wait_help_documents_the_timeout_exit(capsys):
    from mempalace import cli

    with patch("sys.argv", ["mempalace", "daemon", "wait", "--help"]), pytest.raises(SystemExit):
        cli.main()
    out = capsys.readouterr().out
    assert "--timeout" in out and "exit 3" in out
