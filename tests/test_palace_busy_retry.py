"""Short writers ride out a busy palace instead of losing the write.

Evidence (hook.log, 2026-09-19 11:29:36): two sessions ended in the same
second. The first one's transcript ingest held the palace lock; the second
one's diary checkpoint went through the *library* path
(``hooks_cli`` -> ``tool_diary_write``), which never passes through the MCP
writer-lease wait. ``sqlite_exact`` asked for the palace lock with no wait,
``MineAlreadyRunning`` was swallowed by ``_get_collection``'s generic
``except`` and the checkpoint was lost as an opaque "Backend open failed".

These tests use two real processes against a scratch palace: one holds the
palace lock the way a transcript ingest does, the other is a real diary write.
"""

import json
import os
import subprocess
import sys
import time

import pytest

from mempalace.backends.base import PalaceRef
from mempalace.backends.sqlite_exact import SQLiteExactBackend

HOLDER_CODE = """
import sys, time
from mempalace.palace import mine_palace_lock
with mine_palace_lock(sys.argv[1]):
    print("ready", flush=True)
    hold = float(sys.argv[2])
    if hold < 0:
        sys.stdin.read()
    else:
        time.sleep(hold)
"""

DIARY_CODE = """
import json, os, sys
from pathlib import Path
# Same redirect conftest does in-process: never re-download the 79 MB model.
if os.environ.get("MEMPALACE_TEST_ONNX_CACHE"):
    from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2
    ONNXMiniLM_L6_V2.DOWNLOAD_PATH = Path(os.environ["MEMPALACE_TEST_ONNX_CACHE"])
from mempalace.mcp_server import tool_diary_write
result = tool_diary_write(
    agent_name="pi", entry="checkpoint body, verbatim", topic="checkpoint", wing="cc"
)
open(sys.argv[1], "w", encoding="utf-8").write(json.dumps(result))
"""


def _env(tmp_path, palace, wait=None):
    # HOME stays the conftest session HOME: it carries the embedding-model
    # cache, and the lock dir under it is shared by both processes.
    env = os.environ.copy()
    try:
        from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2

        env["MEMPALACE_TEST_ONNX_CACHE"] = str(ONNXMiniLM_L6_V2.DOWNLOAD_PATH)
    except ImportError:
        pass
    env["MEMPALACE_PALACE_PATH"] = str(palace)
    env["MEMPALACE_BACKEND"] = "sqlite_exact"
    env["MEMPALACE_BACKEND_EXPLICIT"] = "sqlite_exact"
    env.pop("MEMPALACE_BACKEND_LOCK_WAIT_SECONDS", None)
    if wait is not None:
        env["MEMPALACE_BACKEND_LOCK_WAIT_SECONDS"] = str(wait)
    return env


def _start_holder(palace, env, hold_seconds):
    holder = subprocess.Popen(
        [sys.executable, "-c", HOLDER_CODE, str(palace), str(hold_seconds)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    assert holder.stdout is not None
    assert holder.stdout.readline().strip() == "ready"
    return holder


def _stop(holder):
    if holder.stdin is not None:
        holder.stdin.close()
    holder.wait(timeout=30)


def _run_diary(env, tmp_path):
    # mcp_server points stdout at stderr on import, so the result goes to a file.
    out = tmp_path / f"diary-result-{time.monotonic_ns()}.json"
    proc = subprocess.run(
        [sys.executable, "-c", DIARY_CODE, str(out)],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert out.exists(), f"diary writer produced no result: {proc.stderr[-2000:]!r}"
    return json.loads(out.read_text(encoding="utf-8"))


def _seed_palace(tmp_path):
    """Create the palace first so the collision is on a live one, as in prod."""
    palace = tmp_path / "palace"
    palace.mkdir()
    env = _env(tmp_path, palace)
    seeded = _run_diary(env, tmp_path)
    assert seeded.get("success"), seeded
    return palace


def test_diary_write_survives_concurrent_ingest(tmp_path):
    """The 11:29:36 collision: the diary write must land once the ingest ends."""
    palace = _seed_palace(tmp_path)
    env = _env(tmp_path, palace)
    holder = _start_holder(palace, env, hold_seconds=4)
    try:
        result = _run_diary(env, tmp_path)
    finally:
        _stop(holder)
    assert result.get("success"), result

    backend = SQLiteExactBackend()
    try:
        col = backend.get_collection(
            palace=PalaceRef(id=str(palace), local_path=str(palace)),
            collection_name="mempalace_drawers",
            create=False,
            options={"read_only": True},
        )
        assert col.get(ids=[result["entry_id"]])["ids"] == [result["entry_id"]]
    finally:
        backend.close()


def test_diary_write_fails_loudly_after_the_cap(tmp_path):
    """Past the cap the write still fails, and says who holds the palace."""
    palace = _seed_palace(tmp_path)
    env = _env(tmp_path, palace, wait=1)
    holder = _start_holder(palace, env, hold_seconds=-1)
    try:
        started = time.monotonic()
        result = _run_diary(env, tmp_path)
        elapsed = time.monotonic() - started
    finally:
        _stop(holder)
    assert not result.get("success"), result
    blob = json.dumps(result)
    assert f"PID {holder.pid}" in blob, result
    assert "Backend open failed" not in blob, result
    assert elapsed < 60


def test_backend_write_waits_for_the_holder(tmp_path, monkeypatch):
    """The wait lives in the backend, so every short writer gets it."""
    monkeypatch.setenv("MEMPALACE_BACKEND_LOCK_WAIT_SECONDS", "10")
    palace = tmp_path / "palace"
    palace.mkdir()
    backend = SQLiteExactBackend()
    col = backend.get_collection(
        palace=PalaceRef(id=str(palace), local_path=str(palace)),
        collection_name="mempalace_drawers",
        create=True,
    )
    holder = _start_holder(palace, os.environ.copy(), hold_seconds=1.5)
    try:
        col.add(ids=["waited"], documents=["doc"], metadatas=[{}], embeddings=[[1, 0]])
    finally:
        _stop(holder)
        backend.close()


def test_backend_lock_wait_default_and_parsing(monkeypatch):
    from mempalace.palace import short_writer_lock_wait_seconds

    monkeypatch.delenv("MEMPALACE_BACKEND_LOCK_WAIT_SECONDS", raising=False)
    assert short_writer_lock_wait_seconds() == 10.0
    monkeypatch.setenv("MEMPALACE_BACKEND_LOCK_WAIT_SECONDS", "0")
    assert short_writer_lock_wait_seconds() == 0.0
    monkeypatch.setenv("MEMPALACE_BACKEND_LOCK_WAIT_SECONDS", "banana")
    assert short_writer_lock_wait_seconds() == 10.0
    monkeypatch.setenv("MEMPALACE_BACKEND_LOCK_WAIT_SECONDS", "-3")
    assert short_writer_lock_wait_seconds() == 10.0
    # A hook has a 30 s budget; no env value may push the wait past it.
    monkeypatch.setenv("MEMPALACE_BACKEND_LOCK_WAIT_SECONDS", "600")
    assert short_writer_lock_wait_seconds() == 20.0


def test_palace_lock_wait_is_jittered(tmp_path, monkeypatch):
    """Two waiters must not probe in lockstep."""
    from mempalace import palace as palace_mod

    sleeps = []

    class RecordingTime:
        """Stands in for palace.py's ``time`` only: patching the real module
        would also record every other thread's sleeps in a full-suite run."""

        def __getattr__(self, name):
            return getattr(time, name)

        @staticmethod
        def sleep(seconds):
            sleeps.append(seconds)
            time.sleep(seconds)

    monkeypatch.setattr(palace_mod, "time", RecordingTime())
    target = tmp_path / "palace"
    target.mkdir()
    holder = _start_holder(target, os.environ.copy(), hold_seconds=-1)
    try:
        with pytest.raises(palace_mod.MineAlreadyRunning):
            with palace_mod.mine_palace_lock(str(target), wait_seconds=2):
                pass
    finally:
        _stop(holder)
    assert len(sleeps) >= 3
    assert len(set(sleeps)) > 1, sleeps
    assert sleeps[0] < sleeps[-2], sleeps  # backs off
    assert max(sleeps) <= palace_mod._PALACE_LOCK_POLL_SECONDS * 1.25 + 1e-9


def test_generic_open_failure_names_its_cause(monkeypatch, tmp_path):
    """A non-lock open failure must say what blocked, not just "Backend open failed".

    palace-write-health cannot tell a real outage from noise when every cause is
    flattened to the same three words (follow-up c72d994a).
    """
    import sqlite3

    from mempalace import mcp_server, palace

    def _boom(*args, **kwargs):
        raise sqlite3.OperationalError("attempt to write a readonly database")

    monkeypatch.setattr(mcp_server, "_selected_backend_name", lambda: "sqlite_exact")
    monkeypatch.setattr(mcp_server._config, "_palace_path_override", None, raising=False)
    monkeypatch.setenv("MEMPALACE_PALACE_PATH", str(tmp_path))
    monkeypatch.setattr(palace, "get_collection", _boom)
    monkeypatch.setattr(mcp_server, "_collection_cache", None)

    assert mcp_server._get_collection(create=True) is None
    result = mcp_server._collection_error_or_no_palace()
    assert result["error"] == "Backend open failed"
    assert "OperationalError" in result["details"]
    assert "attempt to write a readonly database" in result["details"]


def test_hook_log_carries_the_failure_details(monkeypatch, tmp_path):
    """hook.log is what the health agent reads, so the cause must reach it."""
    from mempalace import hooks_cli, mcp_server

    transcript = tmp_path / "session.jsonl"
    transcript.write_text(
        "".join(
            json.dumps({"message": {"role": "user", "content": f"msg {i}"}}) + "\n"
            for i in range(5)
        ),
        encoding="utf-8",
    )
    logged = []
    monkeypatch.setattr(hooks_cli, "_log", logged.append)
    monkeypatch.setattr(
        mcp_server,
        "tool_diary_write",
        lambda **kw: {
            "success": False,
            "error": "Backend open failed",
            "details": "Could not open the selected backend collection: OperationalError: disk I/O error",
        },
    )

    res = hooks_cli._save_diary_direct(str(transcript), "sess1", wing="w", agent_name="claude")

    assert res == {"count": 0}
    failed = [line for line in logged if "checkpoint failed" in line]
    assert failed and "Backend open failed" in failed[0]
    assert "OperationalError: disk I/O error" in failed[0]
