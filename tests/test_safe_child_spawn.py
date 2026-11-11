"""Hook-spawned Python children must not import from the session's folder.

Upstream #2959 (951b14bf, the child-interpreter part). Claude Code runs a hook
in the session's working directory, and the hook spawns ``python -m mempalace
mine ...`` there. ``-m`` puts the working directory first on ``sys.path``, so a
session sitting in a checkout of mempalace itself (any worktree of this repo)
ran *that checkout's* code against the live palace, and any folder holding a
``mempalace/`` package or a ``sqlite3.py`` ran it as the user.

The child now runs with ``PYTHONSAFEPATH=1`` (Python 3.11+), which keeps the
working directory off ``sys.path``. The argv stays ``-m mempalace mine ...`` so
process listings and the hook's per-target PID slots are unchanged.
"""

import os
import sys
import time

import pytest

import mempalace.hooks_cli as hooks_cli


@pytest.fixture
def hook_state(tmp_path, monkeypatch):
    state_dir = tmp_path / "home" / ".mempalace" / "hook_state"
    state_dir.mkdir(parents=True)
    monkeypatch.setattr(hooks_cli, "PALACE_ROOT", state_dir.parent)
    monkeypatch.setattr(hooks_cli, "STATE_DIR", state_dir)
    monkeypatch.setattr(hooks_cli, "_MINE_PID_DIR", state_dir / "mine_pids")
    monkeypatch.setattr(hooks_cli, "_state_dir_initialized", False)
    return state_dir


def _plant_package(folder, marker):
    pkg = folder / "mempalace"
    pkg.mkdir(parents=True)
    body = f"open({str(marker)!r}, 'w').write('hijacked')\n"
    (pkg / "__init__.py").write_text(body, encoding="utf-8")
    (pkg / "__main__.py").write_text(body, encoding="utf-8")


def _wait_for_exit(pid_file, timeout=60.0):
    """Reap the hook's child (this test process is its parent)."""
    pid = int(pid_file.read_text().split()[0])
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        done, _status = os.waitpid(pid, os.WNOHANG)
        if done == pid:
            return
        time.sleep(0.1)
    raise AssertionError(f"spawned child {pid} still running after {timeout}s")


@pytest.mark.skipif(sys.version_info < (3, 11), reason="PYTHONSAFEPATH needs Python 3.11+")
@pytest.mark.skipif(os.name == "nt", reason="POSIX process check")
def test_spawned_mine_does_not_import_from_session_folder(tmp_path, hook_state, monkeypatch):
    session = tmp_path / "session"
    marker = tmp_path / "hijacked.txt"
    _plant_package(session, marker)
    monkeypatch.chdir(session)
    monkeypatch.delenv("PYTHONSAFEPATH", raising=False)
    # The real child must touch nothing: a scratch HOME and a command that
    # only prints the version.
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    cmd = [sys.executable, "-m", "mempalace", "--version"]
    assert hooks_cli._spawn_mine(cmd) is True
    _wait_for_exit(hooks_cli._pid_file_for_cmd(cmd))

    assert not marker.exists(), "the mine child imported mempalace from the session folder"


def test_spawn_mine_child_env_sets_safe_path(hook_state, monkeypatch):
    seen = {}

    class _Proc:
        pid = 999999

    def fake_popen(cmd, **kwargs):
        seen.update(kwargs)
        return _Proc()

    monkeypatch.setattr(hooks_cli.subprocess, "Popen", fake_popen)
    assert hooks_cli._spawn_mine([sys.executable, "-m", "mempalace", "mine", "/x"]) is True
    assert seen["env"]["PYTHONSAFEPATH"] == "1"
    assert seen["env"][hooks_cli._MINE_PID_FILE_ENV]


def test_daemon_start_child_env_sets_safe_path(tmp_path, monkeypatch):
    from mempalace import daemon

    seen = {}

    class _Stop(Exception):
        pass

    def fake_popen(cmd, **kwargs):
        seen["cmd"] = cmd
        seen.update(kwargs)
        raise _Stop()

    monkeypatch.setenv("MEMPALACE_DAEMON_STATE_ROOT", str(tmp_path / "daemon-state"))
    monkeypatch.setattr(daemon, "get_client_if_running", lambda *_a, **_k: None)
    monkeypatch.setattr(daemon.subprocess, "Popen", fake_popen)
    with pytest.raises(_Stop):
        daemon.start_daemon(str(tmp_path / "palace"))
    assert seen["cmd"][1:3] == ["-m", "mempalace.daemon"]
    assert seen["env"]["PYTHONSAFEPATH"] == "1"
