"""A hook-spawned mine removes its own PID slot on every exit path.

Upstream #2917 (f36b1b69, the PID-slot part). The hook claims
``hook_state/mine_pids/mine_<sha>.pid`` and passes it to the child in
``MEMPALACE_MINE_PID_FILE``. The project miner removed the slot in a
``finally`` inside ``_mine_impl``, but:

* the conversation miner (every transcript ingest the hooks spawn) never
  removed it, so one dead slot was left per transcript (963 on the Mini,
  2026-11-11), and a slot whose PID the OS later reuses reads as a live mine
  for that transcript until the 2 h timeout;
* the project miner skipped its cleanup when the palace lock was busy, since
  ``MineAlreadyRunning`` is raised before ``_mine_impl`` runs.
"""

import os

import pytest

from mempalace import convo_miner, miner
from mempalace.palace import MineAlreadyRunning

EXCHANGES = (
    "> What is memory?\nMemory is persistence.\n\n"
    "> Why does it matter?\nIt enables continuity.\n\n"
    "> How do we build it?\nWith structured storage.\n"
)


@pytest.fixture
def own_slot(tmp_path, monkeypatch):
    slot = tmp_path / "mine_pids" / "mine_0123456789abcdef.pid"
    slot.parent.mkdir()
    slot.write_text(f"{os.getpid()} 1700000000")
    monkeypatch.setenv("MEMPALACE_MINE_PID_FILE", str(slot))
    return slot


def test_convo_mine_removes_its_slot(tmp_path, own_slot):
    convos = tmp_path / "convos"
    convos.mkdir()
    (convos / "chat.txt").write_text(EXCHANGES, encoding="utf-8")
    convo_miner.mine_convos(str(convos), str(tmp_path / "palace"), wing="test_convos")
    assert not own_slot.exists()


def test_convo_mine_removes_its_slot_when_the_palace_is_busy(tmp_path, own_slot, monkeypatch):
    def busy(_palace_path, *a, **k):
        raise MineAlreadyRunning("held by another mine")

    monkeypatch.setattr(convo_miner, "mine_palace_lock", busy)
    convos = tmp_path / "convos"
    convos.mkdir()
    (convos / "chat.txt").write_text(EXCHANGES, encoding="utf-8")
    with pytest.raises(MineAlreadyRunning):
        convo_miner.mine_convos(str(convos), str(tmp_path / "palace"), wing="test_convos")
    assert not own_slot.exists()


def test_project_mine_removes_its_slot_when_the_palace_is_busy(tmp_path, own_slot, monkeypatch):
    def busy(_palace_path, *a, **k):
        raise MineAlreadyRunning("held by another mine")

    monkeypatch.setattr(miner, "mine_palace_lock", busy)
    project = tmp_path / "project"
    project.mkdir()
    (project / "notes.md").write_text("# Notes\n\nSome text.\n", encoding="utf-8")
    with pytest.raises(MineAlreadyRunning):
        miner.mine(str(project), str(tmp_path / "palace"))
    assert not own_slot.exists()


def test_slot_of_another_process_is_left_alone(tmp_path, monkeypatch):
    slot = tmp_path / "mine_0123456789abcdef.pid"
    slot.write_text(f"{os.getpid() + 1} 1700000000")
    monkeypatch.setenv("MEMPALACE_MINE_PID_FILE", str(slot))
    convos = tmp_path / "convos"
    convos.mkdir()
    (convos / "chat.txt").write_text(EXCHANGES, encoding="utf-8")
    convo_miner.mine_convos(str(convos), str(tmp_path / "palace"), wing="test_convos")
    assert slot.exists()


def _cli_mine_args(tmp_path):
    import argparse

    return argparse.Namespace(
        dir=str(tmp_path / "convos"),
        palace=str(tmp_path / "palace"),
        mode="convos",
        wing=None,
        agent="mempalace",
        limit=0,
        dry_run=False,
        no_gitignore=False,
        include_ignored=[],
        extract="exchange",
    )


def test_cli_mine_removes_its_slot_when_forwarded_to_a_hub(tmp_path, own_slot, monkeypatch):
    """``mempalace mine`` can return before mine()/mine_convos() run."""
    from mempalace import cli

    monkeypatch.setattr(cli, "_mine_args_forwardable", lambda *_a, **_k: True)
    monkeypatch.setattr(cli, "_forward_mine_to_hub", lambda *_a, **_k: True)
    cli.cmd_mine(_cli_mine_args(tmp_path))
    assert not own_slot.exists()


def test_cli_mine_removes_its_slot_when_it_fails_early(tmp_path, own_slot, monkeypatch):
    from mempalace import cli

    def boom(*_a, **_k):
        raise RuntimeError("failed before mining")

    monkeypatch.setattr(cli, "_mine_args_forwardable", boom)
    with pytest.raises(RuntimeError):
        cli.cmd_mine(_cli_mine_args(tmp_path))
    assert not own_slot.exists()
