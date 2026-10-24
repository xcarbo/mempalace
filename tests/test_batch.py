"""The batch job and `memp submit` / `memp job` (fork).

A batch is several palace writes queued as one daemon job, so an agent saving
a session (/preclean) submits once and moves on. These pin what it promises:
steps run in order under one lock, later steps can use earlier results, a
check-and-set conflict can be parked instead of lost, a busy palace re-queues
the batch with nothing written, and `memp submit` confirms without waiting.
"""

import argparse
import json
import multiprocessing
import os
import time

import pytest

from mempalace import daemon, mcp_server, service


@pytest.fixture
def batch_palace(monkeypatch, config, palace_path, kg):
    monkeypatch.setattr(mcp_server, "_config", config)
    monkeypatch.setattr(mcp_server, "_get_kg", lambda *a, **kw: kg)
    monkeypatch.setenv(daemon.STATE_ROOT_ENV, os.path.join(os.path.dirname(palace_path), "state"))
    return palace_path


def _drawer(wing="proj", room="decisions", content="We chose the shared queue."):
    return {"tool": "add-drawer", "args": {"wing": wing, "room": room, "content": content}}


# ── validation ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "payload, message",
    [
        ({}, "non-empty 'steps'"),
        ({"steps": [{"tool": "delete-drawer", "args": {"drawer_id": "x"}}]}, "not batchable"),
        ({"steps": [{"tool": "search", "args": {}}]}, "not batchable"),
        (
            {"steps": [{"tool": "kg-add", "args": {"subject": "${d1.drawer_id}"}}]},
            "names no earlier step",
        ),
        ({"steps": [{**_drawer(), "id": "a"}, {**_drawer(), "id": "a"}]}, "used twice"),
        ({"steps": [{**_drawer(), "on_conflict": "park"}]}, "needs the batch 'wing'"),
    ],
)
def test_validate_batch_refuses(payload, message):
    with pytest.raises(ValueError, match=message):
        service.validate_batch(payload)


def test_batch_tool_names_accept_the_cli_spelling():
    assert service.batch_tool_name("add-drawer") == "mempalace_add_drawer"
    assert service.batch_tool_name("kg_add") == "mempalace_kg_add"
    assert service.batch_tool_name("mempalace_update_drawer") == "mempalace_update_drawer"


# ── running ──────────────────────────────────────────────────────────────


def test_batch_runs_in_order_and_threads_ids(batch_palace, kg):
    result = service.run_batch(
        {
            "palace_path": batch_palace,
            "wing": "proj",
            "steps": [
                {**_drawer(), "id": "d1"},
                {
                    "tool": "kg-add",
                    "args": {
                        "subject": "action:${d1.drawer_id}",
                        "predicate": "belongs_to",
                        "object": "wing_proj",
                    },
                },
            ],
        }
    )
    assert result["success"] is True, result
    assert [d["tool"] for d in result["done"]] == ["mempalace_add_drawer", "mempalace_kg_add"]
    drawer_id = result["done"][0]["drawer_id"]
    assert drawer_id
    facts = kg.query_entity(f"action:{drawer_id}")
    assert any(f.get("object") == "wing_proj" for f in facts), facts


def test_batch_parks_a_conflicting_update_verbatim(batch_palace):
    added = service.run_batch({"palace_path": batch_palace, "steps": [{**_drawer(), "id": "r"}]})
    roadmap_id = added["done"][0]["drawer_id"]
    proposed = "# Roadmap\n\nThe new plan, exactly as written."

    result = service.run_batch(
        {
            "palace_path": batch_palace,
            "wing": "proj",
            "steps": [
                {
                    "tool": "update-drawer",
                    "args": {
                        "drawer_id": roadmap_id,
                        "content": proposed,
                        "if_unchanged": "0" * 64,
                    },
                    "on_conflict": "park",
                },
                _drawer(content="A later step still runs."),
            ],
        }
    )
    assert result["success"] is True, result
    assert len(result["parked"]) == 1 and len(result["done"]) == 1
    parked = mcp_server.tool_get_drawer(result["parked"][0]["parked_as"])
    assert parked["room"] == "follow-ups"
    assert "Status: open" in parked["content"]
    assert parked["content"].endswith(proposed)
    # The roadmap itself was not touched.
    assert mcp_server.tool_get_drawer(roadmap_id)["content"] == "We chose the shared queue."


def test_batch_conflict_without_park_stops_and_reports(batch_palace):
    added = service.run_batch({"palace_path": batch_palace, "steps": [_drawer()]})
    result = service.run_batch(
        {
            "palace_path": batch_palace,
            "steps": [
                {
                    "tool": "update-drawer",
                    "args": {
                        "drawer_id": added["done"][0]["drawer_id"],
                        "content": "x",
                        "if_unchanged": "0" * 64,
                    },
                },
                _drawer(content="never reached"),
            ],
        }
    )
    assert result["success"] is False
    assert "step 1" in result["error"] and result["done"] == []


def _hold_lock(palace_path, ready, release):
    from mempalace.palace import mine_palace_lock

    with mine_palace_lock(palace_path):
        open(ready, "w").close()
        for _ in range(3000):
            if os.path.exists(release):
                return
            time.sleep(0.01)


def test_busy_palace_refuses_the_whole_batch_with_nothing_written(batch_palace, tmp_path):
    ready, release = str(tmp_path / "ready"), str(tmp_path / "release")
    holder = multiprocessing.get_context("spawn").Process(
        target=_hold_lock, args=(batch_palace, ready, release)
    )
    holder.start()
    try:
        for _ in range(1000):
            if os.path.exists(ready):
                break
            time.sleep(0.01)
        assert os.path.exists(ready)
        result = service.run_batch({"palace_path": batch_palace, "steps": [_drawer()]})
    finally:
        open(release, "w").close()
        holder.join(timeout=10)
    assert result["error_class"] == daemon.LOCK_REFUSAL_ERROR_CLASS
    assert mcp_server.tool_list_drawers(wing="proj").get("drawers") in ([], None)


# ── batch robustness ─────────────────────────────────────────────────────


def _step(content="We chose the shared queue.", **kw):
    return {
        "tool": "add-drawer",
        "args": {"wing": "proj", "room": "decisions", "content": content},
        **kw,
    }


@pytest.mark.parametrize(
    "steps, message",
    [
        (
            [_step(id="d1"), {"tool": "kg-add", "args": {"subject": "${d1.drawerid}"}}],
            "no field 'drawerid'",
        ),
        (
            [
                {
                    "tool": "update-drawer",
                    "id": "r",
                    "args": {"drawer_id": "x", "content": "y", "if_unchanged": "0" * 64},
                    "on_conflict": "park",
                },
                {"tool": "kg-add", "args": {"subject": "${r.drawer_id}"}},
            ],
            "may be parked",
        ),
    ],
)
def test_validate_batch_refuses_references_it_can_prove_bad(steps, message):
    with pytest.raises(ValueError, match=message):
        service.validate_batch({"wing": "proj", "steps": steps})


def test_a_bad_reference_at_run_time_fails_with_the_done_list(batch_palace, monkeypatch):
    # Field checks are static and cover the known tools; anything they miss
    # must still fail cleanly at run time.
    monkeypatch.setattr(service, "_BATCH_REF_FIELDS", {})
    result = service.run_batch(
        {
            "palace_path": batch_palace,
            "steps": [_step(id="d1"), {"tool": "kg-add", "args": {"subject": "${d1.nope}"}}],
        }
    )
    assert result["success"] is False
    assert result["failed_step"] == 2
    assert "${d1.nope}" in result["error"]
    assert len(result["done"]) == 1 and result["done"][0]["drawer_id"]


def test_a_handler_exception_mid_batch_records_what_was_written(batch_palace, monkeypatch):
    def boom(**kwargs):
        raise RuntimeError("kg is gone")

    monkeypatch.setitem(mcp_server.TOOLS["mempalace_kg_add"], "handler", boom)
    result = service.run_batch(
        {
            "palace_path": batch_palace,
            "steps": [
                _step(id="d1"),
                {"tool": "kg-add", "args": {"subject": "action:${d1.drawer_id}"}},
                _step(content="never reached"),
            ],
        }
    )
    assert result["success"] is False
    assert result["failed_step"] == 2
    assert "RuntimeError" in result["error"] and "kg is gone" in result["error"]
    assert [d["step"] for d in result["done"]] == [1]
    assert result["done"][0]["drawer_id"]


def _conflicting_update(roadmap_id, content):
    return {
        "tool": "update-drawer",
        "args": {"drawer_id": roadmap_id, "content": content, "if_unchanged": "0" * 64},
        "on_conflict": "park",
    }


def test_a_park_whose_kg_link_fails_still_counts_as_parked(batch_palace, monkeypatch):
    added = service.run_batch({"palace_path": batch_palace, "steps": [_step()]})
    roadmap_id = added["done"][0]["drawer_id"]

    def boom(**kwargs):
        raise RuntimeError("kg is locked")

    monkeypatch.setitem(mcp_server.TOOLS["mempalace_kg_add"], "handler", boom)
    result = service.run_batch(
        {
            "palace_path": batch_palace,
            "wing": "proj",
            "steps": [_conflicting_update(roadmap_id, "new plan"), _step(content="still runs")],
        }
    )
    assert result["success"] is True, result
    assert result["parked"][0]["parked_as"]
    assert len(result["done"]) == 1


def test_a_lock_refusal_after_a_park_is_not_requeued(batch_palace, monkeypatch):
    from mempalace.palace import MineAlreadyRunning

    added = service.run_batch({"palace_path": batch_palace, "steps": [_step()]})
    roadmap_id = added["done"][0]["drawer_id"]
    real_add = mcp_server.TOOLS["mempalace_add_drawer"]["handler"]

    def add_then_refuse(**kwargs):
        if kwargs.get("room") == "decisions":
            raise MineAlreadyRunning("nested open refused")
        return real_add(**kwargs)

    monkeypatch.setitem(mcp_server.TOOLS["mempalace_add_drawer"], "handler", add_then_refuse)
    result = service.run_batch(
        {
            "palace_path": batch_palace,
            "wing": "proj",
            "steps": [_conflicting_update(roadmap_id, "new plan"), _step()],
        }
    )
    assert result["success"] is False
    assert result.get("error_class") != daemon.LOCK_REFUSAL_ERROR_CLASS, "would re-run the park"


def test_a_park_that_cannot_be_filed_fails_the_batch_and_keeps_the_text(batch_palace, monkeypatch):
    added = service.run_batch({"palace_path": batch_palace, "steps": [_step()]})
    roadmap_id = added["done"][0]["drawer_id"]
    proposed = "# Roadmap\n\nThe new plan, exactly as written."

    real_add = mcp_server.TOOLS["mempalace_add_drawer"]["handler"]

    def add_refusing_follow_ups(**kwargs):
        if kwargs.get("room") == "follow-ups":
            return {"success": False, "error": "palace busy, write not filed"}
        return real_add(**kwargs)

    monkeypatch.setitem(
        mcp_server.TOOLS["mempalace_add_drawer"], "handler", add_refusing_follow_ups
    )
    result = service.run_batch(
        {
            "palace_path": batch_palace,
            "wing": "proj",
            "steps": [
                {
                    "tool": "update-drawer",
                    "args": {
                        "drawer_id": roadmap_id,
                        "content": proposed,
                        "if_unchanged": "0" * 64,
                    },
                    "on_conflict": "park",
                },
                _step(content="never reached"),
            ],
        }
    )
    assert result["success"] is False
    assert result["failed_step"] == 1
    assert "park" in result["error"]
    assert result["parked"] == []
    assert result["unfiled_content"] == proposed
    assert result["done"] == []


# ── memp submit / memp job ───────────────────────────────────────────────


def _args(**kw):
    return argparse.Namespace(palace=None, json=False, **kw)


def test_submit_queues_without_a_daemon_and_job_reports_it(
    batch_palace, tmp_path, capsys, monkeypatch
):
    from mempalace import cli

    monkeypatch.setenv("MEMPALACE_PALACE_PATH", batch_palace)
    batch = tmp_path / "b.json"
    batch.write_text(json.dumps({"wing": "proj", "steps": [_drawer()]}))

    cli.cmd_submit(_args(file=str(batch)))
    out = capsys.readouterr()
    job_id = out.out.split()[1]
    assert out.out.startswith("queued ")
    assert "not running" in out.err

    # Sent again while it waits: the same job, not a second one.
    cli.cmd_submit(_args(file=str(batch)))
    assert capsys.readouterr().out.split()[1] == job_id

    cli.cmd_job(_args(job_id=job_id))
    assert "queued" in capsys.readouterr().out

    store = daemon.QueueStore(daemon.queue_path(daemon.canonical_palace_path(batch_palace)))
    assert store.get(job_id).kind == "batch"
    assert oct(os.stat(store.path).st_mode & 0o777) == "0o600"


def test_submit_refuses_a_bad_batch_and_queues_nothing(batch_palace, tmp_path, capsys, monkeypatch):
    from mempalace import cli

    monkeypatch.setenv("MEMPALACE_PALACE_PATH", batch_palace)
    batch = tmp_path / "b.json"
    batch.write_text(json.dumps({"steps": [{"tool": "delete-drawer", "args": {}}]}))
    with pytest.raises(SystemExit) as exc:
        cli.cmd_submit(_args(file=str(batch)))
    assert exc.value.code == 2
    assert "nothing queued" in capsys.readouterr().err
    assert not daemon.queue_path(daemon.canonical_palace_path(batch_palace)).exists()


def test_job_with_an_unknown_id_exits_1(batch_palace, capsys, monkeypatch):
    from mempalace import cli

    monkeypatch.setenv("MEMPALACE_PALACE_PATH", batch_palace)
    with pytest.raises(SystemExit) as exc:
        cli.cmd_job(_args(job_id="nope"))
    assert exc.value.code == 1


def test_running_daemon_files_a_submitted_batch(batch_palace, tmp_path, capsys, monkeypatch):
    """End to end: submit returns at once, the daemon's worker files it."""
    import threading

    from mempalace import cli

    monkeypatch.setenv("MEMPALACE_PALACE_PATH", batch_palace)
    monkeypatch.setattr(daemon, "SHUTDOWN_DRAIN_SECONDS", 1.0)
    server = threading.Thread(
        target=daemon.run_server, kwargs={"palace_path": batch_palace, "port": 0}, daemon=True
    )
    server.start()
    client = None
    for _ in range(600):
        client = daemon.get_client_if_running(batch_palace)
        if client is not None:
            break
        time.sleep(0.05)
    assert client is not None, "daemon never came up"
    try:
        batch = tmp_path / "b.json"
        batch.write_text(json.dumps({"wing": "proj", "steps": [_drawer()]}))
        started = time.monotonic()
        cli.cmd_submit(_args(file=str(batch)))
        assert time.monotonic() - started < 2, "submit waited on the write"
        job_id = capsys.readouterr().out.split()[1]
        finished = client.wait(job_id, timeout=20)
        assert finished["state"] == "succeeded", finished
        assert finished["result"]["done"][0]["drawer_id"]
    finally:
        client.shutdown()
        server.join(timeout=10)
