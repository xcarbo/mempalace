"""Access levels (mempalace/levels.py): stamping by folder and the read filter.

A seeded sqlite_exact palace with one owners drawer (filed from an owners
folder) and one team drawer (filed from a team folder). Every read path at
team level must return no owners drawer; an owners caller sees both. With no
levels file and no MEMPALACE_LEVEL the whole thing is off.

No real palace data: the owners word is invented ("zanzibarquill").
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from mempalace import levels

OWNERS_WORD = "zanzibarquill"


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _fresh_level_caches():
    levels._folder_cache.clear()
    levels._source_cache.clear()
    yield
    levels._folder_cache.clear()
    levels._source_cache.clear()


@pytest.fixture
def folders(tmp_path):
    code = tmp_path / "code"
    owners = code / "llcsetup"
    dataroom = code / "fpmdataroom"
    team = code / "forgepoint"
    for d in (owners, dataroom, team):
        d.mkdir(parents=True)
    return {"code": code, "owners": owners, "dataroom": dataroom, "team": team}


@pytest.fixture
def levels_file(tmp_path, folders, monkeypatch):
    path = tmp_path / "levels.json"
    path.write_text(
        json.dumps(
            {
                "default": "team",
                "owners_folders": [str(folders["owners"]), str(folders["dataroom"])],
            }
        )
    )
    monkeypatch.setenv("MEMPALACE_LEVELS_FILE", str(path))
    monkeypatch.delenv("MEMPALACE_LEVEL", raising=False)
    return path


@pytest.fixture
def palace(monkeypatch, config, palace_path, kg, levels_file, folders):
    """A sqlite_exact palace with one owners and one team drawer, filed by folder."""
    from mempalace import mcp_server

    monkeypatch.setenv("MEMPALACE_BACKEND", "sqlite_exact")
    monkeypatch.setattr(mcp_server, "_config", config)
    monkeypatch.setattr(mcp_server, "_get_kg", lambda *a, **kw: kg)
    monkeypatch.setattr(mcp_server, "_collection_cache", None)
    monkeypatch.setattr(mcp_server, "_metadata_cache", None)
    kg.access_resolver = lambda ids: levels.drawer_access_lookup(ids, palace_path)

    monkeypatch.chdir(folders["owners"])
    owners = mcp_server.tool_add_drawer(
        wing="llcsetup",
        room="legal",
        content=f"The operating agreement names the {OWNERS_WORD} clause for member buyouts.",
        source_file=str(folders["owners"] / "notes.md"),
    )
    assert owners.get("success"), owners
    monkeypatch.chdir(folders["team"])
    team = mcp_server.tool_add_drawer(
        wing="forgepoint",
        room="portal",
        content="The portal build uses SvelteKit and deploys to Fly for the whole team.",
        source_file=str(folders["team"] / "README.md"),
    )
    assert team.get("success"), team
    return {
        "owners_id": owners["drawer_id"],
        "team_id": team["drawer_id"],
        "palace_path": palace_path,
        "folders": folders,
        "kg": kg,
    }


def _raw_meta(palace_path, drawer_id):
    from mempalace.palace import get_collection

    col = levels.unwrap_collection(get_collection(palace_path, create=False))
    got = col.get(ids=[drawer_id], include=["metadatas"])
    return got["metadatas"][0] if got["ids"] else None


def _as_team(monkeypatch, palace):
    monkeypatch.chdir(palace["folders"]["team"])


def _as_owners(monkeypatch, palace):
    monkeypatch.chdir(palace["folders"]["owners"])


def _dump(obj) -> str:
    """The result as text, minus the caller's own query echoed back."""
    if isinstance(obj, dict):
        obj = {k: v for k, v in obj.items() if k not in ("query", "entity")}
    return json.dumps(obj, default=str)


# ---------------------------------------------------------------------------
# unit: parsing, folders, resolution
# ---------------------------------------------------------------------------


def test_parse_level_names_numbers_and_typos():
    assert levels.parse_level("owners") == 1
    assert levels.parse_level("TEAM") == 2
    assert levels.parse_level("3") == 3
    assert levels.parse_level(2) == 2
    for bad in ("own", "", "0", 7, True, None):
        with pytest.raises(levels.LevelError):
            levels.parse_level(bad)


def test_stored_access_missing_reads_team_and_garbage_reads_owners():
    assert levels.stored_access({}) == levels.TEAM
    assert levels.stored_access({"access": 1}) == levels.OWNERS
    assert levels.stored_access({"access": "nonsense"}) == levels.OWNERS


def test_levels_are_off_without_a_file_or_env(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMPALACE_LEVELS_FILE", str(tmp_path / "absent.json"))
    monkeypatch.delenv("MEMPALACE_LEVEL", raising=False)
    assert not levels.levels_active()
    assert not levels.filter_needed()
    assert levels.access_where() is None
    assert levels.current_level() == levels.TEAM


def test_folder_level_owners_inside_and_sibling(levels_file, folders):
    assert levels.folder_level(str(folders["owners"])) == levels.OWNERS
    assert levels.folder_level(str(folders["owners"] / "sub" / "deep")) == levels.OWNERS
    assert levels.folder_level(str(folders["dataroom"])) == levels.OWNERS
    assert levels.folder_level(str(folders["team"])) == levels.TEAM
    # A sibling whose name only starts with an owners folder's is not inside it.
    assert levels.folder_level(str(folders["code"] / "llcsetup-old")) == levels.TEAM


def test_folder_level_follows_a_symlinked_code_dir(tmp_path, levels_file, folders):
    link = tmp_path / "linked-code"
    link.symlink_to(folders["code"])
    assert levels.folder_level(str(link / "llcsetup")) == levels.OWNERS


def test_broken_levels_file_makes_every_folder_owners(tmp_path, monkeypatch, folders):
    bad = tmp_path / "levels.json"
    bad.write_text("{not json")
    monkeypatch.setenv("MEMPALACE_LEVELS_FILE", str(bad))
    assert levels.levels_active()
    assert levels.folder_level(str(folders["team"])) == levels.OWNERS


def test_inherited_owners_env_in_a_team_folder_stays_owners(levels_file, folders, monkeypatch):
    monkeypatch.chdir(folders["team"])
    assert levels.resolve_level() == levels.TEAM
    monkeypatch.setenv("MEMPALACE_LEVEL", "owners")
    assert levels.resolve_level() == levels.OWNERS
    # The env can never open an owners folder up.
    monkeypatch.setenv("MEMPALACE_LEVEL", "team")
    monkeypatch.chdir(folders["owners"])
    assert levels.resolve_level() == levels.OWNERS


def test_a_typo_in_mempalace_level_fails_loudly(levels_file, monkeypatch):
    monkeypatch.setenv("MEMPALACE_LEVEL", "ownrs")
    with pytest.raises(levels.LevelError):
        levels.current_level()


def _git(*args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def _make_repo_with_worktree(repo: Path, worktree: Path):
    _git("init", "-q", "-b", "main", cwd=repo)
    _git(
        "-c",
        "user.email=t@t",
        "-c",
        "user.name=t",
        "commit",
        "-q",
        "--allow-empty",
        "-m",
        "x",
        cwd=repo,
    )
    _git("worktree", "add", "-q", str(worktree), cwd=repo)


def test_a_worktree_of_a_team_repo_stamps_team(tmp_path, levels_file, folders):
    wt = tmp_path / "elsewhere" / "wt-team"
    _make_repo_with_worktree(folders["team"], wt)
    assert levels.folder_level(str(wt)) == levels.TEAM


def test_a_worktree_of_an_owners_repo_stamps_owners(tmp_path, levels_file, folders):
    wt = tmp_path / "elsewhere" / "wt-owners"
    _make_repo_with_worktree(folders["dataroom"], wt)
    assert levels.folder_level(str(wt)) == levels.OWNERS
    assert levels.folder_level(str(wt / "subdir")) == levels.OWNERS


def test_a_herdr_spawn_worktree_resolves_through_its_meta_repo(tmp_path, levels_file, folders):
    run = tmp_path / "herdr-spawn" / "job-1"
    (run / "worktree").mkdir(parents=True)
    (run / "meta.json").write_text(json.dumps({"repo": str(folders["owners"])}))
    assert levels.folder_level(str(run / "worktree")) == levels.OWNERS
    (run / "meta.json").write_text(json.dumps({"repo": str(folders["team"])}))
    levels._folder_cache.clear()
    assert levels.folder_level(str(run / "worktree")) == levels.TEAM


# ---------------------------------------------------------------------------
# safe by default
# ---------------------------------------------------------------------------


def test_without_levels_get_collection_is_not_wrapped(monkeypatch, palace_path, tmp_path):
    from mempalace.palace import get_collection

    monkeypatch.setenv("MEMPALACE_LEVELS_FILE", str(tmp_path / "absent.json"))
    monkeypatch.setenv("MEMPALACE_BACKEND", "sqlite_exact")
    col = get_collection(palace_path, create=True)
    assert not getattr(col, "_levels_wrapper", False)
    col.add(ids=["d1"], documents=["plain words"], metadatas=[{"wing": "w", "room": "r"}])
    meta = col.get(ids=["d1"], include=["metadatas"])["metadatas"][0]
    assert "access" not in meta, "levels off must not stamp anything"


# ---------------------------------------------------------------------------
# stamping
# ---------------------------------------------------------------------------


def test_writes_stamp_the_folder_level(palace):
    assert _raw_meta(palace["palace_path"], palace["owners_id"])["access"] == levels.OWNERS
    assert _raw_meta(palace["palace_path"], palace["team_id"])["access"] == levels.TEAM


def test_inherited_owners_env_in_a_team_folder_stamps_owners(palace, monkeypatch):
    from mempalace import mcp_server

    _as_team(monkeypatch, palace)
    monkeypatch.setenv("MEMPALACE_LEVEL", "owners")
    r = mcp_server.tool_add_drawer(
        wing="scratch", room="r", content="spawned worker notes, owners env"
    )
    assert _raw_meta(palace["palace_path"], r["drawer_id"])["access"] == levels.OWNERS


def test_a_team_writer_cannot_stamp_more_open_than_itself(palace, monkeypatch):
    from mempalace.palace import get_collection

    _as_owners(monkeypatch, palace)
    col = get_collection(palace["palace_path"], create=False)
    col.upsert(
        ids=["x1"],
        documents=["claims to be team"],
        metadatas=[{"wing": "w", "room": "r", "access": 3}],
    )
    assert _raw_meta(palace["palace_path"], "x1")["access"] == levels.OWNERS


# ---------------------------------------------------------------------------
# the read filter, every tool, team level
# ---------------------------------------------------------------------------


def test_team_search_never_returns_the_owners_drawer(palace, monkeypatch):
    from mempalace import mcp_server
    from mempalace.searcher import search_memories

    _as_team(monkeypatch, palace)
    for result in (
        mcp_server.tool_search(OWNERS_WORD),
        mcp_server.tool_search(OWNERS_WORD, limit=50),
        search_memories(OWNERS_WORD, palace["palace_path"], n_results=50),
        search_memories("operating agreement buyouts", palace["palace_path"], n_results=50),
    ):
        text = _dump(result)
        assert OWNERS_WORD not in text
        assert palace["owners_id"] not in text


def test_owners_search_finds_it(palace, monkeypatch):
    from mempalace import mcp_server

    _as_owners(monkeypatch, palace)
    assert OWNERS_WORD in _dump(mcp_server.tool_search(OWNERS_WORD))


def test_team_listings_hide_the_owners_wing(palace, monkeypatch):
    from mempalace import mcp_server

    _as_team(monkeypatch, palace)
    wings = mcp_server.tool_list_wings()
    assert "llcsetup" not in _dump(wings) and "forgepoint" in _dump(wings)
    for result in (
        mcp_server.tool_list_rooms(),
        mcp_server.tool_get_taxonomy(),
        mcp_server.tool_status(),
        mcp_server.tool_list_drawers(),
        mcp_server.tool_graph_stats(),
    ):
        assert "llcsetup" not in _dump(result), result
    _as_owners(monkeypatch, palace)
    mcp_server._metadata_cache = None
    assert "llcsetup" in _dump(mcp_server.tool_list_wings())


def test_a_hidden_id_is_not_found(palace, monkeypatch):
    from mempalace import mcp_server

    _as_team(monkeypatch, palace)
    got = mcp_server.tool_get_drawer(palace["owners_id"])
    assert OWNERS_WORD not in _dump(got)
    assert "not found" in _dump(got).lower()
    upd = mcp_server.tool_update_drawer(palace["owners_id"], content="overwritten by team")
    assert not upd.get("success"), upd
    dele = mcp_server.tool_delete_drawer(palace["owners_id"])
    assert not dele.get("success"), dele
    meta = _raw_meta(palace["palace_path"], palace["owners_id"])
    assert meta is not None and meta["access"] == levels.OWNERS


def test_team_check_duplicate_and_diary_do_not_reveal_owners_text(palace, monkeypatch):
    from mempalace import mcp_server

    _as_owners(monkeypatch, palace)
    mcp_server.tool_diary_write("claude", f"owners diary: {OWNERS_WORD} again", wing="llcsetup")
    _as_team(monkeypatch, palace)
    dup = mcp_server.tool_check_duplicate(
        f"The operating agreement names the {OWNERS_WORD} clause for member buyouts.",
        threshold=0.1,
    )
    assert OWNERS_WORD not in _dump(dup)
    assert OWNERS_WORD not in _dump(mcp_server.tool_diary_read("claude", wing=""))
    assert OWNERS_WORD not in _dump(mcp_server.tool_diary_read("claude", wing="llcsetup"))


def test_team_kg_hides_facts_from_owners_drawers(palace, monkeypatch):
    from mempalace import mcp_server

    _as_owners(monkeypatch, palace)
    mcp_server.tool_kg_add(
        "Buyout", "governed_by", OWNERS_WORD, source_drawer_id=palace["owners_id"]
    )
    mcp_server.tool_kg_add(f"action:{palace['owners_id']}", "belongs_to", "wing_llcsetup")
    # A fact with no drawer behind it, written by an owners session: owners.
    mcp_server.tool_kg_add("LLC", "banks_with", "Quillbank")
    assert OWNERS_WORD in _dump(mcp_server.tool_kg_query("Buyout"))
    _as_team(monkeypatch, palace)
    mcp_server.tool_kg_add("Portal", "built_with", "SvelteKit", source_drawer_id=palace["team_id"])
    mcp_server.tool_kg_add("Team", "meets_on", "Mondays")
    assert "Quillbank" not in _dump(mcp_server.tool_kg_query("LLC"))
    stats = _dump(mcp_server.tool_kg_stats())
    assert "banks_with" not in stats and "governed_by" not in stats and "built_with" in stats
    assert OWNERS_WORD not in _dump(mcp_server.tool_kg_query("Buyout"))
    assert "wing_llcsetup" not in _dump(mcp_server.tool_kg_timeline())
    assert OWNERS_WORD not in _dump(mcp_server.tool_kg_timeline())
    assert "SvelteKit" in _dump(mcp_server.tool_kg_query("Portal"))
    assert "Mondays" in _dump(mcp_server.tool_kg_query("Team"))


def test_team_tunnels_and_traversal_skip_owners_wings(palace, monkeypatch):
    from mempalace import mcp_server

    _as_owners(monkeypatch, palace)
    mcp_server.tool_create_tunnel(
        "llcsetup",
        "legal",
        "forgepoint",
        "portal",
        label="owners link",
        source_drawer_id=palace["owners_id"],
        target_drawer_id=palace["team_id"],
    )
    assert "llcsetup" in _dump(mcp_server.tool_list_tunnels())
    _as_team(monkeypatch, palace)
    for result in (
        mcp_server.tool_list_tunnels(),
        mcp_server.tool_list_tunnels("forgepoint"),
        mcp_server.tool_follow_tunnels("forgepoint", "portal"),
        mcp_server.tool_traverse_graph("legal"),
        mcp_server.tool_find_tunnels(),
    ):
        assert "llcsetup" not in _dump(result), result
        assert OWNERS_WORD not in _dump(result)


def test_team_wake_up_layers_hide_owners(palace, monkeypatch):
    from mempalace.layers import MemoryStack

    _as_team(monkeypatch, palace)
    stack = MemoryStack(palace_path=palace["palace_path"])
    text = stack.wake_up() + stack.recall() + _dump(stack.search(OWNERS_WORD))
    # search() echoes the query back; the owners drawer's own words must not appear.
    assert "member buyouts" not in text and "llcsetup" not in text
    assert "SvelteKit" in text  # the team drawer is there


def test_the_raw_chroma_fast_paths_decline_when_filtering(palace, monkeypatch):
    from mempalace import fastread
    from mempalace.backends.chroma import _sqlite_wing_room_counts

    _as_team(monkeypatch, palace)
    assert fastread.get_drawer(palace["palace_path"], "mempalace_drawers", "x") is None
    assert fastread.get_rows(palace["palace_path"], "mempalace_drawers", row_id="x") is None
    assert _sqlite_wing_room_counts(palace["palace_path"], "mempalace_drawers") is None


def test_owners_admin_commands_refuse_team(palace, monkeypatch, capsys):
    from mempalace.cli import _require_owners_level

    _as_team(monkeypatch, palace)
    with pytest.raises(SystemExit) as exc:
        _require_owners_level("repair")
    assert exc.value.code == 3
    _as_owners(monkeypatch, palace)
    _require_owners_level("repair")  # no exit


def test_kg_replication_dump_needs_owners(palace, monkeypatch):
    _as_team(monkeypatch, palace)
    with pytest.raises(levels.LevelDenied):
        palace["kg"].dump_rows("triples")


# ---------------------------------------------------------------------------
# memp level set
# ---------------------------------------------------------------------------


def _memp(argv, monkeypatch, palace_path):
    from mempalace import cli

    monkeypatch.setattr(sys, "argv", ["memp", "--palace", palace_path, *argv])
    from mempalace import cli_api

    # The flat api commands undo mcp_server's stdout redirect by dup2-ing the
    # process's original fd 1 back, which bypasses pytest's capture. Keep
    # them on sys.stdout here.
    monkeypatch.setattr(cli_api, "_restore_real_stdout", lambda: None)
    monkeypatch.setattr(sys, "stdout", sys.stdout)
    try:
        cli.main()
        return 0
    except SystemExit as exc:
        return exc.code or 0
    finally:
        # The flat api commands write to the real stdout; flush it so a
        # capfd read sees the output.
        for stream in (sys.stdout, sys.__stdout__, sys.stderr):
            stream.flush()


def test_team_cannot_open_an_owners_drawer_with_level_set(palace, monkeypatch, capsys):
    _as_team(monkeypatch, palace)
    rc = _memp(["level", "set", palace["owners_id"], "team"], monkeypatch, palace["palace_path"])
    assert rc == 1
    assert "not found" in capsys.readouterr().err
    assert _raw_meta(palace["palace_path"], palace["owners_id"])["access"] == levels.OWNERS


def test_team_can_make_a_team_drawer_more_private(palace, monkeypatch, capsys):
    _as_team(monkeypatch, palace)
    rc = _memp(["level", "set", palace["team_id"], "owners"], monkeypatch, palace["palace_path"])
    assert rc == 0, capsys.readouterr()
    meta = _raw_meta(palace["palace_path"], palace["team_id"])
    assert meta["access"] == levels.OWNERS and meta["access_src"] == "override"
    log = levels.override_log_path().read_text().strip().splitlines()
    assert json.loads(log[-1])["to"] == "owners"
    # Now hidden from team, and a team session cannot open it again.
    rc = _memp(["level", "set", palace["team_id"], "team"], monkeypatch, palace["palace_path"])
    assert rc == 1


def test_opening_needs_an_owners_session(palace, monkeypatch, capsys):
    _as_owners(monkeypatch, palace)
    rc = _memp(["level", "set", "--wing", "llcsetup", "team"], monkeypatch, palace["palace_path"])
    assert rc == 0, capsys.readouterr()
    meta = _raw_meta(palace["palace_path"], palace["owners_id"])
    assert meta["access"] == levels.TEAM and meta["access_src"] == "override"
    # Override survives an owners re-write of the same id.
    from mempalace.palace import get_collection

    get_collection(palace["palace_path"], create=False).update(
        ids=[palace["owners_id"]], metadatas=[{"room": "legal-2"}]
    )
    assert _raw_meta(palace["palace_path"], palace["owners_id"])["access"] == levels.TEAM


def test_team_session_opening_a_visible_extended_drawer_is_refused(palace, monkeypatch, capsys):
    from mempalace.palace import get_collection

    _as_owners(monkeypatch, palace)
    col = levels.unwrap_collection(get_collection(palace["palace_path"], create=False))
    col.update(ids=[palace["team_id"]], metadatas=[{"access": 3}])
    _as_team(monkeypatch, palace)
    rc = _memp(["level", "set", palace["team_id"], "team"], monkeypatch, palace["palace_path"])
    assert rc == 0  # 3 -> 2 is more private: allowed
    col.update(ids=[palace["team_id"]], metadatas=[{"access": 2}])
    rc = _memp(["level", "set", palace["team_id"], "extended"], monkeypatch, palace["palace_path"])
    assert rc == 3  # 2 -> 3 opens it: owners only


def test_level_set_refuses_when_levels_are_off(monkeypatch, palace_path, tmp_path, capsys):
    monkeypatch.setenv("MEMPALACE_LEVELS_FILE", str(tmp_path / "absent.json"))
    rc = _memp(["level", "set", "drawer_x", "owners"], monkeypatch, palace_path)
    assert rc == 2
    assert "levels are off" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# backfill-levels
# ---------------------------------------------------------------------------


def test_backfill_dry_run_counts_and_changes_nothing(palace, monkeypatch, capsys):
    from mempalace.palace import get_collection

    _as_owners(monkeypatch, palace)
    col = levels.unwrap_collection(get_collection(palace["palace_path"], create=False))
    # Pre-levels rows: no access at all. One transcript from the owners folder
    # (Claude encodes the folder in the projects dir), one in the owners wing
    # with no source, one team row.
    enc = str(palace["folders"]["owners"]).replace("/", "-").replace(".", "-")
    col.upsert(
        ids=["old_t1", "old_d1", "old_team"],
        documents=["old transcript chunk", "old diary chunk", "old team chunk"],
        metadatas=[
            {"wing": "sessions", "room": "r", "source_file": f"/h/.claude/projects/{enc}/a.jsonl"},
            {"wing": "llcsetup", "room": "diary"},
            {"wing": "forgepoint", "room": "r", "source_file": "/h/code/forgepoint/x.md"},
        ],
    )
    rc = _memp(["backfill-levels", "--dry-run", "--json"], monkeypatch, palace["palace_path"])
    assert rc == 0
    out = capsys.readouterr().out
    report = json.loads(out)
    drawers = report["collections"]["mempalace_drawers"]
    assert drawers["to_change"] == 2 and drawers["changed"] == 0
    assert OWNERS_WORD not in out and "old transcript chunk" not in out
    assert "access" not in _raw_meta(palace["palace_path"], "old_t1")

    rc = _memp(["backfill-levels", "--yes", "--json"], monkeypatch, palace["palace_path"])
    assert rc == 0
    capsys.readouterr()
    assert _raw_meta(palace["palace_path"], "old_t1")["access"] == levels.OWNERS
    assert _raw_meta(palace["palace_path"], "old_d1")["access_src"] == "backfill"
    assert "access" not in _raw_meta(palace["palace_path"], "old_team")


def test_backfill_needs_owners(palace, monkeypatch, capsys):
    _as_team(monkeypatch, palace)
    rc = _memp(["backfill-levels", "--dry-run"], monkeypatch, palace["palace_path"])
    assert rc == 3


# ---------------------------------------------------------------------------
# daemon jobs and hooks
# ---------------------------------------------------------------------------


def test_a_daemon_job_runs_at_its_submitters_level(palace, monkeypatch):
    from mempalace.service import execute_job

    _as_team(monkeypatch, palace)  # the daemon's own cwd is a team folder
    payload = levels.attach_level({})
    assert payload["access_level"] == levels.TEAM
    _as_owners(monkeypatch, palace)
    payload = levels.attach_level({"access_level": "team"})
    assert payload["access_level"] == levels.OWNERS  # never more open than the submitter

    _as_team(monkeypatch, palace)
    result = execute_job(
        "mcp_tool",
        {
            "name": "mempalace_add_drawer",
            "arguments": {
                "wing": "w",
                "room": "r",
                "content": "filed by an owners job via the daemon",
            },
            "access_level": levels.OWNERS,
            "palace_path": palace["palace_path"],
        },
    )
    drawer_id = (result.get("result") or result).get("drawer_id")
    assert drawer_id, result
    assert _raw_meta(palace["palace_path"], drawer_id)["access"] == levels.OWNERS
    assert levels.current_level() == levels.TEAM  # the override did not leak out


def test_a_daemon_job_with_no_level_runs_as_owners_when_levels_are_on(palace, monkeypatch):
    assert levels.job_level({}) == levels.OWNERS
    assert levels.job_level({"access_level": "garbage"}) == levels.OWNERS


def test_hook_level_comes_from_the_transcript_cwd(palace, monkeypatch, tmp_path):
    from mempalace import hooks_cli

    transcript = tmp_path / "t.jsonl"
    transcript.write_text(
        json.dumps({"type": "user", "cwd": str(palace["folders"]["owners"])}) + "\n"
    )
    _as_team(monkeypatch, palace)
    assert hooks_cli._ingest_level(str(transcript)) == levels.OWNERS
    transcript.write_text(
        json.dumps({"type": "user", "cwd": str(palace["folders"]["team"])}) + "\n"
    )
    assert hooks_cli._ingest_level(str(transcript)) == levels.TEAM
    monkeypatch.setenv("MEMPALACE_LEVEL", "owners")
    assert hooks_cli._ingest_level(str(transcript)) == levels.OWNERS
    monkeypatch.delenv("MEMPALACE_LEVEL")
    grok = f"/x/.grok/sessions/{str(palace['folders']['owners']).replace('/', '%2F')}/s1/chat_history.jsonl"
    assert hooks_cli._ingest_level(grok) == levels.OWNERS


def test_hook_pins_the_session_level_in_its_env(palace, monkeypatch, tmp_path):
    from mempalace import hooks_cli

    transcript = tmp_path / "t.jsonl"
    transcript.write_text(
        json.dumps({"type": "user", "cwd": str(palace["folders"]["owners"])}) + "\n"
    )
    _as_team(monkeypatch, palace)
    monkeypatch.delenv("MEMPALACE_LEVEL", raising=False)
    hooks_cli._pin_hook_level(
        {"session_id": "s1", "transcript_path": str(transcript)}, "claude-code"
    )
    assert os.environ.get("MEMPALACE_LEVEL") == "1"
    monkeypatch.delenv("MEMPALACE_LEVEL")


# ---------------------------------------------------------------------------
# the chroma backend gets the same filter
# ---------------------------------------------------------------------------


def test_chroma_backend_filters_too(monkeypatch, palace_path, levels_file, folders):
    from mempalace.palace import get_collection

    monkeypatch.setenv("MEMPALACE_BACKEND", "chroma")
    monkeypatch.chdir(folders["owners"])
    col = get_collection(palace_path, create=True)
    col.upsert(
        ids=["o1"],
        documents=[f"owners {OWNERS_WORD} words"],
        metadatas=[{"wing": "llcsetup", "room": "r"}],
    )
    monkeypatch.chdir(folders["team"])
    col.upsert(
        ids=["t1"], documents=["team words"], metadatas=[{"wing": "forgepoint", "room": "r"}]
    )
    col.upsert(
        ids=["u1"], documents=["unstamped old words"], metadatas=[{"wing": "old", "room": "r"}]
    )
    raw = levels.unwrap_collection(col)
    raw.update(ids=["u1"], metadatas=[{"wing": "old"}])  # still no access key? stamp-free row:
    raw.upsert(ids=["u2"], documents=["legacy row"], metadatas=[{"wing": "old", "room": "r"}])
    got = col.get(include=["metadatas"])
    assert "o1" not in got["ids"] and {"t1", "u2"} <= set(got["ids"])
    q = col.query(query_texts=[OWNERS_WORD], n_results=10, include=["documents"])
    assert "o1" not in q["ids"][0]
    lex = col.lexical_search(query=OWNERS_WORD, n_results=10)
    assert all(h.id != "o1" for h in lex.hits)
    assert col.count() == 3


# ---------------------------------------------------------------------------
# mining: drawers and closets both carry the miner's level
# ---------------------------------------------------------------------------


def _mine_owners_notes(palace, monkeypatch):
    import yaml

    from mempalace.miner import mine

    project = palace["folders"]["owners"] / "docs"
    project.mkdir()
    (project / "buyout.md").write_text(
        (
            f"# Buyout terms\n\nThe {OWNERS_WORD} clause sets the member buyout price "
            "from the last audited valuation, payable over three years.\n\n"
        )
        * 6
    )
    with open(project / "mempalace.yaml", "w") as f:
        yaml.dump({"wing": "llcsetup", "rooms": [{"name": "general", "description": "all"}]}, f)
    _as_owners(monkeypatch, palace)
    mine(str(project), palace["palace_path"])


def test_mined_drawers_and_closets_carry_the_owners_level(palace, monkeypatch):
    from mempalace.palace import get_closets_collection, get_collection

    _mine_owners_notes(palace, monkeypatch)
    drawers = levels.unwrap_collection(get_collection(palace["palace_path"], create=False))
    mined = drawers.get(where={"wing": "llcsetup"}, include=["metadatas"])
    assert len(mined["ids"]) >= 2
    assert {m.get("access") for m in mined["metadatas"]} == {levels.OWNERS}
    closets = levels.unwrap_collection(get_closets_collection(palace["palace_path"], create=False))
    cl = closets.get(include=["metadatas"])
    assert cl["ids"], "the miner wrote no closets"
    assert {m.get("access") for m in cl["metadatas"]} == {levels.OWNERS}

    from mempalace import mcp_server
    from mempalace.searcher import search_memories

    _as_team(monkeypatch, palace)
    for result in (
        search_memories(f"{OWNERS_WORD} buyout valuation", palace["palace_path"], n_results=20),
        mcp_server.tool_search("buyout price audited valuation", limit=20),
    ):
        text = _dump(result)
        assert "audited valuation" not in text and "llcsetup" not in text


def test_human_cli_commands_hide_owners(palace, monkeypatch, capfd):
    _mine_owners_notes(palace, monkeypatch)
    _as_team(monkeypatch, palace)
    capfd.readouterr()
    seen = ""
    for argv in (
        ["search", OWNERS_WORD],
        ["search", "buyout valuation", "--json"],
        ["status"],
        ["wake-up"],
        ["list-wings"],
        ["list-drawers"],
        ["get-drawer", "--drawer-id", palace["owners_id"]],
    ):
        _memp(argv, monkeypatch, palace["palace_path"])
        out = capfd.readouterr()
        text = out.out + out.err
        seen += text
        # get-drawer echoes the requested id (which names its wing) in "not found".
        assert "llcsetup" not in text.replace(palace["owners_id"], ""), (argv, text)
        assert "member buyouts" not in text and "audited valuation" not in text, (argv, text)
    assert "forgepoint" in seen  # the commands did print the team drawer
    assert "Drawer not found" in seen
    _as_owners(monkeypatch, palace)
    rc = _memp(["list-wings"], monkeypatch, palace["palace_path"])
    out = capfd.readouterr()
    assert "llcsetup" in out.out + out.err, (rc, out)


def test_level_set_by_wing_and_room_and_by_ids_file(palace, monkeypatch, capsys, tmp_path):
    from mempalace import mcp_server

    _as_team(monkeypatch, palace)
    legal = mcp_server.tool_add_drawer(
        wing="forgepoint", room="legal", content="LLC terms filed in forgepoint"
    )
    sec = mcp_server.tool_add_drawer(
        wing="sec-2026-11-03", room="findings", content="open box weakness"
    )
    rc = _memp(
        ["level", "set", "--wing", "forgepoint", "--room", "legal", "owners"],
        monkeypatch,
        palace["palace_path"],
    )
    assert rc == 0, capsys.readouterr()
    assert _raw_meta(palace["palace_path"], legal["drawer_id"])["access"] == levels.OWNERS
    assert _raw_meta(palace["palace_path"], palace["team_id"])["access"] == levels.TEAM
    ids = tmp_path / "ids.txt"
    ids.write_text(f"# sweep\n{sec['drawer_id']}\ndrawer_does_not_exist\n")
    rc = _memp(
        ["level", "set", "--ids-file", str(ids), "owners"], monkeypatch, palace["palace_path"]
    )
    err = capsys.readouterr().err
    assert rc == 0 and "not found, skipped: 1" in err
    assert _raw_meta(palace["palace_path"], sec["drawer_id"])["access"] == levels.OWNERS
    # A team re-mine/re-write of an overridden drawer cannot make it team again:
    # it is hidden, so the write is skipped.
    assert "LLC terms" not in _dump(mcp_server.tool_search("LLC terms forgepoint"))


# ---------------------------------------------------------------------------
# review fixes: the pin caps reads; broken link fails private
# ---------------------------------------------------------------------------


def test_a_team_pin_still_reads_team_inside_an_owners_folder(palace, monkeypatch):
    from mempalace import mcp_server

    monkeypatch.setenv("MEMPALACE_LEVEL", "team")
    _as_owners(monkeypatch, palace)
    assert levels.current_level() == levels.TEAM
    assert levels.stamp_level() == levels.OWNERS
    assert "member buyouts" not in _dump(mcp_server.tool_search(OWNERS_WORD))
    assert "llcsetup" not in _dump(mcp_server.tool_list_wings())
    # What it writes from the owners folder is still stamped owners.
    r = mcp_server.tool_add_drawer(wing="w", room="r", content="written from inside llcsetup")
    assert r.get("success"), r
    assert _raw_meta(palace["palace_path"], r["drawer_id"])["access"] == levels.OWNERS


def test_a_job_keeps_its_submitters_read_cap(palace, monkeypatch):
    monkeypatch.setenv("MEMPALACE_LEVEL", "team")
    _as_owners(monkeypatch, palace)
    payload = levels.attach_level({})
    assert payload["access_level"] == levels.OWNERS
    assert payload["access_read_level"] == levels.TEAM
    assert levels.job_read_level(payload) == levels.TEAM
    with levels.caller_level(levels.job_read_level(payload), stamp=levels.job_level(payload)):
        assert levels.current_level() == levels.TEAM and levels.stamp_level() == levels.OWNERS


def test_a_broken_levels_symlink_fails_private(tmp_path, monkeypatch, folders):
    link = tmp_path / "levels.json"
    link.symlink_to(tmp_path / "moved-away" / "palace-levels.json")
    monkeypatch.setenv("MEMPALACE_LEVELS_FILE", str(link))
    monkeypatch.delenv("MEMPALACE_LEVEL", raising=False)
    assert levels.levels_active()
    monkeypatch.chdir(folders["team"])
    assert levels.current_level() == levels.OWNERS  # reads everything...
    assert levels.stamp_level() == levels.OWNERS  # ...and stamps owners: nothing leaks


def test_writes_from_an_owners_folder_under_a_team_pin_succeed(palace, monkeypatch):
    from mempalace import mcp_server

    monkeypatch.setenv("MEMPALACE_LEVEL", "team")
    _as_owners(monkeypatch, palace)
    d = mcp_server.tool_diary_write("claude", "diary from inside llcsetup under a team pin")
    assert d.get("success"), d
    u = mcp_server.tool_update_drawer(palace["team_id"], content="rewritten from inside llcsetup")
    assert u.get("success"), u
    # The rewrite carries owners words now, so it is owners and gone from team view.
    assert _raw_meta(palace["palace_path"], palace["team_id"])["access"] == levels.OWNERS


# ---------------------------------------------------------------------------
# review fixes (fork): source folder, transcripts, odd stamps, jobs, tunnels
# ---------------------------------------------------------------------------


def test_mining_an_owners_folder_from_a_team_cwd_stamps_owners(palace, monkeypatch):
    import yaml

    from mempalace.miner import mine
    from mempalace.palace import get_closets_collection, get_collection

    project = palace["folders"]["owners"] / "more"
    project.mkdir()
    (project / "terms.md").write_text(("# Terms\n\nCapital calls are due in thirty days.\n\n") * 8)
    with open(project / "mempalace.yaml", "w") as f:
        yaml.dump({"wing": "llcsetup", "rooms": [{"name": "general", "description": "all"}]}, f)
    _as_team(monkeypatch, palace)  # a cron job or Gen in a team folder
    mine(str(project), palace["palace_path"])
    drawers = levels.unwrap_collection(get_collection(palace["palace_path"], create=False))
    metas = drawers.get(where={"wing": "llcsetup"}, include=["metadatas"])["metadatas"]
    mined = [m for m in metas if "terms.md" in str(m.get("source_file"))]
    assert mined and {m["access"] for m in mined} == {levels.OWNERS}
    closets = levels.unwrap_collection(get_closets_collection(palace["palace_path"], create=False))
    cl = closets.get(include=["metadatas"])["metadatas"]
    assert cl and {m["access"] for m in cl if "terms.md" in str(m.get("source_file"))} == {
        levels.OWNERS
    }


def test_a_transcript_that_visits_an_owners_folder_is_owners(palace, monkeypatch, tmp_path):
    from mempalace import hooks_cli

    t = tmp_path / "t.jsonl"
    lines = [{"type": "user", "cwd": str(palace["folders"]["team"])}] * 300
    lines.append({"type": "user", "cwd": str(palace["folders"]["owners"])})
    t.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    _as_team(monkeypatch, palace)
    assert hooks_cli._ingest_level(str(t)) == levels.OWNERS
    enc = str(palace["folders"]["team"]).replace("/", "-").replace(".", "-")
    proj = tmp_path / ".claude" / "projects" / enc
    proj.mkdir(parents=True)
    (proj / "s.jsonl").write_text(t.read_text())
    assert levels.source_level(str(proj / "s.jsonl")) == levels.OWNERS


def test_a_string_stamp_is_hidden_from_team(palace, monkeypatch):
    from mempalace.palace import get_collection

    raw = levels.unwrap_collection(get_collection(palace["palace_path"], create=False))
    raw.upsert(
        ids=["odd1", "odd2"],
        documents=["odd stamp owners words", "odd stamp string one"],
        metadatas=[
            {"wing": "w", "room": "r", "access": "owners"},
            {"wing": "w", "room": "r", "access": "1"},
        ],
    )
    _as_team(monkeypatch, palace)
    col = get_collection(palace["palace_path"], create=False)
    assert not col.get(ids=["odd1", "odd2"], include=[])["ids"]
    assert "odd stamp" not in _dump(col.get(include=["documents"]))


def test_a_job_with_no_level_cannot_touch_owners_drawers(palace, monkeypatch):
    from mempalace.service import execute_job

    _as_team(monkeypatch, palace)
    result = execute_job(
        "mcp_tool",
        {
            "name": "mempalace_delete_drawer",
            "arguments": {"drawer_id": palace["owners_id"]},
            "palace_path": palace["palace_path"],
        },
    )
    assert not result.get("success", True) or result.get("error"), result
    assert _raw_meta(palace["palace_path"], palace["owners_id"]) is not None
    result = execute_job(
        "mcp_tool",
        {
            "name": "mempalace_add_drawer",
            "arguments": {"wing": "w", "room": "r", "content": "levelless job words"},
            "palace_path": palace["palace_path"],
        },
    )
    drawer_id = (result.get("result") or result).get("drawer_id")
    assert _raw_meta(palace["palace_path"], drawer_id)["access"] == levels.OWNERS


def test_new_chunks_of_an_overridden_source_keep_the_override(palace, monkeypatch, capsys):
    from mempalace.palace import get_collection

    _as_team(monkeypatch, palace)
    src = str(palace["folders"]["team"] / "plan.md")
    col = get_collection(palace["palace_path"], create=False)
    col.upsert(
        ids=["p0"],
        documents=["plan part zero"],
        metadatas=[{"wing": "w", "room": "r", "source_file": src}],
    )
    assert (
        _memp(["level", "set", "--source", src, "owners"], monkeypatch, palace["palace_path"]) == 0
    )
    capsys.readouterr()
    # A re-mine by a team process adds chunk 1: it must not come out team.
    col.upsert(
        ids=["p1"],
        documents=["plan part one"],
        metadatas=[{"wing": "w", "room": "r", "source_file": src}],
    )
    assert _raw_meta(palace["palace_path"], "p1")["access"] == levels.OWNERS


def test_tunnels_from_hidden_drawers_or_owners_sessions_stay_hidden(palace, monkeypatch):
    from mempalace import mcp_server

    _as_team(monkeypatch, palace)
    team_two = mcp_server.tool_add_drawer(
        wing="forgepoint", room="ops", content="ops runbook for the team"
    )
    _as_owners(monkeypatch, palace)
    mcp_server.tool_add_drawer(
        wing="forgepoint", room="acquisition", content="buyout of a vendor, owners"
    )
    # Same wing as team drawers, but the room holds only owners drawers.
    mcp_server.tool_create_tunnel(
        "forgepoint", "acquisition", "forgepoint", "ops", label="owners label A"
    )
    _as_team(monkeypatch, palace)
    mcp_server.tool_create_tunnel(
        "forgepoint",
        "ops",
        "forgepoint",
        "portal",
        label="team label B",
        source_drawer_id=team_two["drawer_id"],
    )
    listed = _dump(mcp_server.tool_list_tunnels())
    assert "owners label A" not in listed and "acquisition" not in listed
    assert "team label B" in listed


def test_hub_forwarding_is_off_with_levels(palace, monkeypatch):
    from mempalace import cli, mcp_server

    assert cli._hub_forward_disabled()
    assert mcp_server._hub_proxy_target() is None


def test_chroma_raw_bm25_fallback_filters(monkeypatch, palace_path, levels_file, folders):
    from mempalace.palace import get_collection
    from mempalace.searcher import _bm25_only_via_sqlite

    monkeypatch.setenv("MEMPALACE_BACKEND", "chroma")
    monkeypatch.chdir(folders["owners"])
    col = get_collection(palace_path, create=True)
    col.upsert(
        ids=["o1"],
        documents=[f"owners {OWNERS_WORD} words"],
        metadatas=[{"wing": "llcsetup", "room": "r"}],
    )
    monkeypatch.chdir(folders["team"])
    col.upsert(
        ids=["t1"],
        documents=[f"team words near {OWNERS_WORD}x"],
        metadatas=[{"wing": "forgepoint", "room": "r"}],
    )
    out = _bm25_only_via_sqlite(OWNERS_WORD, palace_path, n_results=10)
    assert "llcsetup" not in _dump(out)
    monkeypatch.chdir(folders["owners"])
    assert "llcsetup" in _dump(_bm25_only_via_sqlite(OWNERS_WORD, palace_path, n_results=10))
