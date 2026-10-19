"""`mempalace wings split` and the per-project wing the hooks now file transcripts under."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from mempalace import hooks_cli, wing_split
from mempalace.backends import PalaceRef
from mempalace.backends.sqlite_exact import SQLiteExactBackend
from mempalace.config import MempalaceConfig

CLAUDE = "/Users/me/.claude/projects/{key}/{sid}.jsonl"


def _claude(key: str, sid: str = "s1") -> str:
    return CLAUDE.format(key=key, sid=sid)


def _collection(tmp_path, name="mempalace_drawers"):
    backend = SQLiteExactBackend()
    palace = PalaceRef(id=str(tmp_path), local_path=str(tmp_path))
    return backend.get_collection(palace=palace, collection_name=name, create=True)


def _fill(col, rows):
    """rows: [(id, wing, source_file, document)]"""
    col.add(
        ids=[r[0] for r in rows],
        documents=[r[3] for r in rows],
        metadatas=[
            {"wing": r[1], "room": "general", "source_file": r[2], "extra": "kept"} for r in rows
        ],
        embeddings=[[1.0, float(i)] for i in range(len(rows))],
    )


def _transcript(path: Path, cwd: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"type": "user", "cwd": cwd}) + "\n" * 2 + "x" * 200)
    return str(path)


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: h))
    return h


# ── project keys ─────────────────────────────────────────────────────────────


def test_project_key_claude_nested_subagent_and_codex_unreadable():
    assert wing_split.project_key(_claude("-Users-me-dev-acme")) == "-Users-me-dev-acme"
    nested = "/u/.claude/projects/-Users-me-dev-acme/s1/subagents/a.jsonl"
    assert wing_split.project_key(nested) == "-Users-me-dev-acme"
    assert wing_split.project_key("/u/.codex/sessions/2026/rollout-x.jsonl") is None
    assert wing_split.project_key("/private/tmp/notes.txt") is None
    assert wing_split.project_key(None) is None


def test_project_key_codex_reads_cwd(tmp_path):
    rollout = tmp_path / ".codex" / "sessions" / "r.jsonl"
    rollout.parent.mkdir(parents=True)
    rollout.write_text(json.dumps({"payload": {"cwd": "/work/acme"}}) + "\n")
    assert wing_split.project_key(str(rollout)) == "acme"


def test_project_key_grok_and_pi_normalise_to_the_claude_form():
    grok = "/Users/me/.grok/sessions/%2FUsers%2Fme%2Fdev%2Facme/abc/chat_history.jsonl"
    pi = "/Users/me/.pi/agent/sessions/--Users-me-dev-acme--/2026_x.jsonl"
    assert wing_split.project_key(grok) == "-Users-me-dev-acme"
    assert wing_split.project_key(pi) == "-Users-me-dev-acme"


# ── target resolution: same derivation as the hook ───────────────────────────


def test_resolve_target_uses_the_cwd_and_honours_a_palace_wing_pin(tmp_path, home):
    repo = tmp_path / "code" / "utilities" / "torq"
    repo.mkdir(parents=True)
    (repo / ".palace-wing").write_text("torq-terminal\n")
    sample = _transcript(tmp_path / "t.jsonl", str(repo))
    assert wing_split.resolve_target("-x-utilities-torq", [sample]) == ("torq-terminal", "cwd")


def test_resolve_target_collapses_a_linked_worktree_to_the_main_project(tmp_path, home):
    main = tmp_path / "code" / "acme"
    (main / ".git" / "worktrees" / "wt1").mkdir(parents=True)
    wt = tmp_path / "spawn" / "worktree"
    wt.mkdir(parents=True)
    (wt / ".git").write_text(f"gitdir: {main}/.git/worktrees/wt1\n")
    sample = _transcript(tmp_path / "t.jsonl", str(wt))
    assert wing_split.resolve_target("-spawn-worktree", [sample]) == ("acme", "cwd")


def test_resolve_target_sends_home_directory_sessions_to_the_workstation_wing(tmp_path, home):
    sample = _transcript(tmp_path / "t.jsonl", str(home))
    assert wing_split.resolve_target("-x", [sample]) == (hooks_cli._workstation_wing(), "home")
    # With the transcript gone, the encoded home directory says the same.
    encoded_home = str(home).replace("/", "-")
    assert wing_split.resolve_target(encoded_home, []) == (hooks_cli._workstation_wing(), "home")


def test_resolve_target_falls_back_to_the_encoded_folder_when_transcripts_are_gone():
    target, how = wing_split.resolve_target("-Users-me-dev-customer-portal", ["/nope/a.jsonl"])
    assert (target, how) == ("customer-portal", "path")


def test_resolve_target_reads_a_sibling_transcript_when_the_sample_is_gone(tmp_path, home):
    repo = tmp_path / "code" / "acme"
    repo.mkdir(parents=True)
    keydir = tmp_path / ".claude" / "projects" / "-code-acme"
    _transcript(keydir / "other.jsonl", str(repo))
    gone = str(keydir / "gone.jsonl")
    assert wing_split.resolve_target("-code-acme", [gone]) == ("acme", "cwd")


def test_a_derivation_that_names_no_project_stays_in_sessions():
    assert wing_split.resolve_target("-", []) == ("sessions", "none")


def test_snap_to_existing_prefers_the_longest_suffix():
    known = {"pangaea", "gamedev-pangaea", "other"}
    assert wing_split.snap_to_existing("volumes-xdata-codexd-gamedev-pangaea", known) == (
        "gamedev-pangaea"
    )
    assert wing_split.snap_to_existing("volumes-xdata-nothing", known) is None


# ── plan + apply against a real sqlite_exact collection ──────────────────────


def _palace(tmp_path):
    col = _collection(tmp_path)
    _fill(
        col,
        [
            ("drawer_sessions_general_a", "sessions", _claude("-Users-me-dev-acme"), "alpha one"),
            ("drawer_sessions_general_b", "sessions", _claude("-Users-me-dev-acme", "s2"), "beta"),
            ("drawer_sessions_general_c", "sessions", _claude("-Users-me-dev-beta"), "gamma"),
            ("drawer_sessions_general_d", "sessions", "/private/tmp/agta.txt", "delta"),
            ("drawer_other_general_e", "other", _claude("-Users-me-dev-acme"), "epsilon"),
        ],
    )
    return col


def test_plan_counts_targets_and_leaves_unattributable_drawers(tmp_path):
    col = _palace(tmp_path)
    plan = wing_split.plan_split(col, "sessions", {"acme", "other"})
    assert plan["unresolved"] == 1
    assert {k: (v["target"], v["drawers"], v["existing"]) for k, v in plan["projects"].items()} == {
        "-Users-me-dev-acme": ("acme", 2, True),
        "-Users-me-dev-beta": ("beta", 1, False),
    }
    assert wing_split.plan_targets(plan) == {"acme": 2, "beta": 1}


def test_apply_moves_only_the_wing_and_never_the_id_document_or_index(tmp_path):
    col = _palace(tmp_path)
    before = col.get(ids=["drawer_sessions_general_a"], include=["documents", "metadatas"])
    plan = wing_split.plan_split(col, "sessions", set())
    result = wing_split.apply_split(col, plan)
    assert result["moved"] == 3
    assert result["per_target"] == {"acme": 2, "beta": 1}

    moved = col.get(ids=["drawer_sessions_general_a"], include=["documents", "metadatas"])
    assert moved["ids"] == ["drawer_sessions_general_a"]  # id keeps its old wing prefix
    assert moved["documents"] == before["documents"]
    meta = moved["metadatas"][0]
    assert meta["wing"] == "acme"
    assert {k: v for k, v in meta.items() if k not in ("wing", "last_modified")} == {
        k: v for k, v in before["metadatas"][0].items() if k != "wing"
    }
    # The unattributable drawer and the other wing's drawer did not move.
    assert col.get(ids=["drawer_sessions_general_d"], include=["metadatas"])["metadatas"][0][
        "wing"
    ] == ("sessions")
    assert col.get(ids=["drawer_other_general_e"], include=["metadatas"])["metadatas"][0][
        "wing"
    ] == ("other")
    assert col.count() == 5

    # FTS still finds the moved drawer by its text.
    db = sqlite3.connect(tmp_path / "sqlite_exact.sqlite3")
    assert db.execute("select count(*) from docs_fts where docs_fts match 'alpha'").fetchone()[
        0
    ] == (1)


def test_apply_twice_changes_nothing_the_second_time(tmp_path):
    col = _palace(tmp_path)
    plan = wing_split.plan_split(col, "sessions", set())
    wing_split.apply_split(col, plan)
    assert wing_split.apply_split(col, plan)["moved"] == 0


def test_an_interrupted_apply_resumes_with_the_remainder(tmp_path, monkeypatch):
    col = _palace(tmp_path)
    plan = wing_split.plan_split(col, "sessions", set())
    monkeypatch.setattr(wing_split, "_UPDATE_BATCH", 1)
    real_update = col.update
    calls = []

    def flaky(**kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return real_update(**kwargs)

    monkeypatch.setattr(col, "update", flaky)
    with pytest.raises(KeyboardInterrupt):
        wing_split.apply_split(col, plan)
    monkeypatch.setattr(col, "update", real_update)
    assert wing_split.apply_split(col, plan)["moved"] == 2  # 3 total, 1 already moved
    assert col.get(where={"wing": "sessions"}, include=["metadatas"])["ids"] == [
        "drawer_sessions_general_d"
    ]


def test_apply_rekeys_the_closets_the_same_way(tmp_path):
    col = _palace(tmp_path)
    closets = _collection(tmp_path, "mempalace_closets")
    _fill(closets, [("closet_1", "sessions", _claude("-Users-me-dev-acme"), "index line")])
    plan = wing_split.plan_split(col, "sessions", set())
    result = wing_split.apply_split(col, plan, closets_col=closets)
    assert result["closets_moved"] == 1
    assert closets.get(ids=["closet_1"], include=["metadatas"])["metadatas"][0]["wing"] == "acme"


def test_apply_leaves_the_hallways_file_alone(tmp_path):
    config = MempalaceConfig(palace_path=str(tmp_path))
    hallway_file = Path(config.hallway_file)
    hallway_file.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"schema_version": 1, "hallways": [{"wing": "sessions", "id": 1}]})
    hallway_file.write_text(payload)
    col = _palace(tmp_path)
    wing_split.apply_split(col, wing_split.plan_split(col, "sessions", set()))
    assert hallway_file.read_text() == payload


# ── plan file, pending marker ────────────────────────────────────────────────


def test_plan_dir_override_keeps_the_palace_untouched(tmp_path):
    config = MempalaceConfig(palace_path=str(tmp_path / "palace"))
    plan = {"wing": "sessions", "projects": {"k": {"target": "acme", "drawers": 1}}}
    elsewhere = tmp_path / "scratch"
    path = wing_split.save_split_plan(config, plan, str(elsewhere))
    assert Path(path).parent == elsewhere
    assert not (tmp_path / "palace").exists()
    assert wing_split.load_split_plan(config, "sessions", str(elsewhere))["wing"] == "sessions"


def test_load_plan_refuses_another_wings_plan_and_bad_targets(tmp_path):
    config = MempalaceConfig(palace_path=str(tmp_path))
    wing_split.save_split_plan(
        config, {"wing": "sessions", "projects": {"k": {"target": "acme", "drawers": 1}}}
    )
    with pytest.raises(ValueError, match="not 'other'"):
        path = wing_split.split_plan_path(config, "sessions")
        Path(wing_split.split_plan_path(config, "other")).write_text(Path(path).read_text())
        wing_split.load_split_plan(config, "other")
    wing_split.save_split_plan(
        config, {"wing": "sessions", "projects": {"k": {"target": "a/b", "drawers": 1}}}
    )
    with pytest.raises(ValueError):
        wing_split.load_split_plan(config, "sessions")


def test_pending_marker_pins_the_plan_targets(tmp_path):
    config = MempalaceConfig(palace_path=str(tmp_path))
    plan = {"wing": "sessions", "projects": {"k": {"target": "acme", "drawers": 1}}}
    wing_split.write_split_pending(config, "sessions", plan)
    wing_split.assert_split_pending_matches_plan(config, "sessions", plan)
    plan["projects"]["k"]["target"] = "elsewhere"
    with pytest.raises(ValueError, match="changed since this split started"):
        wing_split.assert_split_pending_matches_plan(config, "sessions", plan)


# ── hook: which wing a new transcript is filed under ─────────────────────────


def test_ingest_wing_is_the_project_for_a_repo_root_and_a_subfolder(tmp_path, home):
    repo = tmp_path / "code" / "acme"
    (repo / ".git").mkdir(parents=True)
    (repo / "sub" / "deep").mkdir(parents=True)
    assert hooks_cli._ingest_wing(_transcript(tmp_path / "a.jsonl", str(repo))) == "acme"
    # A subfolder session walks up to the repo that contains it.
    assert hooks_cli._ingest_wing(_transcript(tmp_path / "b.jsonl", str(repo / "sub"))) == "acme"
    deep = _transcript(tmp_path / "c.jsonl", str(repo / "sub" / "deep"))
    assert hooks_cli._ingest_wing(deep) == "acme"


def test_subfolder_without_a_repo_keeps_the_cwd_leaf(tmp_path, home):
    plain = tmp_path / "scratch" / "notes"
    plain.mkdir(parents=True)
    assert hooks_cli._ingest_wing(_transcript(tmp_path / "a.jsonl", str(plain))) == "notes"


def test_walk_up_honours_the_nearest_palace_wing_pin(tmp_path, home):
    repo = tmp_path / "code" / "utilities" / "torq"
    (repo / ".git").mkdir(parents=True)
    (repo / ".palace-wing").write_text("torq-terminal\n")
    nested = repo / "pkg" / "inner"
    nested.mkdir(parents=True)
    assert hooks_cli._ingest_wing(_transcript(tmp_path / "a.jsonl", str(nested))) == "torq-terminal"
    (repo / "pkg" / ".palace-wing").write_text("torq-pkg\n")
    assert hooks_cli._ingest_wing(_transcript(tmp_path / "b.jsonl", str(nested))) == "torq-pkg"


def test_walk_up_stops_at_home_and_the_projects_root(tmp_path, home):
    # A repo AT ~ (dotfiles) or at the projects root must not swallow its children.
    (home / ".git").mkdir()
    (home / "code").mkdir()
    (home / "code" / ".git").mkdir()
    loose = home / "code" / "loose"
    loose.mkdir()
    assert hooks_cli._ingest_wing(_transcript(tmp_path / "a.jsonl", str(loose))) == "loose"
    docs = home / "Documents"
    docs.mkdir()
    assert hooks_cli._ingest_wing(_transcript(tmp_path / "b.jsonl", str(docs))) == "documents"


def test_walk_up_from_a_worktree_subfolder_collapses_to_the_main_project(tmp_path, home):
    main = tmp_path / "code" / "acme"
    (main / ".git" / "worktrees" / "w").mkdir(parents=True)
    wt = tmp_path / "spawn" / "worktree"
    (wt / "src").mkdir(parents=True)
    (wt / ".git").write_text(f"gitdir: {main}/.git/worktrees/w\n")
    assert hooks_cli._ingest_wing(_transcript(tmp_path / "a.jsonl", str(wt / "src"))) == "acme"


def test_ingest_wing_honours_the_palace_wing_pin_and_a_linked_worktree(tmp_path, home):
    main = tmp_path / "code" / "utilities" / "torq"
    (main / ".git" / "worktrees" / "w").mkdir(parents=True)
    (main / ".palace-wing").write_text("torq-terminal\n")
    wt = tmp_path / "spawn" / "worktree"
    wt.mkdir(parents=True)
    (wt / ".git").write_text(f"gitdir: {main}/.git/worktrees/w\n")
    assert hooks_cli._ingest_wing(_transcript(tmp_path / "a.jsonl", str(wt))) == "torq-terminal"


def test_ingest_wing_sends_home_to_the_workstation_wing_and_tmp_to_its_leaf(tmp_path, home):
    assert (
        hooks_cli._ingest_wing(_transcript(tmp_path / "a.jsonl", str(home)))
        == hooks_cli._workstation_wing()
    )
    assert hooks_cli._ingest_wing(_transcript(tmp_path / "b.jsonl", "/tmp")) == "tmp"


def test_ingest_wing_without_a_cwd_falls_back_to_sessions(tmp_path):
    bare = tmp_path / "x.jsonl"
    bare.write_text("{}\n" * 5)
    assert hooks_cli._ingest_wing(str(bare)) == "sessions"
    assert hooks_cli._ingest_wing(str(tmp_path / "missing.jsonl")) == "sessions"


# ── CLI: plan is the default, --yes applies ──────────────────────────────────


def test_cli_plans_by_default_and_applies_only_with_yes(tmp_path, monkeypatch, capsys):
    from mempalace import cli

    palace = tmp_path / "palace"
    palace.mkdir()
    _fill(
        _collection(palace),
        [("drawer_sessions_general_a", "sessions", _claude("-Users-me-dev-acme"), "alpha")],
    )
    plans = tmp_path / "plans"
    base = ["mempalace", "--palace", str(palace), "wings", "split", "--wing", "sessions"]
    base += ["--plan-dir", str(plans)]

    def wing_of_a():
        return _collection(palace).get(ids=["drawer_sessions_general_a"], include=["metadatas"])[
            "metadatas"
        ][0]["wing"]

    monkeypatch.setattr("sys.argv", base)
    cli.main()
    assert "acme" in capsys.readouterr().out
    assert (plans / "split-sessions.json").is_file()
    assert wing_of_a() == "sessions"

    monkeypatch.setattr("sys.argv", base + ["--yes"])
    cli.main()
    assert wing_of_a() == "acme"
    assert not (plans / "split-sessions.pending").exists()


# ── gone worktrees and the allow-list ────────────────────────────────────────


def test_collapse_worktree_key_finds_the_repo_either_side_of_the_marker():
    known = {"xcmail", "mempalace", "mini-utils"}
    assert wing_split.collapse_worktree_key(
        "-Users-me-utilities-xcmail--worktrees-polish", known
    ) == ("xcmail")
    assert wing_split.collapse_worktree_key(
        "-Users-me--herdr-worktrees-mempalace-memaudit2", known
    ) == ("mempalace")
    assert wing_split.collapse_worktree_key(
        "-Users-me--herdr-worktrees-mini-utils-x-eval", known
    ) == ("mini-utils")
    # A branch name never becomes a wing.
    assert (
        wing_split.collapse_worktree_key("-Users-me--herdr-worktrees-unknown-thing", known) is None
    )
    assert wing_split.collapse_worktree_key("-Users-me-dev-acme", known) is None


def test_filter_plan_keeps_only_allowed_targets_and_counts_the_rest_as_staying():
    plan = {
        "wing": "sessions",
        "unresolved": 1,
        "projects": {
            "a": {"target": "acme", "drawers": 3},
            "b": {"target": "junk", "drawers": 2},
        },
    }
    out = wing_split.filter_plan(plan, {"acme"})
    assert list(out["projects"]) == ["a"]
    assert out["unresolved"] == 3


# ── frozen wings: homes that moved to another palace ─────────────────────────


def test_frozen_wings_never_become_plan_targets_and_a_stale_plan_is_refused(tmp_path, monkeypatch):
    frozen = tmp_path / "frozen.txt"
    frozen.write_text("# moved to pcore-hel1\nacme  # Phil\n")
    monkeypatch.setenv("MEMPALACE_FROZEN_WINGS", str(frozen))
    (tmp_path / "p").mkdir()
    col = _palace(tmp_path / "p")
    plan = wing_split.plan_split(col, "sessions", set())
    assert [e["target"] for e in plan["projects"].values()] == ["beta"]
    assert plan["unresolved"] == 1 + 2  # the unattributable drawer + acme's two

    config = MempalaceConfig(palace_path=str(tmp_path))
    wing_split.save_split_plan(
        config, {"wing": "sessions", "projects": {"k": {"target": "acme", "drawers": 1}}}
    )
    with pytest.raises(ValueError, match="frozen wings"):
        wing_split.load_split_plan(config, "sessions")
    assert wing_split.filter_plan({"projects": {}, "unresolved": 0}, {"acme"})["projects"] == {}
