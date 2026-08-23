"""The standing register of permanent fork deltas.

``xdev-patches`` diverges from ``MemPalace/mempalace`` in a handful of places
where the divergence is deliberate and *silent-revert-shaped*: a clean upstream
merge can quietly restore upstream's value, nothing raises, and the suite stays
green. Two of the 3.7.1 merge's most dangerous moments were exactly this shape.

Every entry below is one deliberate divergence plus a test that fails LOUDLY
when a merge reverts it. Adding a delta without a test here is how the last one
got through.

When an upstream merge turns one of these red, the fix is to restore OUR value —
never to re-point the assertion at upstream's.

Register:

1. ``.mcp.json`` must not exist — asserted in ``test_codex_plugin_manifest.py``
   (upstream's own test, inverted, so it stays where upstream will touch it).
2. ``NORMALIZE_VERSION = 3`` (upstream: 2).
3. ``searcher._open_search_collection`` opens ``read_only=True``.
4. ``hooks_cli._main_worktree_root`` exists and is used for wing derivation.
5. ``ids.ID_RECIPE == "v3"`` — identical to upstream today, pinned because a
   change on EITHER side moves every drawer id in a 200k-drawer palace.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

from mempalace import hooks_cli, ids, searcher
from mempalace.palace import NORMALIZE_VERSION


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_mcp_json_stays_deleted():
    """Delta 1 — the retired MCP server must not be re-registered.

    Duplicated from ``test_codex_plugin_manifest.py`` on purpose: that file is
    upstream's and a merge could replace it wholesale, taking the guard with it.
    This copy is ours and has no upstream counterpart to be overwritten by.
    """
    assert not (REPO_ROOT / ".mcp.json").exists(), (
        "FORK DELTA REVERTED: .mcp.json is back. It re-arms the MemPalace MCP "
        "server retired 2026-06-16, giving every Claude Code session opened in "
        "this repo the mcp__mempalace__* write surface against the live palace. "
        "Delete it (see 3119884)."
    )


def test_normalize_version_is_three():
    """Delta 2 — upstream is 2; a revert freezes the oversized-drawer backlog.

    The v3 re-mine that heals the 6,066 oversized drawers is keyed on this
    constant. Reverted to 2, the re-mine does nothing and reports success.
    ``palace.py`` is one of upstream's most heavily edited files, so this is a
    live merge risk, not a theoretical one.
    """
    assert NORMALIZE_VERSION == 3, (
        f"FORK DELTA REVERTED: NORMALIZE_VERSION is {NORMALIZE_VERSION}, ours is 3 "
        "(upstream ships 2, commit 3354575 bumped it). At 2 the queued v3 re-mine "
        "silently no-ops and the 6,066-drawer oversized backlog stays frozen."
    )


def test_search_collection_opens_read_only():
    """Delta 3 — reverted once already, by stage 5 of the 3.7.1 merge itself.

    ``_open_search_collection`` is the single funnel every search opens through:
    the memp CLI, the :4109 read API, the session hooks and the cron fleet.
    Since upstream ``c6e8783`` a non-read-only open runs ``_init_schema`` inside
    ``mine_palace_lock``, so without this flag every search started during a
    mine raises ``MineAlreadyRunning`` before reading a byte.

    Asserted on the source rather than by calling it, so the test needs no
    palace and stays hermetic.
    """
    src = inspect.getsource(searcher._open_search_collection)
    tree = ast.parse(src.lstrip())

    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "get_collection"
    ]
    assert calls, (
        "FORK DELTA REVERTED: _open_search_collection no longer calls "
        "get_collection — the read_only guard cannot be verified. Re-read the "
        "function before assuming this is safe."
    )

    for call in calls:
        flag = next((kw for kw in call.keywords if kw.arg == "read_only"), None)
        assert flag is not None and flag.value.value is True, (
            "FORK DELTA REVERTED: _open_search_collection must pass "
            "read_only=True. Without it, any search started while a mine holds "
            "mine_palace_lock raises MineAlreadyRunning — for the CLI, :4109, "
            "the hooks and the whole cron fleet at once."
        )


def test_worktree_wing_derivation_is_present():
    """Delta 4 — take-ours resolution in ``hooks_cli``.

    Without it a session running in a ``herdr-spawn --worktree`` checkout files
    into a wing literally named ``worktree``, because the leaf path segment is
    that string. ``.claude/worktrees/<slug>`` checkouts split a project's memory
    into a sibling wing the same way.
    """
    assert hasattr(hooks_cli, "_main_worktree_root"), (
        "FORK DELTA REVERTED: hooks_cli._main_worktree_root is gone. Every agent "
        "running in a linked worktree will file its memory into a wing named "
        "'worktree' instead of the project's."
    )

    src = inspect.getsource(hooks_cli)
    assert src.count("_main_worktree_root(") >= 2, (
        "FORK DELTA REVERTED: _main_worktree_root is defined but no longer "
        "called — wing derivation has been re-pointed at the raw cwd."
    )


def test_id_recipe_is_v3():
    """Delta 5 — pinned, not diverged.

    Identical to upstream today, so there is nothing to defend; it is here
    because a change on EITHER side re-derives every drawer id in a ~206k-drawer
    palace, and that is not something to discover after a merge.
    """
    assert ids.ID_RECIPE == "v3", (
        f"ID_RECIPE is {ids.ID_RECIPE!r}, expected 'v3'. This moves every drawer "
        "id in the live palace. If the change is intended, migrate the palace "
        "first, then update this pin deliberately."
    )
