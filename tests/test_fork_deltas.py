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
6. Backend writes take ``short_writer_palace_lock`` (upstream: bare
   ``mine_palace_lock``, wait=0).
7. ``palace.mine_palace_lock`` removes its lock file on release (upstream leaves
   it, so every next process "reclaims" a dead holder and logs it).
8. An exchange drawer with no ``convo_chunker_version`` still counts as mined
   (upstream 3.11 counts it stale, so its first convo mine re-mines every
   transcript still on disk). #2591 was taken forward-only in 5878576.
9. Hook transcript ingest files under ``hooks_cli._ingest_wing`` (the diary's
   own derivation, so ``.palace-wing`` pins and worktrees resolve), not the
   literal ``sessions`` wing, and not upstream's second derivation.
10. ``wing_split`` takes its targets from ``hooks_cli`` (upstream: suffix match
    else snake_case, which names a project differently from the hook), and
    ``apply_split`` does not drop the source wing's hallway records (upstream
    does; here that would empty ``hallways.json``).
11. ``hooks_cli._wing_from_cwd`` walks up from a subfolder cwd to the nearest
    ``.palace-wing``/``.git`` (stopping at ``~`` and the projects root). Upstream
    names the wing after the cwd leaf, so ``repo/src`` files under ``src``.
"""

from __future__ import annotations

import ast
import os
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


def test_backend_writes_take_the_waiting_palace_lock():
    """Delta 6 — upstream's backends take a bare ``mine_palace_lock(path)``.

    A merge that restores the bare call binds our ``wait_seconds`` default of 0
    and nothing raises: the suite pins the wait to 0 anyway (conftest), so it
    stays green while every session hook that ends in the same second as
    another one loses its diary checkpoint again (hook.log 2026-09-19 11:29:36,
    "Backend open failed", 34 times before the fix).
    """
    backends = Path(inspect.getfile(searcher)).parent / "backends"
    for name, expected in (("sqlite_exact.py", 3), ("chroma.py", 1)):
        tree = ast.parse((backends / name).read_text(encoding="utf-8"))
        calls = [
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        ]
        assert "mine_palace_lock" not in calls, (
            f"FORK DELTA REVERTED: backends/{name} calls bare mine_palace_lock(); "
            "ours is short_writer_palace_lock(). Bare means wait=0: library "
            "writers (the session hooks) are refused the instant a transcript "
            "ingest holds the palace."
        )
        assert calls.count("short_writer_palace_lock") == expected, (
            f"backends/{name}: expected {expected} short_writer_palace_lock() "
            f"call sites, found {calls.count('short_writer_palace_lock')}. A new "
            "write path must take the waiting lock too; then update this count."
        )


def test_palace_lock_file_is_removed_on_release(tmp_path, monkeypatch):
    """Delta 7 — upstream's ``mine_palace_lock`` never unlinks its file.

    Every short writer (each ``memp`` invocation opens the backend under this
    lock) then leaves a file naming a PID that is dead a moment later, and the
    NEXT process's residue GC "reclaims" it with an INFO line on stderr — one
    per call, forever, in every agent's tool output and in hook.log (found
    2026-09-27, 22 lines that day in hook.log alone). A merge that drops the
    ``_cleanup_mine_lock_file`` call in the release path restores exactly that
    and nothing raises.
    """
    from mempalace.palace import mine_palace_lock

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    lock_dir = tmp_path / ".mempalace" / "locks"
    with mine_palace_lock(str(tmp_path / "palace")):
        assert list(lock_dir.glob("mine_palace_*.lock")), "lock file must exist while held"
    assert not list(lock_dir.glob("mine_palace_*.lock")), (
        "FORK DELTA REVERTED: mine_palace_lock left its lock file behind on "
        "release. Ours calls _cleanup_mine_lock_file(lock_path) in the outer "
        "finally; without it every memp call logs a lock-residue GC reclaim."
    )


def test_exchange_drawer_without_chunker_stamp_counts_as_mined(tmp_path):
    """Delta 8 — #2591 was taken forward-only (5878576).

    Upstream 3.11 stamps exchange drawers with ``convo_chunker_version`` and
    its ``palace/mined.py::_meta_is_current`` reads a missing stamp as stale.
    Pulled in by any pick that touches the mined-set checks (#2567, #2593),
    that silently turns the next ``mine --mode convos`` into a re-mine of
    every conversation file still on disk. If the re-mine has been decided
    and run, retire this delta deliberately; never just re-point it.
    """
    from mempalace.backends.base import PalaceRef
    from mempalace.backends.sqlite_exact import SQLiteExactBackend
    from mempalace.palace import file_already_mined, prefetch_mined_set

    backend = SQLiteExactBackend()
    palace = PalaceRef(id=str(tmp_path), local_path=str(tmp_path))
    col = backend.get_collection(palace=palace, collection_name="mempalace_drawers", create=True)
    src = str(tmp_path / "session.jsonl")
    col.add(
        ids=["d1"],
        documents=["> question\nanswer"],
        metadatas=[
            {
                "wing": "w",
                "room": "r",
                "source_file": src,
                "source_mtime": 1.0,
                "extract_mode": "exchange",
                "normalize_version": NORMALIZE_VERSION,
                "chunk_total": 1,
            }
        ],
        embeddings=[[1.0, 0.0]],
    )
    assert file_already_mined(col, src, extract_mode="exchange"), (
        "an unstamped exchange drawer reads as stale: the next convo mine "
        "re-mines every transcript on disk"
    )
    assert src in prefetch_mined_set(col, extract_mode="exchange")


def test_transcript_ingest_files_under_the_derived_project_wing():
    """Delta 9: the hook must not hard-code ``sessions`` or grow a second derivation."""
    ingest = inspect.getsource(hooks_cli._ingest_transcript)
    assert "_ingest_wing(" in ingest
    assert '"sessions"' not in ingest
    derive = inspect.getsource(hooks_cli._ingest_wing)
    assert "_wing_from_transcript_path(" in derive


def test_wing_split_shares_the_hooks_derivation_and_keeps_hallways():
    """Delta 10: split targets match what a new transcript of the project gets."""
    from mempalace import wing_split

    resolve = inspect.getsource(wing_split.resolve_target)
    assert "_wing_from_jsonl_cwd" in resolve and "_wing_from_transcript_path" in resolve
    assert "hallway" not in inspect.getsource(wing_split.apply_split).split('"""')[2].lower()


def test_wing_from_cwd_walks_up_to_the_project_root(tmp_path, monkeypatch):
    """Delta 11: a subfolder session belongs to the repo that contains it."""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    repo = tmp_path / "code" / "acme"
    (repo / ".git").mkdir(parents=True)
    (repo / "src").mkdir()
    assert hooks_cli._wing_from_cwd(str(repo / "src")) == "acme"


def test_chromadb_is_not_imported_at_module_load():
    """Delta 12: no module-level chromadb import on the memp / hook / server path.

    Upstream imports chromadb eagerly in backends/chroma.py, mcp_server.py and
    repair.py. That is 0.3 to 0.55 s per process on a fleet that is all
    sqlite_exact, and a merge that restores any one of the three imports passes
    every other test. Checked in a fresh interpreter, because this suite has
    long since imported chromadb itself.
    """
    import subprocess
    import sys

    # Answer through the exit code: mcp_server points stdout at stderr on import.
    probe = (
        "import sys; import mempalace.mcp_server, mempalace.repair, mempalace.palace; "
        "sys.exit(3 if 'chromadb' in sys.modules else 0)"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert out.returncode != 3, "a module-level chromadb import came back"
    assert out.returncode == 0, out.stderr[-2000:]


def test_sqlite_exact_offset_page_seeks_instead_of_skipping(tmp_path):
    """Delta 13: a deep get(limit, offset) page never puts OFFSET on the row read.

    SQL OFFSET reads every skipped row in full. prefetch_mined_set and
    prefetch_content_hashes page the whole palace that way, so a one-file
    hook mine on the 328k-drawer Mini palace took 30 minutes holding the
    palace lock (2026-10-24). The fork seeks to the page's first rowid with
    an index-only lookup instead. A merge restoring the plain OFFSET passes
    every behavioural test, since the rows returned are the same.
    """
    from mempalace.backends import PalaceRef
    from mempalace.backends.sqlite_exact import SQLiteExactBackend

    col = SQLiteExactBackend().get_collection(
        palace=PalaceRef(id=str(tmp_path), local_path=str(tmp_path)),
        collection_name="mempalace_drawers",
        create=True,
    )
    col.add(
        ids=[f"d{i}" for i in range(5)],
        documents=[f"doc {i}" for i in range(5)],
        metadatas=[{"n": i} for i in range(5)],
        embeddings=[[float(i), 1.0] for i in range(5)],
    )
    statements = []
    conn = col._handle.conn
    conn.set_trace_callback(statements.append)
    try:
        page = col.get(limit=2, offset=3, include=["metadatas"])
    finally:
        conn.set_trace_callback(None)
    assert page.ids == ["d3", "d4"]
    reads = [
        s
        for s in statements
        if "FROM documents" in s and "metadata_json" in s and "ORDER BY rowid" in s
    ]
    assert reads, statements
    assert all("OFFSET" not in s for s in reads), reads
    # Metadata-only sweeps also leave the document text unread.
    assert all("document," not in s.split("FROM")[0] for s in reads), reads


def test_convo_mine_takes_the_lock_after_its_prefetch(tmp_path, monkeypatch):
    """Delta 14: the palace-wide prefetch runs before the mine takes the lock.

    Upstream wraps the whole convo mine in mine_palace_lock, so the read-only
    prefetch (a full-palace scan) held off every other writer, and hook diary
    saves were dropped after their 10 s wait (134 in the Mini's hook.log by
    2026-10-24). Same rows filed either way, so only the order shows it.
    """
    import contextlib

    from mempalace import convo_miner

    events = []
    real_mined, real_hashes = convo_miner.prefetch_mined_set, convo_miner.prefetch_content_hashes

    @contextlib.contextmanager
    def recording_lock(palace_path, wait_seconds=0.0):
        events.append("lock")
        yield
        events.append("unlock")

    def mined(*a, **k):
        events.append("prefetch_mined")
        return real_mined(*a, **k)

    def hashes(*a, **k):
        events.append("prefetch_hashes")
        return real_hashes(*a, **k)

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(convo_miner, "mine_palace_lock", recording_lock)
    monkeypatch.setattr(convo_miner, "prefetch_mined_set", mined)
    monkeypatch.setattr(convo_miner, "prefetch_content_hashes", hashes)
    convo_dir = tmp_path / "convos"
    convo_dir.mkdir()
    (convo_dir / "chat.txt").write_text(
        "> what is the plan\nWe ship the palace fix first, then the daemon.\n\n"
        "> and after that\nThe batch job, then preclean uses it.\n"
    )
    convo_miner.mine_convos(str(convo_dir), str(tmp_path / "palace"), wing="t")

    assert "lock" in events and events[-1] == "unlock", events
    lock_at = events.index("lock")
    assert all(
        events.index(p) < lock_at for p in ("prefetch_mined", "prefetch_hashes") if p in events
    )
    assert "prefetch_mined" in events, events


def _hold_lock_until_released(palace_path, ready_flag, release_flag):
    """Child process: hold the palace write lock until told to let go."""
    import os as _os
    import time as _time

    from mempalace.palace import mine_palace_lock

    with mine_palace_lock(palace_path):
        open(ready_flag, "w").close()
        for _ in range(3000):
            if _os.path.exists(release_flag):
                return
            _time.sleep(0.01)


def test_reads_work_while_another_process_holds_the_write_lock(tmp_path, monkeypatch, capsys):
    """Delta 15: `memp search` (plain and --json) and `memp status` never wait
    on, or fail at, the palace write lock.

    Delta 3 made _open_search_collection read-only, but plain `memp search`
    opened through another door, the closets open in search_memories was
    writable, and so was `memp status`. While any mine ran, plain search and
    status raised MineAlreadyRunning and --json search waited 10 s
    (2026-10-24). Behavioural, because the fix lives in four call sites.
    """
    import multiprocessing
    import time

    from mempalace import miner, searcher
    from mempalace.palace import MineAlreadyRunning, get_backend_for_palace, get_collection

    monkeypatch.setenv("HOME", str(tmp_path))
    # Every fleet palace is sqlite_exact; the suite's default backend is chroma.
    monkeypatch.setenv("MEMPALACE_BACKEND", "sqlite_exact")
    palace = str(tmp_path / "palace")
    col = get_collection(palace, create=True)
    col.add(
        ids=["d1"], documents=["ship the palace fix first"], metadatas=[{"wing": "w", "room": "r"}]
    )
    # Drop the cached writable handle, or the reads below would reuse it and a
    # writable reopen would go unseen.
    get_backend_for_palace(palace).close_palace(palace)

    ready, release = str(tmp_path / "ready"), str(tmp_path / "release")
    holder = multiprocessing.get_context("spawn").Process(
        target=_hold_lock_until_released, args=(palace, ready, release)
    )
    holder.start()
    try:
        for _ in range(1000):
            if os.path.exists(ready):
                break
            time.sleep(0.01)
        assert os.path.exists(ready), "child never took the palace lock"

        started = time.time()
        try:
            searcher.search("palace fix", palace)
            result = searcher.search_memories("palace fix", palace)
            miner.status(palace)
        except MineAlreadyRunning as e:
            raise AssertionError(f"FORK DELTA REVERTED: a read took the write lock: {e}")
        out = capsys.readouterr().out
        assert "Error opening palace" not in out, out
        assert result.get("results"), result
        assert time.time() - started < 8, "a read waited on the write lock"
    finally:
        open(release, "w").close()
        holder.join(timeout=10)
        if holder.is_alive():
            holder.terminate()


def test_daemon_is_a_shared_queue_not_the_palace_owner():
    """Delta 16: run_server takes no lifetime mine_palace_lock.

    Upstream's daemon owns the palace while it runs, so every write outside it
    fails with MineAlreadyRunning: every memp add-drawer, cron job and :4109
    write route on a box with the daemon on. The behaviour is pinned in
    tests/test_daemon.py (test_daemon_does_not_hold_the_writer_lease); this
    catches a merge that brings the lease back before that test is even read.
    """
    from mempalace import daemon

    src = inspect.getsource(daemon.run_server)
    assert "mine_palace_lock(" not in src, "FORK DELTA REVERTED: the daemon holds the palace lock"
