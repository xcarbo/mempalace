"""Shared service operations used by daemon-backed entry points.

The MCP server remains the owner of MCP transport details. This module owns the
small, transport-neutral execution surface the daemon needs: classify known
tools and execute durable background jobs without printing directly to the
caller's terminal.
"""

from __future__ import annotations

import contextlib
import io
import functools
import json
import os
import re
import sys
from typing import Any

from .config import MempalaceConfig

_EXPLICIT_BACKEND_ENV = "MEMPALACE_BACKEND_EXPLICIT"
_PALACE_PATH_ENV = "MEMPALACE_PALACE_PATH"


def _restores_palace_env(fn):
    """Restore ``MEMPALACE_PALACE_PATH`` after the call that stamped it.

    Each ``run_*`` entrypoint sets the variable so downstream config resolution sees THIS
    call's palace. In the daemon that costs nothing — one call, one process. In any process
    that makes two calls it leaks: ``config.py`` reads this variable with priority OVER the
    config file, so the first call's palace silently becomes the second caller's, and a
    tmpdir palace that has since been removed reads back as ``PalaceNotFoundError`` from an
    unrelated code path.

    ``daemon.py`` already saves and restores exactly these keys around its own dispatch.
    This gives the service entrypoints the same discipline.
    """

    @functools.wraps(fn)
    def _wrapped(*args, **kwargs):
        previous = os.environ.get(_PALACE_PATH_ENV)
        try:
            return fn(*args, **kwargs)
        finally:
            if previous is None:
                os.environ.pop(_PALACE_PATH_ENV, None)
            else:
                os.environ[_PALACE_PATH_ENV] = previous

    return _wrapped


_BACKEND_ENV = "MEMPALACE_BACKEND"
# Env vars a job may mutate via _apply_backend / palace_path injection. They are
# snapshotted per job and restored afterward so a job that switches the backend
# (e.g. qdrant) cannot poison every later job in the same daemon process —
# including mcp_tool jobs, which read MempalaceConfig (and thus the leaked env).
_PER_JOB_ENV = (_PALACE_PATH_ENV, _BACKEND_ENV, _EXPLICIT_BACKEND_ENV)


READ_TOOLS = frozenset(
    {
        "mempalace_status",
        "mempalace_list_wings",
        "mempalace_list_rooms",
        "mempalace_get_taxonomy",
        "mempalace_get_aaak_spec",
        "mempalace_traverse",
        "mempalace_find_tunnels",
        "mempalace_graph_stats",
        "mempalace_list_tunnels",
        "mempalace_list_hallways",
        "mempalace_follow_tunnels",
        "mempalace_search",
        "mempalace_check_duplicate",
        "mempalace_get_drawer",
        "mempalace_list_drawers",
        "mempalace_diary_read",
        "mempalace_memories_filed_away",
        "mempalace_kg_query",
        "mempalace_kg_stats",
        "mempalace_kg_timeline",
    }
)

WRITE_TOOLS = frozenset(
    {
        "mempalace_add_drawer",
        "mempalace_checkpoint",
        "mempalace_delete_by_source",
        "mempalace_delete_drawer",
        "mempalace_update_drawer",
        "mempalace_diary_write",
        "mempalace_kg_add",
        "mempalace_kg_invalidate",
        "mempalace_create_tunnel",
        "mempalace_delete_tunnel",
        "mempalace_delete_hallway",
        "mempalace_hook_settings",
    }
)

MAINTENANCE_TOOLS = frozenset({"mempalace_mine", "mempalace_sync", "mempalace_reconnect"})


def classify_tool(name: str) -> str:
    """Return ``read``, ``write``, ``maintenance``, or ``unknown`` for an MCP tool."""
    if name in READ_TOOLS:
        return "read"
    if name in WRITE_TOOLS:
        return "write"
    if name in MAINTENANCE_TOOLS:
        return "maintenance"
    return "unknown"


def _apply_backend(backend: str | None) -> None:
    if not backend:
        return
    backend_name = str(backend).strip().lower()
    from .backends import get_backend_class

    get_backend_class(backend_name)
    os.environ[_EXPLICIT_BACKEND_ENV] = backend_name
    os.environ[_BACKEND_ENV] = backend_name


def _capture(fn):
    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        result = fn()
    return result, stdout.getvalue(), stderr.getvalue()


def execute_job(kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Execute one daemon job and return a JSON-serializable result."""

    def _run():
        if kind == "mine":
            return run_mine(payload)
        if kind == "sync":
            return run_sync(payload)
        if kind == "diary_write":
            return run_diary_write(payload)
        if kind == "mcp_tool":
            return run_mcp_tool(payload)
        if kind == "batch":
            return run_batch(payload)
        return {"success": False, "error": f"unknown daemon job kind: {kind}", "exit_code": 2}

    # Per-job env isolation: snapshot the backend/palace env vars and restore
    # them after the job so one job's _apply_backend / palace_path injection
    # can't leak into the next job in the same long-lived process.
    saved_env = {key: os.environ.get(key) for key in _PER_JOB_ENV}
    try:
        result, stdout, stderr = _capture(_run)
    finally:
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    if result is None:
        result = {}
    if not isinstance(result, dict):
        result = {"success": True, "value": result}
    result.setdefault("success", True)
    result.setdefault("exit_code", 0 if result.get("success") else 1)
    if stdout:
        result["stdout"] = stdout
    if stderr:
        result["stderr"] = stderr
    return result


@_restores_palace_env
def run_mine(payload: dict[str, Any]) -> dict[str, Any]:
    """Run the same mine operation as the CLI, without daemon transport concerns."""
    palace_path = os.path.abspath(
        os.path.expanduser(payload.get("palace_path") or MempalaceConfig().palace_path)
    )
    os.environ["MEMPALACE_PALACE_PATH"] = palace_path
    _apply_backend(payload.get("backend"))

    source = payload.get("source") or payload.get("dir")
    source_adapter = payload.get("source_adapter")
    mode = payload.get("mode") or "projects"
    wing = payload.get("wing")
    agent = payload.get("agent") or "mempalace"
    limit = int(payload.get("limit") or 0)
    dry_run = bool(payload.get("dry_run"))

    if payload.get("redetect_origin") and not source_adapter:
        from .cli import _run_pass_zero

        _run_pass_zero(project_dir=source, palace_dir=palace_path, llm_provider=None)

    from .daemon import LOCK_REFUSAL_ERROR_CLASS
    from .palace import MineAlreadyRunning, MineValidationError

    try:
        if source_adapter:
            from .cli import mine_source_adapter

            mine_source_adapter(
                source_name=source_adapter,
                source_path=source,
                palace_path=palace_path,
                dry_run=dry_run,
            )
        elif mode == "convos":
            from .convo_miner import mine_convos

            mine_convos(
                convo_dir=source,
                palace_path=palace_path,
                wing=wing,
                agent=agent,
                limit=limit,
                dry_run=dry_run,
                extract_mode=payload.get("extract") or "exchange",
            )
        elif mode == "extract":
            from .format_miner import mine_formats

            mine_formats(
                format_dir=source,
                palace_path=palace_path,
                wing=wing,
                agent=agent,
                limit=limit,
                dry_run=dry_run,
            )
        elif mode == "projects":
            include_ignored = payload.get("include_ignored") or []
            from .miner import mine

            mine(
                project_dir=source,
                palace_path=palace_path,
                wing_override=wing,
                agent=agent,
                limit=limit,
                dry_run=dry_run,
                respect_gitignore=not bool(payload.get("no_gitignore")),
                include_ignored=include_ignored,
                max_chunks_per_file=payload.get("max_chunks_per_file"),
            )
        else:
            return {"success": False, "error": f"invalid mine mode: {mode}", "exit_code": 2}
    except MineAlreadyRunning as exc:
        return {
            "success": False,
            "error": str(exc),
            "error_class": LOCK_REFUSAL_ERROR_CLASS,
            "exit_code": 1,
        }
    except MineValidationError as exc:
        return {
            "success": False,
            "error": str(exc),
            "error_class": "MineValidationError",
            "exit_code": 1,
        }
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        return {
            "success": code == 0,
            "error": str(exc),
            "error_class": "SystemExit",
            "exit_code": code,
        }
    except Exception as exc:
        return {"success": False, "error": f"mine failed: {exc}", "exit_code": 1}

    result_mode = "source" if source_adapter else mode
    return {
        "success": True,
        "kind": "mine",
        "mode": result_mode,
        "dry_run": dry_run,
        "exit_code": 0,
    }


@_restores_palace_env
def run_sync(payload: dict[str, Any]) -> dict[str, Any]:
    """Run sync and render the same operator-facing summary shape as the CLI."""
    palace_path = os.path.abspath(
        os.path.expanduser(payload.get("palace_path") or MempalaceConfig().palace_path)
    )
    os.environ["MEMPALACE_PALACE_PATH"] = palace_path
    _apply_backend(payload.get("backend"))

    from .backends import detect_backend_for_path
    from .daemon import LOCK_REFUSAL_ERROR_CLASS
    from .palace import MineAlreadyRunning, _backend_artifact_label, resolve_backend_name

    if not os.path.isdir(palace_path):
        print(f"\n  No palace found at {palace_path}")
        return {"success": True, "exit_code": 0}

    try:
        backend_name = resolve_backend_name(palace_path)
    except Exception as exc:
        return {
            "success": False,
            "error": f"Could not resolve palace backend: {exc}",
            "exit_code": 1,
        }

    if detect_backend_for_path(palace_path) is None:
        print(
            f"\n  Palace dir at {palace_path} exists but has no "
            f"{_backend_artifact_label(backend_name)} yet."
        )
        print("  Run: mempalace mine <dir>")
        return {"success": True, "exit_code": 0}

    project_dirs = []
    if payload.get("dir"):
        project_dirs.append(os.path.expanduser(str(payload["dir"])))
    project_dirs.extend(os.path.expanduser(str(root)) for root in payload.get("root") or [])
    project_dirs = project_dirs or None
    dry_run = bool(payload.get("dry_run", True))

    print(f"\n{'=' * 55}")
    print("  MemPalace Sync -- Gitignore-aware drawer prune")
    print(f"{'=' * 55}")
    print(f"  Palace:   {palace_path}")
    if payload.get("wing"):
        print(f"  Wing:     {payload['wing']}")
    if project_dirs:
        for project_dir in project_dirs:
            print(f"  Project:  {project_dir}")
    print(
        "  Mode:     DRY RUN (no deletions)" if dry_run else "  Mode:     APPLY (deleting drawers)"
    )
    print(f"{'-' * 55}\n")

    try:
        from .sync import sync_palace
        from .wal import _wal_log

        report = sync_palace(
            palace_path=palace_path,
            project_dirs=project_dirs,
            wing=payload.get("wing"),
            dry_run=dry_run,
            wal_log=_wal_log,
        )
    except MineAlreadyRunning as exc:
        return {
            "success": False,
            "error": str(exc),
            "error_class": LOCK_REFUSAL_ERROR_CLASS,
            "exit_code": 1,
        }
    except ValueError as exc:
        return {"success": False, "error": str(exc), "exit_code": 2}
    except Exception as exc:
        return {"success": False, "error": f"sync failed: {exc}", "exit_code": 1}

    removed_suffix = "(would remove)" if dry_run else "(removed)"
    print(f"  Scanned:        {report['scanned']}")
    print(f"  Kept:           {report['kept']}")
    print(f"  Gitignored:     {report['gitignored']}  {removed_suffix}")
    print(f"  Missing:        {report['missing']}  {removed_suffix}")
    print(f"  Unresolved:     {report['unresolved']}  (kept)")
    print(f"  No source:      {report['no_source']}  (kept)")
    print(f"  Out of scope:   {report['out_of_scope']}  (kept)")

    by_source = report.get("by_source") or {}
    if by_source:
        top = sorted(by_source.items(), key=lambda kv: -kv[1])[:5]
        label = "Top sources to remove" if dry_run else "Top sources removed"
        print(f"\n  {label}:")
        for src, n in top:
            print(f"    {src}  ({n})")

    if report["unresolved"]:
        print("\n  Unresolved drawers are kept: nothing here could show their source file is gone.")
        unresolved_sources = report.get("unresolved_by_source") or {}
        if unresolved_sources:
            top = sorted(unresolved_sources.items(), key=lambda kv: -kv[1])[:5]
            for src, n in top:
                print(f"    {src}  ({n})")
            rest = len(unresolved_sources) - len(top)
            if rest:
                print(f"    and {rest} more source file(s)")

    if dry_run:
        if report["gitignored"] + report["missing"] > 0:
            print("\n  Re-run with --apply to commit these deletions.")
    else:
        print(
            f"\n  Removed {report['removed_drawers']} drawers, {report['removed_closets']} closets."
        )

    print(f"\n{'=' * 55}\n")
    return {"success": True, "report": report, "exit_code": 0}


def _lock_refusal_result(exc: Exception) -> dict[str, Any]:
    """A write refused the palace lock: the daemon re-queues it unspent.

    Fork delta 16: the daemon no longer owns the lock for its lifetime, so a
    diary or tool write can meet another writer mid-call. ``mine_palace_lock``
    wraps the write itself, so nothing was filed and a retry is safe.
    """
    from .daemon import LOCK_REFUSAL_ERROR_CLASS

    return {
        "success": False,
        "error": f"palace busy, write deferred: {exc}",
        "error_class": LOCK_REFUSAL_ERROR_CLASS,
        "exit_code": 1,
    }


@_restores_palace_env
def run_diary_write(payload: dict[str, Any]) -> dict[str, Any]:
    palace_path = payload.get("palace_path")
    if palace_path:
        os.environ["MEMPALACE_PALACE_PATH"] = os.path.abspath(os.path.expanduser(palace_path))
    _apply_backend(payload.get("backend"))

    from .mcp_server import tool_diary_write
    from .palace import MineAlreadyRunning

    try:
        result = tool_diary_write(
            agent_name=payload.get("agent_name") or "mempalace",
            entry=payload.get("entry") or "",
            topic=payload.get("topic") or "general",
            wing=payload.get("wing") or "",
        )
    except MineAlreadyRunning as exc:
        return _lock_refusal_result(exc)
    result.setdefault("exit_code", 0 if result.get("success") else 1)
    return result


def run_mcp_tool(payload: dict[str, Any]) -> dict[str, Any]:
    """Execute an MCP tool by name over the daemon queue.

    The daemon is a durable, retried write surface — not a general MCP transport.
    Restrict ``mcp_tool`` to write-classified tools only: read tools would
    exfiltrate verbatim palace content into the queue DB and the job result
    (stored world-readable-by-default without the perms fix, and returned over
    /jobs), and maintenance tools already have their own dedicated kinds
    (mine/sync). No internal caller currently uses ``mcp_tool``; this allowlist
    bounds the blast radius of the generic escape hatch.
    """
    name = payload.get("name")
    arguments = payload.get("arguments") or {}
    if not isinstance(arguments, dict):
        return {"success": False, "error": "arguments must be an object", "exit_code": 2}
    classification = classify_tool(name) if name else "unknown"
    if classification != "write":
        return {
            "success": False,
            "error": f"daemon mcp_tool only accepts write tools; {name!r} is {classification}",
            "exit_code": 2,
        }
    from .mcp_server import TOOLS
    from .palace import MineAlreadyRunning

    if name not in TOOLS:
        return {"success": False, "error": f"unknown MCP tool: {name}", "exit_code": 2}
    try:
        result = TOOLS[name]["handler"](**arguments)
    except MineAlreadyRunning as exc:
        return _lock_refusal_result(exc)
    if isinstance(result, dict):
        # Several write tools signal failure with a bare {"error": ...} and no
        # explicit success flag (e.g. tool_create_tunnel / tool_delete_tunnel
        # validation paths). Infer failure from the "error" key so the daemon
        # does not persist a failed write as succeeded with exit_code 0.
        if "success" not in result:
            result["success"] = "error" not in result
        result.setdefault("exit_code", 0 if result.get("success") else 1)
        return result
    return {"success": True, "value": result, "exit_code": 0}


def print_job_result(result: dict[str, Any]) -> int:
    """Replay captured daemon job output and return the intended process exit code."""
    stdout = result.get("stdout")
    stderr = result.get("stderr")
    if stdout:
        print(stdout, end="")
    if stderr:
        print(stderr, end="", file=sys.stderr)
    if not result.get("success", True) and result.get("error") and not stderr:
        print(f"mempalace: {result['error']}", file=sys.stderr)
    return int(result.get("exit_code", 0 if result.get("success", True) else 1) or 0)


# ── batch: one job, several writes (fork) ─────────────────────────────────
#
# A save such as /preclean is several writes in order: drawers, a KG triple
# for each new drawer id, then a check-and-set roadmap update. A batch carries
# them as one job so the agent submits once and moves on. Deletes are not
# batchable: a fire-and-forget queue is the wrong place for them.

BATCH_TOOLS = frozenset(
    {
        "mempalace_add_drawer",
        "mempalace_update_drawer",
        "mempalace_kg_add",
        "mempalace_kg_invalidate",
        "mempalace_diary_write",
        "mempalace_checkpoint",
    }
)

_STEP_REF = re.compile(r"\$\{([A-Za-z0-9_-]+)\.([A-Za-z0-9_]+)\}")

# Fields a later step may read from an earlier one's result, per tool: the keys
# its success result carries. A ``${id.field}`` outside this set is refused at
# submit time; tools not listed are checked only when the batch runs.
_BATCH_REF_FIELDS = {
    "mempalace_add_drawer": {"drawer_id", "wing", "room", "chunks", "chunk_ids", "reason"},
    "mempalace_update_drawer": {"drawer_id", "wing", "room", "chunks", "chunk_ids", "noop"},
    "mempalace_kg_add": {"triple_id", "fact"},
    "mempalace_kg_invalidate": {"fact", "ended"},
    "mempalace_diary_write": {"entry_id", "agent", "topic", "timestamp", "chunks", "chunk_ids"},
}


def batch_tool_name(name: str) -> str:
    """``add-drawer`` / ``add_drawer`` / ``mempalace_add_drawer`` -> the MCP name."""
    name = str(name or "").strip().replace("-", "_")
    return name if name.startswith("mempalace_") else f"mempalace_{name}"


def validate_batch(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Check a batch before it is queued; return its normalised steps.

    Raises ValueError naming the first bad step, so ``memp submit`` refuses a
    malformed batch at once instead of the daemon failing it later.
    """
    steps = payload.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("a batch needs a non-empty 'steps' list")
    wing = payload.get("wing")
    if wing is not None and (not isinstance(wing, str) or not wing.strip()):
        raise ValueError("'wing' must be a non-empty string")
    seen: dict[str, dict[str, str]] = {}
    out = []
    for i, step in enumerate(steps, 1):
        if not isinstance(step, dict):
            raise ValueError(f"step {i} is not an object")
        tool = batch_tool_name(step.get("tool"))
        if tool not in BATCH_TOOLS:
            allowed = ", ".join(sorted(t.removeprefix("mempalace_") for t in BATCH_TOOLS))
            raise ValueError(f"step {i}: tool {step.get('tool')!r} is not batchable ({allowed})")
        args = step.get("args") or {}
        if not isinstance(args, dict):
            raise ValueError(f"step {i}: 'args' must be an object")
        step_id = step.get("id")
        if step_id is not None:
            if not isinstance(step_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", step_id):
                raise ValueError(f"step {i}: 'id' must be letters, digits, '-' or '_'")
            if step_id in seen:
                raise ValueError(f"step {i}: id {step_id!r} is used twice")
        for ref, field in _STEP_REF.findall(json.dumps(args)):
            if ref not in seen:
                raise ValueError(f"step {i}: ${{{ref}.…}} names no earlier step")
            if seen[ref]["on_conflict"] == "park":
                # A parked step files a follow-up instead and has no result.
                raise ValueError(
                    f"step {i}: ${{{ref}.{field}}} reads a step that may be parked "
                    "(on_conflict 'park'), which leaves no result to read"
                )
            fields = _BATCH_REF_FIELDS.get(seen[ref]["tool"])
            if fields is not None and field not in fields:
                raise ValueError(
                    f"step {i}: ${{{ref}.{field}}}: step {ref!r} "
                    f"({seen[ref]['tool'].removeprefix('mempalace_')}) has no field {field!r} "
                    f"({', '.join(sorted(fields))})"
                )
        on_conflict = step.get("on_conflict", "fail")
        if on_conflict not in ("fail", "park"):
            raise ValueError(f"step {i}: on_conflict must be 'fail' or 'park'")
        if on_conflict == "park" and not wing:
            raise ValueError(f"step {i}: on_conflict 'park' needs the batch 'wing'")
        if step_id:
            seen[step_id] = {"tool": tool, "on_conflict": on_conflict}
        out.append({"id": step_id, "tool": tool, "args": args, "on_conflict": on_conflict})
    return out


class _BatchRefError(ValueError):
    """A ``${id.field}`` with no value when its step runs."""


def _ref_value(match: re.Match, results: dict[str, dict[str, Any]]) -> str:
    step_id, field = match.groups()
    if step_id not in results:
        raise _BatchRefError(f"{match.group(0)}: step {step_id!r} has no result")
    if field not in results[step_id]:
        raise _BatchRefError(f"{match.group(0)}: step {step_id!r} returned no {field!r}")
    return str(results[step_id][field])


def _resolve_refs(value: Any, results: dict[str, dict[str, Any]]) -> Any:
    if isinstance(value, str):
        return _STEP_REF.sub(lambda m: _ref_value(m, results), value)
    if isinstance(value, dict):
        return {k: _resolve_refs(v, results) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_refs(v, results) for v in value]
    return value


def _park_conflict(tools: dict, wing: str, step: dict, args: dict, result: dict) -> dict:
    """File a refused check-and-set write as an open follow-up, verbatim."""
    from datetime import date

    drawer_id = args.get("drawer_id") or result.get("drawer_id")
    note = (
        "# Action: Merge a parked update (it conflicted)\n"
        "Status: open\n"
        f"Added: {date.today().isoformat()}\n"
        f"Project: {wing}\n\n"
        f"A queued update to drawer {drawer_id} was not written: the drawer changed "
        f"after it was read (expected {str(result.get('expected_sha256'))[:12]}, "
        f"found {str(result.get('actual_sha256'))[:12]}). The proposed content is "
        "below, verbatim. Re-read the drawer, merge, and write it with "
        "`memp update-drawer --if-unchanged`.\n\n---\n\n"
        f"{args.get('content') or ''}"
    )
    parked = tools["mempalace_add_drawer"]["handler"](
        wing=wing, room="follow-ups", content=note, added_by="batch-park"
    )
    if parked.get("success") and parked.get("drawer_id"):
        try:
            tools["mempalace_kg_add"]["handler"](
                subject=f"action:{parked['drawer_id']}",
                predicate="belongs_to",
                object=f"wing_{wing}",
            )
        except Exception:  # noqa: BLE001 - the follow-up is filed; the link is optional
            pass
    return parked


@_restores_palace_env
def run_batch(payload: dict[str, Any]) -> dict[str, Any]:
    """Run a batch's steps in order under one palace lock.

    The lock is taken once, non-blocking, before the first step: a busy palace
    is a refusal with nothing written, so the daemon re-queues the whole batch
    unspent. Holding it throughout means a batch never lands half-filed
    because another writer cut in. A step that fails stops the batch; the
    result lists what was written and names the failing step, including when
    the step raised or a ``${id.field}`` had no value. A check-and-set
    conflict on a step marked ``on_conflict: park`` files the proposed content
    as an open follow-up in the batch's wing and carries on; if that follow-up
    cannot be filed the batch fails, with the proposed text in
    ``unfiled_content`` (the failed job is kept until acknowledged).
    """
    from .mcp_server import TOOLS
    from .palace import MineAlreadyRunning, mine_palace_lock

    try:
        steps = validate_batch(payload)
    except ValueError as exc:
        return {"success": False, "error": str(exc), "exit_code": 2}
    palace_path = os.path.abspath(
        os.path.expanduser(payload.get("palace_path") or MempalaceConfig().palace_path)
    )
    os.environ["MEMPALACE_PALACE_PATH"] = palace_path
    _apply_backend(payload.get("backend"))
    wing = payload.get("wing")
    done: list[dict[str, Any]] = []
    parked: list[dict[str, Any]] = []
    results: dict[str, dict[str, Any]] = {}

    def failed(n: int, step: dict[str, Any], why: Any, **extra: Any) -> dict[str, Any]:
        return {
            "success": False,
            "error": f"step {n} ({step['tool']}) failed: {why}",
            "failed_step": n,
            "done": done,
            "parked": parked,
            "exit_code": 1,
            **extra,
        }

    try:
        with mine_palace_lock(palace_path):
            for n, step in enumerate(steps, 1):
                summary = {"step": n, "id": step["id"], "tool": step["tool"]}
                try:
                    args = _resolve_refs(step["args"], results)
                    result = TOOLS[step["tool"]]["handler"](**args)
                    if not isinstance(result, dict):
                        result = {"success": True, "value": result}
                    ok = result.get("success", "error" not in result)
                    if not ok and result.get("conflict") and step["on_conflict"] == "park":
                        try:
                            park = _park_conflict(TOOLS, wing, step, args, result)
                        except Exception as exc:  # noqa: BLE001 - keep the text, fail the batch
                            park = {"success": False, "error": f"{type(exc).__name__}: {exc}"}
                        if not (park.get("success") and park.get("drawer_id")):
                            return failed(
                                n,
                                step,
                                "its update conflicted and could not be parked: "
                                f"{park.get('error')}",
                                unfiled_content=args.get("content"),
                            )
                        parked.append({**summary, "parked_as": park["drawer_id"]})
                        continue
                except MineAlreadyRunning:
                    raise
                except _BatchRefError as exc:
                    return failed(n, step, exc)
                except Exception as exc:  # noqa: BLE001 - a step must not lose the done list
                    return failed(n, step, f"{type(exc).__name__}: {exc}")
                if not ok:
                    return failed(n, step, result.get("error"))
                if step["id"]:
                    results[step["id"]] = result
                done.append({**summary, "drawer_id": result.get("drawer_id")})
    except MineAlreadyRunning as exc:
        if done or parked:
            # Raised after a write (a nested open): report it, never re-queue it.
            return {"success": False, "error": str(exc), "done": done, "exit_code": 1}
        return _lock_refusal_result(exc)
    return {"success": True, "done": done, "parked": parked, "exit_code": 0}
