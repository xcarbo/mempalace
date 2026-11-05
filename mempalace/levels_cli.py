"""``memp level show|set`` and ``memp backfill-levels``: the access-level commands.

Both print counts, levels, wing names and drawer ids. Neither ever prints
drawer content.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from typing import Optional

from . import levels

_PAGE = 2000
_WRITE_BATCH = 500
_COLLECTIONS = ("mempalace_drawers", "mempalace_closets")
BACKFILL_SRC = "backfill"


def register(sub):
    """Add the ``level`` and ``backfill-levels`` subcommands. Returns the level parser."""
    p_level = sub.add_parser(
        "level",
        help="Show this session's access level, or set a drawer's level (owners/team)",
    )
    level_sub = p_level.add_subparsers(dest="level_action")
    p_show = level_sub.add_parser("show", help="Show this session's level and where it comes from")
    p_show.add_argument("--cwd", default=None, help="Resolve for this folder instead of the cwd")
    p_show.add_argument("--json", action="store_true", help="Print JSON")
    p_set = level_sub.add_parser(
        "set",
        help="Set the level of one drawer, every drawer from one source file, or a whole wing",
        description=(
            "memp level set <drawer-id> owners|team\n"
            "memp level set --source <source_file> [--source ...] owners|team\n"
            "memp level set --wing <wing> [--room <room>] owners|team\n"
            "memp level set --ids-file <file, one drawer id per line> owners|team\n\n"
            "Writes access plus access_src=override, so a backfill or re-mine keeps it. "
            "Making drawers more private works from any session; opening them to the team "
            "needs an owners-level session. A drawer this session cannot see is not found."
        ),
        formatter_class=__import__("argparse").RawDescriptionHelpFormatter,
    )
    p_set.add_argument("args", nargs="+", metavar="[DRAWER_ID] LEVEL")
    p_set.add_argument(
        "--source",
        action="append",
        default=None,
        help="Every drawer whose source_file is this (repeatable)",
    )
    p_set.add_argument("--wing", default=None, help="Every drawer in this wing")
    p_set.add_argument("--room", default=None, help="With --wing: only this room")
    p_set.add_argument(
        "--ids-file",
        default=None,
        help="A file of drawer ids, one per line ('#' comments allowed)",
    )
    p_set.add_argument("--dry-run", action="store_true", help="Count only, change nothing")

    p_backfill = sub.add_parser(
        "backfill-levels",
        help="Stamp owners on drawers that came from an owners folder (owners sessions only)",
        description=(
            "Reads every drawer's wing and source_file (never its content) and stamps "
            "access=owners where the folder rule says so. Only ever makes drawers more "
            "private; overrides are left alone. Prints counts per wing and level."
        ),
    )
    mode = p_backfill.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Counts only, change nothing")
    mode.add_argument("--yes", action="store_true", help="Apply the stamps")
    p_backfill.add_argument("--json", action="store_true", help="Print the counts as JSON")
    return p_level


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _palace_path(args) -> str:
    from .config import MempalaceConfig

    if getattr(args, "palace", None):
        return os.path.expanduser(args.palace)
    return MempalaceConfig().palace_path


def _fail(msg: str, code: int = 2):
    print(f"mempalace: {msg}", file=sys.stderr)
    sys.exit(code)


def _open(palace: str, name: str, *, create: bool = False):
    from .palace import get_collection

    try:
        return get_collection(palace, collection_name=name, create=create)
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        if name == "mempalace_closets":
            return None  # a palace with no closets yet
        _fail(f"cannot open {name} at {palace}: {exc}")


def _iter_rows(col, where: Optional[dict] = None):
    """(id, metadata) for every row, paged; never reads documents."""
    offset = 0
    while True:
        kwargs = {"include": ["metadatas"], "limit": _PAGE, "offset": offset}
        if where:
            kwargs["where"] = where
        got = col.get(**kwargs)
        ids = got["ids"]
        metas = got.get("metadatas") or [None] * len(ids)
        for doc_id, meta in zip(ids, metas):
            yield doc_id, meta or {}
        if len(ids) < _PAGE:
            return
        offset += len(ids)


def _batched_update(col, items: list[tuple[str, dict]]) -> None:
    for start in range(0, len(items), _WRITE_BATCH):
        chunk = items[start : start + _WRITE_BATCH]
        col.update(ids=[i for i, _ in chunk], metadatas=[m for _, m in chunk])


# ---------------------------------------------------------------------------
# level show
# ---------------------------------------------------------------------------


def cmd_level(args):
    action = getattr(args, "level_action", None)
    if action == "show":
        return _cmd_show(args)
    if action == "set":
        return _cmd_set(args)
    _fail("usage: memp level show | memp level set ...")


def _cmd_show(args):
    path = levels.levels_file_path()
    config = levels.load_levels_config()
    cwd = os.path.abspath(os.path.expanduser(args.cwd)) if args.cwd else os.getcwd()
    try:
        env = levels.env_level()
    except levels.LevelError as exc:
        _fail(str(exc))
    active = levels.levels_active()
    folder = levels.folder_level(cwd, config) if config is not None else None
    stamp = levels.resolve_level(cwd) if active else None
    level = (env if env is not None else folder) if active else None
    out = {
        "active": active,
        "levels_file": str(path),
        "levels_file_present": config is not None,
        "levels_file_broken": (config.broken if config is not None else "") or None,
        "cwd": cwd,
        "folder_level": levels.level_label(folder) if folder is not None else None,
        "env_level": levels.level_label(env) if env is not None else None,
        "level": levels.level_label(level) if level is not None else None,
        "level_number": level,
        "stamp_level": levels.level_label(stamp) if stamp is not None else None,
        "stamp_level_number": stamp,
    }
    if args.json:
        print(json.dumps(out, indent=2))
        return
    if not active:
        print("levels: off (no levels file, no MEMPALACE_LEVEL): every drawer is readable")
        print(f"  levels file: {path} (absent)")
        return
    print(f"level: {out['level']} ({level}) reads; writes stamp {out['stamp_level']} ({stamp})")
    print(f"  folder {cwd}: {out['folder_level'] or 'n/a (no levels file)'}")
    print(f"  MEMPALACE_LEVEL: {out['env_level'] or 'unset'}")
    print(
        f"  levels file: {path}"
        + (f" (BROKEN: {config.broken})" if config and config.broken else "")
    )


# ---------------------------------------------------------------------------
# level set
# ---------------------------------------------------------------------------


def _cmd_set(args):
    if not levels.levels_active():
        _fail(
            "levels are off here (no levels file, no MEMPALACE_LEVEL): nothing would "
            "enforce a level, so none is set"
        )
    positional = list(args.args)
    target_raw = positional.pop()
    drawer_id = positional.pop() if positional else None
    if positional:
        _fail("give one drawer id, --source or --wing, then the level")
    selectors = [s for s in (drawer_id, args.source, args.wing, args.ids_file) if s]
    if len(selectors) != 1:
        _fail("give exactly one of: a drawer id, --source <file>, --wing <wing>, --ids-file <f>")
    if args.room and not args.wing:
        _fail("--room needs --wing")
    try:
        target = levels.parse_level(target_raw)
        caller = levels.current_level()
    except levels.LevelError as exc:
        _fail(str(exc))

    palace = _palace_path(args)
    col = _open(palace, "mempalace_drawers")
    # Selection runs through the level filter: a drawer this session cannot
    # see does not exist to it, so it cannot be opened from here.
    missing: list[str] = []
    if drawer_id or args.ids_file:
        if drawer_id:
            wanted = [drawer_id]
            selector = {"drawer_id": drawer_id}
        else:
            try:
                lines = (
                    open(os.path.expanduser(args.ids_file), encoding="utf-8").read().splitlines()
                )
            except OSError as exc:
                _fail(f"cannot read --ids-file: {exc}")
            wanted = [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]
            selector = {"ids_file": args.ids_file, "ids_given": len(wanted)}
        rows = {}
        for one in wanted:
            found = _rows_for_id(col, one)
            if not found:
                missing.append(one)
            rows.update(found)
    elif args.source:
        rows = {}
        for source in args.source:
            found = dict(_iter_rows(col, {"source_file": source}))
            if not found:
                missing.append(source)
            rows.update(found)
        selector = {"source": args.source}
    else:
        where = {"wing": args.wing}
        if args.room:
            where = {"$and": [{"wing": args.wing}, {"room": args.room}]}
        rows = dict(_iter_rows(col, where))
        selector = {"wing": args.wing, **({"room": args.room} if args.room else {})}
    if not rows:
        _fail(f"not found: no drawer matches {json.dumps(selector)}", code=1)
    if missing:
        # Partial match: say what was not found (hidden or absent look the same).
        print(f"mempalace: not found, skipped: {len(missing)}: {missing[:20]}", file=sys.stderr)

    before = Counter(levels.stored_access(m) for m in rows.values())
    opening = [i for i, m in rows.items() if levels.stored_access(m) < target]
    if opening and caller != levels.OWNERS:
        print(
            f"mempalace: refused: {len(opening)} drawer(s) would open to "
            f"{levels.level_label(target)}; that needs an owners-level session "
            f"(this one is {levels.level_label(caller)})",
            file=sys.stderr,
        )
        sys.exit(3)
    summary = {
        "selector": selector,
        "drawers": len(rows),
        "from": {levels.level_label(k): v for k, v in sorted(before.items())},
        "to": levels.level_label(target),
        "not_found": len(missing),
    }
    if args.dry_run:
        print(json.dumps({"dry_run": True, **summary}, indent=2))
        return

    ids = sorted(rows)
    if not hasattr(col, "set_access"):
        col = levels.wrap_collection(col)
    for start in range(0, len(ids), _WRITE_BATCH):
        col.set_access(ids[start : start + _WRITE_BATCH], target)
    sources = {m.get("source_file") for m in rows.values() if m.get("source_file")}
    closets = _restamp_closets(palace, sources)
    summary["closets_restamped"] = closets
    levels.log_override(
        {
            **summary,
            "caller_level": levels.level_label(caller),
            "palace": palace,
            "ids": ids if len(ids) <= 200 else None,
        }
    )
    print(json.dumps(summary, indent=2))


def _rows_for_id(col, drawer_id: str) -> dict:
    """The row for ``drawer_id`` plus its chunks (a logical drawer), as seen here."""
    rows = dict(_iter_rows(col, {"parent_drawer_id": drawer_id}))
    got = col.get(ids=[drawer_id], include=["metadatas"])
    for doc_id, meta in zip(got["ids"], got.get("metadatas") or []):
        rows[doc_id] = meta or {}
    return rows


def _restamp_closets(palace: str, sources: set) -> int:
    """Give each source's closets the most private level among its drawers.

    A closet is an index line built from a source file's drawers, so it is as
    private as the most private of them.
    """
    if not sources:
        return 0
    drawers = levels.unwrap_collection(_open(palace, "mempalace_drawers"))
    closets_col = _open(palace, "mempalace_closets")
    if closets_col is None:
        return 0
    closets = levels.unwrap_collection(closets_col)
    changed = 0
    for source in sorted(sources):
        drawer_levels = [
            levels.stored_access(m) for _, m in _iter_rows(drawers, {"source_file": source})
        ]
        if not drawer_levels:
            continue
        want = min(drawer_levels)
        items = [
            (cid, {levels.ACCESS_KEY: want})
            for cid, meta in _iter_rows(closets, {"source_file": source})
            if levels.stored_access(meta) != want
        ]
        _batched_update(closets, items)
        changed += len(items)
    return changed


# ---------------------------------------------------------------------------
# backfill-levels
# ---------------------------------------------------------------------------


def cmd_backfill_levels(args):
    config = levels.load_levels_config()
    if config is None:
        _fail(
            f"no levels file at {levels.levels_file_path()}: nothing says which folders are owners"
        )
    if config.broken:
        _fail(f"levels file {levels.levels_file_path()} is broken: {config.broken}")
    try:
        caller = levels.current_level()
    except levels.LevelError as exc:
        _fail(str(exc))
    if caller != levels.OWNERS:
        _fail(
            f"backfill-levels needs an owners-level session (this one is "
            f"{levels.level_label(caller)}); run it from an owners folder or with "
            "MEMPALACE_LEVEL=owners",
            code=3,
        )
    owners_wings = set(config.owners_wings)
    palace = _palace_path(args)
    report: dict = {"palace": palace, "dry_run": bool(args.dry_run), "collections": {}}
    for name in _COLLECTIONS:
        col = _open(palace, name)
        if col is None:
            continue
        inner = levels.unwrap_collection(col)
        before: dict = defaultdict(Counter)
        after: dict = defaultdict(Counter)
        reasons: Counter = Counter()
        changes: list[tuple[str, dict]] = []
        for doc_id, meta in _iter_rows(inner):
            wing = meta.get("wing") or "?"
            stored = levels.stored_access(meta)
            before[wing][stored] += 1
            new = stored
            if meta.get(levels.ACCESS_SRC_KEY) == levels.OVERRIDE_SRC:
                reasons["override kept"] += 1
            else:
                by_folder = levels.source_level(str(meta.get("source_file") or ""), config)
                by_wing = levels.OWNERS if wing in owners_wings else config.default
                want = min(by_folder, by_wing)
                if want < stored:
                    new = want
                    reasons["owners by folder" if by_folder <= want else "owners by wing"] += 1
                    changes.append(
                        (doc_id, {levels.ACCESS_KEY: new, levels.ACCESS_SRC_KEY: BACKFILL_SRC})
                    )
            after[wing][new] += 1
        if args.yes and changes:
            _batched_update(inner, changes)
        report["collections"][name] = {
            "to_change": len(changes),
            "changed": len(changes) if args.yes else 0,
            "reasons": dict(reasons),
            "by_wing": {
                wing: {
                    "before": {levels.level_label(k): v for k, v in sorted(before[wing].items())},
                    "after": {levels.level_label(k): v for k, v in sorted(after[wing].items())},
                }
                for wing in sorted(before)
            },
            "totals_after": {
                levels.level_label(k): v
                for k, v in sorted(sum((after[w] for w in after), Counter()).items())
            },
        }
    if args.yes:
        levels.log_override(
            {
                "action": "backfill-levels",
                "palace": palace,
                "changed": {n: c["changed"] for n, c in report["collections"].items()},
            }
        )
    if args.json:
        print(json.dumps(report, indent=2))
        return
    _print_backfill(report)


def _print_backfill(report: dict) -> None:
    mode = "DRY RUN, nothing changed" if report["dry_run"] else "APPLIED"
    print(f"backfill-levels ({mode}) on {report['palace']}")
    for name, data in report["collections"].items():
        print(f"\n{name}: {data['to_change']} chunk(s) to owners; after: {data['totals_after']}")
        for reason, n in sorted(data["reasons"].items()):
            print(f"  {reason}: {n}")
        for wing, counts in data["by_wing"].items():
            if counts["before"] != counts["after"] or "owners" in counts["after"]:
                print(f"  {wing:30} before {counts['before']}  after {counts['after']}")
