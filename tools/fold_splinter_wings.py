#!/usr/bin/env python3
"""Fold `wing_*` splinter wings into their existing bare twins.

One-off migration (2026-07-08). Historical sessions filed drawers under
notation-prefixed wings (``wing_cc``) instead of the canonical bare wing
(``cc``). This script re-keys ONLY the ``wing`` metadata field on affected
drawers (and closets, defensively) so the splinters merge into their twins.

Folding rule for a wing named ``wing_X``:
  0. -> an explicit ``--map wing_X=target`` if one was given (see below)
  1. -> ``X``                    if a bare wing ``X`` already exists
  2. -> ``X.replace("_", "-")``  if THAT bare wing already exists
  3. -> SKIP (never create a new wing; no-twin splinters are left untouched
     for a human mapping decision — see the follow-up drawer in
     wing ``mempalace`` / room ``follow-ups``)

``--map`` exists because rules 1-2 only find a twin whose name is the splinter
minus the prefix. Most of the no-twin splinters belong to a wing under a
DIFFERENT name — ``wing_iptv`` belongs to ``utilities-iptv``, ``wing_router`` to
``mail-router`` — which no automatic rule can infer. That is the human mapping
decision rule 3 defers, so this is how the answer gets supplied once made.

A mapped target must ALREADY EXIST as a bare wing; an unknown target aborts the
run rather than silently creating a wing. That keeps the original invariant:
this script never invents a wing, it only merges into one you already have.

"Exists" means: appears as a ``wing`` value on at least one drawer in the
drawer collection and does not itself start with ``wing_``.

Deliberately NOT ``memp migrate-wings`` / ``migrate_wing_names``: that
normalizer maps hyphens to underscores and would rename tens of thousands of
canonical hyphenated wings the wrong way.

Rooms, content, and drawer IDs are untouched. KG triples are notation-only
(``wing_cc`` there already means bare wing ``cc``) and are not rewritten.

Usage:
    python tools/fold_splinter_wings.py             # dry-run (default)
    python tools/fold_splinter_wings.py --apply     # perform the fold
    python tools/fold_splinter_wings.py \
        --map wing_iptv=utilities-iptv --map wing_router=mail-router --apply
"""

import argparse
import sys
import time
from collections import defaultdict

from mempalace.config import MempalaceConfig
from mempalace.migrate import (
    _apply_topics_by_wing_renames,
    _apply_wing_updates,
    _iter_collection_items,
)
from mempalace.palace import get_closets_collection, get_collection

SPLINTER_PREFIX = "wing_"

# Transient "peer writer active / read-only" locks from the live palace.
LOCK_RETRIES = 8
LOCK_WAIT_SECONDS = 15


def fold_target(wing, existing_bare_wings, explicit_map=None, only=None):
    """Return the bare twin for a splinter wing, or None to skip."""
    if not isinstance(wing, str) or not wing.startswith(SPLINTER_PREFIX):
        return None
    # Allowlist, when given, wins over every rule below. A fold is a bulk
    # metadata rewrite on a live palace, so "exactly the wings I approved" has
    # to be expressible — rules 1-2 will otherwise happily fold a splinter whose
    # target is still an open question (wing_fsra has a 3-drawer bare twin, but
    # its drawers may belong in fsra-fsp or fsra-risk instead).
    if only and wing not in only:
        return None
    if explicit_map and wing in explicit_map:
        return explicit_map[wing]
    bare = wing[len(SPLINTER_PREFIX) :]
    if not bare:
        return None
    if bare in existing_bare_wings:
        return bare
    hyphenated = bare.replace("_", "-")
    if hyphenated in existing_bare_wings:
        return hyphenated
    return None


def parse_map(pairs):
    """Parse ``--map wing_x=target`` arguments into {splinter: target}."""
    out = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"--map expects wing_x=target, got: {pair!r}")
        src, _, dst = pair.partition("=")
        src, dst = src.strip(), dst.strip()
        if not src.startswith(SPLINTER_PREFIX):
            raise SystemExit(f"--map source must start with {SPLINTER_PREFIX!r}: {src!r}")
        if not dst:
            raise SystemExit(f"--map target is empty for {src!r}")
        if dst.startswith(SPLINTER_PREFIX):
            raise SystemExit(f"--map target must be a bare wing, not a splinter: {dst!r}")
        out[src] = dst
    return out


def plan_fold(items, existing_bare_wings, explicit_map=None, only=None):
    """Pure planner over ``(id, metadata)`` pairs.

    Returns ``(summary, updates, skipped, sample_ids)`` where ``summary`` is
    ``{(old, new): count}``, ``updates`` is ``[(id, new_metadata), ...]``,
    ``skipped`` is ``{no_twin_wing: count}``, and ``sample_ids`` maps each
    folded wing to a few drawer IDs for post-apply spot checks.
    """
    summary = defaultdict(int)
    updates = []
    skipped = defaultdict(int)
    sample_ids = defaultdict(list)
    for rec_id, meta in items:
        meta = dict(meta or {})
        wing = meta.get("wing")
        if not isinstance(wing, str) or not wing.startswith(SPLINTER_PREFIX):
            continue
        target = fold_target(wing, existing_bare_wings, explicit_map, only)
        if target is None:
            skipped[wing] += 1
            continue
        summary[(wing, target)] += 1
        if len(sample_ids[wing]) < 3:
            sample_ids[wing].append(rec_id)
        meta["wing"] = target
        updates.append((rec_id, meta))
    return summary, updates, skipped, sample_ids


def _apply_with_lock_retry(apply_fn, label):
    """Run apply_fn, retrying on transient peer-writer / read-only locks."""
    for attempt in range(1, LOCK_RETRIES + 1):
        try:
            apply_fn()
            return
        except Exception as exc:
            msg = str(exc).lower()
            transient = any(
                token in msg for token in ("peer", "read-only", "readonly", "locked", "lock")
            )
            if not transient or attempt == LOCK_RETRIES:
                raise
            print(
                f"  {label}: transient lock ({exc}); "
                f"retry {attempt}/{LOCK_RETRIES - 1} in {LOCK_WAIT_SECONDS}s..."
            )
            time.sleep(LOCK_WAIT_SECONDS)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--apply",
        action="store_true",
        help="perform the fold (default is dry-run)",
    )
    parser.add_argument(
        "--palace",
        default=None,
        help="palace path (default: configured palace)",
    )
    parser.add_argument(
        "--map",
        action="append",
        dest="maps",
        metavar="wing_x=target",
        help="explicit fold for a no-twin splinter; repeatable. Target must "
        "already exist as a bare wing.",
    )
    parser.add_argument(
        "--only",
        action="append",
        dest="only",
        metavar="wing_x",
        help="fold ONLY these splinter wings; repeatable. Everything else is "
        "left untouched even if rules 1-2 would match it.",
    )
    args = parser.parse_args(argv)
    explicit_map = parse_map(args.maps)
    only = set(args.only or [])

    palace_path = args.palace or MempalaceConfig().palace_path
    print(f"Palace: {palace_path}")
    print(f"Mode:   {'APPLY' if args.apply else 'DRY RUN'}\n")

    drawers = get_collection(palace_path, create=False)
    d_items = list(_iter_collection_items(drawers))
    total_before = len(d_items)

    # Bare wings that exist today (targets must already exist — never create).
    existing_bare_wings = {
        (m or {}).get("wing")
        for _, m in d_items
        if isinstance((m or {}).get("wing"), str)
        and (m or {}).get("wing")
        and not (m or {}).get("wing").startswith(SPLINTER_PREFIX)
    }

    # Validate explicit targets BEFORE planning: a typo here would otherwise
    # scatter drawers into a wing that does not exist, and this script's whole
    # safety story is that it never creates one.
    if explicit_map:
        unknown = {src: dst for src, dst in explicit_map.items() if dst not in existing_bare_wings}
        if unknown:
            print("ERROR: --map targets that are not existing bare wings:")
            for src, dst in sorted(unknown.items()):
                print(f"  {src} -> {dst}")
            print("\nNo changes made. Register the wing first, or fix the spelling.")
            return 2
        print("Explicit mappings:")
        for src, dst in sorted(explicit_map.items()):
            print(f"  {src:30} -> {dst}")
        print()

    d_summary, d_updates, d_skipped, sample_ids = plan_fold(
        d_items, existing_bare_wings, explicit_map, only
    )

    # Closets: same rule, judged against the same drawer-derived bare wings.
    closets = None
    c_summary, c_updates = defaultdict(int), []
    c_skipped = defaultdict(int)
    try:
        closets = get_closets_collection(palace_path, create=False)
        c_summary, c_updates, c_skipped, _ = plan_fold(
            _iter_collection_items(closets), existing_bare_wings, explicit_map, only
        )
    except Exception:
        closets = None

    # topics_by_wing registry keys, same folding rule.
    topic_renames = {}
    try:
        from mempalace.miner import _load_known_entities_raw

        tbw = _load_known_entities_raw().get("topics_by_wing")
        if isinstance(tbw, dict):
            for key in tbw:
                target = fold_target(key, existing_bare_wings, explicit_map, only)
                if target is not None:
                    topic_renames[key] = target
    except Exception:
        pass

    print("Fold plan (drawers / closets):")
    merged = defaultdict(lambda: [0, 0])
    for key, count in d_summary.items():
        merged[key][0] = count
    for key, count in c_summary.items():
        merged[key][1] = count
    for (old, new), (dc, cc) in sorted(merged.items()):
        print(f"  {old:40} -> {new:30} {dc:5} drawer(s)  {cc:3} closet(s)")
    print(
        f"\n  {len(merged)} source wing(s), "
        f"{len(d_updates)} drawer(s), {len(c_updates)} closet record(s)"
    )
    if topic_renames:
        print(f"  topics_by_wing: {len(topic_renames)} key(s) re-keyed")

    all_skipped = defaultdict(int)
    for w, n in d_skipped.items():
        all_skipped[w] += n
    for w, n in c_skipped.items():
        all_skipped[w] += n
    if all_skipped:
        print(f"\nSkipped (no bare twin — left untouched): {len(all_skipped)} wing(s)")
        for wing, count in sorted(all_skipped.items()):
            print(f"  {wing:40} {count:5} record(s)")

    print("\nSpot-check sample drawer IDs (per folded wing):")
    for wing, ids in sorted(sample_ids.items()):
        print(f"  {wing}: {', '.join(ids)}")

    if not d_updates and not c_updates and not topic_renames:
        print("\nNothing to fold.")
        return 0

    if not args.apply:
        print("\nDRY RUN — no changes made. Re-run with --apply to fold.")
        return 0

    print("\nApplying...")
    _apply_with_lock_retry(lambda: _apply_wing_updates(drawers, d_updates), "drawers")
    if closets is not None and c_updates:
        _apply_with_lock_retry(lambda: _apply_wing_updates(closets, c_updates), "closets")
    if topic_renames:
        _apply_topics_by_wing_renames(topic_renames)

    # Verify: re-read and confirm the folded wings are gone and total unchanged.
    d_after = list(_iter_collection_items(drawers))
    total_after = len(d_after)
    remaining = defaultdict(int)
    for _, m in d_after:
        w = (m or {}).get("wing")
        if isinstance(w, str) and w.startswith(SPLINTER_PREFIX):
            remaining[w] += 1
    folded_sources = {old for (old, _new) in d_summary}
    leftovers = {w: n for w, n in remaining.items() if w in folded_sources}

    print(f"\nApplied {len(d_updates)} drawer update(s), {len(c_updates)} closet update(s).")
    print(f"Total drawers before: {total_before}, after: {total_after}")
    if total_after != total_before:
        print("  WARNING: total drawer count changed!")
    if leftovers:
        print(f"  WARNING: folded wings still have records: {dict(leftovers)}")
    else:
        print("All folded source wings now have 0 drawers.")
    print(f"Remaining splinter wings (expected, no twin): {len(remaining)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
