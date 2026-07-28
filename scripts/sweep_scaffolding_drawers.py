#!/usr/bin/env python3
"""Sweep agent-harness scaffolding drawers out of the palace.

Companion to the mine-time fix in ``normalize.strip_noise``: that stops NEW
scaffolding from being filed, this removes what already landed.

A drawer is a candidate ONLY if ``strip_noise`` reduces its entire body to
nothing. That is the same predicate the miner now applies, so the sweep can
never remove a drawer the current miner would keep, and it never judges by
length — the user's short words are safe by construction.

Dry-run by default. Nothing is deleted without --apply.

    python scripts/sweep_scaffolding_drawers.py --wing sessions
    python scripts/sweep_scaffolding_drawers.py --wing sessions --apply
"""

import argparse
import collections
import sys

from mempalace.normalize import strip_noise
from mempalace.searcher import get_collection

DEFAULT_PALACE = "~/.mempalace/palace"
PAGE = 5000


def iter_drawers(col, wing):
    """Page through a wing, yielding (id, document)."""
    offset = 0
    while True:
        where = {"wing": wing} if wing else None
        batch = col.get(where=where, include=["documents"], limit=PAGE, offset=offset)
        ids = batch.get("ids") or []
        docs = batch.get("documents") or []
        if not ids:
            return
        yield from zip(ids, docs)
        offset += len(ids)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--palace", default=DEFAULT_PALACE)
    ap.add_argument("--wing", default="sessions", help="wing to sweep (default: sessions)")
    ap.add_argument("--apply", action="store_true", help="actually delete (default: dry run)")
    ap.add_argument("--limit", type=int, default=0, help="stop after N candidates (0 = all)")
    args = ap.parse_args()

    col = get_collection(args.palace)

    scanned = 0
    candidates = []
    bodies = collections.Counter()

    for did, doc in iter_drawers(col, args.wing):
        scanned += 1
        body = doc or ""
        if not body.strip():
            continue
        # The predicate: the CURRENT miner would file nothing for this body.
        if strip_noise(body).strip():
            continue
        candidates.append(did)
        bodies[" ".join(body.split())[:70]] += 1
        if args.limit and len(candidates) >= args.limit:
            break

    print(f"wing={args.wing!r}  scanned={scanned}  candidates={len(candidates)}")
    if scanned:
        print(f"share: {100 * len(candidates) / scanned:.2f}% of the wing")
    print("\ntop candidate bodies:")
    for body, n in bodies.most_common(15):
        print(f"  {n:6d}x  {body!r}")

    if not args.apply:
        print("\nDRY RUN — nothing deleted. Re-run with --apply to commit.")
        return 0

    if not candidates:
        print("\nnothing to delete")
        return 0

    print(f"\ndeleting {len(candidates)} drawers...")
    for i in range(0, len(candidates), 500):
        col.delete(ids=candidates[i : i + 500])
        print(f"  deleted {min(i + 500, len(candidates))}/{len(candidates)}")
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
