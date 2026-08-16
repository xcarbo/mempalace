"""Palace time machine — reconstruct what was known at a past date.

``memp as-of 2026-05-01 --wing mempalace`` answers "what did the palace
hold, and what was true, at end-of-day 2026-05-01": drawer counts and
room breakdown at that date, the most recently filed drawers leading up
to it, the roadmap as it stood, and the knowledge-graph facts valid at
that instant (the KG's half-open temporal intervals make this exact).

Time basis is ``filed_at`` — when the palace learned something — with
``authored_at`` displayed where present. Read-only.

Honest limitation, stated in the output: drawer *content* is shown as
currently stored. Updates overwrite in place (there is no content
history), so a drawer updated since the target date shows today's words
under yesterday's filing date. Facts from the KG do not have this
caveat — supersession keeps their temporal boundaries.
"""

from collections import Counter
from datetime import datetime

from .knowledge_graph import KnowledgeGraph
from .palace import get_collection

PREVIEW_CHARS = 160
ROADMAP_CHARS = 2500
MAX_KG_FACTS = 20


def _normalize_cutoff(date: str) -> str:
    """Validate and expand a date to an inclusive end-of-day cutoff."""
    raw = (date or "").strip()
    if "T" in raw:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return parsed.replace(tzinfo=None).isoformat()
    parsed = datetime.strptime(raw, "%Y-%m-%d")
    return parsed.strftime("%Y-%m-%dT23:59:59.999999")


def _drawer_entry(d: dict) -> dict:
    meta = d.get("metadata") or {}
    text = d.get("content_preview") or d.get("content") or ""
    return {
        "drawer_id": d.get("drawer_id") or d.get("id"),
        "room": meta.get("room", "?"),
        "filed_at": meta.get("filed_at"),
        "authored_at": meta.get("authored_at"),
        "preview": " ".join(text[:PREVIEW_CHARS].split()),
    }


def _kg_facts_at(wing: str, cutoff: str, kg_path: str = None) -> list:
    """Facts valid at the cutoff for both wing-entity notations."""
    # The KG accepts YYYY-MM-DD or second-precision Z datetimes only —
    # trim the end-of-day cutoff's microseconds for it.
    kg_cutoff = cutoff[:19] + "Z" if "T" in cutoff else cutoff
    kg = KnowledgeGraph(db_path=kg_path) if kg_path else KnowledgeGraph()
    try:
        facts = []
        seen = set()
        for entity in (f"wing_{wing}", wing):
            try:
                result = kg.query_entity(entity, as_of=kg_cutoff, direction="both")
            except Exception:
                continue
            for f in result or []:
                # belongs_to triples are palace bookkeeping (drawer → wing
                # plumbing, mostly undated); they drown out substantive
                # facts and say nothing about what was true at the date.
                if f.get("predicate") == "belongs_to":
                    continue
                key = (f.get("subject"), f.get("predicate"), f.get("object"))
                if key in seen:
                    continue
                seen.add(key)
                facts.append(f)
        return facts[:MAX_KG_FACTS]
    finally:
        close = getattr(kg, "close", None)
        if close:
            close()


def snapshot(
    palace_path: str,
    date: str,
    wing: str = None,
    latest: int = 10,
    collection_name: str = None,
    kg_path: str = None,
) -> dict:
    """Build the as-of view. Raises ValueError on an unparseable date."""
    try:
        cutoff = _normalize_cutoff(date)
    except ValueError as e:
        raise ValueError(f"unparseable date {date!r} (want YYYY-MM-DD or ISO datetime): {e}") from e

    # Imported here, not at module top: mcp_server is heavy and the CLI
    # dispatch path has usually imported it already by the time we run.
    from .mcp_server import _collapse_drawer_rows, _fetch_drawer_rows

    col = get_collection(palace_path, collection_name=collection_name, create=False, read_only=True)
    if col is None:
        raise ValueError(f"No palace found at {palace_path}")

    where = {"wing": wing} if wing else None
    ids, documents, metadatas = _fetch_drawer_rows(col, where=where)
    drawers = _collapse_drawer_rows(ids, documents, metadatas)

    def _filed(d):
        return (d.get("metadata") or {}).get("filed_at") or ""

    dated = [d for d in drawers if _filed(d)]
    at_date = [d for d in dated if _filed(d) <= cutoff]
    at_date.sort(key=_filed, reverse=True)

    rooms_at_date = Counter((d.get("metadata") or {}).get("room", "?") for d in at_date)

    roadmap = next(
        (d for d in at_date if (d.get("metadata") or {}).get("room") == "roadmap"),
        None,
    )
    roadmap_content = ""
    if roadmap:
        # Collapsed rows carry only a preview; fetch the full drawer body.
        from .mcp_server import _logical_drawer_record

        record = _logical_drawer_record(col, roadmap.get("drawer_id") or roadmap.get("id"))
        roadmap_content = (record or {}).get("content") or roadmap.get("content_preview") or ""

    return {
        "date": date,
        "cutoff": cutoff,
        "wing": wing,
        "total_at_date": len(at_date),
        "total_now": len(drawers),
        "undated_excluded": len(drawers) - len(dated),
        "rooms": dict(rooms_at_date.most_common()),
        "latest": [_drawer_entry(d) for d in at_date[: max(1, latest)]],
        "roadmap": (
            {
                "drawer_id": roadmap.get("drawer_id") or roadmap.get("id"),
                "filed_at": _filed(roadmap),
                "content": roadmap_content[:ROADMAP_CHARS],
            }
            if roadmap
            else None
        ),
        "kg_facts": _kg_facts_at(wing, cutoff, kg_path=kg_path) if wing else [],
        "caveat": (
            "Drawer content is shown as currently stored; in-place updates "
            "overwrite history, so a drawer updated since this date shows "
            "today's content under its original filing date. KG facts are "
            "temporally exact."
        ),
    }


def render(snap: dict) -> str:
    """Human-readable rendering of a snapshot dict."""
    lines = []
    scope = f"wing {snap['wing']}" if snap["wing"] else "whole palace"
    lines.append("=" * 60)
    lines.append(f"  Palace as of {snap['date']} — {scope}")
    lines.append("=" * 60)
    lines.append(
        f"  Drawers then: {snap['total_at_date']}   now: {snap['total_now']}"
        + (f"   (+{snap['undated_excluded']} undated excluded)" if snap["undated_excluded"] else "")
    )
    if snap["rooms"]:
        lines.append("\n  Rooms at that date:")
        for room, count in list(snap["rooms"].items())[:15]:
            lines.append(f"    {room:24s} {count}")
    if snap["latest"]:
        lines.append("\n  Most recently filed as of then:")
        for e in snap["latest"]:
            lines.append(f"    [{(e['filed_at'] or '?')[:10]}] {e['room']}: {e['preview']}")
            lines.append(f"        {e['drawer_id']}")
    if snap["roadmap"]:
        lines.append(f"\n  Roadmap as it stood ({snap['roadmap']['filed_at'][:10]}):")
        for line in snap["roadmap"]["content"].splitlines()[:25]:
            lines.append(f"    {line}")
        lines.append(f"    … full drawer: {snap['roadmap']['drawer_id']}")
    if snap["kg_facts"]:
        lines.append("\n  Facts valid at that instant:")
        for f in snap["kg_facts"]:
            lines.append(f"    {f.get('subject')} → {f.get('predicate')} → {f.get('object')}")
    lines.append(f"\n  NOTE: {snap['caveat']}")
    return "\n".join(lines)
