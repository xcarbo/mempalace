"""Read one drawer straight from ``chroma.sqlite3``, without importing chromadb.

`memp get-drawer` is the palace's most-called command by a wide margin — the
retrieval log shows 8,416 ``get_drawer`` events against 246 searches, and 7,068
of those were a process that fetched exactly one drawer and exited. Each paid
~0.69s, of which ~0.31s is ``import chromadb`` alone (it pulls
``chromadb.utils.embedding_functions`` -> onnxruntime -> numpy on the way in)
and the rest is opening a PersistentClient. None of that is needed to answer
"give me the document with this id": the documents live in ordinary sqlite
tables, and a by-id read never touches the HNSW index.

This module serves exactly that shape with the stdlib alone (~0.12s end to
end). It is an accelerator, never an authority: anything it does not
understand — a palace it cannot open, array-valued metadata, a row shape it
did not expect — returns ``None`` so the caller falls back to the chromadb
path.

The two agree on every *value*: same content, same ``content_sha256`` (so the
check-and-set token is unchanged), same metadata keys and types.
``tests/test_fastread.py`` pins that against the canonical readers in
``mcp_server``. They do NOT agree on the key *order* inside ``metadata``.
Chroma's order comes out of its Rust layer and is not reproducible from the
sqlite rows — it is neither insertion order nor its reverse — so this reader
sorts keys instead, which is at least deterministic. JSON objects are
unordered and every known consumer parses rather than diffs the raw bytes;
nothing hashes this payload except ``content_sha256``, which covers content
alone.

Schema notes (chromadb 1.5.x), which is why this stays fallback-guarded:

* ``embeddings`` holds one row per stored chunk, keyed by ``embedding_id``
  (mempalace's drawer id) and scoped to a segment.
* ``embedding_metadata`` holds one row per metadata key, with the value in
  whichever of ``string_value``/``int_value``/``float_value``/``bool_value``
  matches its type. The document itself is the reserved ``chroma:document``
  key, which chromadb surfaces separately and never inside ``metadatas``.
* ``embedding_metadata_array`` holds list-valued metadata. mempalace writes
  none today; if a row ever appears we bail rather than silently drop it.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Optional

# Reserved chromadb metadata keys never appear in a `metadatas` cell.
_DOCUMENT_KEY = "chroma:document"


def _connect(db_path: str) -> Optional[sqlite3.Connection]:
    """Open ``db_path`` for reading, or return ``None``.

    A plain ``mode=ro`` open of a WAL database fails with "unable to open
    database file" when the ``-shm`` file is absent, which is exactly the state
    a cleanly-closed palace is left in. Fall back to a normal read-write handle
    in that case — we only ever SELECT through it, and it is the same open
    chromadb itself would perform. ``immutable=1`` is deliberately NOT used: it
    promises sqlite the file cannot change, and against a live palace that can
    hand back torn or stale pages.
    """
    if not os.path.isfile(db_path):
        return None
    uri = f"file:{Path(db_path).as_uri()[len('file://') :]}?mode=ro"
    try:
        return sqlite3.connect(uri, uri=True)
    except sqlite3.OperationalError:
        try:
            return sqlite3.connect(db_path)
        except sqlite3.Error:
            return None
    except sqlite3.Error:
        return None


def _metadata_segment_id(conn: sqlite3.Connection, collection_name: str) -> Optional[str]:
    """Return the METADATA segment id for ``collection_name``.

    Scoping matters: a palace holds more than one collection (drawers and
    closets), and their ids share one ``embeddings`` table.
    """
    row = conn.execute(
        """
        SELECT s.id FROM segments s
        JOIN collections c ON c.id = s.collection
        WHERE c.name = ? AND s.scope = 'METADATA'
        """,
        (collection_name,),
    ).fetchone()
    return row[0] if row else None


def _rows_to_meta(rows) -> dict[str, dict[str, Any]]:
    """Fold ``(embedding_id, key, *value_columns)`` rows into per-id dicts.

    Exactly one value column is populated per row. ``bool_value`` is checked
    before ``int_value`` because sqlite stores booleans as integers and the
    column, not the storage class, carries the type.
    """
    out: dict[str, dict[str, Any]] = {}
    for eid, key, sval, ival, fval, bval in rows:
        if bval is not None:
            value: Any = bool(bval)
        elif ival is not None:
            value = ival
        elif fval is not None:
            value = fval
        else:
            value = sval
        out.setdefault(eid, {})[key] = value
    return out


def _fetch(conn, segment_id, where_sql, params) -> dict[str, dict[str, Any]]:
    return _rows_to_meta(
        conn.execute(
            f"""
            SELECT e.embedding_id, m.key, m.string_value, m.int_value,
                   m.float_value, m.bool_value
            FROM embeddings e
            JOIN embedding_metadata m ON m.id = e.id
            WHERE e.segment_id = ? AND {where_sql}
            ORDER BY e.embedding_id, m.key
            """,
            (segment_id, *params),
        )
    )


def _has_array_metadata(conn, segment_id, embedding_ids) -> bool:
    """True if any row carries list-valued metadata this reader would drop."""
    if not embedding_ids:
        return False
    marks = ",".join("?" * len(embedding_ids))
    row = conn.execute(
        f"""
        SELECT 1 FROM embedding_metadata_array a
        JOIN embeddings e ON e.id = a.id
        WHERE e.segment_id = ? AND e.embedding_id IN ({marks}) LIMIT 1
        """,
        (segment_id, *embedding_ids),
    ).fetchone()
    return row is not None


def _split_document(meta: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Peel the reserved document key out of a metadata dict."""
    clean = {k: v for k, v in meta.items() if k != _DOCUMENT_KEY}
    return meta.get(_DOCUMENT_KEY) or "", clean


def _chunk_index(meta: dict[str, Any]) -> int:
    """Mirror of ``mcp_server._chunk_index`` — unparseable sorts first."""
    try:
        return int(meta.get("chunk_index", 0))
    except (TypeError, ValueError):
        return 0


def _response_safe_meta(meta: dict[str, Any]) -> dict[str, Any]:
    """Mirror of ``mcp_server._response_safe_meta`` — basename ``source_file``."""
    safe = dict(meta)
    if safe.get("source_file"):
        safe["source_file"] = Path(safe["source_file"]).name
    return safe


def get_drawer(palace_path: str, collection_name: str, drawer_id: str) -> Optional[dict]:
    """Return the ``get_drawer`` payload for ``drawer_id``, or ``None``.

    ``None`` means "this reader declines" — either the drawer is genuinely
    absent or the palace is in a shape it will not guess at. The caller must
    fall back to the chromadb path rather than reporting a miss, because this
    function cannot tell those two cases apart and must never be the thing that
    reports a drawer lost.
    """
    db_path = os.path.join(palace_path, "chroma.sqlite3")
    conn = _connect(db_path)
    if conn is None:
        return None
    try:
        segment_id = _metadata_segment_id(conn, collection_name)
        if segment_id is None:
            return None

        # A direct hit wins, exactly as _logical_drawer_record tries it first.
        found = _fetch(conn, segment_id, "e.embedding_id = ?", (drawer_id,))
        if found:
            if _has_array_metadata(conn, segment_id, [drawer_id]):
                return None
            content, meta = _split_document(found[drawer_id])
            return _payload(drawer_id, content, meta, None)

        # Otherwise the id may be a logical parent over chunk rows.
        group = _fetch(
            conn,
            segment_id,
            """e.id IN (SELECT m2.id FROM embedding_metadata m2
                        WHERE m2.key = 'parent_drawer_id' AND m2.string_value = ?)""",
            (drawer_id,),
        )
        if not group:
            return None
        if _has_array_metadata(conn, segment_id, list(group)):
            return None

        docs_metas = {cid: _split_document(m) for cid, m in group.items()}
        # (chunk_index, chunk_id) — the same ordering _logical_chunk_group uses,
        # so ties on a missing/duplicate index resolve identically.
        ordered = sorted(docs_metas.items(), key=lambda kv: (_chunk_index(kv[1][1]), kv[0]))
        chunk_ids = [cid for cid, _ in ordered]
        content = "".join(doc for _, (doc, _m) in ordered)
        first_meta = ordered[0][1][1]
        return _payload(drawer_id, content, first_meta, chunk_ids)
    except (sqlite3.Error, KeyError, TypeError, ValueError):
        return None
    finally:
        conn.close()


def get_rows(
    palace_path: str,
    collection_name: str,
    *,
    row_id: Optional[str] = None,
    parent_id: Optional[str] = None,
) -> "Optional[list[dict]]":
    """Raw stored rows for one drawer id, straight from sqlite.

    The layer under :func:`get_drawer`, for callers that need the rows as
    chromadb's ``get()`` would return them rather than a stitched
    ``get_drawer`` payload — notably the local HTTP API, whose
    ``/drawers/{id}`` contract is one row and whose ``/drawers/{id}/full``
    contract reports per-chunk completeness. Metadata is returned **verbatim**
    (no ``source_file`` basenaming), because those endpoints have always
    exposed the stored value and a client may hand it back as a
    ``source_file`` filter, which only matches the full path.

    Pass exactly one of ``row_id`` (exact ``embedding_id``) or ``parent_id``
    (every chunk whose ``parent_drawer_id`` matches). Rows come back sorted by
    ``(chunk_index, id)`` — the ordering ``mcp_server._logical_chunk_group``
    uses — each as ``{'id', 'document', 'metadata', 'chunk_index'}``.

    Returns ``None`` when this reader declines (unopenable palace, unknown
    collection, array-valued metadata) and the caller must fall back to
    chromadb. An empty list means "read fine, matched nothing" — still fall
    back before reporting a miss, since a sqlite reader must never be the
    thing that declares a drawer lost.
    """
    if (row_id is None) == (parent_id is None):
        raise ValueError("pass exactly one of row_id or parent_id")

    db_path = os.path.join(palace_path, "chroma.sqlite3")
    conn = _connect(db_path)
    if conn is None:
        return None
    try:
        segment_id = _metadata_segment_id(conn, collection_name)
        if segment_id is None:
            return None

        if row_id is not None:
            found = _fetch(conn, segment_id, "e.embedding_id = ?", (row_id,))
        else:
            found = _fetch(
                conn,
                segment_id,
                """e.id IN (SELECT m2.id FROM embedding_metadata m2
                            WHERE m2.key = 'parent_drawer_id' AND m2.string_value = ?)""",
                (parent_id,),
            )
        if not found:
            return []
        if _has_array_metadata(conn, segment_id, list(found)):
            return None

        rows = []
        for stored_id, raw_meta in found.items():
            document, meta = _split_document(raw_meta)
            rows.append(
                {
                    "id": stored_id,
                    "document": document,
                    "metadata": meta,
                    "chunk_index": _chunk_index(meta),
                }
            )
        rows.sort(key=lambda r: (r["chunk_index"], r["id"]))
        return rows
    except (sqlite3.Error, KeyError, TypeError, ValueError):
        return None
    finally:
        conn.close()


def _payload(drawer_id: str, content: str, meta: dict, chunk_ids: Optional[list]) -> dict:
    """Build the payload ``mcp_server._drawer_payload`` would have returned."""
    safe_meta = _response_safe_meta(meta)
    payload = {
        "drawer_id": drawer_id,
        "content": content,
        "content_sha256": hashlib.sha256(str(content or "").encode("utf-8")).hexdigest(),
        "wing": safe_meta.get("wing", ""),
        "room": safe_meta.get("room", ""),
        "metadata": safe_meta,
    }
    if chunk_ids is not None:
        payload["chunks"] = len(chunk_ids)
        payload["chunk_ids"] = chunk_ids
        payload["metadata"]["chunks"] = len(chunk_ids)
        payload["metadata"]["chunk_ids"] = chunk_ids
    return payload


def try_cli_fast_path(argv: list[str]) -> Optional[int]:
    """Serve ``memp get-drawer --drawer-id <id>`` without importing chromadb.

    Returns a process exit code when it handled the call, else ``None`` so the
    normal CLI path runs. Deliberately narrow: it claims only the plain shape
    with no global options, so anything unusual (``--palace``, a different
    backend, extra flags) still goes the long way round.
    """
    if not argv or argv[0] != "get-drawer":
        return None

    rest = argv[1:]
    pretty = False
    drawer_id = None
    i = 0
    while i < len(rest):
        tok = rest[i]
        if tok == "--pretty":
            pretty = True
        elif tok == "--drawer-id" and i + 1 < len(rest):
            drawer_id = rest[i + 1]
            i += 1
        elif tok.startswith("--drawer-id="):
            drawer_id = tok.split("=", 1)[1]
        else:
            return None  # anything else: let the full parser deal with it
        i += 1

    if not drawer_id:
        return None

    from .config import MempalaceConfig

    try:
        cfg = MempalaceConfig()
        if getattr(cfg, "backend", "chroma") not in ("chroma", None, ""):
            return None
        payload = get_drawer(cfg.palace_path, cfg.collection_name, drawer_id)
    except Exception:
        return None

    if payload is None:
        return None

    print(json.dumps(payload, indent=2 if pretty else None, ensure_ascii=False))
    return 0
