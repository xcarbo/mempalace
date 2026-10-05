#!/usr/bin/env python3
"""Seed a new palace with a chosen set of wings copied from an existing one.

Built for the ForgePoint palace on fp-hel1 (2026-10-05): a second, separate
palace that starts from the ForgePoint drawers already filed in Chris's.

What is copied, and how:

* Drawer and closet rows for the named wings, through the destination
  backend's ``upsert``. Ids, documents, metadata and the stored vectors are
  kept as they are, so nothing is re-embedded and every drawer id (pins,
  cross-references, KG ``action:`` subjects) still resolves in the new palace.
  The source and destination must use the same embedder; this refuses
  otherwise.
* Knowledge-graph triples that name a copied wing or drawer, plus the
  entities those triples use. Inserted with ``INSERT OR IGNORE`` by id.

The source is opened read-only (``mode=ro``) and never written. Re-runs are
idempotent (upsert by id), so ``--since`` picks up drawers filed after the
first run.

Dry-run by default. Nothing is written without --apply.

    python scripts/seed_palace.py --dest /tmp/fp-seed/palace-exact \\
        --dest-kg /tmp/fp-seed/knowledge_graph.sqlite3 \\
        --wing forgepoint --wing fpsites --exclude-room diary
    python scripts/seed_palace.py ... --apply
"""

import argparse
import json
import os
import sqlite3
import sys

import numpy as np

from mempalace import palace
from mempalace.knowledge_graph import KnowledgeGraph

DEFAULT_SRC = "~/.mempalace/palace-exact"
DEFAULT_SRC_KG = "~/.mempalace/knowledge_graph.sqlite3"
DB_NAME = "sqlite_exact.sqlite3"
BATCH = 500


def ro(path):
    """Open a SQLite file read-only; fail if it does not exist."""
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        sys.exit(f"source not found: {path}")
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def select_rows(src, wings, exclude_rooms, since):
    """Return {collection_name: [(id, document, metadata_json, embedding)]}."""
    marks = ",".join("?" * len(wings))
    sql = (
        "SELECT d.id, d.document, d.metadata_json, d.embedding "
        "FROM documents d JOIN collections c ON c.id = d.collection_id "
        f"WHERE c.name = ? AND d.wing IN ({marks})"
    )
    params = list(wings)
    if exclude_rooms:
        sql += f" AND COALESCE(d.room, '') NOT IN ({','.join('?' * len(exclude_rooms))})"
        params += exclude_rooms
    if since:
        sql += " AND d.updated_at >= ?"
        params.append(since)
    out = {}
    for (name,) in src.execute("SELECT name FROM collections ORDER BY id"):
        out[name] = src.execute(sql, [name] + params).fetchall()
    return out


def check_embedder(src, dest_db):
    """Refuse to mix vectors from different embedders."""
    src_meta = dict(src.execute("SELECT key, value FROM meta"))
    if not os.path.isfile(dest_db):
        return src_meta
    dst = sqlite3.connect(f"file:{dest_db}?mode=ro", uri=True)
    dst_meta = dict(dst.execute("SELECT key, value FROM meta"))
    dst.close()
    for key, value in src_meta.items():
        if key.startswith("embedder_model:") and key in dst_meta and dst_meta[key] != value:
            sys.exit(f"embedder mismatch on {key}: source {value!r}, destination {dst_meta[key]!r}")
    return src_meta


def copy_drawers(dest, rows_by_collection):
    for name, rows in rows_by_collection.items():
        if not rows:
            continue
        col = palace.get_collection(dest, name, create=True, backend="sqlite_exact")
        for i in range(0, len(rows), BATCH):
            chunk = rows[i : i + BATCH]
            col.upsert(
                ids=[r[0] for r in chunk],
                documents=[r[1] for r in chunk],
                metadatas=[json.loads(r[2]) for r in chunk],
                embeddings=[np.frombuffer(r[3], dtype=np.float32).tolist() for r in chunk],
            )
        print(f"  {name}: upserted {len(rows)}")


def select_kg(src_kg, wings, drawer_ids):
    """Triples naming a copied wing or drawer, and the entities they use."""
    keys = set(wings) | {f"wing_{w}" for w in wings}
    likes = [f"%{w}%" for w in wings]
    cond = " OR ".join(["subject LIKE ? OR object LIKE ?"] * len(likes))
    params = [p for like in likes for p in (like, like)]
    triples = {}
    for row in src_kg.execute(f"SELECT * FROM triples WHERE {cond}", params):
        triples[row[0]] = row
    base_ids = {i.split("_chunk_")[0] for i in drawer_ids}
    for row in src_kg.execute("SELECT * FROM triples WHERE source_drawer_id IS NOT NULL"):
        if row[10] in base_ids:
            triples[row[0]] = row
    names = {t[1] for t in triples.values()} | {t[3] for t in triples.values()} | keys
    entities = [e for e in src_kg.execute("SELECT * FROM entities") if e[0] in names]
    return list(triples.values()), entities


def copy_kg(dest_kg, triples, entities):
    KnowledgeGraph(dest_kg).close()  # creates the canonical schema
    con = sqlite3.connect(dest_kg)
    tcols = [r[1] for r in con.execute("PRAGMA table_info(triples)")]
    ecols = [r[1] for r in con.execute("PRAGMA table_info(entities)")]
    with con:
        con.executemany(
            f"INSERT OR IGNORE INTO entities ({','.join(ecols)}) VALUES ({','.join('?' * len(ecols))})",
            [e[: len(ecols)] for e in entities],
        )
        con.executemany(
            f"INSERT OR IGNORE INTO triples ({','.join(tcols)}) VALUES ({','.join('?' * len(tcols))})",
            [t[: len(tcols)] for t in triples],
        )
    con.close()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--src", default=DEFAULT_SRC, help="source palace dir (read-only)")
    ap.add_argument("--src-kg", default=DEFAULT_SRC_KG, help="source knowledge graph (read-only)")
    ap.add_argument("--dest", required=True, help="destination palace dir")
    ap.add_argument("--dest-kg", required=True, help="destination knowledge graph file")
    ap.add_argument("--wing", action="append", required=True, help="wing to copy (repeatable)")
    ap.add_argument("--exclude-room", action="append", default=[], help="room to skip (repeatable)")
    ap.add_argument("--since", help="only rows updated at or after this ISO time")
    ap.add_argument("--apply", action="store_true", help="write; default is a dry run")
    args = ap.parse_args()

    src_dir = os.path.expanduser(args.src)
    dest = os.path.expanduser(args.dest)
    dest_kg = os.path.expanduser(args.dest_kg)
    if os.path.realpath(src_dir) == os.path.realpath(dest):
        sys.exit("source and destination are the same palace")

    src = ro(os.path.join(src_dir, DB_NAME))
    check_embedder(src, os.path.join(dest, DB_NAME))
    rows = select_rows(src, args.wing, args.exclude_room, args.since)
    drawer_ids = [r[0] for r in rows.get("mempalace_drawers", [])]
    src_kg = ro(args.src_kg)
    triples, entities = select_kg(src_kg, args.wing, drawer_ids)

    print(f"source  {src_dir}")
    print(f"dest    {dest}")
    for name, r in rows.items():
        print(f"  {name}: {len(r)} rows")
    print(f"  kg: {len(triples)} triples, {len(entities)} entities")
    if not args.apply:
        print("dry run: nothing written (pass --apply)")
        return
    os.makedirs(dest, exist_ok=True)
    copy_drawers(dest, rows)
    copy_kg(dest_kg, triples, entities)
    print("done")


if __name__ == "__main__":
    main()
