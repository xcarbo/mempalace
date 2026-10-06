"""seed_palace.py must copy KG rows by column name, not by position.

A long-lived knowledge graph gained columns by ALTER TABLE over time, so its
`triples` columns are in a different order from a freshly created one. A
positional copy then writes `extracted_at` into `source_drawer_id` (seen
seeding the Phil palace, 2026-10-06).
"""

import importlib.util
import sqlite3
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "seed_palace.py"
_spec = importlib.util.spec_from_file_location("seed_palace", _SCRIPT)
seed = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(seed)

# Column order of an older, ALTER-TABLE-grown KG: extracted_at before source_drawer_id.
_LEGACY_TRIPLES = """CREATE TABLE triples (
    id TEXT PRIMARY KEY, subject TEXT NOT NULL, predicate TEXT NOT NULL, object TEXT NOT NULL,
    valid_from TEXT, valid_to TEXT, confidence REAL DEFAULT 1.0, source_closet TEXT,
    source_file TEXT, extracted_at TEXT DEFAULT CURRENT_TIMESTAMP,
    source_drawer_id TEXT, adapter_name TEXT)"""
_LEGACY_ENTITIES = """CREATE TABLE entities (
    id TEXT PRIMARY KEY, name TEXT NOT NULL, type TEXT DEFAULT 'unknown',
    properties TEXT DEFAULT '{}', created_at TEXT DEFAULT CURRENT_TIMESTAMP)"""


def _legacy_kg(path):
    con = sqlite3.connect(path)
    con.execute(_LEGACY_TRIPLES)
    con.execute(_LEGACY_ENTITIES)
    con.execute(
        "INSERT INTO entities (id, name, created_at) VALUES "
        "('drawer:d1', 'drawer:d1', '2026-01-01'), ('wing_w', 'wing_w', '2026-01-02')"
    )
    con.execute(
        "INSERT INTO triples (id, subject, predicate, object, valid_to, extracted_at, "
        "source_drawer_id, adapter_name) VALUES "
        "('t1', 'drawer:d1', 'belongs_to', 'wing_w', '2026-09-27', '2026-07-27 18:35:56', 'd1', 'ad')"
    )
    con.commit()
    return con


def test_copy_kg_maps_columns_by_name(tmp_path):
    src = _legacy_kg(tmp_path / "src.sqlite3")
    triples, entities = seed.select_kg(src, ["w"], ["d1"])
    dest = tmp_path / "dest.sqlite3"
    seed.copy_kg(str(dest), triples, entities)

    con = sqlite3.connect(dest)
    con.row_factory = sqlite3.Row
    row = con.execute("SELECT * FROM triples WHERE id = 't1'").fetchone()
    assert row["extracted_at"] == "2026-07-27 18:35:56"
    assert row["source_drawer_id"] == "d1"
    assert row["adapter_name"] == "ad"
    assert row["valid_to"] == "2026-09-27"
    ent = con.execute("SELECT * FROM entities WHERE id = 'wing_w'").fetchone()
    assert ent["created_at"] == "2026-01-02"


def test_select_kg_finds_triples_by_source_drawer(tmp_path):
    src = _legacy_kg(tmp_path / "src.sqlite3")
    src.execute(
        "INSERT INTO triples (id, subject, predicate, object, source_drawer_id) "
        "VALUES ('t2', 'x', 'mentions', 'y', 'd1')"
    )
    triples, _ = seed.select_kg(src, ["nomatch"], ["d1_chunk_000000"])
    assert {t["id"] for t in triples} == {"t1", "t2"}
