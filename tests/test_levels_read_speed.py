"""The level filter's reads use the ``access`` index and page once.

With levels on, every read ANDs the access clause into its ``where``. Read
from ``metadata_json`` that clause parsed every row's JSON, and a paged sweep
re-ran it for every page: ``memp status`` on an 88k-drawer palace was 27 full
scans, a transcript ingest about 100. These tests pin the fast path and, more
importantly, that it returns exactly the rows the slow path did.
"""

from __future__ import annotations

import pytest

from mempalace import levels
from mempalace.backends import sqlite_exact as se
from mempalace.backends.base import PalaceRef
from mempalace.backends.sqlite_exact import SQLiteExactBackend

# Every shape an ``access`` value can take in a stored row: the integers the
# wrapper stamps, plus the strays a write past the wrapper could leave.
ACCESS_VALUES = [1, 2, 3, 2, 1, "2", "owners", True, False, 2.5, 0.5, None, "missing"]


def _open(path, *, read_only=False):
    backend = SQLiteExactBackend()
    ref = PalaceRef(id=str(path), local_path=str(path))
    col = backend.get_collection(
        palace=ref,
        collection_name="mempalace_drawers",
        create=not read_only,
        options={"read_only": read_only},
    )
    return backend, col


def _seed(col, n_per_value=1):
    ids, docs, metas, embs = [], [], [], []
    i = 0
    for _ in range(n_per_value):
        for value in ACCESS_VALUES:
            meta = {"wing": f"w{i % 3}", "source_file": f"/s/{i % 5}.md"}
            if value != "missing":
                meta["access"] = value
            ids.append(f"d{i:05d}")
            docs.append(f"doc {i}")
            metas.append(meta)
            embs.append([1.0, float(i % 7)])
            i += 1
    col.add(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
    return ids


def _level_where(level):
    """The clause ``levels.access_where`` builds at ``level`` with levels on."""
    if level <= levels.MISSING_ACCESS_LEVEL:
        return {"$or": [{"access": {"$gte": level}}, {"access": {"$eq": None}}]}
    return {"access": {"$gte": level}}


ACCESS_WHERES = [
    None,
    _level_where(2),
    _level_where(3),
    {"access": 2},
    {"access": {"$ne": 2}},
    {"access": {"$in": [1, None]}},
    {"access": {"$gt": "1"}},
    {"access": {"$lt": 2}},
    {"$and": [{"wing": "w1"}, _level_where(2)]},
]


@pytest.mark.parametrize("where", ACCESS_WHERES, ids=repr)
def test_access_column_matches_the_json_path(tmp_path, where):
    """Read through the generated column or through the JSON, the same rows."""
    _b, col = _open(tmp_path)
    _seed(col)
    assert col._handle.untyped_columns == frozenset({"access"})
    fast = col.get(where=where, include=["metadatas"])
    fast_page = col.get(where=where, limit=3, offset=2, include=["metadatas"])
    fast_count = col.count_where(where)

    col._handle.untyped_columns = frozenset()
    col._handle.indexed_untyped = frozenset()
    col._handle.page_candidates.clear()
    slow = col.get(where=where, include=["metadatas"])
    slow_page = col.get(where=where, limit=3, offset=2, include=["metadatas"])
    assert fast.ids == slow.ids
    assert fast_page.ids == slow_page.ids
    assert fast_count == len(slow.ids)


def test_level_clause_alone_is_forced_onto_the_access_index(tmp_path):
    _b, col = _open(tmp_path)
    _seed(col)
    statements: list[str] = []
    col._handle.conn.set_trace_callback(statements.append)
    try:
        col.get(where=_level_where(2), limit=2, offset=0, include=["metadatas"])
        col.count_where(_level_where(2))
        col.get(where={"$and": [{"wing": "w1"}, _level_where(2)]}, limit=2, offset=0)
    finally:
        col._handle.conn.set_trace_callback(None)
    candidate = [s for s in statements if s.startswith("SELECT rowid FROM documents")]
    counts = [s for s in statements if s.startswith("SELECT COUNT(*) FROM documents")]
    assert "INDEXED BY idx_documents_access" in candidate[0]
    assert "INDEXED BY idx_documents_access" in counts[0]
    assert "json_extract" not in candidate[0]
    # A caller filter on another key keeps the planner's choice.
    assert "INDEXED BY" not in candidate[1]


def _sweep(col, where, page=7):
    seen, offset = [], 0
    while True:
        got = col.get(where=where, limit=page, offset=offset, include=["metadatas"])
        if not got.ids:
            return seen
        seen.extend(got.ids)
        offset += len(got.ids)


def _scans(statements):
    return [s for s in statements if s.startswith("SELECT rowid FROM documents")]


def test_a_paged_sweep_scans_the_candidates_twice_not_per_page(tmp_path):
    """The first page keeps nothing (a one-off read leaves no list behind);
    the second scans and keeps; every later page reuses it."""
    _b, col = _open(tmp_path)
    _seed(col, n_per_value=10)
    where = _level_where(2)
    statements: list[str] = []
    col._handle.conn.set_trace_callback(statements.append)
    try:
        seen = _sweep(col, where)
    finally:
        col._handle.conn.set_trace_callback(None)
    assert seen == col.get(where=where).ids
    assert len(seen) > 7 * 3
    assert len(_scans(statements)) == 2, _scans(statements)


def test_a_one_page_read_keeps_nothing(tmp_path):
    _b, col = _open(tmp_path)
    _seed(col)
    col.get(where=_level_where(2), limit=5, offset=0)
    assert len(col._handle.page_candidates) == 0


def test_interleaved_sweeps_keep_their_own_candidates(tmp_path):
    """Two readers sharing a handle (the API, the daemon) page different
    filters in turn; neither sweep goes back to a scan per page."""
    _b, col = _open(tmp_path)
    _seed(col, n_per_value=10)
    a, b = _level_where(2), _level_where(3)
    statements: list[str] = []
    col._handle.conn.set_trace_callback(statements.append)
    seen_a, seen_b, offset = [], [], 0
    try:
        while True:
            pa = col.get(where=a, limit=5, offset=offset).ids
            pb = col.get(where=b, limit=5, offset=offset).ids
            if not pa and not pb:
                break
            seen_a += pa
            seen_b += pb
            offset += 5
    finally:
        col._handle.conn.set_trace_callback(None)
    assert seen_a == col.get(where=a).ids and seen_b == col.get(where=b).ids
    assert len(_scans(statements)) == 4, _scans(statements)


def test_the_cached_candidates_drop_on_this_connections_write(tmp_path):
    _b, col = _open(tmp_path)
    _seed(col)
    where = _level_where(2)
    first = col.get(where=where, limit=100, offset=0).ids
    col.add(ids=["zz-new"], documents=["new"], metadatas=[{"access": 2}], embeddings=[[0.0, 1.0]])
    again = col.get(where=where, limit=100, offset=0).ids
    assert again == first + ["zz-new"]
    col.delete(ids=["zz-new"])
    assert col.get(where=where, limit=100, offset=0).ids == first


def test_the_cached_candidates_drop_on_another_connections_write(tmp_path):
    _b1, writer = _open(tmp_path)
    _seed(writer)
    _b2, reader = _open(tmp_path, read_only=True)
    where = _level_where(2)
    first = reader.get(where=where, limit=100, offset=0).ids
    writer.add(
        ids=["zz-other"], documents=["new"], metadatas=[{"access": 3}], embeddings=[[0.0, 1.0]]
    )
    assert reader.get(where=where, limit=100, offset=0).ids == first + ["zz-other"]
    writer.update(ids=["zz-other"], metadatas=[{"access": 1}])
    assert reader.get(where=where, limit=100, offset=0).ids == first


def _strip_access_column(path):
    """Turn an upgraded palace file back into one from before the column."""
    import sqlite3

    conn = sqlite3.connect(str(path / "sqlite_exact.sqlite3"))
    conn.execute("DROP INDEX idx_documents_access")
    conn.execute("ALTER TABLE documents DROP COLUMN access")
    conn.commit()
    conn.close()


def test_a_palace_without_the_access_column_reads_the_json(tmp_path):
    """A read-only open of a palace no writer has upgraded has no column: the
    compiler must not name it, or every filtered read fails."""
    backend, col = _open(tmp_path)
    _seed(col)
    expected = col.get(where=_level_where(2)).ids
    backend.close()
    _strip_access_column(tmp_path)

    _b, ro = _open(tmp_path, read_only=True)
    assert ro._handle.untyped_columns == frozenset()
    assert ro._handle.indexed_untyped == frozenset()
    statements: list[str] = []
    ro._handle.conn.set_trace_callback(statements.append)
    try:
        assert _sweep(ro, _level_where(2), page=3) == expected
        assert ro.count_where(_level_where(2)) == len(expected)
    finally:
        ro._handle.conn.set_trace_callback(None)
    assert all("access" not in s.replace('"access"', "") for s in _scans(statements))


def test_a_read_only_handle_picks_up_a_later_upgrade(tmp_path, monkeypatch):
    """A live read-only handle (opened while a writer holds the WAL, so not
    immutable) sees a writer's upgrade on a later read."""
    import sqlite3

    backend, col = _open(tmp_path)
    _seed(col)
    backend.close()
    _strip_access_column(tmp_path)
    other = sqlite3.connect(str(tmp_path / "sqlite_exact.sqlite3"))
    other.execute("SELECT COUNT(*) FROM documents").fetchone()  # holds the WAL open
    _b, ro = _open(tmp_path, read_only=True)
    assert not ro._handle.immutable
    assert ro._handle.indexed_untyped == frozenset()

    _open(tmp_path)  # a writable open upgrades the file again
    monkeypatch.setattr(se, "_SCHEMA_RECHECK_SECONDS", 0.0)
    ro.count_where(_level_where(2))
    assert ro._handle.indexed_untyped == frozenset({"access"})
    other.close()


def test_the_level_wrapper_count_where_is_filtered(tmp_path, monkeypatch):
    """Left to __getattr__, count_where would count drawers the caller cannot see."""
    path = tmp_path / "levels.json"
    path.write_text('{"default": "team", "owners_folders": []}')
    monkeypatch.setenv("MEMPALACE_LEVELS_FILE", str(path))
    monkeypatch.delenv("MEMPALACE_LEVEL", raising=False)
    _b, inner = _open(tmp_path / "p")
    _seed(inner)
    col = levels.wrap_collection(inner)
    visible = len(inner.get(where=_level_where(2)).ids)
    assert col.count_where(None) == visible < inner.count()
    in_wing = len(inner.get(where={"$and": [{"wing": "w1"}, _level_where(2)]}).ids)
    assert col.count_where({"wing": "w1"}) == in_wing < inner.count_where({"wing": "w1"})


def test_the_level_wrapper_counts_with_one_query(tmp_path, monkeypatch):
    path = tmp_path / "levels.json"
    path.write_text('{"default": "team", "owners_folders": []}')
    monkeypatch.setenv("MEMPALACE_LEVELS_FILE", str(path))
    monkeypatch.delenv("MEMPALACE_LEVEL", raising=False)
    _b, inner = _open(tmp_path / "p")
    _seed(inner)
    col = levels.wrap_collection(inner)
    expected = len(inner.get(where=_level_where(2)).ids)

    def no_paging(**_kw):
        raise AssertionError("count() paged the collection instead of counting in SQL")

    monkeypatch.setattr(inner, "get", no_paging)
    assert col.count() == expected
    assert col.estimated_count() == expected
