import math
import sqlite3
import threading

import pytest

from _chroma_palace_helper import make_minimal_chroma_sqlite, make_minimal_sqlite_exact_sqlite

import mempalace.backends.sqlite_exact as sqlite_exact_module
from mempalace.backends import (
    BackendMismatchError,
    CollectionNotInitializedError,
    DimensionMismatchError,
    PalaceRef,
    QueryResult,
    UnsupportedCapabilityError,
    available_backends,
)
from mempalace.backends.sqlite_exact import SQLiteExactBackend


def _collection(tmp_path, name="mempalace_drawers", create=True):
    backend = SQLiteExactBackend()
    palace = PalaceRef(id=str(tmp_path), local_path=str(tmp_path))
    return backend, backend.get_collection(palace=palace, collection_name=name, create=create)


def test_sqlite_exact_missing_collection_error_names_collection(tmp_path):
    """CollectionNotInitializedError must identify the missing collection, not
    the palace path — consistent with line 287 and the other backends."""
    backend, _ = _collection(tmp_path, name="mempalace_drawers")
    palace = PalaceRef(id=str(tmp_path), local_path=str(tmp_path))
    with pytest.raises(CollectionNotInitializedError) as exc:
        backend.get_collection(palace=palace, collection_name="does_not_exist", create=False)
    assert "does_not_exist" in str(exc.value)
    assert str(tmp_path) not in str(exc.value)

    with pytest.raises(CollectionNotInitializedError) as exc2:
        backend.delete_collection(str(tmp_path), "also_missing")
    assert "also_missing" in str(exc2.value)
    assert str(tmp_path) not in str(exc2.value)


def test_registry_exposes_sqlite_exact():
    assert "sqlite_exact" in available_backends()


def test_sqlite_exact_add_query_filters_and_persistence(tmp_path):
    backend, col = _collection(tmp_path)
    col.add(
        ids=["a", "b", "c"],
        documents=[
            "alpha vector memory",
            "beta sqlite exact memory",
            "gamma filtered memory",
        ],
        metadatas=[
            {"wing": "alpha", "room": "notes", "chunk_index": 0, "tags": "core,vector"},
            {"wing": "alpha", "room": "notes", "chunk_index": 1, "tags": "sqlite,exact"},
            {"wing": "gamma", "room": "archive", "chunk_index": 2, "tags": "old"},
        ],
        embeddings=[[1.0, 0.0], [0.0, 1.0], [0.2, 0.8]],
    )

    ranked = col.query(query_embeddings=[[1.0, 0.0]], n_results=3)
    assert ranked.ids[0] == ["a", "c", "b"]
    assert ranked.distances[0][0] == pytest.approx(0.0)

    filtered = col.get(
        where={
            "$and": [
                {"wing": "alpha"},
                {"chunk_index": {"$gte": 1}},
                {"tags": {"$contains": "sqlite"}},
            ]
        },
        include=["documents", "metadatas", "embeddings"],
    )
    assert filtered.ids == ["b"]
    assert filtered.documents == ["beta sqlite exact memory"]
    assert filtered.embeddings == [[0.0, 1.0]]

    col.update(ids=["b"], metadatas=[{"room": "lab"}])
    assert col.get(ids=["b"]).metadatas[0]["room"] == "lab"

    backend.close_palace(str(tmp_path))
    reopened = backend.get_collection(
        palace=PalaceRef(id=str(tmp_path), local_path=str(tmp_path)),
        collection_name="mempalace_drawers",
        create=False,
    )
    assert reopened.count() == 3
    assert reopened.get(ids=["a"]).documents == ["alpha vector memory"]


def test_sqlite_exact_write_failure_rolls_back_whole_batch(tmp_path):
    _backend, col = _collection(tmp_path)

    with pytest.raises(Exception):
        col.add(
            ids=["dup", "dup"],
            documents=["first write", "duplicate write"],
            metadatas=[{}, {}],
            embeddings=[[1.0, 0.0], [0.0, 1.0]],
        )

    assert col.count() == 0


def test_sqlite_exact_enforces_collection_dimension(tmp_path):
    _backend, col = _collection(tmp_path)
    col.add(ids=["a"], documents=["two dims"], metadatas=[{}], embeddings=[[1.0, 0.0]])

    with pytest.raises(DimensionMismatchError):
        col.add(ids=["b"], documents=["three dims"], metadatas=[{}], embeddings=[[1.0, 0.0, 0.0]])
    with pytest.raises(DimensionMismatchError):
        col.upsert(
            ids=["b"], documents=["three dims"], metadatas=[{}], embeddings=[[1.0, 0.0, 0.0]]
        )
    with pytest.raises(DimensionMismatchError):
        col.update(ids=["a"], embeddings=[[1.0, 0.0, 0.0]])
    with pytest.raises(DimensionMismatchError):
        col.query(query_embeddings=[[1.0, 0.0, 0.0]], n_results=1)

    assert col.count() == 1
    assert col.get(ids=["a"]).documents == ["two dims"]


def test_sqlite_exact_get_preserves_requested_id_order_and_duplicates(tmp_path):
    _backend, col = _collection(tmp_path)
    col.add(
        ids=["a", "b"],
        documents=["doc a", "doc b"],
        metadatas=[{}, {}],
        embeddings=[[1, 0], [0, 1]],
    )

    result = col.get(ids=["b", "a", "b"], include=["documents"])

    assert result.ids == ["b", "a", "b"]
    assert result.documents == ["doc b", "doc a", "doc b"]


def _doc_select_sql(col, action):
    """Run ``action`` while tracing SQL; return (result, [documents SELECTs]).

    The documents-table scan in ``_rows`` is the only statement that is both
    ``FROM documents`` and ``ORDER BY rowid`` (``count`` lacks the ORDER BY),
    so filtering on both isolates it from collection-id lookups and commits.
    """
    statements = []
    conn = col._handle.conn
    conn.set_trace_callback(statements.append)
    try:
        result = action()
    finally:
        conn.set_trace_callback(None)
    selects = [s for s in statements if "FROM documents" in s and "ORDER BY rowid" in s]
    return result, selects


def _seed(col, n):
    col.add(
        ids=[f"d{i}" for i in range(n)],
        documents=[f"doc {i}" for i in range(n)],
        metadatas=[{"wing": "w", "n": i} for i in range(n)],
        embeddings=[[float(i), 1.0] for i in range(n)],
    )


def test_sqlite_exact_get_unfiltered_page_pushes_limit_offset(tmp_path):
    _backend, col = _collection(tmp_path)
    _seed(col, 10)

    result, selects = _doc_select_sql(
        col, lambda: col.get(limit=3, offset=2, include=["documents"])
    )

    assert result.ids == ["d2", "d3", "d4"]
    assert result.documents == ["doc 2", "doc 3", "doc 4"]
    assert len(selects) == 1
    assert "LIMIT" in selects[0]
    assert "OFFSET" in selects[0]


def test_sqlite_exact_get_filtered_page_compiles_without_sql_page(tmp_path):
    _backend, col = _collection(tmp_path)
    _seed(col, 6)

    # A translatable filter compiles to SQL, but the compiled scan is fetched
    # unordered (no ORDER BY — see the planner note in _rows) and sorted in
    # Python, so LIMIT/OFFSET must not reach SQL; the page is taken in Python
    # over the rowid-sorted filtered rows.
    result, statements = _traced_sql(
        col,
        lambda: col.get(where={"wing": "w"}, limit=2, offset=1, include=["metadatas"]),
    )
    scans = [s for s in statements if "metadata_json" in s and "FROM documents" in s]

    assert result.ids == ["d1", "d2"]
    assert len(scans) == 1
    assert "ORDER BY" not in scans[0]
    assert "LIMIT" not in scans[0]
    assert "OFFSET" not in scans[0]
    # The compiled predicate reached SQL (no Python fallback scan).
    assert "wing" in scans[0]


def _traced_sql(col, action):
    """Run ``action`` while tracing every SQL statement on the connection."""
    statements = []
    conn = col._handle.conn
    conn.set_trace_callback(statements.append)
    try:
        result = action()
    finally:
        conn.set_trace_callback(None)
    return result, statements


def test_sqlite_exact_filtered_get_skips_embedding_column(tmp_path):
    """The filtered get() must not drag 1.5KB embedding blobs through the scan
    unless the caller asked for embeddings — that fetch was the dominant cost
    of the 2026-08-09 cutover latency tail (775ms per hydration call)."""
    _backend, col = _collection(tmp_path)
    _seed(col, 4)

    _result, statements = _traced_sql(
        col, lambda: col.get(where={"wing": "w"}, include=["documents", "metadatas"])
    )
    scans = [s for s in statements if "FROM documents" in s and "metadata_json" in s]
    assert scans and all("embedding" not in s for s in scans)

    with_embed, statements = _traced_sql(
        col, lambda: col.get(where={"wing": "w"}, include=["embeddings"])
    )
    scans = [s for s in statements if "FROM documents" in s and "metadata_json" in s]
    assert scans and any("embedding" in s for s in scans)
    assert with_embed.embeddings and with_embed.embeddings[0] == [0.0, 1.0]


def test_sqlite_exact_filtered_get_uses_index_without_analyze(tmp_path):
    """The scoped hydration filter must hit the generated-column index on a
    freshly built store with no sqlite_stat1 rows. Keeping ORDER BY rowid in
    the compiled SQL makes the stat-less planner walk the whole
    (collection_id) index to avoid the sort — 183ms per call on 177k rows —
    which silently re-creates the cutover latency tail on every fresh build."""
    _backend, col = _collection(tmp_path)
    col.add(
        ids=["a", "b"],
        documents=["da", "db"],
        metadatas=[
            {"source_file": "f.md", "parent_drawer_id": "p1", "chunk_index": 0},
            {"source_file": "g.md", "parent_drawer_id": "p2", "chunk_index": 1},
        ],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
    )
    conn = col._handle.conn
    assert not conn.execute(
        "SELECT name FROM sqlite_master WHERE name LIKE 'sqlite_stat%'"
    ).fetchall(), "test premise: no ANALYZE stats present"
    where = {"$and": [{"source_file": "f.md"}, {"parent_drawer_id": "p1"}]}
    result, statements = _traced_sql(col, lambda: col.get(where=where, include=["metadatas"]))
    assert result.ids == ["a"]
    scan = next(s for s in statements if "FROM documents" in s and "metadata_json" in s)
    # The trace callback yields the statement with parameters expanded.
    plan = " | ".join(row[3] for row in conn.execute("EXPLAIN QUERY PLAN " + scan).fetchall())
    # Either scoped generated-column index is fine; a bare (collection_id)
    # walk is the regression.
    assert "idx_documents_source_file" in plan or "idx_documents_parent_drawer_id" in plan, plan
    assert "USING INDEX idx_documents_collection " not in plan, plan


def test_sqlite_exact_untranslatable_get_falls_back_to_python_scan(tmp_path):
    _backend, col = _collection(tmp_path)
    col.add(
        ids=["a", "b"],
        documents=["da", "db"],
        metadatas=[{'we"ird': "x", "wing": "w"}, {'we"ird': "y", "wing": "w"}],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
    )
    # Unquotable JSON-path key → _WhereNotTranslatable → Python scan, same rows.
    result = col.get(where={'we"ird': "x"}, include=["metadatas"])
    assert result.ids == ["a"]
    assert result.metadatas[0]['we"ird'] == "x"


def _seed_equivalence_fixture(col):
    """Rows exercising every metadata shape the filter grammar can meet:
    missing keys, None values, bools (stored as JSON true/false), ints,
    floats, unicode text, empty strings, and mixed types under one key."""
    col.add(
        ids=["r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7"],
        documents=[
            "alpha beta",
            "beta gamma",
            "Ünïcode dräwer 日本語",
            "",
            "delta",
            "epsilon zeta",
            "eta theta",
            "iota kappa",
        ],
        metadatas=[
            {"wing": "w1", "n": 0, "flag": True, "score": 1.5},
            {"wing": "w1", "n": 1, "flag": False, "score": 0.0},
            {"wing": "wü", "n": 2, "label": "Ünïcode 值"},
            {"wing": "w2", "n": None, "label": ""},
            {"wing": "w2", "n": 4},
            {"n": "5"},  # n as str on this row, int elsewhere
            {"wing": "w3", "score": -2},
            {},
        ],
        embeddings=[[float(i), 1.0] for i in range(8)],
    )


_EQUIVALENCE_FILTERS = [
    None,
    {},
    {"wing": "w1"},
    {"wing": "wü"},
    {"label": "Ünïcode 值"},
    {"label": ""},
    {"n": None},
    {"missing": None},
    {"flag": True},
    {"flag": False},
    {"n": 1},
    {"n": "5"},
    {"score": 1.5},
    {"wing": {"$ne": "w1"}},
    {"missing": {"$ne": "x"}},
    {"n": {"$ne": None}},
    {"n": {"$in": [0, "5", None]}},
    {"n": {"$nin": [0, 1]}},
    {"n": {"$in": []}},
    {"n": {"$gt": 0}},
    {"n": {"$gte": 0}},
    {"n": {"$lt": 2}},
    {"score": {"$lte": 1.5}},
    {"n": {"$gt": "4"}},  # str operand: only str values compare
    {"wing": {"$contains": "w"}},
    {"n": {"$contains": "5"}},
    {"label": {"$contains": ""}},
    {"$and": [{"wing": "w1"}, {"flag": True}]},
    {"$or": [{"wing": "w3"}, {"n": 4}]},
    {"$and": [{"$or": [{"wing": "w1"}, {"wing": "w2"}]}, {"n": {"$ne": 1}}]},
    {"$and": []},
    {"$or": []},
]

_EQUIVALENCE_WHERE_DOCS = [
    None,
    {"$contains": "beta"},
    {"$contains": "日本語"},
    {"$contains": ""},
    {"$or": [{"$contains": "alpha"}, {"$contains": "kappa"}]},
]


def test_sqlite_exact_get_compiled_sql_matches_python_scan(tmp_path, monkeypatch):
    """The compiled-SQL filter path and the Python fallback must return the
    same rows for every expressible filter shape — a silent difference in
    WHICH rows come back would never show up in a recall metric."""
    import mempalace.backends.sqlite_exact as se

    _backend, col = _collection(tmp_path)
    _seed_equivalence_fixture(col)

    def forced(*_a, **_k):
        raise se._WhereNotTranslatable("forced fallback")

    for where in _EQUIVALENCE_FILTERS:
        for where_doc in _EQUIVALENCE_WHERE_DOCS:
            kwargs = dict(
                where=where,
                where_document=where_doc,
                include=["documents", "metadatas", "embeddings"],
            )
            compiled = col.get(**kwargs)
            with monkeypatch.context() as m:
                m.setattr(se, "_compile_where", forced)
                m.setattr(se, "_compile_where_document", forced)
                fallback = col.get(**kwargs)
            label = f"where={where!r} where_document={where_doc!r}"
            assert compiled.ids == fallback.ids, label
            assert compiled.documents == fallback.documents, label
            assert compiled.metadatas == fallback.metadatas, label
            assert compiled.embeddings == fallback.embeddings, label


def test_sqlite_exact_index_version_migrates_v2_to_v3(tmp_path):
    backend, col = _collection(tmp_path)
    _seed(col, 3)
    conn = col._handle.conn
    # Simulate a v2 palace: drop the v3-only indexes and stamp version 2.
    conn.execute("DROP INDEX IF EXISTS idx_documents_source_file")
    conn.execute("DROP INDEX IF EXISTS idx_documents_parent_drawer_id")
    conn.execute("UPDATE meta SET value = '2' WHERE key = 'index_version'")
    conn.commit()
    backend.close()

    backend2, col2 = _collection(tmp_path)
    conn2 = col2._handle.conn
    names = {
        row[0]
        for row in conn2.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index' AND name LIKE 'idx_documents_%'"
        ).fetchall()
    }
    assert {"idx_documents_source_file", "idx_documents_parent_drawer_id"} <= names
    version = conn2.execute("SELECT value FROM meta WHERE key = 'index_version'").fetchone()[0]
    assert version == "3"
    backend2.close()


def test_sqlite_exact_get_offset_only_and_limit_only_push(tmp_path):
    _backend, col = _collection(tmp_path)
    _seed(col, 5)

    limit_only, limit_sql = _doc_select_sql(col, lambda: col.get(limit=2))
    assert limit_only.ids == ["d0", "d1"]
    assert len(limit_sql) == 1
    assert "LIMIT" in limit_sql[0]
    assert "OFFSET" not in limit_sql[0]

    offset_only, offset_sql = _doc_select_sql(col, lambda: col.get(offset=3))
    assert offset_only.ids == ["d3", "d4"]
    assert len(offset_sql) == 1
    assert "OFFSET" in offset_sql[0]
    # SQLite requires a LIMIT before OFFSET; an offset-only page uses LIMIT -1.
    assert "LIMIT" in offset_sql[0]


def test_sqlite_exact_get_negative_bounds_use_python_slice(tmp_path):
    _backend, col = _collection(tmp_path)
    _seed(col, 5)

    # Negative limit means Python "all but last", which a SQL LIMIT (negative ==
    # unbounded in SQLite) cannot express, so it must stay on the slice path.
    neg_limit, neg_limit_sql = _doc_select_sql(col, lambda: col.get(limit=-1))
    assert neg_limit.ids == ["d0", "d1", "d2", "d3"]
    assert len(neg_limit_sql) == 1
    assert "LIMIT" not in neg_limit_sql[0]

    # Negative offset means Python "last N"; it must not reach SQL either.
    neg_offset, neg_offset_sql = _doc_select_sql(col, lambda: col.get(offset=-2))
    assert neg_offset.ids == ["d3", "d4"]
    assert len(neg_offset_sql) == 1
    assert "OFFSET" not in neg_offset_sql[0]


def test_sqlite_exact_get_pages_tile_without_overlap(tmp_path):
    _backend, col = _collection(tmp_path)
    _seed(col, 10)

    seen = []
    offset = 0
    while True:
        page = col.get(limit=4, offset=offset)
        if not page.ids:
            break
        seen.extend(page.ids)
        offset += len(page.ids)

    assert seen == [f"d{i}" for i in range(10)]
    # The same set, same rowid order, as a single unfiltered scan.
    assert col.get().ids == seen


def test_sqlite_exact_get_limit_zero_pushes_empty_page(tmp_path):
    _backend, col = _collection(tmp_path)
    _seed(col, 3)

    # limit=0 is a real bound, not "no limit": it pushes LIMIT 0 and returns
    # nothing, matching the old rows[:0] slice. Guards the `is not None` check
    # against an `if limit:` regression that would treat 0 as unbounded.
    result, selects = _doc_select_sql(col, lambda: col.get(limit=0))
    assert result.ids == []
    assert len(selects) == 1
    assert "LIMIT" in selects[0]


def test_sqlite_exact_get_offset_zero_is_a_full_scan(tmp_path):
    _backend, col = _collection(tmp_path)
    _seed(col, 3)

    # offset=0 with no limit is not a page request, so it stays on the full scan.
    result, selects = _doc_select_sql(col, lambda: col.get(offset=0))
    assert result.ids == ["d0", "d1", "d2"]
    assert len(selects) == 1
    assert "LIMIT" not in selects[0]
    assert "OFFSET" not in selects[0]


def test_sqlite_exact_get_ids_with_page_slices_in_python(tmp_path):
    _backend, col = _collection(tmp_path)
    _seed(col, 5)

    # ids force the Python path even with a page: the requested order is kept,
    # then offset/limit slice the reordered list with no SQL LIMIT/OFFSET.
    result, selects = _doc_select_sql(
        col, lambda: col.get(ids=["d4", "d3", "d2", "d1"], offset=1, limit=2)
    )
    assert result.ids == ["d3", "d2"]
    assert len(selects) == 1
    assert "LIMIT" not in selects[0]
    assert "OFFSET" not in selects[0]


def test_sqlite_exact_upsert_delete_and_multi_collection_isolation(tmp_path):
    backend, drawers = _collection(tmp_path, "drawers")
    palace = PalaceRef(id=str(tmp_path), local_path=str(tmp_path))
    closets = backend.get_collection(palace=palace, collection_name="closets", create=True)

    drawers.upsert(
        ids=["same"], documents=["drawer one"], metadatas=[{"kind": "drawer"}], embeddings=[[1, 0]]
    )
    closets.upsert(
        ids=["same"], documents=["closet one"], metadatas=[{"kind": "closet"}], embeddings=[[0, 1]]
    )
    drawers.upsert(
        ids=["same"],
        documents=["drawer replaced"],
        metadatas=[{"kind": "drawer", "version": 2}],
        embeddings=[[1, 0]],
    )

    assert drawers.count() == 1
    assert closets.count() == 1
    assert drawers.get(ids=["same"]).documents == ["drawer replaced"]
    assert closets.get(ids=["same"]).documents == ["closet one"]

    drawers.delete(where={"version": {"$in": [2, 3]}})
    assert drawers.count() == 0
    assert closets.count() == 1


def test_sqlite_exact_lexical_search_and_python_fallback(tmp_path, monkeypatch):
    _backend, col = _collection(tmp_path)
    col.add(
        ids=["a", "b", "c"],
        documents=[
            "ordinary project note",
            "rareterm rareterm sqlite exact note",
            "rareterm unrelated archive",
        ],
        metadatas=[
            {"wing": "w", "room": "a"},
            {"wing": "w", "room": "b"},
            {"wing": "old", "room": "b"},
        ],
        embeddings=[[1, 0], [0, 1], [0.5, 0.5]],
    )

    hits = col.lexical_search(query="rareterm sqlite", n_results=2, where={"wing": "w"}).hits
    assert [hit.id for hit in hits] == ["b"]

    monkeypatch.setattr(col, "_fts_available", lambda _cur: False)
    fallback_hits = col.lexical_search(query="rareterm sqlite", n_results=2).hits
    assert fallback_hits[0].id == "b"


def test_sqlite_exact_lexical_search_filters_after_full_fts_window(tmp_path):
    _backend, col = _collection(tmp_path)
    ids = [f"old-{i}" for i in range(12)] + ["target"]
    col.add(
        ids=ids,
        documents=["needle shared lexical note" for _ in ids],
        metadatas=[{"wing": "old"} for _ in range(12)] + [{"wing": "target"}],
        embeddings=[[1.0, 0.0] for _ in ids],
    )

    hits = col.lexical_search(query="needle", n_results=1, where={"wing": "target"}).hits

    assert [hit.id for hit in hits] == ["target"]


def test_sqlite_exact_lexical_filtered_rescores_window_not_fts_rank(tmp_path):
    """The filtered lane must rank by the window-relative Okapi rescore, not
    by FTS5's whole-corpus bm25 — and must fetch a window wider than
    ``n_results`` to do it.

    Out-of-scope "banana" filler drives FTS5's corpus IDF for "banana"
    negative, so FTS5 ranks the apple-stuffed docB above docA ("apple
    banana"). Within the wing-scoped window, "banana" is the discriminative
    term and the Okapi rescore ranks docA first. The pre-fix code (LIMIT
    n_results by raw FTS rank) returned docB here — the exact failure mode
    behind the 2026-08-09 golden regressions (ct-fp-three-buckets,
    ct-thewill-adgm, arc-liq4life-sdlt-legal).
    """
    _backend, col = _collection(tmp_path)
    filler_ids = [f"noise-{i}" for i in range(30)]
    col.add(
        ids=["docA", "docB", *filler_ids],
        documents=[
            "apple banana",
            "apple apple apple apple",
            *["banana banana" for _ in filler_ids],
        ],
        metadatas=[
            {"wing": "w"},
            {"wing": "w"},
            *[{"wing": "elsewhere"} for _ in filler_ids],
        ],
        embeddings=[[1.0, 0.0] for _ in range(len(filler_ids) + 2)],
    )

    hits = col.lexical_search(query="apple banana", n_results=1, where={"wing": "w"}).hits

    assert [hit.id for hit in hits] == ["docA"]
    assert hits[0].score > 0


def test_sqlite_exact_lexical_short_tokens_skip_fts_but_keep_python_fallback(tmp_path):
    """2-char tokens are dropped from FTS candidate generation (unicode61 has
    no substring matching, so they only flood the OR match set), matching the
    chroma lane's ≥3 floor. A query of ONLY short tokens must not go dark:
    it skips FTS entirely and the Python BM25 scan still finds the doc.
    """
    _backend, col = _collection(tmp_path)
    col.add(
        ids=["short", "long"],
        documents=["ab ab ab", "needle note"],
        metadatas=[{"wing": "w"}, {"wing": "w"}],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
    )

    fts_hits = col.lexical_search(query="ab needle", n_results=5).hits
    assert [hit.id for hit in fts_hits] == ["long"]

    fallback_hits = col.lexical_search(query="ab", n_results=5).hits
    assert [hit.id for hit in fallback_hits] == ["short"]


def test_sqlite_exact_logical_filters_evaluate_sibling_predicates(tmp_path):
    _backend, col = _collection(tmp_path)
    col.add(
        ids=["a", "b"],
        documents=["alpha document", "beta document"],
        metadatas=[
            {"wing": "w", "room": "wrong", "kind": "note"},
            {"wing": "w", "room": "right", "kind": "note"},
        ],
        embeddings=[[1, 0], [0, 1]],
    )

    result = col.get(where={"$and": [{"wing": "w"}], "room": "right"})

    assert result.ids == ["b"]


def test_sqlite_exact_close_palace_marks_existing_collections_closed(tmp_path):
    backend, col = _collection(tmp_path)
    palace = PalaceRef(id=str(tmp_path), local_path=str(tmp_path))
    col.add(ids=["a"], documents=["doc"], metadatas=[{}], embeddings=[[1, 0]])

    backend.close_palace(palace)

    assert not col.health().ok
    with pytest.raises(Exception):
        col.count()


def test_palace_wrapper_embeds_for_sqlite_exact(tmp_path, monkeypatch):
    import mempalace.backends.embedding_wrapper as embedding_wrapper
    from mempalace.palace import get_collection

    monkeypatch.setenv("MEMPALACE_BACKEND_EXPLICIT", "sqlite_exact")
    monkeypatch.setattr(
        embedding_wrapper,
        "_embed_texts",
        lambda texts, is_query=False: [[float(len(text)), 1.0] for text in texts],
    )

    col = get_collection(str(tmp_path), create=True)
    col.add(ids=["a"], documents=["abcd"], metadatas=[{"wing": "w"}])

    result = col.query(query_texts=["abcd"], n_results=1)
    assert result.ids == [["a"]]


def test_backend_mismatch_protection(tmp_path, monkeypatch):
    from mempalace.palace import get_collection

    make_minimal_chroma_sqlite(tmp_path)
    monkeypatch.setenv("MEMPALACE_BACKEND_EXPLICIT", "sqlite_exact")

    with pytest.raises(BackendMismatchError):
        get_collection(str(tmp_path), create=True)


def test_mixed_backend_artifacts_are_rejected_even_when_chroma_selected(tmp_path, monkeypatch):
    from mempalace.palace import resolve_backend_name

    make_minimal_chroma_sqlite(tmp_path)
    make_minimal_sqlite_exact_sqlite(tmp_path)
    monkeypatch.setenv("MEMPALACE_BACKEND_EXPLICIT", "chroma")

    with pytest.raises(BackendMismatchError):
        resolve_backend_name(str(tmp_path))


def test_sqlite_exact_detect_matches_palace_with_sqlite_header(tmp_path):
    """A real SQLite database at ``<path>/sqlite_exact.sqlite3`` registers
    as sqlite_exact. Mirrors the chroma analog at
    ``test_chroma_detect_matches_palace_with_sqlite_header``.
    """
    make_minimal_sqlite_exact_sqlite(tmp_path)
    assert SQLiteExactBackend.detect(str(tmp_path)) is True
    assert SQLiteExactBackend.detect(str(tmp_path.parent)) is False


def test_sqlite_exact_detect_rejects_empty_sqlite_exact_sqlite(tmp_path):
    """A 0-byte ``sqlite_exact.sqlite3`` is not a sqlite_exact palace (#1893).

    Same root cause as the chroma side: bare ``sqlite3.connect()`` against
    a missing path leaves a 0-byte file behind because the SQLite header is
    written on the first statement, not on connect. Detection must reject
    that artifact so it cannot trip ``BackendMismatchError`` against a real
    non-sqlite_exact backend marker in the same directory.
    """
    (tmp_path / "sqlite_exact.sqlite3").write_bytes(b"")
    assert SQLiteExactBackend.detect(str(tmp_path)) is False


def test_sqlite_exact_detect_rejects_non_sqlite_file(tmp_path):
    """A non-SQLite file at the ``sqlite_exact.sqlite3`` path is not
    sqlite_exact. Defends against partial writes / garbage content / anything
    that lands at the canonical path but isn't actually a SQLite database.
    """
    (tmp_path / "sqlite_exact.sqlite3").write_bytes(b"not a sqlite file" * 4)
    assert SQLiteExactBackend.detect(str(tmp_path)) is False


def test_sqlite_exact_exact_ranking_uses_cosine(tmp_path):
    _backend, col = _collection(tmp_path)
    halfway = [0.5, math.sqrt(0.75)]
    col.add(
        ids=["half", "orthogonal", "same"],
        documents=["half", "orthogonal", "same"],
        metadatas=[{}, {}, {}],
        embeddings=[halfway, [0.0, 1.0], [1.0, 0.0]],
    )

    result = col.query(query_embeddings=[[1.0, 0.0]], n_results=3)
    assert result.ids[0] == ["same", "half", "orthogonal"]
    assert result.distances[0] == pytest.approx([0.0, 0.5, 1.0])


def test_search_union_uses_sqlite_exact_lexical_search(tmp_path, monkeypatch):
    import mempalace.backends.embedding_wrapper as embedding_wrapper
    from mempalace.palace import get_collection
    from mempalace.searcher import search_memories

    def fake_embed(texts, is_query=False):
        vectors = []
        for text in texts:
            if text == "rareterm":
                vectors.append([1.0, 0.0])
            elif "rareterm" in text:
                vectors.append([0.0, 1.0])
            else:
                vectors.append([0.5, math.sqrt(0.75)])
        return vectors

    monkeypatch.setenv("MEMPALACE_BACKEND_EXPLICIT", "sqlite_exact")
    monkeypatch.setattr(embedding_wrapper, "_embed_texts", fake_embed)
    # Disable the archive-mode vector floor (_VECTOR_CANDIDATE_FLOOR=60): on
    # this 4-doc corpus it would let the vector lane return every doc, so the
    # rare doc could never demonstrate arriving through the lexical lane.
    monkeypatch.setenv("MEMPALACE_ARCHIVE_WINGS", "")

    col = get_collection(str(tmp_path), create=True)
    col.add(
        ids=["d1", "d2", "d3", "rare"],
        documents=[
            "ordinary support note",
            "ordinary billing note",
            "ordinary project note",
            "rareterm rareterm rareterm policy note",
        ],
        metadatas=[
            {"wing": "w", "room": "r", "source_file": "/tmp/d1.md", "chunk_index": 0},
            {"wing": "w", "room": "r", "source_file": "/tmp/d2.md", "chunk_index": 0},
            {"wing": "w", "room": "r", "source_file": "/tmp/d3.md", "chunk_index": 0},
            {"wing": "w", "room": "r", "source_file": "/tmp/rare.md", "chunk_index": 0},
        ],
    )

    result = search_memories(
        "rareterm",
        str(tmp_path),
        n_results=1,
        candidate_strategy="union",
    )

    assert result["results"][0]["source_file"] == "rare.md"
    assert result["results"][0]["matched_via"] == "bm25_backend"


def test_search_union_reports_unsupported_lexical_capability(monkeypatch, tmp_path):
    import mempalace.searcher as searcher

    class NoLexicalCollection:
        def query(self, **_kwargs):
            return QueryResult(
                ids=[["a"]],
                documents=[["ordinary note"]],
                metadatas=[[{"source_file": "/tmp/a.md", "chunk_index": 0}]],
                distances=[[0.5]],
            )

        def lexical_search(self, **_kwargs):
            raise UnsupportedCapabilityError("no lexical support")

    monkeypatch.setattr(searcher, "get_collection", lambda *_args, **_kwargs: NoLexicalCollection())
    monkeypatch.setattr(
        searcher,
        "get_closets_collection",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("no closets")),
    )

    result = searcher.search_memories(
        "anything",
        str(tmp_path),
        n_results=1,
        candidate_strategy="union",
    )

    assert result["unsupported_capability"] == "supports_lexical_search"


def test_search_vector_disabled_fallback_is_chroma_only(tmp_path, monkeypatch):
    from mempalace.searcher import search_memories

    monkeypatch.setenv("MEMPALACE_BACKEND_EXPLICIT", "sqlite_exact")

    result = search_memories("anything", str(tmp_path), vector_disabled=True)

    assert result["unsupported_capability"] == "chroma_hnsw_fallback"
    assert result["backend"] == "sqlite_exact"


def test_concurrent_first_open_single_connection_no_leak(tmp_path, monkeypatch):
    """Two threads first-opening the same palace concurrently must share one
    handle and one sqlite connection.

    The barrier inside the patched ``sqlite3.connect`` releases immediately
    only when both threads pass the cache-miss check together: the broken
    interleaving, which also ran ``_init_schema`` concurrently on a fresh
    file and surfaced "database is locked". With creation serialized under
    ``_clients_lock`` the second thread waits on the lock instead, the
    winner's barrier times out, and exactly one connection is ever created.
    """
    created = []
    barrier = threading.Barrier(2)
    real_connect = sqlite3.connect

    def racing_connect(*args, **kwargs):
        try:
            barrier.wait(timeout=1.0)
        except threading.BrokenBarrierError:
            pass
        conn = real_connect(*args, **kwargs)
        created.append(conn)
        return conn

    monkeypatch.setattr(sqlite_exact_module.sqlite3, "connect", racing_connect)

    backend = SQLiteExactBackend()
    palace = PalaceRef(id=str(tmp_path), local_path=str(tmp_path))
    results = [None, None]
    errors = []

    def open_collection(i):
        try:
            results[i] = backend.get_collection(
                palace=palace, collection_name="drawers", create=True
            )
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=open_collection, args=(i,), daemon=True) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not any(t.is_alive() for t in threads)
    assert errors == []
    assert len(created) == 1
    assert results[0]._handle is results[1]._handle

    backend.close()
    with pytest.raises(sqlite3.ProgrammingError):
        created[0].execute("SELECT 1")
