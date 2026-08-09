"""Tests for the sqlite_exact matmul query path (vector cache + SQL filters).

Three things are pinned here:

1. The SQL-compiled filter path returns *exactly* the rows the reference
   Python evaluator (``_matches_where`` / ``_matches_where_document``) accepts,
   via a differential sweep over adversarial metadata and filters.
2. The vector cache appends on pure inserts, rebuilds on mutation, and stays
   correct across separate connections (cross-process protocol).
3. The read path performs no writes.
"""

import sqlite3

import numpy as np
import pytest

from mempalace.backends.sqlite_exact import (
    SQLiteExactBackend,
    SQLiteExactCollection,
    _matches_where,
    _matches_where_document,
)

DIM = 6


def _mk_collection(tmp_path, name="drawers"):
    backend = SQLiteExactBackend()
    col = backend.get_collection(str(tmp_path), name, create=True)
    return backend, col


def _seed(col, n=40, seed=0):
    rng = np.random.default_rng(seed)
    vecs = rng.normal(size=(n, DIM)).astype(np.float32)
    ids = [f"d{i:03d}" for i in range(n)]
    metas = []
    for i in range(n):
        meta = {
            "wing": f"w{i % 4}",
            "room": ["decisions", "technical", "events"][i % 3],
            "authored_at": f"2026-0{1 + i % 9}-15T12:00:00",
            "chunk_index": i % 5,
            "score": round(float(i) / 7.0, 3),
            "flag": bool(i % 2),
        }
        if i % 7 == 0:
            meta["rare"] = "yes"
        if i % 11 == 0:
            meta["nullish"] = None
        metas.append(meta)
    docs = [f"drawer {i} body with token{i % 6} and shared words" for i in range(n)]
    col.add(documents=docs, ids=ids, metadatas=metas, embeddings=vecs.tolist())
    return ids, docs, metas, vecs


# ---------------------------------------------------------------------------
# 1. Differential: SQL filter compilation vs the Python reference evaluator
# ---------------------------------------------------------------------------

FILTER_CASES = [
    None,
    {"wing": "w1"},
    {"wing": {"$eq": "w2"}},
    {"wing": {"$ne": "w0"}},
    {"missing_key": {"$ne": "anything"}},  # missing key matches $ne
    {"missing_key": "anything"},  # missing key fails equality
    {"nullish": None},  # explicit JSON null equals None
    {"flag": True},
    {"flag": {"$ne": False}},
    {"chunk_index": 3},
    {"chunk_index": {"$gt": 2}},
    {"chunk_index": {"$gte": 2, "$lt": 4}},
    {"score": {"$lte": 2.0}},
    {"wing": {"$gt": "w1"}},  # string ordering
    {"wing": {"$gt": 5}},  # type mismatch → never matches (Python TypeError)
    {"chunk_index": {"$gt": "2"}},  # numeric actual vs str operand → no match
    {"wing": {"$in": ["w0", "w3"]}},
    {"wing": {"$in": []}},
    {"wing": {"$nin": ["w0", "w3"]}},
    {"missing_key": {"$nin": ["a"]}},  # missing key matches $nin
    {"missing_key": {"$in": ["a"]}},
    {"chunk_index": {"$in": [0, 4]}},
    {"flag": {"$in": [True]}},
    {"rare": {"$contains": "ye"}},
    {"wing": {"$contains": "3"}},
    {"chunk_index": {"$contains": "0"}},  # falsy actual stringifies to ""
    {"$and": [{"wing": "w1"}, {"room": "technical"}]},
    {"$or": [{"wing": "w0"}, {"rare": "yes"}]},
    {"$and": [{"$or": [{"wing": "w0"}, {"wing": "w1"}]}, {"chunk_index": {"$lt": 3}}]},
    {"$and": []},
    {"$or": []},
    {"wing": "w1", "room": "decisions"},  # implicit AND
    {"authored_at": {"$gte": "2026-03"}},
    {"authored_at": {"$gt": "2026-03", "$lt": "2026-07"}},
]

DOC_FILTER_CASES = [
    None,
    {"$contains": "token2"},
    {"$contains": "absent-token"},
    {"$contains": ""},
    {"$and": [{"$contains": "shared"}, {"$contains": "token1"}]},
    {"$or": [{"$contains": "token0"}, {"$contains": "token5"}]},
]


@pytest.mark.parametrize("where", FILTER_CASES)
@pytest.mark.parametrize("where_document", DOC_FILTER_CASES)
def test_query_filter_matches_python_reference(tmp_path, where, where_document):
    backend, col = _mk_collection(tmp_path)
    try:
        ids, docs, metas, vecs = _seed(col)
        expected = {
            ids[i]
            for i in range(len(ids))
            if _matches_where(metas[i], where) and _matches_where_document(docs[i], where_document)
        }
        q = vecs[0].tolist()
        result = col.query(
            query_embeddings=[q],
            n_results=len(ids),
            where=where,
            where_document=where_document,
        )
        assert set(result.ids[0]) == expected
    finally:
        backend.close()


@pytest.mark.parametrize("where", [w for w in FILTER_CASES if w])
def test_lexical_filter_matches_python_reference(tmp_path, where):
    backend, col = _mk_collection(tmp_path)
    try:
        ids, docs, metas, _ = _seed(col)
        allowed = {ids[i] for i in range(len(ids)) if _matches_where(metas[i], where)}
        result = col.lexical_search(query="shared words", n_results=len(ids), where=where)
        got = {hit.id for hit in result.hits}
        # Every doc contains "shared words", so the filtered lexical lane must
        # return exactly the allowed set (bounded by n_results=len(ids)).
        assert got == allowed
        for hit in result.hits:
            assert _matches_where(hit.metadata, where)
    finally:
        backend.close()


def test_untranslatable_filter_key_falls_back_to_python_scan(tmp_path):
    backend, col = _mk_collection(tmp_path)
    try:
        rng = np.random.default_rng(1)
        vecs = rng.normal(size=(4, DIM)).astype(np.float32)
        weird_key = 'we"ird\\key'
        metas = [{weird_key: "yes" if i % 2 else "no"} for i in range(4)]
        col.add(
            documents=[f"doc {i}" for i in range(4)],
            ids=[f"x{i}" for i in range(4)],
            metadatas=metas,
            embeddings=vecs.tolist(),
        )
        result = col.query(
            query_embeddings=[vecs[0].tolist()], n_results=10, where={weird_key: "yes"}
        )
        assert set(result.ids[0]) == {"x1", "x3"}
    finally:
        backend.close()


def test_query_ranking_and_distances_are_exact_cosine(tmp_path):
    backend, col = _mk_collection(tmp_path)
    try:
        ids, _, _, vecs = _seed(col, n=30, seed=3)
        q = np.random.default_rng(9).normal(size=DIM).astype(np.float32)
        result = col.query(query_embeddings=[q.tolist()], n_results=30)
        # Reference: exact cosine distances, float64
        sims = (vecs / np.linalg.norm(vecs, axis=1, keepdims=True)) @ (q / np.linalg.norm(q))
        ref = sorted(zip((1.0 - sims).tolist(), ids))
        assert result.ids[0] == [doc_id for _, doc_id in ref]
        for got, (want, _) in zip(result.distances[0], ref):
            assert got == pytest.approx(want, abs=1e-5)
    finally:
        backend.close()


def test_query_include_embeddings_returns_raw_stored_vectors(tmp_path):
    backend, col = _mk_collection(tmp_path)
    try:
        # Deliberately NOT normalized: include=embeddings must return the
        # stored vector, not the cache's normalized copy.
        vec = [3.0, 0.0, 4.0, 0.0, 0.0, 0.0]
        col.add(documents=["one"], ids=["a"], metadatas=[{}], embeddings=[vec])
        result = col.query(query_embeddings=[vec], n_results=1, include=["embeddings", "distances"])
        assert result.embeddings[0][0] == pytest.approx(vec)
        assert result.distances[0][0] == pytest.approx(0.0, abs=1e-6)
    finally:
        backend.close()


def test_zero_norm_query_vector_returns_unit_distance(tmp_path):
    backend, col = _mk_collection(tmp_path)
    try:
        _seed(col, n=5)
        result = col.query(query_embeddings=[[0.0] * DIM], n_results=3)
        assert len(result.ids[0]) == 3
        assert all(d == pytest.approx(1.0) for d in result.distances[0])
    finally:
        backend.close()


# ---------------------------------------------------------------------------
# 2. Vector cache lifecycle
# ---------------------------------------------------------------------------


def _load_calls(monkeypatch):
    calls = []
    orig = SQLiteExactCollection._load_vector_rows

    def spy(self, cur, collection_id, dim, min_rowid):
        calls.append(min_rowid)
        return orig(self, cur, collection_id, dim, min_rowid)

    monkeypatch.setattr(SQLiteExactCollection, "_load_vector_rows", spy)
    return calls


def test_cache_appends_on_add_and_rebuilds_on_mutation(tmp_path, monkeypatch):
    backend, col = _mk_collection(tmp_path)
    try:
        ids, _, metas, vecs = _seed(col, n=10)
        calls = _load_calls(monkeypatch)

        q = vecs[0].tolist()
        col.query(query_embeddings=[q], n_results=3)
        assert calls == [0]  # first build is a full load

        col.query(query_embeddings=[q], n_results=3)
        assert calls == [0]  # cache hit: no load at all

        col.add(
            documents=["new doc"],
            ids=["new1"],
            metadatas=[{"wing": "w9"}],
            embeddings=[vecs[0].tolist()],
        )
        result = col.query(query_embeddings=[q], n_results=3)
        assert calls == [0, 10]  # pure insert → incremental append, not rebuild
        assert "new1" in result.ids[0]

        col.upsert(
            documents=["flip"],
            ids=["new1"],
            metadatas=[{"wing": "w9"}],
            embeddings=[(-vecs[0]).tolist()],
        )
        result = col.query(query_embeddings=[q], n_results=11)
        assert calls == [0, 10, 0]  # overwrite → generation bump → full rebuild
        assert result.ids[0][-1] == "new1"  # flipped vector now ranks last

        col.delete(ids=["new1"])
        result = col.query(query_embeddings=[q], n_results=11)
        assert calls == [0, 10, 0, 0]
        assert "new1" not in result.ids[0]
    finally:
        backend.close()


def test_update_with_embeddings_invalidates_cache(tmp_path):
    backend, col = _mk_collection(tmp_path)
    try:
        ids, _, _, vecs = _seed(col, n=6)
        q = vecs[2].tolist()
        assert col.query(query_embeddings=[q], n_results=1).ids[0] == ["d002"]
        col.update(ids=["d002"], embeddings=[(-vecs[2]).tolist()])
        assert col.query(query_embeddings=[q], n_results=1).ids[0] != ["d002"]
    finally:
        backend.close()


def test_metadata_only_update_is_visible_without_cache_rebuild(tmp_path, monkeypatch):
    backend, col = _mk_collection(tmp_path)
    try:
        ids, _, _, vecs = _seed(col, n=6)
        calls = _load_calls(monkeypatch)
        q = vecs[0].tolist()
        col.query(query_embeddings=[q], n_results=1)
        col.update(ids=["d000"], metadatas=[{"wing": "moved"}])
        result = col.query(query_embeddings=[q], n_results=6, where={"wing": "moved"})
        assert result.ids[0] == ["d000"]
        assert calls == [0]  # filters read SQL live; no cache invalidation
    finally:
        backend.close()


def test_writes_from_a_second_connection_are_seen(tmp_path):
    """Cross-process cache protocol, simulated with two backend instances
    (two sqlite connections on the same file — exactly what two processes do)."""
    backend_a, col_a = _mk_collection(tmp_path)
    backend_b = SQLiteExactBackend()
    try:
        ids, _, metas, vecs = _seed(col_a, n=8)
        col_b = backend_b.get_collection(str(tmp_path), "drawers", create=False)
        q = vecs[3].tolist()
        assert col_b.query(query_embeddings=[q], n_results=1).ids[0] == ["d003"]

        # append via A → B must see it (max_rowid moved)
        col_a.add(documents=["late"], ids=["late1"], metadatas=[{}], embeddings=[vecs[3].tolist()])
        assert "late1" in col_b.query(query_embeddings=[q], n_results=2).ids[0]

        # mutation via A → B must see it (generation bumped in the DB)
        col_a.upsert(
            documents=["late"], ids=["late1"], metadatas=[{}], embeddings=[(-vecs[3]).tolist()]
        )
        result = col_b.query(query_embeddings=[q], n_results=9)
        assert result.ids[0][-1] == "late1"

        col_a.delete(ids=["d003"])
        assert "d003" not in col_b.query(query_embeddings=[q], n_results=9).ids[0]
    finally:
        backend_a.close()
        backend_b.close()


# ---------------------------------------------------------------------------
# 3. Read-path purity + pragmas
# ---------------------------------------------------------------------------


def test_read_path_performs_no_writes(tmp_path):
    backend, col = _mk_collection(tmp_path)
    try:
        ids, _, _, vecs = _seed(col, n=12)
        db_path = tmp_path / "sqlite_exact.sqlite3"
        watcher = sqlite3.connect(str(db_path))
        try:
            before = watcher.execute("PRAGMA data_version").fetchone()[0]
            q = vecs[0].tolist()
            for _ in range(5):
                col.query(query_embeddings=[q], n_results=5)
                col.query(query_embeddings=[q], n_results=5, where={"wing": "w1"})
                col.lexical_search(query="shared words", n_results=5)
                col.lexical_search(query="shared words", n_results=5, where={"wing": "w1"})
                col.get(ids=["d000"])
                col.count()
            after = watcher.execute("PRAGMA data_version").fetchone()[0]
            assert after == before, "read path wrote to the database"
        finally:
            watcher.close()
    finally:
        backend.close()


def test_connection_pragmas(tmp_path):
    backend, col = _mk_collection(tmp_path)
    try:
        handle = col._handle
        assert handle.conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert handle.conn.execute("PRAGMA busy_timeout").fetchone()[0] == 10000
        assert handle.conn.execute("PRAGMA synchronous").fetchone()[0] == 1  # NORMAL
    finally:
        backend.close()


def test_generated_columns_and_indexes_exist(tmp_path):
    backend, col = _mk_collection(tmp_path)
    try:
        _seed(col, n=4)
        conn = col._handle.conn
        cols = {row[1] for row in conn.execute("PRAGMA table_xinfo(documents)").fetchall()}
        assert {"wing", "room", "authored_at"} <= cols
        indexes = {row[1] for row in conn.execute("PRAGMA index_list(documents)").fetchall()}
        assert {"idx_documents_wing", "idx_documents_room", "idx_documents_authored_at"} <= indexes
        # and the wing filter actually uses its index
        plan = conn.execute(
            "EXPLAIN QUERY PLAN SELECT rowid FROM documents WHERE collection_id = 1 AND wing = 'w1'"
        ).fetchall()
        assert any("idx_documents_wing" in str(tuple(row)) for row in plan)
    finally:
        backend.close()


def test_legacy_palace_without_generated_columns_migrates(tmp_path):
    """A pre-existing sqlite_exact DB (no generated columns) opens and gains
    the columns + indexes on first connect."""
    db_path = tmp_path / "sqlite_exact.sqlite3"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE collections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            dimension INTEGER,
            created_at TEXT NOT NULL
        );
        CREATE TABLE documents (
            collection_id INTEGER NOT NULL,
            id TEXT NOT NULL,
            document TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            embedding BLOB NOT NULL,
            dim INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (collection_id, id)
        );
        INSERT INTO collections(name, dimension, created_at) VALUES ('drawers', 3, 'x');
        """
    )
    vec = np.asarray([1.0, 0.0, 0.0], dtype=np.float32).tobytes()
    conn.execute(
        "INSERT INTO documents VALUES (1, 'old1', 'legacy doc', '{\"wing\":\"legacy\"}', ?, 3, 'x', 'x')",
        (vec,),
    )
    conn.commit()
    conn.close()

    backend = SQLiteExactBackend()
    try:
        col = backend.get_collection(str(tmp_path), "drawers", create=False)
        result = col.query(
            query_embeddings=[[1.0, 0.0, 0.0]], n_results=5, where={"wing": "legacy"}
        )
        assert result.ids[0] == ["old1"]
    finally:
        backend.close()
