"""SQLite exact-vector backend for MemPalace.

This backend is intentionally simple and local-first. It is a correctness
backend, not a high-throughput ANN backend: vectors are stored as float32
blobs and query uses exact cosine distance over the matching collection.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

from .base import (
    BackendClosedError,
    BaseBackend,
    BaseCollection,
    CollectionNotInitializedError,
    DimensionMismatchError,
    GetResult,
    HealthStatus,
    LexicalHit,
    LexicalResult,
    PalaceNotFoundError,
    PalaceRef,
    QueryResult,
    UnsupportedFilterError,
    _IncludeSpec,
)

logger = logging.getLogger(__name__)

_DB_FILENAME = "sqlite_exact.sqlite3"
_TOKEN_RE = re.compile(r"\w{2,}", re.UNICODE)
# FTS candidate-generation window for lexical_search, mirroring the chroma
# lane's ``max_candidates=500``: FTS5 rank selects the window, the shared
# Okapi rescore ranks it. See ``_rescore_lexical_candidates``.
_FTS_CANDIDATE_WINDOW = 500
_SUPPORTED_OPERATORS = frozenset(
    {"$eq", "$ne", "$in", "$nin", "$and", "$or", "$contains", "$gt", "$gte", "$lt", "$lte"}
)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj or {}, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _json_loads(text: str | None) -> dict:
    if not text:
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _encode_vector(vector: list[float]) -> bytes:
    return _as_vector_array(vector).tobytes()


def _as_vector_array(vector: list[float]) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float32)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("embedding must be a non-empty 1D vector")
    return arr


def _decode_vector(blob: bytes | None) -> list[float]:
    if not blob:
        return []
    return np.frombuffer(blob, dtype=np.float32).astype(float).tolist()


def _decode_array(blob: bytes | None) -> Optional[np.ndarray]:
    if not blob:
        return None
    arr = np.frombuffer(blob, dtype=np.float32)
    if arr.size == 0:
        return None
    return arr


def _tokenize(text: str) -> list[str]:
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


def _bm25_scores(query: str, documents: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    query_terms = set(_tokenize(query))
    n_docs = len(documents)
    if not query_terms or n_docs == 0:
        return [0.0] * n_docs

    tokenized = [_tokenize(d) for d in documents]
    doc_lens = [len(toks) for toks in tokenized]
    if not any(doc_lens):
        return [0.0] * n_docs
    avgdl = sum(doc_lens) / n_docs or 1.0

    df = {term: 0 for term in query_terms}
    for toks in tokenized:
        for term in set(toks) & query_terms:
            df[term] += 1

    idf = {term: np.log((n_docs - df[term] + 0.5) / (df[term] + 0.5) + 1.0) for term in query_terms}

    scores = []
    for toks, dl in zip(tokenized, doc_lens):
        if dl == 0:
            scores.append(0.0)
            continue
        tf: dict[str, int] = {}
        for token in toks:
            if token in query_terms:
                tf[token] = tf.get(token, 0) + 1
        score = 0.0
        for term, freq in tf.items():
            num = freq * (k1 + 1)
            den = freq + k1 * (1 - b + b * dl / avgdl)
            score += float(idf[term]) * num / den
        scores.append(score)
    return scores


def _validate_where(where: Optional[dict]) -> None:
    if not where:
        return
    stack = [where]
    while stack:
        node = stack.pop()
        if not isinstance(node, dict):
            continue
        for key, value in node.items():
            if key.startswith("$") and key not in _SUPPORTED_OPERATORS:
                raise UnsupportedFilterError(f"operator {key!r} not supported by sqlite_exact")
            if isinstance(value, dict):
                stack.append(value)
            elif isinstance(value, list):
                stack.extend(item for item in value if isinstance(item, dict))


def _coerce_comparable(value: Any):
    if isinstance(value, bool):
        return int(value)
    return value


def _compare(actual: Any, op: str, expected: Any) -> bool:
    actual = _coerce_comparable(actual)
    expected = _coerce_comparable(expected)
    if op == "$eq":
        return actual == expected
    if op == "$ne":
        return actual != expected
    if op == "$in":
        return actual in (expected or [])
    if op == "$nin":
        return actual not in (expected or [])
    if op == "$contains":
        return str(expected) in str(actual or "")
    try:
        if op == "$gt":
            return actual > expected
        if op == "$gte":
            return actual >= expected
        if op == "$lt":
            return actual < expected
        if op == "$lte":
            return actual <= expected
    except TypeError:
        return False
    raise UnsupportedFilterError(f"operator {op!r} not supported by sqlite_exact")


def _matches_where(meta: dict, where: Optional[dict]) -> bool:
    if not where:
        return True
    if not isinstance(where, dict):
        return False
    for key, expected in where.items():
        if key == "$and":
            if not all(_matches_where(meta, clause) for clause in expected or []):
                return False
            continue
        if key == "$or":
            if not any(_matches_where(meta, clause) for clause in expected or []):
                return False
            continue
        if key.startswith("$"):
            raise UnsupportedFilterError(f"operator {key!r} not supported by sqlite_exact")
        actual = meta.get(key)
        if isinstance(expected, dict):
            for op, operand in expected.items():
                if not _compare(actual, op, operand):
                    return False
        elif actual != expected:
            return False
    return True


class _WhereNotTranslatable(Exception):
    """Internal: this filter cannot be compiled to SQL; use the Python scan.

    Never escapes the backend — every raise site falls back to the row-scan
    path, which evaluates the same filter via :func:`_matches_where`.
    """


# Metadata keys promoted to indexed generated columns (see ``_init_schema``).
# All string-valued in every write path: TEXT affinity on the generated column
# is lossless. Do NOT add integer-valued keys (e.g. ``chunk_index``) — TEXT
# affinity would coerce them and break ``IS ?`` comparisons against ints.
# ``source_file`` / ``parent_drawer_id`` serve the searcher's per-hit
# neighbor/hydration ``get(where=...)`` calls, which otherwise scan the
# whole collection per call (measured 775 ms warm, seconds cold, ×2 per
# closet-boosted hit — the entire 2026-08-09 cutover latency tail).
_GENERATED_META_COLUMNS = ("wing", "room", "authored_at", "source_file", "parent_drawer_id")

# JSON path keys we can quote safely inside ``$."<key>"``. A double quote or
# backslash would need JSON-path escaping SQLite does not define; those keys
# fall back to the Python scan.
_SAFE_JSON_KEY_RE = re.compile(r'^[^"\\\x00-\x1f]+$')

_SCALAR_TYPES = (str, int, float, bool, type(None))

_ORDERED_OPS = {"$gt": ">", "$gte": ">=", "$lt": "<", "$lte": "<="}


def _compile_where(where: Optional[dict], prefix: str = "") -> tuple[str, list]:
    """Compile a chroma-style ``where`` dict to a SQL predicate over ``documents``.

    Returns ``(sql, params)``. The predicate is semantically identical to
    :func:`_matches_where` for scalar metadata values (the only kind the write
    paths accept — chroma-compatible metadata is scalar-only). Raises
    :class:`_WhereNotTranslatable` for shapes SQL cannot express faithfully
    (unquotable keys, non-scalar operands); callers fall back to the Python
    scan, so translation failure is a slow path, never a behavior change.

    ``prefix`` qualifies column references (e.g. ``"d."``) for use in joins.
    """
    params: list = []
    sql = _compile_where_node(where or {}, prefix, params)
    return sql, params


def _compile_where_node(node: dict, prefix: str, params: list) -> str:
    if not isinstance(node, dict):
        raise _WhereNotTranslatable("where clause must be a dict")
    parts: list[str] = []
    for key, expected in node.items():
        if key == "$and":
            sub = [_compile_where_node(clause, prefix, params) for clause in expected or []]
            parts.append("(" + " AND ".join(sub) + ")" if sub else "1")
            continue
        if key == "$or":
            sub = [_compile_where_node(clause, prefix, params) for clause in expected or []]
            parts.append("(" + " OR ".join(sub) + ")" if sub else "0")
            continue
        if key.startswith("$"):
            raise UnsupportedFilterError(f"operator {key!r} not supported by sqlite_exact")
        if isinstance(expected, dict):
            for op, operand in expected.items():
                parts.append(_compile_where_op(key, op, operand, prefix, params))
        else:
            parts.append(_compile_where_op(key, "$eq", expected, prefix, params))
    if not parts:
        return "1"
    return "(" + " AND ".join(parts) + ")"


def _compile_where_op(key: str, op: str, operand, prefix: str, params: list) -> str:
    if not _SAFE_JSON_KEY_RE.match(key):
        raise _WhereNotTranslatable(f"key {key!r} cannot be quoted in a JSON path")

    def value_expr() -> str:
        # Each call emits one occurrence of the value expression, appending its
        # path parameter in the same left-to-right order SQLite binds ``?``s.
        if key in _GENERATED_META_COLUMNS:
            return f"{prefix}{key}"
        params.append(f'$."{key}"')
        return f"json_extract({prefix}metadata_json, ?)"

    def type_expr() -> str:
        params.append(f'$."{key}"')
        return f"json_type({prefix}metadata_json, ?)"

    if op in ("$eq", "$ne"):
        if not isinstance(operand, _SCALAR_TYPES):
            raise _WhereNotTranslatable(f"non-scalar operand for {op}")
        # IS treats NULLs as comparable, matching Python's ``None == None`` /
        # ``missing != value`` semantics in _compare.
        sql = f"{value_expr()} IS ?"
        params.append(_coerce_comparable(operand))
        return sql if op == "$eq" else f"NOT ({sql})"

    if op in ("$in", "$nin"):
        values = [_coerce_comparable(v) for v in (operand or [])]
        if any(not isinstance(v, _SCALAR_TYPES) for v in values):
            raise _WhereNotTranslatable(f"non-scalar operand for {op}")
        non_null = [v for v in values if v is not None]
        clauses = []
        if non_null:
            placeholders = ",".join("?" for _ in non_null)
            clauses.append(f"{value_expr()} IN ({placeholders})")
            params.extend(non_null)
        if len(non_null) != len(values):  # None was in the list
            clauses.append(f"{value_expr()} IS NULL")
        membership = " OR ".join(clauses) if clauses else "0"
        # COALESCE(NULL, 0): a missing key is "not in" any list, exactly like
        # Python's ``None in [...]`` / ``None not in [...]``.
        if op == "$in":
            return f"COALESCE(({membership}), 0)"
        return f"NOT COALESCE(({membership}), 0)"

    if op in _ORDERED_OPS:
        sql_op = _ORDERED_OPS[op]
        operand = _coerce_comparable(operand)
        if isinstance(operand, (int, float)) and not isinstance(operand, bool):
            # Python raises TypeError (→ no match) comparing str/list/None to a
            # number; json_type gates SQL to the same numeric-only domain.
            sql = (
                f"({type_expr()} IN ('integer','real','true','false') "
                f"AND {value_expr()} {sql_op} ?)"
            )
            params.append(operand)
            return sql
        if isinstance(operand, str):
            sql = f"({type_expr()} = 'text' AND {value_expr()} {sql_op} ?)"
            params.append(operand)
            return sql
        # None / list / dict operands always raise TypeError in Python → False.
        return "0"

    if op == "$contains":
        operand = _coerce_comparable(operand)
        if not isinstance(operand, _SCALAR_TYPES):
            raise _WhereNotTranslatable("non-scalar operand for $contains")
        needle = str(operand)
        if needle == "":
            return "1"  # Python: "" in anything → True; SQL instr(x,'') is 0
        # Python computes ``str(actual or "")``: every falsy actual (missing,
        # 0, 0.0, False, "") stringifies to "". The CASE mirrors that exactly.
        a, b = value_expr(), value_expr()
        c = value_expr()
        sql = (
            f"instr(CASE WHEN {a} IS NULL OR {b} = 0 OR {c} = '' "
            f"THEN '' ELSE CAST({value_expr()} AS TEXT) END, ?) > 0"
        )
        params.append(needle)
        return sql

    raise UnsupportedFilterError(f"operator {op!r} not supported by sqlite_exact")


def _compile_where_document(where_document: Optional[dict], prefix: str = "") -> tuple[str, list]:
    """Compile a ``where_document`` dict to SQL over ``documents.document``."""
    params: list = []
    sql = _compile_where_document_node(where_document or {}, prefix, params)
    return sql, params


def _compile_where_document_node(node: dict, prefix: str, params: list) -> str:
    if not isinstance(node, dict):
        raise _WhereNotTranslatable("where_document clause must be a dict")
    parts: list[str] = []
    for key, value in node.items():
        if key == "$contains":
            needle = str(value)
            if needle == "":
                parts.append("1")
            else:
                parts.append(f"instr({prefix}document, ?) > 0")
                params.append(needle)
            continue
        if key == "$and":
            sub = [_compile_where_document_node(c, prefix, params) for c in value or []]
            parts.append("(" + " AND ".join(sub) + ")" if sub else "1")
            continue
        if key == "$or":
            sub = [_compile_where_document_node(c, prefix, params) for c in value or []]
            parts.append("(" + " OR ".join(sub) + ")" if sub else "0")
            continue
        raise UnsupportedFilterError(f"where_document operator {key!r} not supported")
    if not parts:
        return "1"
    return "(" + " AND ".join(parts) + ")"


def _matches_where_document(document: str, where_document: Optional[dict]) -> bool:
    if not where_document:
        return True
    if not isinstance(where_document, dict):
        return False
    for key, value in where_document.items():
        if key == "$contains":
            if str(value) not in document:
                return False
            continue
        if key == "$and":
            if not all(_matches_where_document(document, clause) for clause in value or []):
                return False
            continue
        if key == "$or":
            if not any(_matches_where_document(document, clause) for clause in value or []):
                return False
            continue
        raise UnsupportedFilterError(f"where_document operator {key!r} not supported")
    return True


def _validate_write_batch(
    *,
    documents: list[str],
    ids: list[str],
    metadatas: Optional[list[dict]],
    embeddings: Optional[list[list[float]]],
) -> None:
    n = len(ids)
    if len(documents) != n:
        raise ValueError(f"documents length {len(documents)} does not match ids length {n}")
    if metadatas is not None and len(metadatas) != n:
        raise ValueError(f"metadatas length {len(metadatas)} does not match ids length {n}")
    if embeddings is not None and len(embeddings) != n:
        raise ValueError(f"embeddings length {len(embeddings)} does not match ids length {n}")


class _VectorCache:
    """Snapshot of one collection's vectors for the matmul path.

    ``matrix`` is an L2-row-normalized float32 array in ``rowid`` order (so
    ``rowids`` is strictly ascending — required by the ``searchsorted`` mask
    mapping). ``count`` / ``max_rowid`` are the DB-side values at snapshot
    time; ``generation`` is the persisted ``vec_gen:<collection_id>`` counter,
    bumped by every mutating write (in this or any other process).

    Freshness is two-tier. The O(1) tier: ``data_version`` (moves when another
    connection commits) plus the connection's ``total_changes`` (moves on own
    writes) — both unchanged proves *nothing anywhere committed*, so the warm
    read path touches no tables at all. Only when one of them moved does the
    generation / MAX(rowid) / COUNT protocol run, against covering indexes.

    ``filters`` memoizes candidate index arrays per compiled filter, bounded
    at ``_MAX_CACHED_FILTERS``. It is cleared on *any* observed commit —
    metadata can change without vectors changing, and clearing is cheaper than
    proving which writes were metadata-neutral — while the matrix survives
    unless the vector protocol says otherwise.
    """

    __slots__ = (
        "generation",
        "max_rowid",
        "count",
        "rowids",
        "ids",
        "matrix",
        "data_version",
        "total_changes",
        "filters",
    )

    _MAX_CACHED_FILTERS = 32

    def __init__(self, generation, max_rowid, count, rowids, ids, matrix):
        self.generation = generation
        self.max_rowid = max_rowid
        self.count = count
        self.rowids = rowids
        self.ids = ids
        self.matrix = matrix
        self.data_version = -1
        self.total_changes = -1
        self.filters: dict[str, np.ndarray] = {}


class _SQLiteExactHandle:
    def __init__(self, conn: sqlite3.Connection, lock: threading.RLock):
        self.conn = conn
        self.lock = lock
        self.closed = False
        # (collection_id, dim) → _VectorCache. Shared by every collection
        # object on this palace; guarded by ``lock``.
        self.vec_caches: dict[tuple[int, int], _VectorCache] = {}


class SQLiteExactCollection(BaseCollection):
    def __init__(self, handle: _SQLiteExactHandle, collection_name: str):
        self._handle = handle
        self._collection_name = collection_name
        self._closed = False

    def _ensure_open(self) -> None:
        if self._closed or self._handle.closed:
            raise BackendClosedError("SQLiteExactCollection has been closed")

    @contextlib.contextmanager
    def _cursor(self):
        with self._handle.lock:
            self._ensure_open()
            cur = self._handle.conn.cursor()
            try:
                yield cur
            except Exception:
                self._handle.conn.rollback()
                raise
            else:
                self._handle.conn.commit()
            finally:
                cur.close()

    def _collection_id(self, cur) -> int:
        row = cur.execute(
            "SELECT id FROM collections WHERE name = ?",
            (self._collection_name,),
        ).fetchone()
        if row is None:
            raise CollectionNotInitializedError(self._collection_name)
        return int(row[0])

    def _collection_dimension(self, cur, collection_id: int) -> Optional[int]:
        row = cur.execute(
            "SELECT dimension FROM collections WHERE id = ?",
            (collection_id,),
        ).fetchone()
        if row is None or row[0] is None:
            return None
        return int(row[0])

    def _ensure_collection_dimension(self, cur, collection_id: int, dims: list[int]) -> None:
        distinct = {int(dim) for dim in dims}
        if not distinct:
            return
        if len(distinct) > 1:
            raise DimensionMismatchError(
                f"sqlite_exact collection {self._collection_name!r} cannot mix "
                f"embedding dimensions {sorted(distinct)}"
            )
        dim = distinct.pop()
        stored = self._collection_dimension(cur, collection_id)
        if stored is None:
            cur.execute(
                "UPDATE collections SET dimension = ? WHERE id = ?",
                (dim, collection_id),
            )
        elif stored != dim:
            raise DimensionMismatchError(
                f"sqlite_exact collection {self._collection_name!r} expects "
                f"embedding dimension {stored}, got {dim}"
            )

    def _fts_available(self, cur) -> bool:
        row = cur.execute("SELECT value FROM meta WHERE key = 'fts5_available'").fetchone()
        return bool(row and row[0] == "1")

    def _embedder_meta_key(self) -> str:
        return f"embedder_model:{self._collection_name}"

    def get_stored_embedder_identity(self):
        from .base import EmbedderIdentity

        with self._cursor() as cur:
            try:
                cid = self._collection_id(cur)
            except CollectionNotInitializedError:
                return None
            row = cur.execute(
                "SELECT value FROM meta WHERE key = ?",
                (self._embedder_meta_key(),),
            ).fetchone()
            if not row or not row[0]:
                return None
            dim = self._collection_dimension(cur, cid) or 0
            return EmbedderIdentity(model_name=str(row[0]), dimension=int(dim))

    def set_embedder_identity(self, identity) -> None:
        if not identity or not identity.model_name:
            return
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO meta(key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (self._embedder_meta_key(), str(identity.model_name)),
            )

    def _insert_fts(self, cur, collection_id: int, doc_id: str, document: str, rowid: int) -> None:
        """Insert a fresh FTS row keyed to the documents rowid.

        ``docs_fts`` rowids mirror ``documents`` rowids (enforced by the
        one-time rebuild in ``_init_schema``), so removal is an O(1) rowid
        delete instead of a full FTS scan over the UNINDEXED columns — the
        latter made batch ingest O(n²) (observed live: 333 → 150 rows/s and
        falling within the first 20k rows of a 174k build).
        """
        if not self._fts_available(cur):
            return
        cur.execute(
            "INSERT INTO docs_fts(rowid, collection_id, doc_id, document) VALUES (?, ?, ?, ?)",
            (rowid, collection_id, doc_id, document),
        )

    def _replace_fts(self, cur, collection_id: int, doc_id: str, document: str, rowid: int) -> None:
        if not self._fts_available(cur):
            return
        cur.execute("DELETE FROM docs_fts WHERE rowid = ?", (rowid,))
        self._insert_fts(cur, collection_id, doc_id, document, rowid)

    def _document_rowid(self, cur, collection_id: int, doc_id: str) -> Optional[int]:
        row = cur.execute(
            "SELECT rowid FROM documents WHERE collection_id = ? AND id = ?",
            (collection_id, doc_id),
        ).fetchone()
        return int(row[0]) if row else None

    def add(self, *, documents, ids, metadatas=None, embeddings=None):
        _validate_write_batch(
            documents=documents,
            ids=ids,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        if embeddings is None:
            raise ValueError("sqlite_exact requires explicit embeddings")
        metadatas = metadatas or [{} for _ in ids]
        now = _utcnow()
        with self._cursor() as cur:
            collection_id = self._collection_id(cur)
            prepared = []
            for doc_id, doc, meta, emb in zip(ids, documents, metadatas, embeddings):
                arr = _as_vector_array(emb)
                prepared.append((doc_id, doc, meta, arr.tobytes(), int(arr.size)))
            self._ensure_collection_dimension(cur, collection_id, [item[4] for item in prepared])
            for doc_id, doc, meta, emb_blob, dim in prepared:
                cur.execute(
                    """
                    INSERT INTO documents
                        (collection_id, id, document, metadata_json, embedding, dim, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        collection_id,
                        doc_id,
                        doc,
                        _json_dumps(meta),
                        emb_blob,
                        dim,
                        now,
                        now,
                    ),
                )
                # Plain INSERT: the row is new, so no stale FTS row can exist.
                self._insert_fts(cur, collection_id, doc_id, doc, cur.lastrowid)

    def upsert(self, *, documents, ids, metadatas=None, embeddings=None):
        _validate_write_batch(
            documents=documents,
            ids=ids,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        if embeddings is None:
            raise ValueError("sqlite_exact requires explicit embeddings")
        metadatas = metadatas or [{} for _ in ids]
        now = _utcnow()
        with self._cursor() as cur:
            collection_id = self._collection_id(cur)
            prepared = []
            for doc_id, doc, meta, emb in zip(ids, documents, metadatas, embeddings):
                arr = _as_vector_array(emb)
                prepared.append((doc_id, doc, meta, arr.tobytes(), int(arr.size)))
            self._ensure_collection_dimension(cur, collection_id, [item[4] for item in prepared])
            existing = self._existing_rowids(cur, collection_id, [item[0] for item in prepared])
            # An upsert that overwrites an existing row changes that row's
            # embedding in place (same rowid), which append detection cannot
            # see — bump the generation. Pure-insert upserts stay append-only.
            if existing:
                self._bump_vec_generation(cur, collection_id)
            for doc_id, doc, meta, emb_blob, dim in prepared:
                cur.execute(
                    """
                    INSERT INTO documents
                        (collection_id, id, document, metadata_json, embedding, dim, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(collection_id, id) DO UPDATE SET
                        document = excluded.document,
                        metadata_json = excluded.metadata_json,
                        embedding = excluded.embedding,
                        dim = excluded.dim,
                        updated_at = excluded.updated_at
                    """,
                    (
                        collection_id,
                        doc_id,
                        doc,
                        _json_dumps(meta),
                        emb_blob,
                        dim,
                        now,
                        now,
                    ),
                )
                rowid = existing.get(doc_id)
                if rowid is not None:
                    self._replace_fts(cur, collection_id, doc_id, doc, rowid)
                else:
                    # Record the fresh rowid so a duplicate id later in the
                    # same batch replaces its FTS row instead of doubling it.
                    existing[doc_id] = cur.lastrowid
                    self._insert_fts(cur, collection_id, doc_id, doc, cur.lastrowid)

    def update(self, *, ids, documents=None, metadatas=None, embeddings=None):
        if documents is None and metadatas is None and embeddings is None:
            raise ValueError("update requires at least one of documents, metadatas, embeddings")
        n = len(ids)
        for label, value in (
            ("documents", documents),
            ("metadatas", metadatas),
            ("embeddings", embeddings),
        ):
            if value is not None and len(value) != n:
                raise ValueError(f"{label} length {len(value)} does not match ids length {n}")
        with self._cursor() as cur:
            collection_id = self._collection_id(cur)
            updates = []
            for idx, doc_id in enumerate(ids):
                row = cur.execute(
                    """
                    SELECT document, metadata_json, embedding, dim, rowid
                    FROM documents
                    WHERE collection_id = ? AND id = ?
                    """,
                    (collection_id, doc_id),
                ).fetchone()
                if row is None:
                    continue
                doc = documents[idx] if documents is not None else row[0]
                meta = _json_loads(row[1])
                if metadatas is not None:
                    meta.update(metadatas[idx] or {})
                if embeddings is not None:
                    arr = _as_vector_array(embeddings[idx])
                    emb_blob = arr.tobytes()
                    dim = int(arr.size)
                else:
                    emb_blob = row[2]
                    dim = row[3]
                updates.append((doc_id, doc, meta, emb_blob, dim, int(row[4])))
            if embeddings is not None:
                self._ensure_collection_dimension(cur, collection_id, [item[4] for item in updates])
                if updates:
                    # Only an embedding change invalidates the vector cache;
                    # document/metadata updates are read live from SQL.
                    self._bump_vec_generation(cur, collection_id)
            for doc_id, doc, meta, emb_blob, dim, rowid in updates:
                cur.execute(
                    """
                    UPDATE documents
                    SET document = ?, metadata_json = ?, embedding = ?, dim = ?, updated_at = ?
                    WHERE collection_id = ? AND id = ?
                    """,
                    (doc, _json_dumps(meta), emb_blob, dim, _utcnow(), collection_id, doc_id),
                )
                self._replace_fts(cur, collection_id, doc_id, doc, rowid)

    def _rows(
        self, cur, *, where=None, where_document=None, limit=None, offset=None, with_embedding=True
    ) -> list[dict]:
        _validate_where(where)
        _validate_where(where_document)
        collection_id = self._collection_id(cur)
        # Compile the filters to SQL when possible so a filtered read is a
        # (potentially indexed) SQL scan instead of a full-collection fetch
        # plus Python JSON matching. Untranslatable filters keep the Python
        # scan, which is a slow path, never a behavior change — both paths
        # evaluate the same predicate semantics over rowid order.
        compiled_clauses: list[str] = []
        compiled_params: list = []
        compiled = True
        try:
            if where is not None:
                sql_part, w_params = _compile_where(where)
                compiled_clauses.append(sql_part)
                compiled_params.extend(w_params)
            if where_document is not None:
                sql_part, d_params = _compile_where_document(where_document)
                compiled_clauses.append(sql_part)
                compiled_params.extend(d_params)
        except _WhereNotTranslatable:
            compiled = False
            compiled_clauses = []
            compiled_params = []
        needs_python_filter = not compiled and (where is not None or where_document is not None)
        # The embedding blob dominates row width (1.5 KB at 384-d); skip it
        # unless the caller actually wants vectors back.
        select_cols = "rowid, id, document, metadata_json" + (
            ", embedding" if with_embedding else ""
        )
        sql = f"SELECT {select_cols}\nFROM documents\nWHERE collection_id = ?" + "".join(
            f" AND ({clause})" for clause in compiled_clauses
        )
        params = [collection_id, *compiled_params]
        # SQL ORDER BY + LIMIT/OFFSET pushdown only on the no-filter scan:
        # there the (collection_id) index yields rowid order for free. With
        # compiled filter clauses, ``ORDER BY rowid`` makes the planner (no
        # ANALYZE stats) prefer that same index *to avoid the sort* — a full
        # 177k-entry walk instead of the selective generated-column index
        # (measured: 183 ms vs <1 ms). So filtered reads fetch the (small)
        # match set unordered, then sort and slice in Python — identical
        # rowid-order results either way.
        pushed_page = False
        if not compiled_clauses:
            # No-filter scan or Python-fallback scan: the (collection_id)
            # index yields rowid order without a sort step.
            sql += "\nORDER BY rowid"
            if not needs_python_filter and (limit is not None or offset):
                pushed_page = True
                if limit is not None:
                    sql += "\nLIMIT ?"
                    params.append(int(limit))
                elif offset:
                    sql += "\nLIMIT -1"
                if offset:
                    sql += "\nOFFSET ?"
                    params.append(int(offset))
        rows = cur.execute(sql, params).fetchall()
        if compiled_clauses:
            rows.sort(key=lambda row: row[0])
        out = []
        for row in rows:
            doc_id, doc, meta_json = row[1], row[2], row[3]
            meta = _json_loads(meta_json)
            if needs_python_filter:
                if not _matches_where(meta, where):
                    continue
                if not _matches_where_document(doc or "", where_document):
                    continue
            out.append(
                {
                    "id": doc_id,
                    "document": doc or "",
                    "metadata": meta,
                    "embedding": row[4] if with_embedding else None,
                }
            )
        if not pushed_page and (limit is not None or offset):
            # Callers only pass limit/offset here under the push_page contract
            # (non-negative bounds), so plain slicing mirrors SQL LIMIT/OFFSET.
            if offset:
                out = out[offset:]
            if limit is not None:
                out = out[:limit]
        return out

    # ------------------------------------------------------------------
    # Vector cache (the matmul query path)
    # ------------------------------------------------------------------

    @staticmethod
    def _vec_gen_key(collection_id: int) -> str:
        return f"vec_gen:{collection_id}"

    def _read_vec_generation(self, cur, collection_id: int) -> int:
        row = cur.execute(
            "SELECT value FROM meta WHERE key = ?",
            (self._vec_gen_key(collection_id),),
        ).fetchone()
        if not row:
            return 0
        try:
            return int(row[0])
        except (TypeError, ValueError):
            return 0

    def _bump_vec_generation(self, cur, collection_id: int) -> None:
        """Invalidate every process's vector cache for this collection.

        Called by every write that can change an *existing* row's embedding or
        remove a row (upsert-over-existing, update with embeddings, delete).
        Pure appends do not bump — the cache detects them via MAX(rowid) and
        extends incrementally instead of rebuilding.
        """
        cur.execute(
            "INSERT INTO meta(key, value) VALUES (?, '1') "
            "ON CONFLICT(key) DO UPDATE SET value = CAST(CAST(value AS INTEGER) + 1 AS TEXT)",
            (self._vec_gen_key(collection_id),),
        )

    def _load_vector_rows(self, cur, collection_id: int, dim: int, min_rowid: int):
        """Load (rowids, ids, L2-normalized float32 matrix) above ``min_rowid``.

        Returns the raw fetched row count as the fourth element so the caller
        can reconcile against COUNT(*) even if defective blobs were skipped.
        """
        rows = cur.execute(
            "SELECT rowid, id, embedding FROM documents "
            "WHERE collection_id = ? AND dim = ? AND rowid > ? ORDER BY rowid",
            (collection_id, dim, min_rowid),
        ).fetchall()
        fetched = len(rows)
        expected_bytes = dim * 4
        rowids: list[int] = []
        ids: list[str] = []
        buf = bytearray()
        for rowid, doc_id, blob in rows:
            if blob is None or len(blob) != expected_bytes:
                logger.warning(
                    "sqlite_exact: skipping row %s with malformed embedding blob", doc_id
                )
                continue
            rowids.append(int(rowid))
            ids.append(doc_id)
            buf += blob
        if ids:
            matrix = np.frombuffer(bytes(buf), dtype=np.float32).reshape(len(ids), dim)
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            matrix = matrix / norms  # new writable float32 array
        else:
            matrix = np.empty((0, dim), dtype=np.float32)
        return np.asarray(rowids, dtype=np.int64), ids, matrix, fetched

    def _data_version(self, cur) -> int:
        return int(cur.execute("PRAGMA data_version").fetchone()[0])

    def _vector_cache(self, cur, collection_id: int, dim: int) -> _VectorCache:
        """Return a current vector cache for ``(collection_id, dim)``.

        Tier 1 — O(1), the warm path: if the connection's ``total_changes``
        and the DB's ``data_version`` both match the snapshot, no commit has
        happened anywhere (own writes move total_changes, foreign commits move
        data_version) — return the cache without touching a table.

        Tier 2 — after an observed commit (memoized filters dropped first,
        since metadata may have changed without vectors changing):

        * generation changed → a mutation happened somewhere; full rebuild.
        * generation same, MAX(rowid)/COUNT(*) unchanged → cache is current.
        * generation same, both grew consistently → pure appends; load only
          rows above the cached MAX(rowid) and extend.
        * anything else → rebuild (defensive; cannot happen with in-tree
          writers, which bump the generation on every non-append mutation).
        """
        key = (collection_id, dim)
        cache = self._handle.vec_caches.get(key)
        total_changes = self._handle.conn.total_changes
        data_version: Optional[int] = None
        if cache is not None:
            if total_changes == cache.total_changes:
                data_version = self._data_version(cur)
                if data_version == cache.data_version:
                    return cache
            if data_version is None:
                data_version = self._data_version(cur)
            cache.filters.clear()
            generation = self._read_vec_generation(cur, collection_id)
            if cache.generation == generation:
                row = cur.execute(
                    "SELECT COALESCE(MAX(rowid), 0), COUNT(*) FROM documents "
                    "WHERE collection_id = ? AND dim = ?",
                    (collection_id, dim),
                ).fetchone()
                max_rowid, count = int(row[0]), int(row[1])
                if max_rowid == cache.max_rowid and count == cache.count:
                    cache.data_version = data_version
                    cache.total_changes = total_changes
                    return cache
                if max_rowid > cache.max_rowid and count > cache.count:
                    rowids, ids, matrix, fetched = self._load_vector_rows(
                        cur, collection_id, dim, cache.max_rowid
                    )
                    if cache.count + fetched == count:
                        cache = _VectorCache(
                            generation,
                            max_rowid,
                            count,
                            np.concatenate([cache.rowids, rowids]),
                            cache.ids + ids,
                            np.concatenate([cache.matrix, matrix]) if ids else cache.matrix,
                        )
                        cache.data_version = data_version
                        cache.total_changes = total_changes
                        self._handle.vec_caches[key] = cache
                        return cache
            # fall through to a full rebuild
        generation = self._read_vec_generation(cur, collection_id)
        if data_version is None:
            data_version = self._data_version(cur)
        rowids, ids, matrix, fetched = self._load_vector_rows(cur, collection_id, dim, 0)
        cache = _VectorCache(
            generation,
            int(rowids[-1]) if len(rowids) else 0,
            fetched,
            rowids,
            ids,
            matrix,
        )
        cache.data_version = data_version
        cache.total_changes = self._handle.conn.total_changes
        self._handle.vec_caches[key] = cache
        return cache

    def _cached_candidate_indices(
        self, cur, collection_id: int, dim: int, cache: _VectorCache, where, where_document
    ) -> Optional[np.ndarray]:
        """Memoizing wrapper around :meth:`_candidate_indices`.

        The memo lives on the cache snapshot and is cleared by
        ``_vector_cache`` on any observed commit, so a hit is always computed
        against current data. Unfiltered queries bypass it entirely.
        """
        if where is None and where_document is None:
            return None
        try:
            filter_key = json.dumps([where, where_document], sort_keys=True)
        except (TypeError, ValueError):
            filter_key = None
        if filter_key is not None:
            hit = cache.filters.get(filter_key)
            if hit is not None:
                return hit
        cand = self._candidate_indices(cur, collection_id, dim, cache, where, where_document)
        if filter_key is not None and cand is not None:
            if len(cache.filters) >= cache._MAX_CACHED_FILTERS:
                cache.filters.pop(next(iter(cache.filters)))
            cache.filters[filter_key] = cand
        return cand

    def _candidate_indices(
        self, cur, collection_id: int, dim: int, cache: _VectorCache, where, where_document
    ) -> Optional[np.ndarray]:
        """Resolve where/where_document to matrix row indices via SQL.

        Returns ``None`` when unfiltered. Raises ``_WhereNotTranslatable`` for
        filters SQL cannot express; the caller falls back to the Python scan.
        """
        if where is None and where_document is None:
            return None
        clauses: list[str] = []
        params: list = []
        if where is not None:
            sql, w_params = _compile_where(where)
            clauses.append(sql)
            params.extend(w_params)
        if where_document is not None:
            sql, d_params = _compile_where_document(where_document)
            clauses.append(sql)
            params.extend(d_params)
        rows = cur.execute(
            "SELECT rowid FROM documents WHERE collection_id = ? AND dim = ? AND "
            + " AND ".join(f"({c})" for c in clauses),
            (collection_id, dim, *params),
        ).fetchall()
        if not rows:
            return np.empty(0, dtype=np.int64)
        cand = np.fromiter((r[0] for r in rows), dtype=np.int64, count=len(rows))
        # Same transaction as the cache snapshot, so every candidate rowid is
        # present in cache.rowids; searchsorted maps rowid → matrix index.
        pos = np.searchsorted(cache.rowids, cand)
        valid = pos < len(cache.rowids)
        valid[valid] &= cache.rowids[pos[valid]] == cand[valid]
        return pos[valid]

    def _fetch_rows_by_id(self, cur, collection_id: int, ids: list[str], with_embedding: bool):
        """Fetch document/metadata (and optionally the raw stored embedding)."""
        select = "id, document, metadata_json" + (", embedding" if with_embedding else "")
        by_id: dict[str, tuple] = {}
        for start in range(0, len(ids), 900):
            chunk = ids[start : start + 900]
            placeholders = ",".join("?" for _ in chunk)
            for row in cur.execute(
                f"SELECT {select} FROM documents "
                f"WHERE collection_id = ? AND id IN ({placeholders})",
                (collection_id, *chunk),
            ).fetchall():
                by_id[row[0]] = tuple(row)
        return by_id

    def query(
        self,
        *,
        query_texts=None,
        query_embeddings=None,
        n_results=10,
        where=None,
        where_document=None,
        include=None,
    ) -> QueryResult:
        if query_texts is not None:
            raise ValueError(
                "sqlite_exact requires query_embeddings; use palace.get_collection wrapper"
            )
        if query_embeddings is None:
            raise ValueError("query requires query_embeddings")
        if not query_embeddings:
            raise ValueError("query input must be a non-empty list")

        _validate_where(where)
        _validate_where(where_document)
        spec = _IncludeSpec.resolve(include, default_distances=True)
        outer_ids: list[list[str]] = []
        outer_docs: list[list[str]] = []
        outer_metas: list[list[dict]] = []
        outer_dists: list[list[float]] = []
        outer_embeds: list[list[list[float]]] = []

        with self._cursor() as cur:
            collection_id = self._collection_id(cur)
            expected_dim = self._collection_dimension(cur, collection_id)
            for query_vector in query_embeddings:
                q = _as_vector_array(query_vector)
                if expected_dim is not None and int(q.size) != expected_dim:
                    raise DimensionMismatchError(
                        f"sqlite_exact collection {self._collection_name!r} expects "
                        f"embedding dimension {expected_dim}, got {int(q.size)}"
                    )
                dim = int(q.size)
                cache = self._vector_cache(cur, collection_id, dim)
                try:
                    cand_idx = self._cached_candidate_indices(
                        cur, collection_id, dim, cache, where, where_document
                    )
                except _WhereNotTranslatable:
                    top_ids, dists, sel = self._query_python_scan(
                        cur, q, n_results, where, where_document
                    )
                    self._append_query_output(
                        cur,
                        spec,
                        collection_id,
                        top_ids,
                        dists,
                        sel,
                        outer_ids,
                        outer_docs,
                        outer_metas,
                        outer_dists,
                        outer_embeds,
                    )
                    continue

                q_norm = float(np.linalg.norm(q))
                q_unit = (q / q_norm) if q_norm > 0 else np.zeros_like(q)
                scores = cache.matrix @ q_unit  # cosine: rows are L2-normalized

                if cand_idx is None:
                    pool_scores, pool_idx = scores, None
                else:
                    pool_scores, pool_idx = scores[cand_idx], cand_idx
                k = min(int(n_results), len(pool_scores))
                if k <= 0:
                    sel_global = np.empty(0, dtype=np.int64)
                else:
                    part = np.argpartition(-pool_scores, k - 1)[:k]
                    sel_global = part if pool_idx is None else pool_idx[part]
                cos = np.clip(scores[sel_global], -1.0, 1.0)
                dist = 1.0 - cos
                # Sort by distance, ties broken by matrix index = rowid order,
                # matching the old stable-sort-over-row-order behavior.
                order = np.lexsort((sel_global, dist))
                sel_global = sel_global[order]
                dist = dist[order]

                top_ids = [cache.ids[i] for i in sel_global]
                self._append_query_output(
                    cur,
                    spec,
                    collection_id,
                    top_ids,
                    [float(d) for d in dist],
                    None,
                    outer_ids,
                    outer_docs,
                    outer_metas,
                    outer_dists,
                    outer_embeds,
                )

        return QueryResult(
            ids=outer_ids,
            documents=outer_docs,
            metadatas=outer_metas,
            distances=outer_dists,
            embeddings=outer_embeds if spec.embeddings else None,
        )

    def _query_python_scan(self, cur, q: np.ndarray, n_results: int, where, where_document):
        """Legacy per-row scan, kept as the fallback for untranslatable filters."""
        rows = self._rows(cur, where=where, where_document=where_document)
        q_norm = float(np.linalg.norm(q))
        scored = []
        for row in rows:
            vec = _decode_array(row["embedding"])
            if vec is None or vec.size != q.size:
                continue
            denom = q_norm * float(np.linalg.norm(vec))
            cos = 0.0 if denom <= 0 else float(np.dot(q, vec) / denom)
            distance = 1.0 - max(-1.0, min(1.0, cos))
            scored.append((distance, row["id"]))
        scored.sort(key=lambda item: item[0])
        top = scored[:n_results]
        return [doc_id for _, doc_id in top], [float(d) for d, _ in top], None

    def _append_query_output(
        self,
        cur,
        spec,
        collection_id,
        top_ids,
        dists,
        _sel,
        outer_ids,
        outer_docs,
        outer_metas,
        outer_dists,
        outer_embeds,
    ) -> None:
        outer_ids.append(list(top_ids))
        outer_dists.append(list(dists) if spec.distances else [])
        need_rows = spec.documents or spec.metadatas or spec.embeddings
        by_id = (
            self._fetch_rows_by_id(cur, collection_id, top_ids, spec.embeddings)
            if need_rows and top_ids
            else {}
        )
        docs: list[str] = []
        metas: list[dict] = []
        embeds: list[list[float]] = []
        for doc_id in top_ids:
            row = by_id.get(doc_id)
            if row is None:
                docs.append("")
                metas.append({})
                embeds.append([])
                continue
            docs.append(row[1] or "")
            metas.append(_json_loads(row[2]))
            if spec.embeddings:
                embeds.append(_decode_vector(row[3]))
        outer_docs.append(docs if spec.documents else [])
        outer_metas.append(metas if spec.metadatas else [])
        if spec.embeddings:
            outer_embeds.append(embeds)

    def get(
        self,
        *,
        ids=None,
        where=None,
        where_document=None,
        limit=None,
        offset=None,
        include=None,
    ) -> GetResult:
        spec = _IncludeSpec.resolve(include, default_distances=False)
        # Fast path for pages without an ids= post-filter (e.g. the
        # prefetch_mined_set and status sweeps, and the searcher's filtered
        # neighbor/hydration reads): push LIMIT/OFFSET down to _rows, which
        # emits SQL LIMIT/OFFSET when the filter compiled (or slices
        # identically after the Python fallback filter). Safe only with no
        # ids= post-filter and non-negative bounds: SQLite does not honor a
        # negative LIMIT or OFFSET the way a Python slice does, so those keep
        # the slice path.
        push_page = (
            ids is None
            and (limit is None or limit >= 0)
            and (offset is None or offset >= 0)
            and (limit is not None or offset)
        )
        with self._cursor() as cur:
            rows = self._rows(
                cur,
                where=where,
                where_document=where_document,
                limit=limit if push_page else None,
                offset=offset if push_page else None,
                with_embedding=spec.embeddings,
            )
        if not push_page:
            if ids is not None:
                by_id = {row["id"]: row for row in rows}
                rows = [by_id[doc_id] for doc_id in ids if doc_id in by_id]
            if offset:
                rows = rows[offset:]
            if limit is not None:
                rows = rows[:limit]
        return GetResult(
            ids=[row["id"] for row in rows],
            documents=[row["document"] for row in rows] if spec.documents else [],
            metadatas=[row["metadata"] for row in rows] if spec.metadatas else [],
            embeddings=(
                [_decode_vector(row["embedding"]) for row in rows] if spec.embeddings else None
            ),
        )

    def delete(self, *, ids=None, where=None):
        with self._cursor() as cur:
            collection_id = self._collection_id(cur)
            if ids is None:
                rows = self._rows(cur, where=where, with_embedding=False)
                ids = [row["id"] for row in rows]
            deleted = 0
            fts = self._fts_available(cur)
            for doc_id in ids or []:
                # Resolve the rowid before deleting so the FTS row (which
                # mirrors it) can be removed with an O(1) rowid delete.
                rowid = self._document_rowid(cur, collection_id, doc_id)
                if rowid is None:
                    continue
                cur.execute(
                    "DELETE FROM documents WHERE collection_id = ? AND id = ?",
                    (collection_id, doc_id),
                )
                deleted += max(0, cur.rowcount)
                if fts:
                    cur.execute("DELETE FROM docs_fts WHERE rowid = ?", (rowid,))
            if deleted:
                self._bump_vec_generation(cur, collection_id)

    def _existing_rowids(self, cur, collection_id: int, ids: list[str]) -> dict[str, int]:
        found: dict[str, int] = {}
        for start in range(0, len(ids), 900):
            chunk = ids[start : start + 900]
            placeholders = ",".join("?" for _ in chunk)
            for doc_id, rowid in cur.execute(
                f"SELECT id, rowid FROM documents "
                f"WHERE collection_id = ? AND id IN ({placeholders})",
                (collection_id, *chunk),
            ).fetchall():
                found[doc_id] = int(rowid)
        return found

    def count(self) -> int:
        with self._cursor() as cur:
            collection_id = self._collection_id(cur)
            row = cur.execute(
                "SELECT COUNT(*) FROM documents WHERE collection_id = ?",
                (collection_id,),
            ).fetchone()
            return int(row[0]) if row else 0

    def lexical_search(self, *, query: str, n_results: int = 10, where: Optional[dict] = None):
        _validate_where(where)
        with self._cursor() as cur:
            hits = self._lexical_search_fts(cur, query=query, n_results=n_results, where=where)
            if hits is not None:
                return LexicalResult(hits=hits)
            rows = self._rows(cur, where=where, with_embedding=False)
        scores = _bm25_scores(query, [row["document"] for row in rows])
        scored = [
            LexicalHit(
                id=row["id"],
                document=row["document"],
                metadata=row["metadata"],
                score=score,
            )
            for row, score in zip(rows, scores)
            if score > 0
        ]
        scored.sort(key=lambda hit: hit.score, reverse=True)
        return LexicalResult(hits=scored[:n_results])

    def _lexical_search_fts(self, cur, *, query: str, n_results: int, where: Optional[dict]):
        if not self._fts_available(cur):
            return None
        # ≥3 like the chroma lane: unicode61 matches whole tokens only, so
        # 2-char tokens ("to", "of") just explode the OR match set with noise.
        tokens = [t for t in _tokenize(query) if len(t) >= 3]
        if not tokens:
            return None
        fts_query = " OR ".join(tokens)
        collection_id = self._collection_id(cur)
        window = max(_FTS_CANDIDATE_WINDOW, n_results)
        if where:
            hits = self._lexical_search_fts_filtered(
                cur,
                fts_query=fts_query,
                query=query,
                n_results=n_results,
                window=window,
                where=where,
                collection_id=collection_id,
            )
            if hits is not None:
                return hits
            # untranslatable filter → fall through to the full-window
            # post-filter path below
        try:
            limit_sql = "" if where else "LIMIT ?"
            params = (fts_query, collection_id)
            if not where:
                params = (*params, window)
            rows = cur.execute(
                f"""
                SELECT doc_id, bm25(docs_fts) AS rank
                FROM docs_fts
                WHERE docs_fts MATCH ? AND collection_id = ?
                ORDER BY rank
                {limit_sql}
                """,
                params,
            ).fetchall()
        except sqlite3.Error:
            logger.debug("sqlite_exact FTS query failed; using Python lexical scan", exc_info=True)
            return None
        if not rows:
            return []
        ids = [row[0] for row in rows]
        docs = []
        for start in range(0, len(ids), 900):
            chunk_ids = ids[start : start + 900]
            placeholders = ",".join("?" for _ in chunk_ids)
            docs.extend(
                cur.execute(
                    f"""
                    SELECT id, document, metadata_json
                    FROM documents
                    WHERE collection_id = ? AND id IN ({placeholders})
                    """,
                    (collection_id, *chunk_ids),
                ).fetchall()
            )
        by_id = {doc_id: (doc or "", _json_loads(meta_json)) for doc_id, doc, meta_json in docs}
        candidates = []
        for doc_id, _rank in rows:
            doc_meta = by_id.get(doc_id)
            if doc_meta is None:
                continue
            doc, meta = doc_meta
            if not _matches_where(meta, where):
                continue
            candidates.append((doc_id, doc, meta))
        return self._rescore_lexical_candidates(query, candidates, n_results)

    @staticmethod
    def _rescore_lexical_candidates(
        query: str, candidates: list[tuple], n_results: int
    ) -> list[LexicalHit]:
        """Rank FTS *candidates* with the shared Okapi BM25, window-relative IDF.

        FTS5's own ``bm25()`` ranks with whole-corpus IDF and k1=1.2, which is
        a different function from the chroma lane's candidate-window Okapi
        rescore — measured 2026-08-09, corpus-IDF top-N selection systematically
        favors term-stuffed chunks and regressed 3 real click-through goldens
        while the TF-IDF-derived 300-case set (whose targets *are* the
        term-dense doc) still improved. The FTS stage is a candidate generator;
        this rescore is the lane's actual ranking, same as chroma's.
        """
        scores = _bm25_scores(query, [doc for _, doc, _ in candidates])
        hits = [
            LexicalHit(id=doc_id, document=doc, metadata=meta, score=float(score))
            for (doc_id, doc, meta), score in zip(candidates, scores)
            if score > 0
        ]
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:n_results]

    def _lexical_search_fts_filtered(
        self,
        cur,
        *,
        fts_query: str,
        query: str,
        n_results: int,
        window: int,
        where: dict,
        collection_id: int,
    ):
        """Filtered lexical lane via one FTS5 JOIN with the where compiled to SQL.

        The join keeps the window bounded *within the filter scope* (the old
        filtered path read the entire match set), and the window is then
        rescored by :meth:`_rescore_lexical_candidates` — LIMIT ``window``, not
        ``n_results``: a LIMIT at ``n_results`` made FTS5's corpus-IDF bm25 the
        lane's final ranking, which is the regression described there. Returns
        ``None`` when the filter cannot be compiled; the caller falls back to
        the full-window path.
        """
        try:
            where_sql, where_params = _compile_where(where, prefix="d.")
        except _WhereNotTranslatable:
            return None
        try:
            rows = cur.execute(
                f"""
                SELECT f.doc_id, bm25(f) AS rank, d.document, d.metadata_json
                FROM docs_fts f
                JOIN documents d ON d.collection_id = f.collection_id AND d.id = f.doc_id
                WHERE f MATCH ? AND f.collection_id = ? AND ({where_sql})
                ORDER BY rank
                LIMIT ?
                """,
                (fts_query, collection_id, *where_params, window),
            ).fetchall()
        except sqlite3.Error:
            logger.debug(
                "sqlite_exact filtered FTS join failed; using full-window scan", exc_info=True
            )
            return None
        candidates = [(row[0], row[2] or "", _json_loads(row[3])) for row in rows]
        return self._rescore_lexical_candidates(query, candidates, n_results)

    def close(self) -> None:
        self._closed = True

    def health(self) -> HealthStatus:
        if self._closed or self._handle.closed:
            return HealthStatus.unhealthy("collection closed")
        return HealthStatus.healthy()

    def maintenance_state(self) -> dict:
        try:
            rows = self.count()
        except Exception:
            rows = 0
        # vector_index is null by design — exact cosine over every row, no ANN.
        state = {"row_count": rows, "vector_index": None}
        try:
            with self._cursor() as cur:
                page_count = cur.execute("PRAGMA page_count").fetchone()
                freelist = cur.execute("PRAGMA freelist_count").fetchone()
            state["page_count"] = int(page_count[0]) if page_count else 0
            state["freelist_pages"] = int(freelist[0]) if freelist else 0
        except Exception:
            pass
        return state

    def run_maintenance(self, kind: str):
        from .base import MaintenanceResult, UnsupportedMaintenanceKindError

        if kind not in SQLiteExactBackend.maintenance_kinds:
            raise UnsupportedMaintenanceKindError(
                f"sqlite_exact does not support maintenance kind {kind!r}"
            )
        if kind == "analyze":
            # Refresh planner stats. Concurrent runs serialize on the handle lock.
            with self._cursor() as cur:
                cur.execute("ANALYZE")
            return MaintenanceResult(kind="analyze", status="ran")

        # compact → VACUUM. It cannot run inside a transaction, so flip the
        # connection to autocommit for the duration. The handle lock serializes
        # concurrent runs in-process; SQLite's own write lock serializes across
        # processes.
        before = self.maintenance_state()
        with self._handle.lock:
            self._ensure_open()
            conn = self._handle.conn
            prev_isolation = conn.isolation_level
            try:
                conn.commit()
                conn.isolation_level = None
                conn.execute("VACUUM")
            finally:
                conn.isolation_level = prev_isolation
        after = self.maintenance_state()
        reclaimed = max(0, before.get("page_count", 0) - after.get("page_count", 0))
        return MaintenanceResult(
            kind="compact",
            status="ran",
            stats={
                "pages_before": before.get("page_count", 0),
                "pages_after": after.get("page_count", 0),
                "pages_reclaimed": reclaimed,
            },
        )


class SQLiteExactBackend(BaseBackend):
    name = "sqlite_exact"
    capabilities = frozenset(
        {
            "requires_explicit_embeddings",
            "supports_embeddings_in",
            "supports_embeddings_passthrough",
            "supports_embeddings_out",
            "supports_metadata_filters",
            "supports_lexical_search",
            "local_mode",
        }
    )
    # "reindex" is intentionally omitted: sqlite_exact does exact cosine over
    # every row (no ANN index to build), so it has no analogue for it.
    maintenance_kinds = frozenset({"analyze", "compact"})

    def __init__(self):
        self._clients: dict[str, _SQLiteExactHandle] = {}
        self._clients_lock = threading.RLock()
        self._closed = False

    @staticmethod
    def _db_path(palace_path: str) -> str:
        return os.path.join(palace_path, _DB_FILENAME)

    def _connect(self, palace_path: str, create: bool):
        if self._closed:
            raise BackendClosedError("SQLiteExactBackend has been closed")
        db_path = self._db_path(palace_path)
        if not create and not os.path.isfile(db_path):
            raise PalaceNotFoundError(db_path)
        if create:
            os.makedirs(palace_path, exist_ok=True)
            try:
                os.chmod(palace_path, 0o700)
            except (OSError, NotImplementedError):
                pass
        # Hold the registry lock across cache-check + connect + schema init:
        # two threads first-opening the same palace must not each create a
        # connection (the loser leaked unclosed and outlived close()) nor run
        # _init_schema concurrently on a fresh file, which surfaces transient
        # "database is locked" errors before WAL mode is established. Only
        # first-open pays for the I/O under the lock; cache hits are a dict
        # probe.
        with self._clients_lock:
            if self._closed:
                raise BackendClosedError("SQLiteExactBackend has been closed")
            cached = self._clients.get(palace_path)
            if cached is not None and not cached.closed:
                return cached
            conn = sqlite3.connect(db_path, check_same_thread=False)
            try:
                conn.row_factory = sqlite3.Row
                # Per-connection pragmas (WAL itself is persistent, set in
                # _init_schema). busy_timeout lets a reader ride out another
                # process's write transaction instead of failing immediately;
                # synchronous=NORMAL is the documented WAL pairing — commits
                # skip the per-transaction fsync but the WAL still guarantees
                # the DB can never be corrupted by an app or OS crash.
                # Deliberately no mmap_size: a laptop opening this palace over
                # SMB with mmap enabled risks silent corruption (machine.md).
                conn.execute("PRAGMA busy_timeout = 10000")
                conn.execute("PRAGMA synchronous = NORMAL")
                lock = threading.RLock()
                handle = _SQLiteExactHandle(conn, lock)
                with handle.lock:
                    self._init_schema(conn)
            except BaseException:
                conn.close()
                raise
            self._clients[palace_path] = handle
            return handle

    def _init_schema(self, conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS collections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                dimension INTEGER,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS documents (
                collection_id INTEGER NOT NULL,
                id TEXT NOT NULL,
                document TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                embedding BLOB NOT NULL,
                dim INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (collection_id, id),
                FOREIGN KEY(collection_id) REFERENCES collections(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_documents_collection
                ON documents(collection_id);
            """
        )
        columns = {row[1] for row in conn.execute("PRAGMA table_info(collections)").fetchall()}
        if "dimension" not in columns:
            conn.execute("ALTER TABLE collections ADD COLUMN dimension INTEGER")
        # Generated columns for the hot filter keys, so a wing/room/date WHERE
        # is an indexed point lookup instead of a full metadata_json scan.
        # table_xinfo, not table_info: virtual generated columns are "hidden"
        # and table_info omits them, which would re-ALTER on every connect.
        doc_columns = {row[1] for row in conn.execute("PRAGMA table_xinfo(documents)").fetchall()}
        for col in _GENERATED_META_COLUMNS:
            if col not in doc_columns:
                conn.execute(
                    f"ALTER TABLE documents ADD COLUMN {col} TEXT "
                    f"GENERATED ALWAYS AS (json_extract(metadata_json, '$.\"{col}\"')) VIRTUAL"
                )
        # Index version 2: every hot-path index carries `dim` so the vector
        # cache's freshness recheck (MAX(rowid)/COUNT per (collection_id, dim))
        # and the filtered candidate scans are index-only. Without `dim` in the
        # index, each of the ~n entries costs a table probe — measured at
        # 134 ms p50 per query on a 174k palace, ~30× the matmul it guards.
        # Index version 3 = version 2 (every hot-path index carries `dim`)
        # plus generated-column indexes for source_file / parent_drawer_id.
        # v2 → v3 only builds the two new indexes (CREATE IF NOT EXISTS skips
        # the unchanged three); pre-v2 palaces drop the dim-less shapes first.
        version_row = conn.execute("SELECT value FROM meta WHERE key = 'index_version'").fetchone()
        version = version_row[0] if version_row else None
        if version not in ("2", "3"):
            for col in _GENERATED_META_COLUMNS:
                conn.execute(f"DROP INDEX IF EXISTS idx_documents_{col}")
        for col in _GENERATED_META_COLUMNS:
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_documents_{col} "
                f"ON documents(collection_id, {col}, dim)"
            )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_collection_dim "
            "ON documents(collection_id, dim)"
        )
        if version != "3":
            conn.execute(
                "INSERT INTO meta(key, value) VALUES ('index_version', '3') "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value"
            )
        try:
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts
                USING fts5(collection_id UNINDEXED, doc_id UNINDEXED, document)
                """
            )
            conn.execute(
                """
                INSERT INTO meta(key, value)
                VALUES ('fts5_available', '1')
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """
            )
            # One-time migration: align docs_fts rowids with documents rowids
            # so FTS maintenance is a rowid delete, not a full-table scan over
            # the UNINDEXED columns. Palaces written by the pre-alignment code
            # have arbitrary FTS rowids; rebuild once and mark it done.
            aligned = conn.execute(
                "SELECT value FROM meta WHERE key = 'fts_rowid_aligned'"
            ).fetchone()
            if not aligned or aligned[0] != "1":
                conn.execute("DELETE FROM docs_fts")
                conn.execute(
                    "INSERT INTO docs_fts(rowid, collection_id, doc_id, document) "
                    "SELECT rowid, collection_id, id, document FROM documents"
                )
                conn.execute(
                    "INSERT INTO meta(key, value) VALUES ('fts_rowid_aligned', '1') "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value"
                )
        except sqlite3.OperationalError:
            conn.execute(
                """
                INSERT INTO meta(key, value)
                VALUES ('fts5_available', '0')
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """
            )
        conn.commit()

    def get_collection(
        self,
        *args,
        **kwargs,
    ) -> SQLiteExactCollection:
        palace, collection_name, create = self._normalize_args(args, kwargs)
        palace_path = palace.local_path
        if palace_path is None:
            raise PalaceNotFoundError("SQLiteExactBackend requires PalaceRef.local_path")
        if not create and not os.path.isdir(palace_path):
            raise PalaceNotFoundError(palace_path)
        handle = self._connect(palace_path, create=create)
        with handle.lock:
            row = handle.conn.execute(
                "SELECT id FROM collections WHERE name = ?",
                (collection_name,),
            ).fetchone()
            if row is None:
                if not create:
                    raise CollectionNotInitializedError(collection_name)
                handle.conn.execute(
                    "INSERT INTO collections(name, created_at) VALUES (?, ?)",
                    (collection_name, _utcnow()),
                )
                handle.conn.commit()
        return SQLiteExactCollection(handle, collection_name)

    @staticmethod
    def _normalize_args(args, kwargs):
        if "palace" in kwargs:
            palace = kwargs.pop("palace")
            if not isinstance(palace, PalaceRef):
                raise TypeError("palace= must be a PalaceRef instance")
            collection_name = kwargs.pop("collection_name")
            create = bool(kwargs.pop("create", False))
            kwargs.pop("options", None)
            if args or kwargs:
                raise TypeError("unexpected arguments to get_collection")
            return palace, collection_name, create
        if args:
            palace_path = args[0]
            rest = list(args[1:])
            collection_name = kwargs.pop("collection_name", None) or (rest.pop(0) if rest else None)
            if collection_name is None:
                raise TypeError("collection_name is required")
            create = kwargs.pop("create", False)
            if rest:
                create = rest.pop(0)
            if rest or kwargs:
                raise TypeError("unexpected arguments to get_collection")
            return PalaceRef(id=palace_path, local_path=palace_path), collection_name, bool(create)
        if "palace_path" in kwargs:
            palace_path = kwargs.pop("palace_path")
            collection_name = kwargs.pop("collection_name")
            create = bool(kwargs.pop("create", False))
            if kwargs:
                raise TypeError("unexpected arguments to get_collection")
            return PalaceRef(id=palace_path, local_path=palace_path), collection_name, create
        raise TypeError("get_collection requires palace= or a positional palace_path")

    def close_palace(self, palace: PalaceRef | str) -> None:
        path = palace.local_path if isinstance(palace, PalaceRef) else palace
        if path is None:
            return
        with self._clients_lock:
            cached = self._clients.pop(path, None)
        if cached is not None:
            with cached.lock:
                cached.closed = True
                cached.conn.close()

    def close(self) -> None:
        # Flip _closed under the registry lock so a concurrent _connect either
        # sees the flag or finishes before the handle snapshot is taken; a
        # connection can no longer slip into the registry after close().
        # Unlocked readers of _closed elsewhere are advisory fast-fails; the
        # locked recheck in _connect is the authoritative gate.
        with self._clients_lock:
            handles = list(self._clients.values())
            self._clients.clear()
            self._closed = True
        for handle in handles:
            with handle.lock:
                handle.closed = True
                handle.conn.close()

    def health(self, palace: Optional[PalaceRef] = None) -> HealthStatus:
        if self._closed:
            return HealthStatus.unhealthy("backend closed")
        if palace and palace.local_path and not os.path.isfile(self._db_path(palace.local_path)):
            return HealthStatus.unhealthy("sqlite_exact database not found")
        return HealthStatus.healthy()

    @classmethod
    def detect(cls, path: str) -> bool:
        """Return True when ``path`` looks like a sqlite_exact palace.

        Verifies the SQLite magic header rather than file presence alone, for
        the same reason as :py:meth:`mempalace.backends.chroma.ChromaBackend.detect`:
        bare ``sqlite3.connect()`` against a missing path leaves a 0-byte file
        behind because the SQLite header is written on the first statement,
        not on connection. The 16-byte ``SQLite format 3\\x00`` magic prefix
        accepts every real palace while rejecting empty / garbage files. See #1893.
        """
        db_path = os.path.join(path, _DB_FILENAME)
        if not os.path.isfile(db_path):
            return False
        try:
            with open(db_path, "rb") as f:
                return f.read(16) == b"SQLite format 3\x00"
        except OSError:
            return False

    def create_collection(self, palace_path: str, collection_name: str) -> SQLiteExactCollection:
        return self.get_collection(palace_path, collection_name, create=True)

    def get_or_create_collection(self, palace_path: str, collection_name: str):
        return self.get_collection(palace_path, collection_name, create=True)

    def delete_collection(self, palace_path: str, collection_name: str) -> None:
        handle = self._connect(palace_path, create=False)
        with handle.lock:
            row = handle.conn.execute(
                "SELECT id FROM collections WHERE name = ?",
                (collection_name,),
            ).fetchone()
            if row is None:
                raise CollectionNotInitializedError(collection_name)
            collection_id = int(row[0])
            handle.conn.execute("DELETE FROM documents WHERE collection_id = ?", (collection_id,))
            try:
                handle.conn.execute(
                    "DELETE FROM docs_fts WHERE collection_id = ?",
                    (collection_id,),
                )
            except sqlite3.OperationalError:
                pass
            handle.conn.execute("DELETE FROM collections WHERE id = ?", (collection_id,))
            # Bump-then-forget: the generation write invalidates other
            # processes' caches for this collection id; dropping the local
            # entries frees the matrix memory immediately.
            handle.conn.execute(
                "INSERT INTO meta(key, value) VALUES (?, '1') "
                "ON CONFLICT(key) DO UPDATE SET value = CAST(CAST(value AS INTEGER) + 1 AS TEXT)",
                (f"vec_gen:{collection_id}",),
            )
            handle.conn.commit()
            for key in [k for k in handle.vec_caches if k[0] == collection_id]:
                handle.vec_caches.pop(key, None)


__all__ = ["SQLiteExactBackend", "SQLiteExactCollection"]
