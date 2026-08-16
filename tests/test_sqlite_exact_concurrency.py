"""Multi-process concurrency proof for sqlite_exact.

The reason this backend exists: ChromaDB measurably dies under 2 concurrent
accesses (7,586 clean requests at 1 thread, dead in ~20 s at 2 — 2026-08-08
audit). sqlite_exact must sustain one writer plus N reader *processes* on the
same palace file with zero errors. WAL gives readers a consistent snapshot
while the writer commits; busy_timeout absorbs write-lock contention.

The workers run as spawned processes (fresh interpreters, own connections —
exactly the shape of the CLI, the :4109 service, and the cron fleet hitting
one palace).
"""

import multiprocessing
import threading
import time
import traceback

import numpy as np

from mempalace.backends.sqlite_exact import SQLiteExactBackend

DIM = 16
SEED_ROWS = 100
RUN_SECONDS = 2.5


def _vectors(n, seed):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, DIM)).astype(np.float32).tolist()


def _writer_proc(palace_path, deadline, errors, ops):
    try:
        backend = SQLiteExactBackend()
        col = backend.get_collection(palace_path, "drawers", create=False)
        batch = 0
        count = 0
        while time.time() < deadline:
            ids = [f"w{batch}-{i}" for i in range(5)]
            col.add(
                documents=[f"writer batch {batch} row {i} palace text" for i in range(5)],
                ids=ids,
                metadatas=[{"wing": f"w{batch % 3}", "room": "technical"} for _ in ids],
                embeddings=_vectors(5, seed=1000 + batch),
            )
            count += 1
            if batch % 5 == 2:
                col.upsert(
                    documents=["mutated row"],
                    ids=[ids[0]],
                    metadatas=[{"wing": "mutated"}],
                    embeddings=_vectors(1, seed=2000 + batch),
                )
                count += 1
            if batch % 7 == 3:
                col.delete(ids=[ids[1]])
                count += 1
            batch += 1
        backend.close()
        ops.put(("writer", count))
    except Exception:
        errors.put(("writer", traceback.format_exc()))


def _reader_proc(palace_path, deadline, errors, ops, seed):
    try:
        backend = SQLiteExactBackend()
        # read_only=True is load-bearing, not decoration. Since upstream
        # c6e8783 the non-read-only open path runs _init_schema inside
        # mine_palace_lock, so a plain get_collection(create=False) — a pure
        # read — raises MineAlreadyRunning whenever anything else holds the
        # palace, which on this machine is every mine window. Readers must
        # declare themselves.
        # The backend's contract is options={"read_only": True}; the friendly
        # read_only= kwarg lives one layer up on palace.get_collection.
        col = backend.get_collection(
            palace_path, "drawers", create=False, options={"read_only": True}
        )
        q = _vectors(1, seed=seed)[0]
        count = 0
        while time.time() < deadline:
            result = col.query(query_embeddings=[q], n_results=5)
            assert len(result.ids[0]) > 0
            filtered = col.query(query_embeddings=[q], n_results=5, where={"wing": "w1"})
            for meta in filtered.metadatas[0]:
                assert meta.get("wing") == "w1"
            lex = col.lexical_search(query="palace text", n_results=5, where={"room": "technical"})
            for hit in lex.hits:
                assert hit.metadata.get("room") == "technical"
            col.get(ids=[result.ids[0][0]])
            assert col.count() > 0
            count += 1
        backend.close()
        ops.put(("reader", count))
    except Exception:
        errors.put(("reader", traceback.format_exc()))


def test_one_writer_three_reader_processes_stay_clean(tmp_path):
    palace_path = str(tmp_path)
    backend = SQLiteExactBackend()
    col = backend.get_collection(palace_path, "drawers", create=True)
    col.add(
        documents=[f"seed row {i} palace text" for i in range(SEED_ROWS)],
        ids=[f"seed{i}" for i in range(SEED_ROWS)],
        metadatas=[{"wing": f"w{i % 3}", "room": "technical"} for i in range(SEED_ROWS)],
        embeddings=_vectors(SEED_ROWS, seed=0),
    )
    backend.close()

    ctx = multiprocessing.get_context("spawn")
    errors = ctx.Queue()
    ops = ctx.Queue()
    deadline = time.time() + RUN_SECONDS
    procs = [ctx.Process(target=_writer_proc, args=(palace_path, deadline, errors, ops))]
    procs.extend(
        ctx.Process(target=_reader_proc, args=(palace_path, deadline, errors, ops, 10 + i))
        for i in range(3)
    )
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)
        assert not p.is_alive(), "concurrency worker hung"

    failures = []
    while not errors.empty():
        failures.append(errors.get())
    assert not failures, f"concurrent workers raised: {failures}"

    counts = {}
    while not ops.empty():
        role, n = ops.get()
        counts.setdefault(role, []).append(n)
    assert counts.get("writer", [0])[0] > 0, "writer made no progress"
    assert len(counts.get("reader", [])) == 3
    assert all(n > 0 for n in counts["reader"]), "a reader made no progress"

    # And the palace is intact afterwards.
    backend = SQLiteExactBackend()
    try:
        col = backend.get_collection(palace_path, "drawers", create=False)
        assert col.count() >= SEED_ROWS - 1  # writer deletes only its own rows
        result = col.query(query_embeddings=[_vectors(1, seed=99)[0]], n_results=5)
        assert len(result.ids[0]) == 5
    finally:
        backend.close()


def test_threaded_readers_during_writes_stay_clean(tmp_path):
    """In-process variant: 4 reader threads + 1 writer thread on one handle.

    The chroma failure mode this replaces was death at 2 concurrent accesses
    in one process; here the handle lock serializes access and nothing errors.
    """
    palace_path = str(tmp_path)
    backend = SQLiteExactBackend()
    col = backend.get_collection(palace_path, "drawers", create=True)
    col.add(
        documents=[f"seed row {i} palace text" for i in range(50)],
        ids=[f"seed{i}" for i in range(50)],
        metadatas=[{"wing": f"w{i % 3}"} for i in range(50)],
        embeddings=_vectors(50, seed=5),
    )
    deadline = time.time() + 1.0
    failures = []

    def reader(seed):
        try:
            q = _vectors(1, seed=seed)[0]
            while time.time() < deadline:
                result = col.query(query_embeddings=[q], n_results=3, where={"wing": "w1"})
                assert len(result.ids) == 1
        except Exception:
            failures.append(traceback.format_exc())

    def writer():
        try:
            i = 0
            while time.time() < deadline:
                col.add(
                    documents=[f"thread row {i}"],
                    ids=[f"t{i}"],
                    metadatas=[{"wing": "w1"}],
                    embeddings=_vectors(1, seed=500 + i),
                )
                if i % 4 == 0:
                    col.delete(ids=[f"t{i}"])
                i += 1
        except Exception:
            failures.append(traceback.format_exc())

    threads = [threading.Thread(target=reader, args=(20 + i,)) for i in range(4)]
    threads.append(threading.Thread(target=writer))
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    try:
        assert not failures, f"threaded workers raised: {failures}"
    finally:
        backend.close()
