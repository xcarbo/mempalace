#!/usr/bin/env python3
"""Phase-2 parity gates, per collection, tonight's snapshot pair.

Origin: tn-gate-260810-231241. Gate 1: row count (exact == chroma count on
the same snapshot). Gate 2: FULL id-set parity via direct SQL (not sampled).
Gate 3: 20-row spot checks — document byte-identical, full metadata dict
equal, stored vector cosine 1.0 vs the HNSW-extracted vector.
"""

import json
import os
import random
import sqlite3
import sys

import numpy as np

sys.path.insert(0, "/Volumes/xData/codeXD/mempalace")  # 2026-08-11 flip: merged HEAD
from mempalace.backends.sqlite_exact import SQLiteExactBackend  # noqa: E402

CO = "/Volumes/xData/.mempalace-cutover/tn-gate-260810"
OUT = os.path.join(CO, "palace-exact")

src = sqlite3.connect(f"file:{CO}/palace-chroma/chroma.sqlite3?mode=ro", uri=True)
id_map = dict(src.execute("SELECT embedding_id, id FROM embeddings"))
chroma_counts = json.load(open(f"{CO}/chroma_counts.json"))

conn = sqlite3.connect(f"file:{OUT}/sqlite_exact.sqlite3?mode=ro", uri=True)
backend = SQLiteExactBackend()


def chroma_meta(int_id):
    doc, meta = "", {}
    for key, sv, iv, fv, bv in src.execute(
        "SELECT key, string_value, int_value, float_value, bool_value "
        "FROM embedding_metadata WHERE id = ?",
        (int_id,),
    ):
        if key == "chroma:document":
            doc = sv or ""
            continue
        if key.startswith("chroma:"):
            continue
        v = (
            sv
            if sv is not None
            else iv
            if iv is not None
            else fv
            if fv is not None
            else (bool(bv) if bv is not None else None)
        )
        if v is not None:
            meta[key] = v
    return doc, meta


for name in ("mempalace_drawers", "mempalace_closets"):
    ids = json.load(open(f"{CO}/vector_ids_{name}.json"))
    vecs = np.load(f"{CO}/vectors_{name}.npy")
    chroma_ids = json.load(open(f"{CO}/chroma_ids_{name}.json"))
    col = backend.get_collection(OUT, name, create=False)

    n = col.count()
    assert n == len(ids) == chroma_counts[name] == len(chroma_ids), (
        f"{name}: count mismatch exact={n} artifacts={len(ids)} chroma={chroma_counts[name]}"
    )
    print(f"GATE 1 {name}: count {n} == chroma count OK")

    cid = conn.execute("SELECT id FROM collections WHERE name = ?", (name,)).fetchone()[0]
    stored = {
        r[0] for r in conn.execute("SELECT id FROM documents WHERE collection_id = ?", (cid,))
    }
    assert stored == set(chroma_ids), (
        f"{name}: id set differs — {len(stored - set(chroma_ids))} extra, "
        f"{len(set(chroma_ids) - stored)} missing"
    )
    print(f"GATE 2 {name}: FULL id-set parity ({len(stored)} ids) OK")

    random.seed(7)
    for i in random.sample(range(len(ids)), min(20, len(ids))):
        did = ids[i]
        got = col.get(ids=[did], include=["documents", "metadatas", "embeddings"])
        assert got.ids == [did], f"{name}: get({did}) returned {got.ids}"
        ref_doc, ref_meta = chroma_meta(id_map[did])
        assert got.documents[0] == ref_doc, f"{name}: doc mismatch {did}"
        assert got.metadatas[0] == ref_meta, (
            f"{name}: metadata mismatch {did}: {got.metadatas[0]} != {ref_meta}"
        )
        v = np.asarray(got.embeddings[0], dtype=np.float32)
        ref = vecs[i]
        cos = float(v @ ref / (np.linalg.norm(v) * np.linalg.norm(ref)))
        assert cos > 0.999999, f"{name}: vector mismatch {did} cos={cos}"
    print(f"GATE 3 {name}: 20-row doc/metadata/vector spot checks OK")

backend.close()
print("\nALL PARITY GATES PASSED (both collections)")
