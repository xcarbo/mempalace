#!/usr/bin/env python3
"""Build sqlite_exact palace from tonight's fresh snapshot — EVERY collection.

Origin: tn-gate-260810-231241, 2026-08-10. Adapted from the proven
/Volumes/xData/.mempalace-cutover/build_all.py (2026-08-09); paths retargeted
to tonight's snapshot and to the gate worktree's merged HEAD. ANALYZE is NOT
run here — the stat-less query-plan preflight gates first, then analyze.py.
Run under python 3.13.7 (hnswlib).
"""

import json
import os
import pickle
import shutil
import sqlite3
import sys
import time

import numpy as np

CO = "/Volumes/xData/.mempalace-cutover/tn-gate-260810"
SRC = os.path.join(CO, "palace-chroma")
OUT = os.path.join(CO, "palace-exact")
BATCH = 2000

sys.path.insert(0, "/Users/xdev/.local/state/herdr-spawn/tn-gate-260810-231241/worktree")
import hnswlib  # noqa: E402

from mempalace.backends.base import EmbedderIdentity  # noqa: E402
from mempalace.backends.sqlite_exact import SQLiteExactBackend  # noqa: E402

t0 = time.time()
db = sqlite3.connect(f"file:{SRC}/chroma.sqlite3?mode=ro", uri=True)

collections = dict(db.execute("SELECT id, name FROM collections"))
vec_seg = {
    coll: seg
    for seg, coll in db.execute("SELECT id, collection FROM segments WHERE scope = 'VECTOR'")
}
print(f"collections: { {collections[c]: vec_seg.get(c) for c in collections} }", flush=True)

id_map = dict(db.execute("SELECT embedding_id, id FROM embeddings"))
docs, metas = {}, {}
for int_id, key, sv, iv, fv, bv in db.execute(
    "SELECT id, key, string_value, int_value, float_value, bool_value FROM embedding_metadata"
):
    if key == "chroma:document":
        docs[int_id] = sv or ""
        continue
    if key.startswith("chroma:"):
        continue
    value = (
        sv
        if sv is not None
        else iv
        if iv is not None
        else fv
        if fv is not None
        else (bool(bv) if bv is not None else None)
    )
    if value is None:
        continue
    metas.setdefault(int_id, {})[key] = value
print(f"id_map={len(id_map)} docs={len(docs)} {time.time() - t0:.0f}s", flush=True)

if os.path.isdir(OUT):
    shutil.rmtree(OUT)
os.makedirs(OUT)

backend = SQLiteExactBackend()
report = {}

for coll_id, coll_name in collections.items():
    seg = vec_seg.get(coll_id)
    seg_dir = os.path.join(SRC, seg)
    meta = pickle.load(open(os.path.join(seg_dir, "index_metadata.pickle"), "rb"))
    id_to_label = meta["id_to_label"]
    ids = list(id_to_label.keys())
    labels = np.array([id_to_label[i] for i in ids], dtype=np.int64)

    idx = hnswlib.Index(space="cosine", dim=384)
    idx.load_index(seg_dir, is_persistent_index=True, max_elements=meta["total_elements_added"])
    X = (
        np.vstack(
            [
                np.asarray(idx.get_items(labels[s : s + 20000]), dtype=np.float32)
                for s in range(0, len(labels), 20000)
            ]
        )
        if len(labels)
        else np.zeros((0, 384), dtype=np.float32)
    )
    norms = np.linalg.norm(X, axis=1) if len(X) else np.array([1.0])
    print(
        f"\n{coll_name}: {len(ids)} ids, norms min={norms.min():.6f} max={norms.max():.6f} "
        f"zero={(norms < 1e-6).sum()}",
        flush=True,
    )

    # cross-check against the chroma collection count on the same snapshot
    n_chroma = db.execute(
        "SELECT count(*) FROM embeddings WHERE segment_id = "
        "(SELECT id FROM segments WHERE collection = ? AND scope = 'METADATA')",
        (coll_id,),
    ).fetchone()[0]
    print(f"  chroma metadata-segment count: {n_chroma}", flush=True)

    col = backend.get_collection(OUT, coll_name, create=True)
    col.set_embedder_identity(EmbedderIdentity(model_name="minilm", dimension=384))

    missing = 0
    for start in range(0, len(ids), BATCH):
        cids = ids[start : start + BATCH]
        cdocs, cmetas = [], []
        for cid in cids:
            int_id = id_map.get(cid)
            d = docs.get(int_id)
            if d is None:
                missing += 1
                d = ""
            cdocs.append(d)
            cmetas.append(metas.get(int_id, {}))
        col.add(
            documents=cdocs,
            ids=cids,
            metadatas=cmetas,
            embeddings=X[start : start + BATCH].tolist(),
        )
    assert col.count() == len(ids), f"{coll_name}: {col.count()} != {len(ids)}"
    print(
        f"  inserted {len(ids)} rows, {missing} empty documents, {time.time() - t0:.0f}s",
        flush=True,
    )
    report[coll_name] = {"rows": len(ids), "empty_docs": missing, "chroma_meta_count": n_chroma}
    np.save(os.path.join(CO, f"vectors_{coll_name}.npy"), X)
    json.dump(ids, open(os.path.join(CO, f"vector_ids_{coll_name}.json"), "w"))

backend.close()

for fname in ("mempalace_embedder.json", "palace_format.json"):
    src_path = os.path.join(SRC, fname)
    if os.path.isfile(src_path):
        shutil.copy(src_path, os.path.join(OUT, fname))

json.dump(report, open(os.path.join(CO, "build_report.json"), "w"), indent=1)
print(f"\nDONE {time.time() - t0:.0f}s  {json.dumps(report)}")
os.system(f"du -sh {OUT}")
