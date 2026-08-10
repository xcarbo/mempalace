#!/usr/bin/env python3
"""Dump chroma's authoritative id set per collection from the snapshot copy.

Origin: tn-gate-260810-231241. Run under 3.14.6 (chromadb 1.5.9).
"""

import json
import time

import chromadb

CO = "/Volumes/xData/.mempalace-cutover/tn-gate-260810"

t0 = time.time()
client = chromadb.PersistentClient(path=f"{CO}/palace-chroma")
out = {}
for name in ("mempalace_drawers", "mempalace_closets"):
    col = client.get_collection(name)
    n = col.count()
    ids = []
    offset = 0
    while True:
        batch = col.get(limit=50000, offset=offset, include=[])["ids"]
        if not batch:
            break
        ids.extend(batch)
        offset += len(batch)
    print(f"{name}: count()={n} enumerated={len(ids)} {time.time() - t0:.0f}s", flush=True)
    assert len(ids) == n, f"{name}: enumeration {len(ids)} != count {n}"
    json.dump(sorted(ids), open(f"{CO}/chroma_ids_{name}.json", "w"))
    out[name] = n
json.dump(out, open(f"{CO}/chroma_counts.json", "w"))
print("DONE", out)
