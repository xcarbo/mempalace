#!/usr/bin/env python3
"""Fresh consistent snapshot of the live palace for the tn-gate-260810 cutover gate.

Origin: tn-gate-260810-231241 (overnight Phase-3 gate agent), 2026-08-10.
Reads the live palace (resolved path, read-only), writes NOTHING to it.
Order: segment dirs first, sqlite online backup second, so every vector id in
the copied HNSW has its document row in the copied sqlite. A before/after
change probe detects writes during the window; if it fires, rerun.
"""

import os
import shutil
import sqlite3
import sys
import time

LIVE = "/Volumes/xData/.mempalace/palace"  # resolved real path (pwd -P verified)
DEST = "/Volumes/xData/.mempalace-cutover/tn-gate-260810/palace-chroma"


def probe(path):
    db = sqlite3.connect(f"file:{path}/chroma.sqlite3?mode=ro", uri=True)
    n, mx = db.execute("SELECT count(*), max(seq_id) FROM embeddings").fetchone()
    db.close()
    return n, mx


t0 = time.time()
before = probe(LIVE)
print(f"before: embeddings rows={before[0]} max_seq={before[1]}", flush=True)

if os.path.isdir(DEST):
    shutil.rmtree(DEST)
os.makedirs(DEST)

for entry in sorted(os.listdir(LIVE)):
    if entry.startswith("chroma.sqlite3"):
        continue
    src = os.path.join(LIVE, entry)
    dst = os.path.join(DEST, entry)
    if os.path.isdir(src):
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)
    print(f"copied {entry} {time.time() - t0:.0f}s", flush=True)

src_db = sqlite3.connect(f"file:{LIVE}/chroma.sqlite3?mode=ro", uri=True)
dst_db = sqlite3.connect(os.path.join(DEST, "chroma.sqlite3"))
src_db.backup(dst_db)
dst_db.close()
src_db.close()
print(f"sqlite online backup done {time.time() - t0:.0f}s", flush=True)

after = probe(LIVE)
print(f"after: embeddings rows={after[0]} max_seq={after[1]}", flush=True)
if before != after:
    print("CHANGED DURING SNAPSHOT — rerun", flush=True)
    sys.exit(2)

snap = probe(DEST)
print(f"snapshot: embeddings rows={snap[0]} max_seq={snap[1]}", flush=True)
print(f"DONE {time.time() - t0:.0f}s", flush=True)
