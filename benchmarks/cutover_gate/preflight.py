#!/usr/bin/env python3
"""Query-plan preflight (runbook Phase 3, 2026-08-09 rewrite).

Origin: tn-gate-260810-231241. Traces the SQL that backend.get() ACTUALLY
emits (set_trace_callback — never hand-written SQL) for the two hydration
filter shapes, runs EXPLAIN QUERY PLAN on the traced statements, and asserts
a generated-column index is used, not a (collection_id) walk.

Runs twice by design: pass 1 on the stat-less freshly built store (the sharp
test — no sqlite_stat1), then ANALYZE via run_maintenance, then pass 2.
Exit 0 only if BOTH passes use the generated-column index for BOTH shapes.
"""

import os
import sqlite3
import sys
import time

sys.path.insert(0, "/Volumes/xData/codeXD/mempalace")  # 2026-08-11 flip: merged HEAD
from mempalace.backends.sqlite_exact import SQLiteExactBackend  # noqa: E402

CO = "/Volumes/xData/.mempalace-cutover/tn-gate-260810"
OUT = os.path.join(CO, "palace-exact")

backend = SQLiteExactBackend()
col = backend.get_collection(OUT, "mempalace_drawers", create=False)
conn = col._handle.conn

# a real source_file / parent_drawer_id pair from the store
row = conn.execute(
    "SELECT source_file, parent_drawer_id FROM documents "
    "WHERE source_file IS NOT NULL AND parent_drawer_id IS NOT NULL LIMIT 1"
).fetchone()
src_file, parent_id = row
print(f"probe values: source_file={src_file!r} parent_drawer_id={parent_id!r}\n")

SHAPES = {
    "bare-equality {source_file}": {"source_file": src_file},
    "$and {source_file, parent_drawer_id}": {
        "$and": [{"source_file": src_file}, {"parent_drawer_id": parent_id}]
    },
}


def stat1_present():
    r = conn.execute("SELECT count(*) FROM sqlite_master WHERE name='sqlite_stat1'").fetchone()[0]
    if not r:
        return False
    return conn.execute("SELECT count(*) FROM sqlite_stat1").fetchone()[0] > 0


def run_pass(label):
    print(f"=== pass: {label} (sqlite_stat1 populated: {stat1_present()}) ===")
    all_ok = True
    for shape_name, where in SHAPES.items():
        traced = []
        conn.set_trace_callback(lambda stmt: traced.append(stmt))
        res = col.get(where=where, include=["metadatas"])
        conn.set_trace_callback(None)
        stmts = [
            s
            for s in traced
            if s.lstrip().upper().startswith("SELECT")
            and "FROM documents" in s
            and "json_extract" in s
            or ('"source_file"' in s or "source_file" in s)
            and s.lstrip().upper().startswith("SELECT")
        ]
        # keep the statement(s) that carry the filter columns
        stmts = [
            s for s in traced if s.lstrip().upper().startswith("SELECT") and "source_file" in s
        ]
        assert stmts, f"no traced SELECT carrying the filter for {shape_name}: {traced}"
        print(f"\n-- shape: {shape_name} (rows returned: {len(res.ids)})")
        for s in stmts:
            print("TRACED SQL:")
            print("  " + s.replace("\n", "\n  "))
            plan = conn.execute(f"EXPLAIN QUERY PLAN {s}").fetchall()
            plan_txt = "; ".join(p[3] for p in plan)
            print(f"PLAN: {plan_txt}")
            ok = any(f"idx_documents_{c}" in plan_txt for c in ("source_file", "parent_drawer_id"))
            walk = "idx_documents_collection" in plan_txt and not ok
            status = (
                "OK (generated-column index)"
                if ok
                else "FAIL (collection walk)"
                if walk
                else f"FAIL (unrecognized plan)"
            )
            print(f"VERDICT: {status}")
            all_ok = all_ok and ok
    print(f"\npass '{label}': {'ALL OK' if all_ok else 'FAILED'}\n")
    return all_ok


p1 = run_pass("stat-less fresh build")
t0 = time.time()
res = col.run_maintenance("analyze")
print(f"ANALYZE: {res} in {time.time() - t0:.2f}s\n")
p2 = run_pass("post-ANALYZE")
backend.close()

print(f"PREFLIGHT {'PASS' if (p1 and p2) else 'FAIL'} (stat-less={p1}, post-analyze={p2})")
sys.exit(0 if (p1 and p2) else 1)
