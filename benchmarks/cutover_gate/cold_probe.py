#!/usr/bin/env python3
"""Cold-process probe: fresh interpreter, one production search, print total ms.

Origin: tn-gate-260810-231241. The caller times the whole process wall-clock
(interpreter + imports + model load + query) — this script also prints its
in-process split for diagnosis. Usage: cold_probe.py <palace_path>
"""

import os
import sys
import time

t0 = time.perf_counter()
os.environ["MEMPALACE_RETRIEVAL_LOG"] = "0"
for var in ("MEMPALACE_RERANK_URL", "MEMPALACE_RERANK_MODEL"):
    os.environ.pop(var, None)
sys.path.insert(0, "/Users/xdev/.local/state/herdr-spawn/tn-gate-260810-231241/worktree")
from mempalace.searcher import search_memories  # noqa: E402

t_import = time.perf_counter()
r = search_memories(
    "mempalace cutover roadmap decisions",
    palace_path=sys.argv[1],
    n_results=10,
    max_distance=1.5,
)
t_done = time.perf_counter()
n = len(r.get("results") or [])
print(
    f"import_ms={1000 * (t_import - t0):.0f} query_ms={1000 * (t_done - t_import):.0f} "
    f"total_inproc_ms={1000 * (t_done - t0):.0f} hits={n}"
)
