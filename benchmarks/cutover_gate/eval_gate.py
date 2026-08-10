#!/usr/bin/env python3
"""Phase-3 gate harness — ONE leg per invocation (legs must never run
concurrently; the caller sequences them).

Usage: eval_gate.py <label> <palace_path>

Origin: tn-gate-260810-231241, per docs/sqlite-exact-cutover.md (2026-08-09
rewrite). Runs the 300-case set + the FULL golden set through production
search_memories (golden semantics identical to golden_recall.py: wing/room
filters, per-case limit, max_distance 1.5, rank among unique drawer ids,
one retry on failure). Records per-query latency, matched_via, closet-boost
hit counts, and loadavg at start + every 50 cases. Dumps everything to
gate_<label>.json.
"""

import json
import os
import statistics
import sys
import time

label, palace = sys.argv[1], sys.argv[2]
CO = "/Volumes/xData/.mempalace-cutover/tn-gate-260810"
HARNESS = os.path.expanduser("~/.local/state/herdr-spawn/ma-code-260808-224241/harness")
GOLDEN = os.path.expanduser("~/.mempalace/tools/golden_recall.json")

os.environ["MEMPALACE_RETRIEVAL_LOG"] = "0"
for var in ("MEMPALACE_RERANK_URL", "MEMPALACE_RERANK_MODEL"):
    os.environ.pop(var, None)

sys.path.insert(0, "/Users/xdev/.local/state/herdr-spawn/tn-gate-260810-231241/worktree")
from mempalace.searcher import search_memories  # noqa: E402

cases = [json.loads(line) for line in open(os.path.join(HARNESS, "eval_set.jsonl"))]
goldens = json.load(open(GOLDEN))
loadavgs = [{"at": "start", "load": os.getloadavg(), "t": time.time()}]
print(f"[{label}] {len(cases)} cases + {len(goldens)} goldens; load {os.getloadavg()}", flush=True)


def run_query(query, *, wing=None, room=None, n_results=10, max_distance=1.5):
    t0 = time.perf_counter()
    r = search_memories(
        query,
        palace_path=palace,
        wing=wing,
        room=room,
        n_results=n_results,
        max_distance=max_distance,
    )
    ms = (time.perf_counter() - t0) * 1000
    hits = r.get("results") or []
    closet_hits = sum(1 for h in hits if h.get("matched_via") == "drawer+closet")
    return hits, ms, closet_hits, r.get("error")


# warm-up: model + vector-cache build, untimed for the pool but recorded
_, warm_ms, _, warm_err = run_query("warmup palace roadmap")
print(f"[{label}] warmup {warm_ms:.0f}ms err={warm_err}", flush=True)

# ---- 300-case set (unfiltered) ----
recs = []
for i, c in enumerate(cases):
    if i % 50 == 0:
        loadavgs.append({"at": f"case{i}", "load": os.getloadavg(), "t": time.time()})
    try:
        hits, ms, ch, err = run_query(c["query"])
    except Exception as ex:
        recs.append(
            {
                "query": c["query"],
                "target": c["target"],
                "archive": c["archive"],
                "rank": None,
                "ms": None,
                "closet_hits": 0,
                "error": repr(ex)[:200],
            }
        )
        continue
    ids = list(dict.fromkeys(h.get("drawer_id") for h in hits))
    rank = ids.index(c["target"]) + 1 if c["target"] in ids else None
    recs.append(
        {
            "query": c["query"],
            "target": c["target"],
            "archive": c["archive"],
            "rank": rank,
            "ms": ms,
            "closet_hits": ch,
        }
    )
loadavgs.append({"at": "cases_done", "load": os.getloadavg(), "t": time.time()})

# ---- full golden set, golden_recall.py semantics ----
gold = []
for g in goldens:
    wanted = list(
        g.get("expect_any_of") or ([g["expect_drawer_id"]] if "expect_drawer_id" in g else [])
    )
    rank_within = int(g.get("rank_within", 3))

    def attempt():
        hits, ms, ch, err = run_query(
            g["query"],
            wing=g.get("wing"),
            room=g.get("room"),
            n_results=int(g.get("limit", 10)),
            max_distance=float(g.get("max_distance", 1.5)),
        )
        ids = list(dict.fromkeys(h.get("drawer_id") for h in hits))
        ranks = [ids.index(e) + 1 for e in wanted if e in ids]
        return (min(ranks) if ranks else None), ms, ch, len(hits)

    rank, ms, ch, nh = attempt()
    retried = False
    if not nh or rank is None or rank > rank_within:
        time.sleep(2)
        rank, ms2, ch, nh = attempt()
        retried = True
    gold.append(
        {
            "name": g["name"],
            "rank": rank,
            "rank_within": rank_within,
            "passed": rank is not None and rank <= rank_within,
            "known_failure": g.get("known_failure"),
            "ms": ms,
            "retry_ms": ms2 if retried else None,
            "closet_hits": ch,
            "retried": retried,
        }
    )
    if not gold[-1]["passed"]:
        print(f"[{label}] GOLDEN FAIL {g['name']}: rank={rank} want<={rank_within}", flush=True)
loadavgs.append({"at": "goldens_done", "load": os.getloadavg(), "t": time.time()})

# ---- wing-filtered probe ----
wing_lats = []
for _ in range(10):
    _, ms, _, _ = run_query("decisions locked strategy", wing="mempalace")
    wing_lats.append(ms)
loadavgs.append({"at": "end", "load": os.getloadavg(), "t": time.time()})

# ---- summarize ----
ok = [r for r in recs if r["ms"] is not None]
pool = (
    [r["ms"] for r in ok] + [x["ms"] for x in gold] + [x["retry_ms"] for x in gold if x["retry_ms"]]
)


def pct(lats, p):
    lats = sorted(lats)
    return lats[min(len(lats) - 1, int(p / 100 * len(lats)))]


def recall(rs, k):
    return sum(1 for x in rs if x["rank"] and x["rank"] <= k) / max(1, len(rs))


summary = {
    "label": label,
    "palace": palace,
    "n_cases": len(ok),
    "errors": len(recs) - len(ok),
    "R@1": recall(ok, 1),
    "R@5": recall(ok, 5),
    "R@10": recall(ok, 10),
    "MRR": sum(1.0 / x["rank"] for x in ok if x["rank"]) / max(1, len(ok)),
    "goldens_passed": sum(1 for x in gold if x["passed"]),
    "goldens_failed": [x["name"] for x in gold if not x["passed"]],
    "pool_n": len(pool),
    "p50": pct(pool, 50),
    "p95": pct(pool, 95),
    "p99": pct(pool, 99),
    "max": max(pool),
    "mean": statistics.mean(pool),
    "warmup_ms": warm_ms,
    "closet_boost_queries": sum(1 for r in recs if r["closet_hits"])
    + sum(1 for x in gold if x["closet_hits"]),
    "wing_probe_p50": pct(wing_lats, 50),
    "wing_probe_max": max(wing_lats),
    "load_max1m": max(l["load"][0] for l in loadavgs),
}
print(f"[{label}] " + json.dumps(summary, indent=1, default=str), flush=True)

json.dump(
    {
        "summary": summary,
        "records": recs,
        "goldens": gold,
        "wing_lats": wing_lats,
        "loadavgs": loadavgs,
    },
    open(os.path.join(CO, f"gate_{label}.json"), "w"),
    default=str,
)
print(f"[{label}] saved gate_{label}.json", flush=True)
