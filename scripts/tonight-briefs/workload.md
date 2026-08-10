# Task: does MemPalace survive Chris's REAL workload? Simulate it and measure.

You are working in an isolated git worktree of `xcarbo/mempalace` (fork branch `xdev-patches`).
Wing for any palace reference: **mempalace**.

**The question is not "is p95 110 ms good". It is: with Chris's actual concurrent usage on a 24 GB
Mini, does this backend hold up — and does it still hold up with the bigger embedder?** Everything
measured so far is single-process microbenchmarks. That is not how this machine is used.

## HARD RULES

1. **NEVER write to the live palace at `~/.mempalace/palace`** (symlink → `/Volumes/xData/.mempalace`).
   178k rows, Chris's real memory. Read it, copy it, never mutate it.
2. Cutover artifacts under `/Volumes/xData/.mempalace-cutover/` are **read-only**. Build your own
   copies elsewhere on `/Volumes/xData` (~1.3 TB free) and clean up after.
3. **No palace writes** — no `add-drawer`, `update-drawer`, `kg-add`. The controller files everything.
4. Do not touch `~/.mempalace/tools`.
5. Commit to your own branch; do not merge to `xdev-patches`.
6. **You have the machine, but Chris is asleep and it must be usable in the morning.** All scheduled
   cron jobs are paused for 24 h. `:4109` is running (leave it). Another agent may be doing
   non-benchmark analysis; a third is queued behind you. **Do not leave a runaway process, and do
   not fill the disk** — clean up palace copies as you go.

## Chris's actual profile, in his words

> "I will generally run three, four, or five different projects at the same time on the Mini. Each
> project may have one to five agents running. If there's multiple agents, it's probably because I
> have a master agent managing subagents."

So: **3–5 concurrent projects × 1–5 agents each = roughly 5 to 25 concurrent agent processes**, plus
their tool calls, plus session hooks firing on every stop, plus `:4109`, plus whatever Chris is doing
interactively. All on one 24 GB M4.

## What is already known (single-process, so treat as a floor not a forecast)

- Every `memp` CLI call and every session hook is **its own Python process**, and each one measured
  **~1,427 MB RSS** after its first search — because each rebuilds the full float32 vector matrix
  (177,946 × 384 × 4 ≈ 261 MB) plus loads the ONNX embedder, numpy, SQLite page cache.
- Cold start for that first search: **450 ms to ~3 s** depending on cache state.
- Steady-state query latency once warm, with the pending fix: **p50 ~55 ms, p95 ~110–130 ms**.
- Without the fix, closet-heavy queries cost **~700 ms per hydration call** and reached 24 s.
- Concurrency of the backend itself is fine at 2/4/8 reader threads (chroma dies at 2).
- A **future embedder change is likely**: nomic-embed-text-v1.5, either 384-d (MRL-truncated,
  matrix stays ~261 MB) or **768-d native (matrix ~522 MB)**. Chris explicitly wants to know whether
  the full 768 is affordable — he has said he may accept worse latency for better accuracy, but
  nobody has measured what "worse" means under load. The embedder's own idle RSS was measured at
  ~777 MB for nomic vs ~273 MB for the current minilm.

**The arithmetic that motivates this task:** 1.4 GB × 10 concurrent processes = 14 GB on a 24 GB box,
before agents, before Chris's own session. At nomic-768 it is worse. This may be fine, or it may be
the real blocker. Nobody has measured it. That is your job.

## What to build and measure

1. **A realistic workload harness.** Model the profile above: N concurrent worker processes, each
   doing a realistic mix — searches (use the 300-case eval set for queries), `get-drawer` by id,
   an occasional wing-filtered search, and a session-hook-shaped invocation (fresh process, one
   search, exit). Vary N across at least **1, 3, 5, 10, 15, 20**. Include a "master + subagents"
   shape where several processes start near-simultaneously (thundering herd at session start).
2. **Measure at each N**, and report the shape, not just averages:
   - total system RSS, free memory, **swap used and swap-ins/outs** (this is the one that kills a Mac),
   - per-process RSS,
   - query latency p50/p95/p99/max,
   - failures, timeouts, and any crash,
   - wall-clock to first useful result for a newly-started process (the hook budget is 500 ms).
3. **Do this for the configurations that matter**, in this priority order — if you run out of time,
   having 1 and 2 done properly beats all four done badly:
   1. `sqlite_exact` + fix + minilm@384 (the imminent cutover)
   2. `sqlite_exact` + fix + **nomic@768** (Chris's explicit question). If a 768 store does not
      exist yet, you do NOT need to re-embed the corpus to answer the memory question — you can
      measure the resident-matrix effect faithfully with a synthetic 768-wide store of the same row
      count, as long as you say clearly that recall was not measured, only footprint and latency.
   3. `sqlite_exact` + fix + nomic@384
   4. chroma (today's live baseline) — for comparison, and note chroma has its own crash behaviour
      under concurrency; if it dies, that IS the result.
4. **Find the knee.** At what N does the machine start swapping, and at what N does latency or
   failure rate become unacceptable? Give Chris a number: "you can run about X concurrent
   MemPalace-touching processes before it degrades, and here is what degrades first."
5. **Test the obvious mitigation.** There is an existing open follow-up to serve `memp` CLI reads
   via the resident `:4109` service instead of starting a fresh process each time
   (`…46832475` — "the instructed path is 30× slower"). If per-process resident cost is the
   blocker, that turns N × 1.4 GB into 1 × 1.4 GB. **Measure whether routing reads through :4109
   actually fixes the concurrency picture** — even a rough prototype or a proxy measurement
   (e.g. N concurrent HTTP clients against :4109 vs N concurrent CLI processes) answers it.

## Deliver

- **A table of N vs memory, swap, latency, failures**, per configuration.
- **The knee**, stated as a number Chris can plan against.
- **A plain answer to his question**: given 3–5 projects × 1–5 agents, do these latency and memory
  characteristics matter? Is full nomic-768 affordable on this machine under that load, or does it
  only work at 384, or only with a resident service?
- **The single highest-leverage change** to make it comfortable, with evidence.
- Anything you could not verify, and every place you extrapolated rather than measured.

Write `result.md` in the directory the spawn harness gives you, plus `preclean-proposal.md`
(propose drawers; file nothing yourself).

Six measurements on this project pointed the wrong way in the last 24 hours and each was caught only
by someone re-measuring. If a number surprises you, measure it again before reporting it. Say
"GUESS" wherever you extrapolate.
