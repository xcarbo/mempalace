# Task: rebuild the exact store and run the rewritten Phase-3 cutover gate

You are working in an isolated git worktree of `xcarbo/mempalace` (branch `xdev-patches`).
Wing for any palace reference: **mempalace**.

You are running unattended, launched by cron. **Nobody is awake.** Everything you need is below;
there is no one to ask and no release signal coming. If you hit a genuine blocker, write what you
found to `result.md` and stop — a clear "stopped because X" is a good outcome.

## HARD RULES

1. **NEVER write to the live palace at `~/.mempalace/palace`** (symlink → `/Volumes/xData/.mempalace`).
   It is Chris's real memory, ~178k rows, and it is serving him. Read it, snapshot it, never mutate it.
2. **DO NOT flip production config.** Do not touch `~/.mempalace/config.json`. Your job ends at the
   gate verdict. Flipping the live backend is a decision for a human who is awake.
3. **No palace writes** — no `add-drawer`, `update-drawer`, `kg-add`.
4. Do not touch `~/.mempalace/tools`.
5. Commit to your own branch; do not merge to `xdev-patches`.
6. Two agents ran before you tonight and may have left heavy artifacts. Check free space on
   `/Volumes/xData` before building (you need ~1 GB) and clean up your own copies when done.

## What to do

### 1. Rebuild the exact store from a fresh snapshot — with EVERY collection

`/Volumes/xData/.mempalace-cutover/build_all.py` is the template and already does this correctly.
Take a fresh consistent copy of the live palace first (`cp -R` of the resolved path — resolve the
symlink with `pwd -P`; a copy by name archives the link, not the data).

**Both collections are mandatory.** The 2026-08-09 build omitted `mempalace_closets` (2,585 rows).
`search_memories` silently degrades to drawer-only when it is absent — no error — and that omission
both hid a latency bug and made recall look better than it was. Verify by count after building.

**Run `ANALYZE`** on the built store (or `run_maintenance("analyze")`). The merged fix is
stat-independent so it should not need it, but it costs 0.16 s and is belt-and-braces.

### 2. Phase-2 parity, per collection

Row count, full id-set parity, and 20-row spot checks (document byte-identical, metadata equal,
vector cosine 1.0). `/Volumes/xData/.mempalace-cutover/verify_all.py` is the working template.

### 3. Phase-3 gate — the rewritten one in `docs/sqlite-exact-cutover.md`

Read it and follow it exactly; it was rewritten on 2026-08-09 precisely because the old gate passed
twice while real regressions shipped. All criteria required:

- **Recall, per case:** the candidate's golden failure set must be a **subset** of the chroma leg's,
  measured on the same snapshot pair, same code, same config. Aggregate counts do not gate; cases do.
- **300-case aggregates** ≥ chroma.
- **Latency, absolute** (warm, closets present, over 300-case + full goldens): p50 ≤ 150 ms,
  p95 ≤ 500 ms, p99 ≤ 1 s, **max ≤ 2 s**; plus p95 ≤ chroma same-session.
- **Cold-process probe:** fresh interpreter, median of ≥5, ≤ 1.5× chroma's equivalent.
- **Hydration-heavy queries** must be in the run (the 3 highest closet-boost queries), each under the
  2 s max.
- **Query-plan preflight:** `EXPLAIN QUERY PLAN` on the SQL `get()` **actually emits** (trace it —
  do not hand-write approximations, that mistake was made twice) against the freshly built store.
  It must show a generated-column index, not a `(collection_id)` walk.
- **Closet-boost liveness:** at least one A/B query returns `matched_via="drawer+closet"`.
- **Machine discipline:** record loadavg through the run; void any leg that ran concurrently with
  other heavy work.

### 4. Write the verdict

`result.md` must open with a one-line **GATE: PASS** or **GATE: FAIL**, then the full criterion-by-
criterion table with numbers. If PASS, include the exact commands a human should run in the morning
to perform the flip (runbook Phase 4) and the rollback. If FAIL, say which criterion failed, by how
much, and what you think the cause is.

Also write `preclean-proposal.md` alongside — propose drawers for the controller to file, file none.

## Context you need

- The cutover exists because chroma's HNSW drops the true top-1 out of its top-10 on 5.1% of queries
  and chroma SIGBUSes under load (three crashes on 2026-08-09). The exact backend removes both.
- Two prior attempts failed and were rolled back: a lexical-lane ranking defect (fixed, `3360b77`)
  and a `get(where=…)` full-scan that made latency `155 ms + ~700 ms × closet-boosted hits` (fixed,
  merged `53b9cec` — the stat-independent variant).
- Current merged HEAD measures, on a pristine store: p50 57.3 / p95 114.1 / p99 158.9 / max 197.8 ms
  vs chroma 95.9 / 262.3 / 520.6. Recall bit-identical; goldens match chroma's 2-case failure set.
- Prior results worth reading: `~/.local/state/herdr-spawn/{perf-fix,perf-grok,perf-profile}-2608*/result.md`.
- 300-case set: `~/.local/state/herdr-spawn/ma-code-260808-224241/harness/eval_set.jsonl`.
- Golden set: `~/.mempalace/tools/golden_recall.json`.
- Production interpreter **3.14.6** (chromadb 1.5.9). 3.13.7 has 1.5.8 + `hnswlib` (vector extraction).
- Earlier tonight two agents ran before you: a workload ladder and an embedder matrix. If their
  results exist, read them — the embedder matrix may change what width the store should eventually
  hold, though **tonight's gate is on the CURRENT embedder (minilm@384)**; do not re-embed anything.

Six measurements on this project pointed the wrong way in 48 hours, each caught only by re-measuring.
If a number surprises you, measure it again before reporting it.
