# Task: give `get(ids=)` the compiled-SQL path — kill the 835 ms drawer-open scan

You are working in an isolated git worktree of `xcarbo/mempalace` (fork branch `xdev-patches`,
HEAD `53b9cec`). Wing for any palace reference: **mempalace**. It is ~23:4x GST, Chris is asleep.
**Deadline 02:30**, degrade-rather-than-overrun: a correct fix with partial benchmark coverage
beats a rushed fix. If the fix cannot be proven correct in time, report that and stop — do not ship
an unproven fast path.

## The defect (confirmed twice tonight: workload agent root-cause + gate agent code-read)

`mempalace/backends/sqlite_exact.py`, `get()`: when `ids=` is passed with no `where=`, the method
sets `push_page=False` and calls `_rows(cur, where=None)` — a full collection scan (185k rows,
JSON-parsing every row's metadata) filtered afterwards in a Python dict. Warm cost **~835 ms per
call**; chroma does the same op in **1–5 ms**. This is the `get_drawer` path post-cutover, so it is
user-visible on every drawer open the moment the backend flips. `get(where=)` already got a
compiled fast path in `53b9cec` (stat-independent variant); `ids=` never did.

## The job

1. Give `get(ids=[...])` a compiled `WHERE id IN (...)` path with the same design discipline as the
   `53b9cec` `where=` fast path (stat-independence included — study that commit and its tests
   first; match its structure and its paging semantics). Mind SQLite's bound-parameter limits for
   large id lists (chunk if needed) and preserve existing semantics exactly: result ordering,
   include/exclude fields, missing ids silently absent, duplicate ids, empty list, `ids=` combined
   WITH `where=` (whatever the current combined behavior is — preserve it, do not redesign it).
2. **Correctness bar (same as the where= fix):** prove equivalence against the old Python path
   across a real spread of id sets — singletons, batches, missing ids, duplicate ids, empty list,
   ids spanning both collections (drawers + closets), ids= combined with where=. Zero divergence
   or it does not ship. Add these as pytest cases, not just a script.
3. **Micro-bench, before and after:** `benchmarks/workload_sim/get_micro.py` exists on branch
   `bench/workload-sim` @ `35bb0e9` — cherry-pick or run it from that branch's worktree
   (`~/.local/state/herdr-spawn/workload-sim-260809-213800/worktree`). Store to bench against:
   build/copy your own (e.g. from `/Volumes/xData/.mempalace-cutover/tn-gate-260810/palace-exact`
   — COPY the sqlite file, never open the original: connect-time DDL means opening is writing).
   Report warm p50/p95 for `get(ids=)` before vs after, and confirm `get(where=)` and search paths
   are unchanged. Single-process micro-bench; note loadavg alongside.
4. **Full test suite** must pass: `uv run pytest tests/ -v --ignore=tests/benchmarks` — baseline at
   `53b9cec` is 4,085 passed / 31 skipped. `uv run ruff check .` and `ruff format --check .` clean.
5. Commit on your own branch (`fix/get-ids-fastpath` or similar) with a clean message. **Do NOT
   merge to `xdev-patches`. Do NOT flip any production config.** Chris decides in the morning; your
   deliverable is a branch he can merge with one action.

## HARD RULES (same as every worker tonight)

1. **NEVER write to the live palace** at `~/.mempalace/palace`. Read-only. Cutover/gate artifacts
   are read-only too — copy before opening (connect-time DDL!).
2. **No palace writes** — no `add-drawer`/`update-drawer`/`kg-add`. Propose in `preclean-proposal.md`.
3. Do not touch `~/.mempalace/tools`.
4. Another worker (tn-paraphrase) is building an eval set concurrently — low CPU, ignore it. Your
   test-suite and micro-bench runs are fine; just record loadavg with the bench numbers.

## Deliver

`result.md` in your spawn dir: fix summary (what changed, why it is stat-independent), before/after
micro-bench table, the equivalence-proof coverage list, test-suite + lint status, branch + commit
sha, and anything you could not verify. Plus `preclean-proposal.md` (propose, file nothing).
