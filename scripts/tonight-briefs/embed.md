# Task: the embedder × width matrix, end-to-end, apples-to-apples — so Chris can choose

You are working in an isolated git worktree of `xcarbo/mempalace` (fork branch `xdev-patches`).
Wing for any palace reference: **mempalace**.

**Do not recommend a winner and stop. Produce the trade-off table.** Chris has said explicitly he
may accept worse latency for better accuracy — *it depends* — and that no one should assume for him.
Your job is to make the choice decidable: same backend, same eval sets, same machine, **varying only
the embedder and the vector width**.

## HARD RULES

1. **NEVER write to the live palace at `~/.mempalace/palace`** (symlink → `/Volumes/xData/.mempalace`).
   177k rows, the user's real memory. Read it, copy it, never mutate it.
2. `/Volumes/xData/.mempalace-cutover/palace-exact-v2` is a cutover artifact — **read-only**.
   Build your own stores elsewhere under `/Volumes/xData` (~1.3 TB free). Clean up when done.
3. **No palace writes** — no `add-drawer`, `update-drawer`, `kg-add`. The controller files everything.
4. Do not touch `~/.mempalace/tools` (uncommitted changes that are not yours).
5. Commit to your own branch; do not merge to `xdev-patches`.
6. **Bound your resource use.** 24 GB M4 that Chris is actively working on, with **two other agents
   benchmarking concurrently**. Cap ONNX/numpy threads (`MEMPALACE_EMBEDDING_THREADS`, and note that
   config has no upper clamp — a bad value built a 200-thread pool and drove load to 150 today).
   Do the heavy embed pass thread-capped, then run latency measurements as a separate quiet phase,
   repeated, reporting medians **and the machine load at the time**. Never report a single-shot timing.

## Why this matters / where it sits

MemPalace is moving from chromadb to the in-repo `sqlite_exact` backend (chroma's HNSW loses the
true top-1 out of top-10 on 5.1% of queries, and it SIGBUSes under load — three crashes today). That
cutover is nearly done. **Separately**, the 2021 embedder `all-MiniLM-L6-v2` (384-d) is the measured
ceiling on recall. Nothing has been re-embedded yet — the live palace is still MiniLM.

An earlier agent measured candidate embedders by **pool re-ranking** (re-embed a few thousand
candidates, re-rank) and reported, on a 293-case set, distractor-debiased:

| model | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|
| minilm@384 (live) | .355 | .631 | .696 | .474 |
| bge-small@384 | .423 | .730 | .816 | .561 |
| nomic@384 (MRL-truncated) | .515 | .775 | .857 | .633 |
| nomic@768 (native) | .532 | .799 | .870 | .653 |

**Those are pool re-ranks — upper bounds, not end-to-end.** They were never measured through the
real hybrid pipeline on a fully re-embedded corpus. That is the gap you are closing.

## The matrix to measure

Four configurations, **identical in every respect except the vectors**:

| config | model | width | notes |
|---|---|---|---|
| A | minilm | 384 | the live baseline; a store already exists |
| B | nomic-embed-text-v1.5 | 384 | MRL truncation of the same vectors as C |
| C | nomic-embed-text-v1.5 | **768** | native width — the one Chris wants measured |
| D | bge-small | 384 | cheap insurance; include if time allows, drop it before compromising A–C |

**Key efficiency: B is free.** nomic is Matryoshka — embed the corpus ONCE at 768, then slice the
first 384 dims and re-normalise for B. Do not run two embed passes. (Verify the slice-then-normalise
matches what `NomicEmbedONNX` produces at its 384 setting — if it does not, say so; that is a bug.)

Everything else held constant:
- **Backend: `sqlite_exact` on branch `perf/hydration-get-full-scan` (commit `9c66f71`)** — the
  latency fix, which is NOT yet merged to `xdev-patches`. Use it; without it, latency numbers are
  meaningless (p95 was 10.9 s before the fix, 110 ms after). Merge/rebase it into your branch.
- Same corpus: build from `/Volumes/xData/.mempalace-cutover/palace-chroma` (177,946 drawers +
  2,585 closets). **Build every collection** — omitting `mempalace_closets` silently changes both
  recall and latency and cost a whole cycle today. `build_all.py` in that directory is the template.
- Same eval sets, same protocol, same machine, sequential.
- Run `ANALYZE` on every store you build (a freshly bulk-built store has no `sqlite_stat1` and the
  planner picks a full scan without it — this was load-bearing in today's fix).

## What to report, per configuration

1. **Recall, end-to-end through `search_memories`** (not pool re-rank): R@1 / R@5 / R@10 / MRR on
   the 300-case set, AND the 37-case golden set **per case** with its pass/fail set. The golden set
   is the one that caught a regression the 300-case set missed today — report both, never one.
2. **Latency**: p50 / p95 / p99 / max / mean on the 300-case set, plus a wing-filtered probe.
   Repeat runs; report medians of the percentiles and the spread.
3. **Memory**: resident vector matrix at corpus scale (177,946 × width × 4 bytes), process RSS at
   idle and under query load, and the embedder's own idle RSS (it lives in every `memp` call and
   every session hook — measured today: minilm 273 MB, nomic ~777 MB).
4. **Migration cost**: wall-clock for the full re-embed of 177,946 rows at the width, measured not
   extrapolated where you can, and the on-disk store size.
5. **Query-time embed cost**: per-query embed latency for the model, against the project's budgets
   in `CLAUDE.md` (hooks under 500 ms, startup injection under 100 ms).

## The deliverable

A single table Chris can read and decide from, with **recall gain and latency/memory cost side by
side**, plus:

- **The honest exchange rate**: "768 over 384 buys you X points of R@10 and costs Y ms at p95 and
  Z MB resident." That sentence is the point of the whole task.
- Whether 768 changes the *shape* of anything, not just the magnitude — e.g. does the doubled matrix
  push the matmul path past a memory cliff on a 24 GB box shared with everything else Chris runs?
- Any case where 768 is WORSE than 384 (it happens; MRL truncation sometimes denoises).
- What you could not verify.

State a recommendation at the end, clearly marked as your opinion and separate from the data — but
the table is the deliverable and it must stand on its own if Chris disagrees with you.

## Reproduction / inputs

- Chroma snapshot (source of documents + metadata): `/Volumes/xData/.mempalace-cutover/palace-chroma`.
- Build template (all collections): `/Volumes/xData/.mempalace-cutover/build_all.py`.
- A/B harness: `/Volumes/xData/.mempalace-cutover/eval_ab_v2.py` (note: it never passes `wing`, so it
  does not exercise the filtered lexical path — add filtered coverage or say you did not).
- 300-case set: `~/.local/state/herdr-spawn/ma-code-260808-224241/harness/eval_set.jsonl`.
- Golden set: `~/.mempalace/tools/golden_recall.json` (3 cases repointed today; backup alongside).
- Embedder classes: `mempalace/embedding.py` — `NomicEmbedONNX` (`embedding_model: "nomic"`),
  `BgeSmallONNX` (`"bge-small"`), `EmbeddinggemmaONNX`, minilm. All in-process ONNX, hf-cached.
  **Do not use LM Studio** — the production requirement is an always-on in-process embedder.
- Prior embedder work: `~/.local/state/herdr-spawn/embedder-260809-133748/result.md` +
  `/Volumes/xData/codeXD/.reports/mempalace-audit/embedder-ab/`. Migration runbook:
  `docs/embedder-migration.md`.
- Latency-fix diagnosis: `~/.local/state/herdr-spawn/perf-profile-260809-204224/result.md`.
- Production interpreter **3.14.6** (chromadb 1.5.9); 3.13.7 has 1.5.8 + `hnswlib`. Say which you used.
- Note `mempalace_embedder.json` records `{model_name, dimension}` per collection and the backend
  enforces identity — each store needs its own correct identity or reads are refused.

## Output

Write `result.md` in the directory the spawn harness gives you: the matrix table, the exchange-rate
sentence, per-case goldens, memory and migration costs, your separately-marked recommendation, and
everything you could not verify. Plus `preclean-proposal.md` alongside (propose drawers; file nothing).

Five measurements on this project pointed the wrong way today and each was caught only by
re-measuring. If a number surprises you, measure it again before reporting it.
