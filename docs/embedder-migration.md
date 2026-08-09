# Migration runbook: swapping the embedding model on a live palace

**Status: NOT EXECUTED. As of 2026-08-09 the measured winner-if-migrating is
`nomic` (nomic-embed-text-v1.5 at the 384-d Matryoshka truncation — a large
vector-leg win on the 300-case eval set, no schema change, implemented and
ready), but the live migration is deliberately NOT scheduled until the
sandbox rehearsal below has been run through the production search path.
See "What the 2026-08-09 A/B measured" at the bottom for the evidence and
its limits.**

Why a runbook exists anyway: the embedder is recorded in
`~/.mempalace/palace/mempalace_embedder.json` and persisted per-collection
by ChromaDB; vectors from two models never mix. Changing models is therefore
a full re-embed of every row (~177k at time of writing) plus an index
rebuild, gated by golden-recall. It is the single most expensive maintenance
operation the palace has.

Throughout, `$PALACE` = `~/.mempalace/palace` (resolve the symlink with
`pwd -P` before any tar/cp — the palace lives on `/Volumes/xData` and a tar
of the symlink BY NAME archives the link, not the data).

## Decision gates (all four before any re-embed)

1. **Dimension decision — call it out explicitly.** A 384-d model
   (minilm-shaped: bge-small, MRL-truncated gemma/nomic) is NOT a schema
   change: vectors drop into the existing collection width, and
   `sqlite_exact`/`pgvector` blob widths stay valid. A 768-d model (nomic
   full, gemma full) IS a schema change: `mempalace_embedder.json`
   dimension changes, every explicit-vector backend's stored width changes,
   and any code that hardcodes 384 (HNSW extraction tooling,
   `docs/sqlite-exact-cutover.md` step 2) must be revisited. Do not start a
   768-d migration casually.
2. **Measured recall gate.** The candidate must beat minilm on the FULL
   37-case golden set AND the 300-case eval set
   (`.reports/mempalace-audit/embedder-ab/` has the rig: `build_pools.py` →
   `embedder_ab.py`, `build_pools_300.py` → `eval_300.py`). Pool re-ranks
   are upper bounds for the candidate — treat a small win as a tie. One
   golden case is not evidence (the nomic rank-91→7 case from the
   2026-08-08 audit inverted to a *loss* by 2026-08-09 because the target
   drawer's content changed).
3. **In-process gate.** The model must run as in-process ONNX (hf_hub
   download, cached, CPU by default) following the `EmbeddinggemmaONNX`
   pattern in `mempalace/embedding.py`: stable `name()`, lazy load under a
   lock, `MempalaceConfig.embedding_threads` honoured, bounded sub-batches,
   graceful ImportError. No HTTP endpoint, no LM Studio, no external API —
   an embedder behind a service the user has to remember to start violates
   the always-on contract (and local-first if the endpoint is remote).
4. **Asymmetric-prefix check.** If the model needs different query/document
   prefixes (nomic, bge, gemma retrieval prompts): chromadb ≥1.5.9 calls
   `embed_query()` for queries and `__call__` for documents; implement both
   on the EF (`BgeSmallONNX`/`NomicEmbedONNX` are the template, and
   `backends/embedding_wrapper.py::_embed_texts(is_query=)` routes the
   explicit-vector backends the same way since this branch). Caveat that
   remains open: `dedup.py` compares documents via `query_texts`, so under
   an asymmetric model its similarity scores shift — re-check dedup
   thresholds before trusting near-dup verdicts on a migrated palace.

## Phase 1 — Snapshot (rollback anchor)

1. Quiesce writers for the snapshot instant: no active `mine`, cron fleet
   outside its 00:40–05:58 window, Stop-hook saves idle. Readers can stay.
2. `cp -R "$(cd $PALACE && pwd -P)" /Volumes/xData/.mempalace-snapshots/pre-embedder-$(date +%Y%m%d)`
   — a ~1.5 GB copy takes seconds on direct HFS+ (measured 2026-08-09).
   Verify the copy: `chroma.sqlite3` present, size within 1% of source,
   `pragma integrity_check` = ok on the copy.

## Phase 2 — Re-embed + rebuild

The supported path is config flip + full index rebuild:

3. Set the model: `"embedding_model": "<name>"` in `~/.mempalace/config.json`
   (or `MEMPALACE_EMBEDDING_MODEL` for a dry run in a sandbox first —
   `/memp-worktree seed` gives a sealed full-copy sandbox; rehearse there).
4. Run `mempalace repair rebuild-index`. This extracts all drawers,
   recreates the collection (the new EF `name()` is persisted on it), and
   re-embeds every row through the new EF.
5. **Crash expectation (chroma backend):** chromadb's Rust bindings SIGBUS
   (exit 138) under sustained read+write on a palace this size — observed
   killing a batch pass at 91/1,886 on 2026-08-09; integrity_check returned
   ok every time. `rebuild_index` is NOT resumable: a crash mid-rebuild
   leaves a partial collection, and a re-run starts over. So either
   supervise-and-retry from scratch (acceptable at ≤1 h per attempt), or —
   strongly preferred — **sequence this migration AFTER the sqlite_exact
   cutover** (`docs/sqlite-exact-cutover.md`): on `sqlite_exact` a re-embed
   is a resumable in-place vector UPDATE with a state file (the
   `reembed_headers.py` supervisor pattern from
   `.reports/mempalace-audit/header-chunk-ab/run_reembed.sh`), with no
   chroma crash class at all. That runbook's own transition caveat says the
   same thing from the other side: cut over on existing vectors first,
   re-embed second.
6. Wall-clock budget: at the 2026-08-09 measured in-process CPU rates
   (production thread cap of 5, quiet M4 24 GB — minilm ~153 docs/s,
   bge-small ~90/s, nomic ~22-31/s, gemma ~17/s) a 177k-row re-embed is
   ~20 min (minilm), ~33 min (bge-small), ~1.6-2.2 h (nomic), ~2.9 h
   (gemma). Add rebuild extract/upsert overhead on top, and expect worse
   under concurrent load.

## Phase 3 — Identity + verification (gate: golden-recall)

7. Confirm `$PALACE/mempalace_embedder.json` now records the new
   `model_name` and its true dimension (probe-derived; 384 or 768). If it
   still says the old model, stop — the identity write failed and every
   later writer would mix vector spaces.
8. Row count: collection count == pre-migration count (compare against the
   *collection* count, not the raw `embeddings` table, which over-counts
   tombstones).
9. **Golden-recall gate:** `python3 ~/.mempalace/tools/golden_recall.py --json`
   — pass criteria: zero NEW regressions vs the pre-migration run (save one
   the same morning as the baseline), and the case-level ranks not
   collectively worse. Exit 20 (known failures only) is acceptable; exit 1
   is a rollback trigger.
10. Smoke: `memp search` unfiltered, one wing-filtered search, `memp
    get-drawer` on a known id, one `add-drawer` + search-back on a scratch
    wing, then delete the scratch drawer. The add-drawer proves the write
    path embeds with the new model (its vector must be the new dimension).

## Phase 4 — Rollback (keep the snapshot ≥ 2 weeks)

11. Rollback = stop writers, `rm -rf` the migrated palace dir, `cp -R` the
    snapshot back into place, restart `:4109`/hooks. Verified pattern: the
    controller did exactly this on 2026-08-09 (header re-embed rollback)
    and golden-recall came back rank-for-rank identical to baseline.
12. Drawers written AFTER the migration exist only in the migrated palace —
    export them before rolling back (`filed_at > <migration timestamp>`)
    and re-add after, or accept the gap.
13. Do not delete the snapshot until two full weeks of cron windows and one
    laptop SMB session have passed clean.

## Resource envelope (2026-08-09, measured through the production factory)

The in-process embedder lives inside every memp CLI call, every hook fire,
and the `:4109` service — its resident footprint is paid ~70×/day. Per
candidate, probed in a fresh process on a quiet machine, thread cap 5
(the production default), 200-doc sustained sweep:

| model | RSS idle | RSS doc-sweep peak | cold ready | query p50 | docs/s |
|---|---|---|---|---|---|
| minilm | 273 MB | 851 MB | 0.36 s | 6 ms | 153 |
| bge-small | 304 MB | 798 MB | 0.75 s | 2 ms | 90 |
| nomic (len 1024, batch 4) | 777 MB | 2.6 GB | 0.91 s | 5 ms | 22-31 |
| embeddinggemma | 1.6 GB | 3.6 GB | 1.44 s | 37 ms | 17 |

- Every class honours `MempalaceConfig.embedding_threads` (measured ~500%
  CPU at cap 5; uncapped control higher). Caveat: the config accepts any
  positive integer — a bad value (200) built a 200-thread ORT pool and
  drove system load past 150. There is no upper clamp yet.
- nomic at its first profile (len 2048, batch 8) peaked at **7.3 GB RSS**:
  ORT's CPU arena never shrinks and attention buffers scale with
  batch × len². The shipped profile (len 1024, batch 4) cut that to 2.6 GB
  with recall unchanged (only 0.5% of live chunks exceed 1024 tokens).
  Doc-sweep peaks apply to re-embed/repair workloads; query-only processes
  (hooks, searches) stay near the idle figure.
- Warm query latency passes the <500 ms hook budget for every candidate.
  Cold-ready (load + first embed, paid once per process) is 0.4-1.4 s — an
  embedder-touching CLI/hook call pays this today too (minilm 0.36 s).
- Resident vector matrix at corpus scale (interacts with the sqlite_exact
  matmul path): 176,658 × 384 × 4 B ≈ **271 MB**; at 768-d ≈ **543 MB**.
  On a 24 GB shared box this is a real, permanent cost of going 768-d —
  the 384-d Matryoshka option avoids it for ~2 pp of R@10.

## Sandbox rehearsal (the gate between "implemented" and "scheduled")

The full runbook must be rehearsed once in a sealed sandbox before it runs
against the live palace. `/memp-worktree seed` builds a full-copy sandbox
(`./mp` wrapper; the 2026-08-09 spawn left one at
`/Volumes/xData/.mempalace-sandbox/feature-embedder-modernize`). In the
sandbox: set the model, run `./mp repair rebuild-index` (expect hours; the
SIGBUS-restart behaviour is itself part of what the rehearsal measures),
then run golden-recall with `--palace` pointed at the sandbox AND the
`eval_run.py` 300-case harness through production `search_memories`. Only a
sandbox pass on both promotes the migration to schedulable.

## What the 2026-08-09 A/B measured (for the next reader)

Pool-rerank A/B, 37 goldens + 293 usable TF-IDF eval cases, all in-process
ONNX CPU on the M4 (rig: this directory's `.reports/mempalace-audit/
embedder-ab/`; every production class verified vector-identical to the rig).

On the 293-case set (the statistically meaningful one), unfiltered
drawer-level R@k vs minilm's exact ranks, candidates as pool re-ranks
(upper bounds):

| model | R@1 | R@5 | R@10 | MRR@10 | est. true R@10* |
|---|---|---|---|---|---|
| minilm (exact, baseline) | .355 | .631 | .696 | .474 | .703 (validates: exact .696) |
| bge-small@384 | .423 | .730 | .816 | .561 | .782 |
| nomic@384 (MRL, shipped len-1024 profile) | .515 | .775 | .857 | .633 | .833 (at len 2048) |
| nomic@768 | .532 | .799 | .870 | .653 | .853 |
| nomic int8@384 | .529 | .812 | .846 | .645 | — |

\* 5,000 random non-pool chunks embedded as background distractors and the
outrank count scaled by the sampling fraction — converts the pool-rerank
upper bound into an estimate, validated against minilm's known exact ranks
(R@10 within 0.7 pp; R@1 runs ~7 pp optimistic from tie handling, so
compare deltas, not absolutes).

nomic is a large vector-leg win that survives de-biasing, its 384-d MRL
truncation keeps nearly all of it (no schema change), int8 quantization is
recall-neutral, and the memory-tamed len-1024 profile costs no recall.
`embeddinggemma` (both prompt styles) was the slowest model measured (~8
docs/s) and its shipped 384-d config scored *below* minilm on the goldens —
the multilingual onboarding default costs measurable English recall.

Why this did not trigger an immediate live migration:

- The 37-case golden gate is insensitive at this effect size: minilm 19/37
  on the vector leg, candidates 17-21 — within the ±2-case jitter measured
  between two builds (fp32 vs int8) of the *same* nomic model. The flagship
  rank-91→7 nomic case from the 2026-08-08 audit had already inverted by
  2026-08-09 (minilm rank 7, nomic 10) because the target drawer's content
  changed — single cases do not generalize.
- Of production golden-recall's five failures, three are drawers missing
  from the index entirely (no embedder can fix), one is curation debt, and
  the one live regression (`arc-hosting-residency`) got *worse* under every
  candidate (minilm vector rank 15 → 66-73). Better embeddings fix zero
  currently-failing goldens.
- The production pipeline is hybrid (BM25 + vector + blend + rerank), which
  cushions vector-leg gains end-to-end; both eval sets are keyword-biased,
  so the vector leg's paraphrase advantage is under-measured in both
  directions. Hence the sandbox rehearsal gate above.
