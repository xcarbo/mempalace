# Cutover runbook: chroma → sqlite_exact

**Status: NOT EXECUTED. This is the plan the controller follows deliberately —
nothing here runs automatically.**

Why: HNSW at the live `ef=100` drops the true top-1 out of its top-10 on 5.1%
of queries (593-query exact-vs-ANN comparison, 2026-08-08 audit); `ef=500`
still loses 1.9% at 4× the query cost. Exact cosine over the full 174k matrix
costs ~4 ms — invisible against the ~82 ms pipeline — and removes the entire
chroma crash class (SIGBUS, `Error finding id`, `link_lists.bin` runaway,
death under 2 concurrent accesses, 27% dead tombstones in the segment).

Prerequisites: the `sqlite_exact` backend with the matmul query path, SQL
filters, FTS5 lexical lane, and the multi-process concurrency test — all
landed on branch `feat/memaudit2-exact-backend`, merged.

Throughout, `$PALACE` = `~/.mempalace/palace` (resolve the symlink with
`pwd -P` before any tar/rsync — the palace lives on `/Volumes/xData` and a
tar of the symlink BY NAME archives the link, not the data).

## Interim zero-code option (only if the cutover is deferred)

Setting `hnsw:search_ef=500` on the live collection cuts real top-1 loss from
~5.1% to ~1.9% for ~+2 ms/query, with zero code. **Caveat that makes it a
quiesced operation, not a quick win:** it is a `collection.modify()` sysdb
**write** — the exact operation class that caused the concurrent-read panics
fixed in `c45c1b0`. If used: stop the `:4109` service and the cron fleet,
apply, restart. Do not apply while readers are live. Skip it entirely if the
cutover happens this week.

## Phase 1 — Build the exact palace (live palace untouched)

1. Quiesce writers only for the snapshot instant: take a consistent copy of
   the live palace (sqlite online backup of `chroma.sqlite3` + `cp -R` of the
   two segment dirs, exactly as the audit spike did). Readers can stay up.
2. Extract vectors from the copy (~2.4 s):
   `hnswlib.Index(space="cosine", dim=384).load_index(<vector segment dir>,
   is_persistent_index=True, max_elements=<slot count>)`, then batched
   `get_items(labels)`; `index_metadata.pickle` in the segment dir carries
   `id_to_label` — no sqlite join needed. Or reuse the verified artifacts at
   `/Volumes/xData/codeXD/.reports/mempalace-audit/spike-artifacts/`
   (`vectors.npy` + `vector_ids.json`) if the palace has not been written
   since 2026-08-08 — verify with a row-count comparison first.
3. Build the exact store into a **new directory** (never inside the chroma
   palace dir — mixed backend artifacts are rejected by design):
   `scratchpad/ab/build_exact.py` from the 2026-08-09 A/B run is the working
   template. It reads docs+metadata from the copied `chroma.sqlite3`
   (`embedding_metadata`, key `chroma:document`), inserts in 2 000-row
   batches with explicit embeddings, and records the embedder identity
   (`minilm`, 384).
   **Build EVERY collection, not just `mempalace_drawers`.** The live palace
   also carries `mempalace_closets` (~2.6k rows); the 2026-08-09 build
   omitted it, and `search_memories` silently degrades to drawer-only search
   when `get_closets_collection` fails — closet rank-boosts (up to 0.40
   effective-distance) vanish without any error. Gate: the built store's
   collection list == the chroma palace's collection list, with per-collection
   row-count parity.
4. Copy `mempalace_embedder.json` and `palace_format.json` from the live
   palace into the new directory.

## Phase 2 — Verify parity (gate: all three must pass)

5. **Row count:** backend `count()` on the new store == number of extracted
   vector ids == chroma `collection.count()` on the copy. (Note: the raw
   `embeddings` sqlite table over-counts — 176,823 rows vs 174,272 active at
   audit time; compare against the *collection* count, not the table.)
6. **Id parity:** the sorted id set of the new store equals
   `vector_ids.json` exactly. Spot-check 20 random ids: document text
   byte-identical to chroma's `chroma:document`, metadata dict equal.
7. **Vector parity:** for 20 random ids, the stored blob decodes to the same
   float32 vector extracted from HNSW (cosine 1.0 within fp32 epsilon).

## Phase 3 — A/B (gate: recall must not regress)

> **Why this gate got stricter (2026-08-09):** the first cutover attempt
> passed the then-current gate — 300-case R@1/5/10/MRR all ≥ chroma — while
> the full 37-golden set regressed 2 → 5 failures and two real click-through
> cases fell out entirely. The 300-case queries are unfiltered TF-IDF
> extractions, so they never exercised the **wing/room-filtered lexical
> path**, which was a different code path with a different ranking function
> (raw FTS5 corpus-IDF bm25, `LIMIT n`) than chroma's (windowed candidates +
> Okapi rescore). Aggregate metrics on one query shape are not a recall gate.

8. Run the A/B harness (`scratchpad/ab/eval_ab.py` shape): both palace
   copies, production `search_memories`, the **full golden set**
   (`golden_recall.py`, all cases) + the 300-case set at
   `~/.local/state/herdr-spawn/ma-code-260808-224241/harness/`.
9. Pass criteria — ALL of:
   * **300-case set:** sqlite_exact R@1/R@5/R@10/MRR ≥ chroma (exact
     retrieval is a superset by construction — a regression means a build
     bug, not a property of exact search).
   * **Full golden set, per-case:** the exact leg's failure set must be a
     subset of the chroma leg's failure set measured on the same snapshot
     pair, same code, same config. No new failing case, no case pushed past
     its `rank_within` bound — aggregate counts are not enough; compare
     case-by-case. The goldens are the only leg that carries real
     click-throughs and the only leg that exercises wing/room-filtered
     searches; a candidate backend that "wins on average" while dropping a
     click-through case is not at parity.
   * **Both lexical code paths exercised:** the run must include filtered
     searches (the goldens do). If the golden set ever loses its filtered
     cases, add explicit wing-filtered probes before trusting this gate.
   * **Latency, absolute and tail-aware.** The first latency criterion here
     ("within ~4× of chroma") failed on 2026-08-09: the A/B's p50 was 109 ms
     while p95 was 10.9 s and the worst query 25 s — a per-hit hydration
     `get(where=...)` full-scan that no median can see. Percentile
     thresholds, all required, measured over the combined 300-case + full
     golden run (which exercises unfiltered, wing/room-filtered, and
     hydration-heavy queries), warm process, closets collection present:
     - p50 ≤ 150 ms, p95 ≤ 500 ms (the CLAUDE.md hook budget), p99 ≤ 1 s,
     - **max ≤ 2 s** — with 337 cases, p99 still hides three queries; the
       2026-08-09 tail lived exactly there,
     - p95 ≤ the chroma leg's p95 measured in the same session on the same
       machine state.
     Also one **cold-process probe** (the `memp` CLI reality): fresh
     interpreter, one query, median of ≥ 5 runs ≤ 1.5× the chroma leg's
     equivalent (interpreter + model load dominate both legs; the backend
     must not add a multi-second rebuild on top).
   * **Hydration-heavy queries included.** The tail scaled with the number
     of closet-boosted hits (worst case: 39 hydration calls in one query).
     The run must include the ≥ 3 queries with the highest closet-boost
     counts from the previous A/B (2026-08-09: `kaimeta-tailscale-ecs`,
     `dup-tie-cjc1295-hair`, `ct-thewill-adgm`); each is individually
     subject to the 2 s max.
   * **Query-plan preflight (catches the ANALYZE / missing-index class).**
     On the freshly built store, before the A/B: trace the SQL that `get()`
     **actually emits** for the two hydration filter shapes
     (`{"source_file": X}` and `{"$and": [{"source_file": X},
     {"parent_drawer_id": Y}]}`) and run `EXPLAIN QUERY PLAN` on those
     traced statements; assert the plan uses a generated-column index — not
     a bare `(collection_id)` index walk. Do not hand-write the probe SQL:
     the stat-less planner's choice is decided by whether the emitted
     statement carries `ORDER BY rowid` (measured 2026-08-09: with ORDER BY
     every filter shape, including bare equality, full-walks on a
     `sqlite_stat1`-free store; without it every shape picks the selective
     index), so a probe that differs from the shipped SQL by only that
     clause proves nothing. A freshly bulk-built store has **no `sqlite_stat1`**; code whose
     plan is correct only after `ANALYZE` will pass every warm benchmark on
     a hand-tuned store and silently regress ~30× on the next fresh build
     (measured 2026-08-09: p95 110 ms with stats vs 3,004 ms without, same
     code, same store). If the shipped code needs stats, `ANALYZE` must be
     a build step (0.16 s on 177k rows) *and* this preflight still gates.
   * **Closet-boost liveness.** Collection-list parity is a Phase-1 gate,
     but assert it end-to-end here too: at least one A/B query must return
     `matched_via = "drawer+closet"`. The 2026-08-09 build shipped without
     `mempalace_closets`; search silently degraded to drawer-only, recall
     numbers looked *better*, and the hydration cost this collection
     triggers was invisible — the latency bug was measured only after the
     rebuild restored closets.
   * **Machine discipline.** Latency gates are void if another benchmark or
     heavy agent ran concurrently (three agents timed each other into
     noise on 2026-08-09). Record `os.getloadavg()` at start and every 50
     cases; rerun any leg whose load exceeded ~2× the idle baseline, and
     never run two legs simultaneously.
10. **If the A/B regresses: stop.** Do not cut over. Diagnose against the
    2026-08-09 baselines (`eval_ab_results.json` in the cutover dir; per-lane
    traces in the 2026-08-09 exact-diag session). A golden-only regression
    with a green 300-case set points at ranking-path divergence (lexical
    lane, closet boosts, fusion pool composition), not at the build.

## Phase 4 — Flip the default

11. Quiesce everything that touches the palace: `launchctl` stop
    `mempalace-api` (:4109), pause the cron fleet (00:40–05:58 window —
    flip outside it), no active `memp` sessions.
12. Final delta sync: re-run Phase 1–2 if the live palace changed since the
    snapshot (drawer adds since the build → re-extract or re-mine the delta;
    id parity gate must pass again).
13. Move the exact store into place as its own palace dir and point config at
    it: set `"backend": "sqlite_exact"` and the palace path in
    `~/.mempalace/config.json`. Do NOT mix artifacts in one dir; keep the
    chroma palace dir intact as the rollback.
14. Restart `:4109` and the hooks. Smoke: `memp search` (unfiltered), a
    wing-filtered search (the class that crashed :4109 — must return, not
    500), `memp get-drawer` on a known id, one `add-drawer` + search-back on
    a scratch wing... then delete the scratch drawer.
15. Watch the first cron fleet window end-to-end (`/agents-status` next
    morning): mining writes, gnome curation reads, no lock errors. WAL +
    `busy_timeout=10000` absorbs writer contention; single-writer discipline
    from `locks.py` still applies at the palace layer.

## Phase 5 — Rollback (keep alive ≥ 2 weeks)

16. Rollback = flip `~/.mempalace/config.json` back to `"backend": "chroma"`
    / the old palace path and restart `:4109`. The chroma palace was never
    modified. **Caveat:** any drawers written *after* the cutover exist only
    in the exact store — before rolling back, export them
    (`updated_at > <cutover timestamp>` in `sqlite_exact.sqlite3`) and
    re-add after rollback, or accept the gap.
17. Do not delete the chroma segment dirs until the exact backend has
    survived two full weeks of cron windows + a laptop SMB session.

## Transition-window caveats

- **One version everywhere.** The vector-cache invalidation protocol
  (`vec_gen` counter in the `meta` table) assumes every writer bumps it on
  non-append mutations. Old-code writers (a stale laptop checkout, an
  un-synced cron env) mutating the store won't bump it and readers may serve
  a stale cache. Merge first, sync both machines, then cut over.
- **SMB from the laptop:** the backend deliberately sets no `mmap_size`, so
  SQLite uses plain reads — safe over SMB. WAL over SMB is still not
  multi-host-safe; the laptop should keep going through `:4109` / `memp`
  against the Mini (as it does today), never open the palace file directly
  over SMB while the Mini has it open.
- The re-embed decision (nomic-v1.5, measured rank 91 → 7 on the failing
  golden) is a **separate migration** — do not fold it into this cutover.
  Sequencing per the audit: cut over on MiniLM vectors (bit-identical A/B),
  then re-embed into the same store as its own verified step.
