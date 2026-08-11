# Task: does the embedder verdict survive queries that DON'T share keywords? Build the paraphrase set and re-run the matrix.

You are working in an isolated git worktree of `xcarbo/mempalace` (fork branch `xdev-patches`).
Wing for any palace reference: **mempalace**. It is ~00:xx GST; Chris is asleep; you have a quiet
machine. **Deadline 04:30, degrade-rather-than-overrun** (a smaller honest result beats a bigger
late one). Resolves open follow-up `3089c388` (paraphrase eval set).

## Why you exist

Tonight's embedder matrix (tn-embed) measured, end-to-end through `search_memories` on 177,946
drawers: minilm@384 wins the golden gate 35/37 vs 29/37 for nomic@384/@768, and nomic's promised
+16 pp R@10 collapsed to +1.0–1.4 pp. Recommendation: do not migrate. **Its own top caveat:** both
eval sets are keyword-biased (goldens = agent keyword queries; the 300-set = TF-IDF terms drawn
from the targets' own text). That is exactly the regime where BM25+minilm looks best and where a
semantic embedder's advantage is least visible. Your job: build a paraphrase eval set with minimal
lexical overlap with the targets, re-run the SAME four configs through the SAME harness, and say
whether the verdict stands.

**Your result.md must open with one line: `PARAPHRASE VERDICT: CONFIRMS` or
`PARAPHRASE VERDICT: OVERTURNS` (or `PARTIALLY OVERTURNS`), then the evidence.**

## HARD RULES (same as every worker tonight)

1. **NEVER write to the live palace** at `~/.mempalace/palace`. Read-only everywhere.
2. **No palace writes** — no `add-drawer`/`update-drawer`/`kg-add`. Propose in `preclean-proposal.md`.
3. Do not touch `~/.mempalace/tools`. Commit only to your own branch.
4. Do NOT delete or modify tn-embed's artifacts — you are REUSING them.
5. Cron fleet is live (00:40 backup etc.); quiet-gate your timed cells if you take latency numbers
   (recall is the point tonight, latency is not — do not burn time on latency rigor).

## Everything is already staged — REUSE, do not rebuild

- **Four stores, built + ANALYZEd + identity-set**, at
  `/Volumes/xData/.mempalace-bench-embed-matrix/store-{A-minilm384,B-nomic384,C-nomic768,D-bge384}`.
- **Harness**: same dir — `eval_run.py`, `goldens_filtered.py`, `aggregate_results.py`,
  `run_phase4_gated.py` (tn-embed's additions; mirror their protocol). `corpus.pkl` has ids, docs,
  metas for both collections — this is your source of drawer TEXT for paraphrase generation.
- **Backend/env**: branch `bench/embed-matrix` (= `9c66f71` + `f4f41fd`) in tn-embed's worktree at
  `~/.local/state/herdr-spawn/tn-embed-260810-105657/worktree` with a working 3.14.6 venv. Reuse
  that venv (read-only) or clone the branch into your own worktree. Config C needs
  `MEMPALACE_EMBEDDING_MODEL=nomic` + `MEMPALACE_NOMIC_DIM=768` in the process env (loud error if
  forgotten). Device pinned cpu. Embed threads: irrelevant (no corpus embedding tonight — stores
  exist; only ~few-hundred query embeds).
- **Existing sets** (for targets, NOT for query text): 300-case
  `~/.local/state/herdr-spawn/ma-code-260808-224241/harness/eval_set.jsonl`; goldens
  `~/.mempalace/tools/golden_recall.json` (37 cases; keep each case's wing/room filter + limit).

## Building the paraphrase set — the part that decides whether tonight means anything

1. **Generate from the DRAWER, not from the old query.** For each case: read the target drawer's
   text from `corpus.pkl`, and write the question a human (Chris) or an agent would actually ask
   when they need that drawer, WITHOUT reusing its distinctive vocabulary. Style mix: natural
   questions ("what did we decide about…", "how did we fix…"), vague recollections ("that thing
   where the backup was silently tiny"), task-context asks ("resuming X, what was blocking it").
2. **Generator**: you write them (you are the LLM), or batch via LM Studio `:1234` if you prefer —
   never an external API. Quality beats volume.
3. **Enforce low overlap programmatically, or the set is worthless**: for every paraphrase, compute
   content-word overlap (non-stopword token Jaccard, and shared-distinctive-term count) against the
   target doc. Reject/rewrite until overlap is under a stated threshold (e.g. Jaccard < 0.15 and
   ≤2 shared distinctive terms). Report the overlap distribution of the final set vs the original
   300-set's (which will be high — that contrast is part of the finding).
4. **Prove the set defeats keyword search**: run a BM25/lexical-only pass (or vector-weight-zero
   config if the pipeline exposes it; else compare against the unfiltered-BM25 scores the backend
   can emit). If BM25-only recall on your set is not SUBSTANTIALLY below its recall on the original
   set, your paraphrases still leak keywords — say so and fix or caveat.
5. **Size and provenance**: target ~150–200 paraphrase cases from the 300-set (deterministic
   sample, seed recorded, spread across wings) + all 37 goldens paraphrased (keep filters/limits —
   run them through the `goldens_filtered.py` semantics). Every case records: target id(s), the
   paraphrase, overlap stats, and which original case it derives from. Commit the set + generator
   script to your branch — the set outlives tonight.
6. **Contamination guards**: never show the generator the original query. Never tune a paraphrase
   against search results (no peeking-then-rewording until it hits — that is target leakage; one
   rewrite loop on overlap stats only).

## Measure

- All four configs A/B/C/D through the identical `search_memories` path, same protocol as tn-embed
  (fresh process per run, sequential). Recall was bit-identical across runs tonight, so **2 runs**
  with an assert-identical check suffice (fall back to 1 run + spot-check if time-squeezed; say so).
- Report per config: R@1/R@5/R@10/MRR on the paraphrase-300 subset; paraphrased-goldens X/37
  per-case with ranks; and the DELTA table — original set vs paraphrase set, side by side.
- If the ranking of configs flips (nomic beats minilm on paraphrases), quantify by how much and on
  which case types; if minilm still wins or ties, that is the finding, report it with equal force.

## Degrade ladder (in order, if 04:30 approaches)

1. Drop config D (bge). 2. Drop the second run (keep assert on goldens only). 3. Shrink the
paraphrase-300 subset to the first 100 sampled cases (never shrink the 37 goldens). 4. If even
that won't land: paraphrased goldens only, all four configs — still decision-relevant.

## Deliver

`result.md` in your spawn dir: the VERDICT line first, delta tables, overlap-distribution proof,
BM25-defeat proof, per-case paraphrased goldens, what was degraded, what you could not verify.
Plus `preclean-proposal.md` (propose drawers + the follow-up `3089c388` resolution; file nothing).
Do not clean up tn-embed's artifacts; leave your own eval outputs on disk beside them.
