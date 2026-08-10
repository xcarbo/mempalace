# Task: you are the ORCHESTRATOR for tonight's MemPalace performance programme

You are a long-lived driver agent in a herdr pane. **You are not doing the measurement work
yourself.** You spawn each worker, wait for it, harvest it, read its result, decide whether the next
job still makes sense, and move on. You run until the programme is done or you hit a real blocker.

**Why you exist:** on 2026-08-09 the controller session staged two agents with "wait for RELEASE"
and then the human went to bed. The controller session only runs when the human types, so the
release never came and ~6 hours were lost. You remove that dependency — you are awake for as long
as this pane lives, so nothing in this programme waits on a human.

## HARD RULES

1. **NEVER write to the live palace at `~/.mempalace/palace`** (symlink → `/Volumes/xData/.mempalace`).
   ~178k rows, Chris's real memory, and it is serving him. Read it, snapshot it, never mutate it.
   Enforce this on your workers too — if a worker's result says it wrote to the live palace, say so
   loudly in your summary.
2. **DO NOT flip production config.** Nothing in `~/.mempalace/config.json` changes tonight. The
   programme ends at a gate verdict. Flipping the live backend is a decision for a human who is awake.
3. **No palace writes at all** — no `add-drawer`, `update-drawer`, `kg-add`, by you or your workers.
   Collect their `preclean-proposal.md` files for the controller to file later.
4. Do not touch `~/.mempalace/tools`.
5. Do not merge anything to `xdev-patches`. Workers commit to their own branches; the human merges.

## The machine, and when to run what

Chris is using the Mini right now and needs it. **Timing-sensitive work must wait for a quiet
machine.** Do not ask anyone — detect it:

- Poll `uptime` / `ps` every few minutes. Treat "quiet" as **1-minute load average below ~2.0 with
  no non-your Python or node process above ~50% CPU for three consecutive checks**.
- While the machine is busy: do the work that does not need quiet — read prior results, sanity-check
  the briefs, verify inputs exist, pre-stage stores. **Do not burn the whole evening polling; do the
  prep first, then wait.**
- Once quiet, run the jobs **strictly one at a time**. Three agents benchmarking simultaneously on
  2026-08-09 corrupted each other's timings and every contended number had to be retaken.
- If Chris starts working again mid-job, let the current worker finish, then re-check before the next.
  Record the loadavg you observed for each job so numbers stay auditable.

## The programme, in order

Briefs live in `/Volumes/xData/codeXD/mempalace/scripts/tonight-briefs/`.

| # | worker | brief | why it is in this position |
|---|---|---|---|
| 1 | `tn-workload` | `workload.md` | Finishes the ladder cut short this morning: N=20, **the `:4109`-vs-CLI comparison**, and the nomic-768 footprint ladder. The `:4109` comparison is the highest-value single measurement in the programme — if routing reads through the resident service removes the ~1 s per-call boot, it changes the architecture answer and may make nomic-768 affordable. Its harness is already committed on branch `bench/workload-sim` and palace copies are kept at `/Volumes/xData/workload-sim-260809/palace-*`, so it should resume in minutes, not rebuild. |
| 2 | `tn-embed`    | `embed.md`    | The embedder matrix: minilm@384 / nomic@384 / nomic@768 / bge-small@384, end-to-end. ~2.5 h, mostly a CPU-saturating embed pass. Chris explicitly wants the exchange rate measured, not assumed — he may trade latency for accuracy. A run sheet from the staged-but-never-run agent is at `~/.local/state/herdr-spawn/embed-matrix-260809-211241/result.md`; read it first, it saves rediscovery. |
| 3 | `tn-gate`     | `gate.md`     | Rebuild the exact store from a fresh snapshot **with both collections**, run Phase-2 parity, then the rewritten Phase-3 gate. Ends in **GATE: PASS** or **GATE: FAIL** plus the exact commands a human would run to flip. |

Spawn each with:

```
herdr-spawn spawn --name <slug> --label <slug> --cwd /Volumes/xData/codeXD/mempalace \
  --worktree --task-file <brief> -- --model fable
```

Then `herdr-spawn wait <id> --timeout <generous ms>` and act on it.

**Critical gotcha, learned the hard way:** `herdr-spawn wait` returns **2 ("stopped without writing a
result")** as a FALSE POSITIVE for agents that pace themselves with a monitor — it happened to three
agents on 2026-08-09. **`result.md` existing is the real signal.** On rc=2, check for
`~/.local/state/herdr-spawn/<id>/result.md`; if it is absent, look at the pane
(`herdr pane read <paneId>`) before concluding anything, and re-park a fresh wait if the agent is
plainly still working. Only harvest once `result.md` exists — **harvest closes the pane**, so check
for `preclean-proposal.md` too before you do.

## Judgement you are expected to exercise

You are not a shell script. Between jobs, read the worker's `result.md` and decide:

- **Does the next job still make sense?** If job 1 shows the `:4109` route collapses the per-call
  cost, that is a major finding — note it prominently, and consider whether job 2's memory framing
  changes (nomic-768 may become affordable, which is exactly Chris's open question).
- **Did a worker contradict a prior result?** Six measurements on this project pointed the wrong way
  in 48 hours. Contradiction is expected and valuable — surface it, do not smooth it over.
- **Is a worker's number implausible?** Prefer sending it back for a re-measure (`herdr agent prompt`)
  over passing a suspicious number up.
- **Running short on time or space?** Jobs are in priority order. Finishing 1 and 3 properly beats
  starting all three badly. Check free space on `/Volumes/xData` before job 2 (needs a few GB).

## Deliverable

Write `result.md` in **your own** spawn directory containing:

1. **A one-screen summary a tired human can read first thing** — what ran, what it found, what
   changed about the decision, and the single most important number.
2. The **GATE: PASS/FAIL** verdict from job 3, verbatim, with the flip commands if it passed.
3. **The answer to Chris's actual question**, as far as tonight got it: given 3–5 projects × 1–5
   agents on a 24 GB Mini, do these latency and memory characteristics matter, and is nomic-768
   affordable — at 384, at 768, or only with a resident service?
4. Per-job: worker id, branch/commit, loadavg observed, and where its artifacts live.
5. **What did not run, and why** — plainly. A partial programme honestly reported is a good outcome.
6. Any worker that broke a hard rule.

Also write `preclean-proposal.md` alongside, consolidating your workers' proposals so the controller
can file them in one pass.

## Standing context

- The live palace is on **chroma** and healthy; it has not been migrated. The current embedder is
  **all-MiniLM-L6-v2, 384-d** — nothing has been re-embedded. nomic and bge-small are merged but
  switched off.
- Merged today: the stat-independent `get()` fast path (`53b9cec`). On a pristine store the exact
  backend measures p50 57.3 / p95 114.1 / p99 158.9 / max 197.8 ms vs chroma 95.9 / 262.3 / 520.6,
  with bit-identical recall and chroma's exact 2-case golden failure set.
- Known this morning from the partial ladder: **no memory knee through N=15** (11 GB, zero swap,
  nothing died); the binding cost is the **~0.8–1.3 s fresh-process boot**, which blows the 500 ms
  hook budget even at N=1.
- An unexplained lead worth confirming if a worker can: `get-drawer` measured ~0.85–1.0 s warm at
  every N, near-constant. Hypothesis: `get(ids=[…])` with no `where` still walks the full collection.
  Same path the get_drawer tool uses post-cutover, so if real it is user-visible.
- The cron fleet is live; overnight jobs (00:40 backup onward) will run. Do not pause the fleet. If
  you need silence for a timed cell you may pause **only** the two 15-minute pollers
  (`mempalace-orphan-reaper`, `palace-disk-guard`) and **must restore them afterwards** — back up the
  crontab first and restore it even if you abort.
