# Recovery: runaway `link_lists.bin` exhausts the host disk and panics the machine

**Severity: host-fatal.** This failure mode is not confined to the palace — on
the canonical Mini it filled the internal disk, starved macOS of swap, and
caused **three watchdog kernel panics in nine hours**, the last two while the
palace was being actively repaired. Recovered; see the as-executed Recovery
section below, which documents two remediation approaches that **do not work**.
Sibling of
[`index-metadata-recovery.md`](./index-metadata-recovery.md): same integrity
gate, same `<uuid>.corrupt-<timestamp>` quarantine, different corruption.

Incident date: 2026-07-27 (host `xcarbo-dev`, macOS 25F84, Darwin 25.5.0, M4 Pro).

## Symptom

The host becomes unstable before anything in mempalace looks wrong:

- Kernel panics with `watchdog timeout: no checkins from watchdogd in N seconds`.
- Free disk space collapses with no obvious cause; `du` on the palace reports
  hundreds of GB.
- SQLite operations start failing with `disk I/O error`:

  ```
  Error opening palace at ~/.mempalace/palace: InternalError('error returned
  from database: (code: 4618) disk I/O error')
  ```

- On restart, the integrity gate quarantines the segment, renaming it to
  `<uuid>.corrupt-<timestamp>.drift-<timestamp>` — and a fresh empty segment
  appears under the original UUID. **The quarantine does not reclaim the space.**

## Observed evidence

Quarantined segment contents:

```
~/.mempalace/palace/9b682425-….corrupt-20260727-075647.drift-20260727-075647/
  data_level0.bin          285M
  header.bin               100B
  index_metadata.pickle     16M
  length.bin               696K
  link_lists.bin           1.9T   <-- apparent size
```

`link_lists.bin`, via `stat`:

| Field | Value |
|---|---|
| Apparent size | 2,115,422,292,248 B (1.92 TiB) |
| Blocks allocated | 483,033,312 × 512 B = **230.3 GiB actually on disk** |
| Birth | 2026-07-05 18:33:54 |
| Last modified | 2026-07-27 07:56:46 (moment of quarantine) |

**The birth date is when the segment was created, not when the runaway
started.** Do not read it as three weeks of growth — see the timeline below.
The nightly backup log recorded 169 GB free ~23 hours before the first panic,
so all 231 GiB was written inside a single day.

It is a **sparse file**: 1.92 TiB apparent, 230 GiB real. `du` and `df`
disagree with `ls -l` by ~1.7 TiB, which makes this easy to misdiagnose.

Host disk state at the time:

```
APFS container:  494.4 GB capacity,  722.5 MB not allocated  (99.9% used)
/System/Volumes/Data:  460Gi size,  232Mi avail,  100% capacity
vm.swapusage:  total = 0.00M  used = 0.00M  free = 0.00M
```

## Why only `link_lists.bin`

The other files in the segment are **normal sizes** — `data_level0.bin` at
285 MB and `length.bin` at 696 KB are both consistent with the element count.
They are immune because both are addressed by fixed stride: element `i` always
lives at `i * size_data_per_element_` and `i * sizeof(float)`. Only
`link_lists.bin` is variable-length, and only it can drift.

### Root cause (confirmed against `chroma-core/hnswlib@master`)

The writer and the reader trust each other, and neither validates. That closes
an amplification loop.

**1. The writer skips non-dirty elements using in-memory levels**
(`persistDirty`, `hnswlib/hnswalg.h:1098-1116`):

```cpp
this->output_link_lists_.seekp(0, std::ios::beg);
for (size_t i = 0; i < cur_element_count && dirty_elements_iter != end; i++) {
    unsigned int linkListSize = element_levels_[i] > 0
        ? size_links_per_element_ * element_levels_[i] : 0;
    if (i == *dirty_elements_iter) { /* write record */ }
    else this->output_link_lists_.seekp(linkListSize + sizeof(unsigned int), cur);
}
```

This requires that the on-disk record for every element `i` is exactly
`4 + size_links_per_element_ * element_levels_[i]` bytes. Seeking past EOF and
then writing is what creates the holes — hence a sparse file.

**2. The reader reconstructs those levels from the file itself, unvalidated**
(`loadLinkLists`, `hnswalg.h:1364-1365`):

```cpp
element_levels_[i] = linkListSize / size_links_per_element_;
linkLists_[i] = (char *)malloc(linkListSize);
input_link_list.read(linkLists_[i], linkListSize);
```

No bounds check, no `% size_links_per_element_` check, no cap against
`maxlevel_`. A single torn record — a writer killed mid-`persistDirty` — means
`linkListSize` is read out of the middle of a link payload, i.e. neighbour node
IDs reinterpreted as a length. That assigns a garbage level *and* consumes the
wrong number of bytes, desynchronising the stream, so **every element after the
torn one also gets a garbage level**. The next `persistDirty` then seeks by
`size_links_per_element_ × garbage` and the file explodes. Every reload makes
it worse.

The `malloc(linkListSize)` on line 1365 is a second failure channel and helps
explain the load spike (33.85) during the failed reindex: loading a corrupt
index attempts multi-GB allocations, so the host was fighting memory pressure
*and* disk exhaustion at once.

### What the evidence rules out

Measured against the quarantined segment, so future triage does not re-chase
these:

- **`header.bin` was completely healthy** — `cur_element_count=178230`,
  `max_elements_=262144`, `size_data_per_element_=1676`, `maxlevel_=4`,
  `enterpoint_node_=45925`, `maxM_=16`, `mult_=0.3607`. Not one corrupt field.
- **`data_level0.bin` was byte-exact**: 298,713,480 = 178,230 × 1676 precisely.
- **`length.bin` is not levels — it is per-element float32 norms.** Every value
  read `1065353216` / `1065353215`, which are `0x3F800000` / `0x3F7FFFFF`, i.e.
  the bit patterns for `1.0f`. An earlier revision of this document read those
  as corrupted level values; they are healthy unit norms. Levels are not stored
  in any file — they are derived from `linkListSize`, which is the whole
  problem.

## Why it panics the host (macOS)

The kernel panic is a *downstream* consequence of disk exhaustion, and the
panic log does not name mempalace anywhere — it is easy to chase the wrong
suspect.

1. Segment growth drives the APFS container to 99.9% full.
2. macOS cannot allocate or extend swapfiles with no free space
   (`vm.swapusage total = 0.00M`).
3. Under memory pressure the compressor fills, the VM subsystem needs to page
   out, and cannot. It stalls **in the kernel holding locks** — the panic log
   showed cores 0/1/2/3/7 all parked at an identical `PC`/`LR`, i.e. every
   core spinning on the same lock.
4. `watchdogd` is a userspace daemon; it never gets scheduled to check in.
5. At 93 seconds `AppleARMWatchdogTimer` panics the machine.

Two misleading signals in that panic log:

- `AppleARMWatchdogTimer` / `AppleInterruptControllerV3` appear in the
  backtrace. These are the watchdog *reporting* machinery, not the cause.
  With `roots installed: 0` and secure boot on, no third-party kext is involved.
- `Compressor Info: … 7 swapfiles and OK swap space` reads as healthy. That is
  the compressor's own accounting; it does not know the **filesystem** cannot
  extend the swapfile.

The crash also suppresses its own evidence: with the disk full, macOS cannot
write `.panic` files to `/Library/Logs/DiagnosticReports/`, so repeated panics
leave no report behind. Detect hard crashes via `last reboot` — a `reboot`
entry with no matching `shutdown time` is an unclean stop.

## Recovery — as executed 2026-07-27 08:26–08:28 UTC−?

Two obvious approaches **both failed** before one worked. Read this section
before writing any automated remediation; the naive versions do not work.

### ✗ Failure 1 — detection by apparent size silently misses the file

An emergency script guarded with `find ... -size +10G` matched **nothing**,
because the file reported:

```
apparent size (st_size)     : 258 MB           <-- does NOT match -size +10G
allocated blocks (st_blocks): 231 GiB          <-- the real problem
```

BSD `find -size` with a suffix tests `st_size`, so a sparse file slips
straight through it.

**Detection must use allocated blocks** (`du`, or `stat -f %b`), never
`ls`/`find -size`.

*Caveat on the measurement:* the `st_size` reading was 1.92 TiB at 07:56 and
258 MB at 08:26 with an unchanged mtime. No writer in this code path produces
that transition, and by then three directories shared the same UUID prefix, so
the two readings were most likely taken on different paths. The rule above
holds regardless — but do not treat "apparent size actively changes" as an
established property of this failure.

### ✗ Failure 2 — truncation is impossible on a full APFS volume

The corrected, block-based script found the file and tried `: > "$f"`:

```
-> FAILED
bash: .../link_lists.bin: No space left on device
```

APFS is copy-on-write: truncating a file still requires writing metadata,
which requires free space. At 104 MiB free the volume was wedged — unable to
free space because freeing space needed space. Any remediation that assumes
truncate will work on the volume it is trying to rescue is broken by
construction.

Note also that free space was still *falling* while the host was up
(232 MiB → 155 MiB → 104 MiB across successive checks), so remediation is
racing an active writer.

### ✓ What worked — `rm` the single runaway file

`rm` drops the inode rather than performing a COW truncate, and succeeds where
truncation cannot:

```bash
rm -f ~/.mempalace/palace/<uuid>.corrupt-<ts>.drift-<ts>/link_lists.bin
```

Result, immediate:

```
before:  439Gi used,  104Mi avail,  100%   load 33.85
after :  208Gi used,  231Gi avail,   48%   load  6.94 (falling)
quarantine dir: 231G -> 301M
```

**Prefer this to `rm -rf` of the whole quarantine directory.** All 231 GiB was
in `link_lists.bin` alone, so removing just that file reclaims 100% of the
space while preserving the rest of the segment for vector salvage:

```
data_level0.bin        285M   <- the actual vectors, PRESERVED
index_metadata.pickle   16M   <- PRESERVED
length.bin             696K   <- PRESERVED
header.bin             100B   <- PRESERVED
```

### ✗ Failure 3 — `repair --mode from-sqlite` cannot read its own archive

Reclaiming the space is not the end. The live segment is empty, so the index
must be rebuilt — and the documented rebuild fails on this palace:

```
Archiving ~/.mempalace/palace → ~/.mempalace/palace.pre-rebuild-<ts>
ERROR: Upsert failed in collection 'mempalace_drawers' after 0 rows:
       OperationalError('unable to open database file')
```

The failure is in `extract_via_sqlite` (`repair.py:1314`), which opens the
source through `sqlite_read_uri()` — i.e. `?mode=ro`. **A WAL-mode database
cannot be opened read-only unless its `-shm` file already exists**, because
SQLite needs the shared-memory index and a read-only connection may not create
it. This palace runs WAL by design (the `.wal_enabled` marker), and
`--archive-existing` leaves an archive with no `-shm` alongside it. The error
message names the *destination* collection, which sends you looking at the
wrong file entirely.

The recovery hint the tool prints — re-run with `--source <archive>` — fails
for the same reason.

**Workaround** (keeps the archive pristine — work on a copy, never the only
good copy of the palace):

```bash
ARCH=~/.mempalace/palace.pre-rebuild-<ts>
SRC=~/.mempalace/palace.rebuild-src
mkdir -p "$SRC" && cp "$ARCH"/chroma.sqlite3 "$ARCH"/*.json "$SRC/"
sqlite3 "$SRC/chroma.sqlite3" \
  "PRAGMA wal_checkpoint(TRUNCATE); PRAGMA journal_mode=DELETE;"

mv ~/.mempalace/palace ~/.mempalace/palace.partial-failed-<ts>   # if a partial dest exists
mempalace --palace ~/.mempalace/palace \
          repair --mode from-sqlite --source "$SRC" --yes
```

Note `--palace` is a **global** flag and must precede the `repair` subcommand;
putting it after gives `invalid choice` on `repair_action`.

### Then

1. **Confirm swap can re-establish.** `sysctl -n vm.swapusage` will still read
   `total = 0.00M` immediately afterward — that is correct, macOS creates
   swapfiles on demand. The point is that it now *can*. Free space on the data
   volume is the number that matters.
2. **Reindex.** The live segment is empty (`link_lists.bin` at 8K), so semantic
   search returns nothing useful until vectors are rebuilt from
   `chroma.sqlite3`. Follow the reindex path in
   [`index-metadata-recovery.md`](./index-metadata-recovery.md), or salvage
   from the preserved `data_level0.bin` above.
3. **Watch disk usage while reindexing.** The reindex is memory-heavy and was
   what drove this host into the wall in the first place. Do not fire and
   forget it.
4. **Verify** with a known-good query before trusting recall.

## What was actually done (2026-07-27)

Upstream report: **[chroma-core/chroma#7510]**(https://github.com/chroma-core/chroma/issues/7510).
(Filed against `chroma` because issues are disabled on `chroma-core/hnswlib`,
where the code lives.)

| Change | Where |
|---|---|
| `reclaim_runaway_link_lists()` — deletes any `link_lists.bin` over its header-derived ceiling, runs first in the pre-open safety pass, and scans quarantined dirs | `mempalace/backends/chroma.py` |
| `palace_disk_guard.py` — 15-min cron (`12-59/15`): header ceiling, free-space floor on **both** volumes, and a write probe | `mempalace-tools`, `~/.agents/palace-disk-guard.toml` |
| `repair --mode from-sqlite` could not read its own archive (WAL + `mode=ro` + no `-shm`) | `mempalace/repair.py`, `config.py` |
| Palace relocated to `/Volumes/xData/.mempalace`, `~/.mempalace` now a symlink | — |
| `backup.sh` — resolve the symlink, refuse an undersized/near-empty archive | `mempalace-tools` |

### Detection was never the gap — reclamation was

The pre-existing link-to-data ratio gate **did** fire and quarantine the
segment at 07:56. It renamed the directory aside and left all 231 GiB in place.
The host then died twice more. When adding safeguards to this class of failure,
verify they *free* bytes rather than merely relabel them.

### Relocating the palace: two traps

The palace now lives on `/Volumes/xData` so a runaway can no longer take the
boot volume's swap with it. Two things bite immediately:

**macOS TCC is per-binary.** launchd-spawned processes are blocked from
external volumes unless that specific binary holds Full Disk Access, and the
grant is revoked silently across macOS updates. Measured on this host:

| Spawned by | Writes to `/Volumes/xData` |
|---|---|
| `/bin/bash` under a LaunchAgent | **blocked** (silent EPERM) |
| `mempalace-api`'s venv python under a LaunchAgent | allowed |
| cron (`/usr/sbin/cron`) children | allowed |

Test before trusting it, and keep the guard's write probe in place — it turns
a silent revocation into a visible trip within 15 minutes.

**`tar` archives a symlink, not its target.** `backup.sh` ran
`tar -C "$HOME" .mempalace`; once `~/.mempalace` became a symlink that produced
a **362-byte tarball that still exited 0 and logged "backup complete"** — a
total, silent backup loss. Resolve with `pwd -P` and archive from the real
parent. The same trap applies to `find ~/.mempalace` (no trailing slash does
not descend). Any backup should assert a floor on its own output.

## Hardening worth doing

- **Reclaim on quarantine.** The integrity gate renames but never frees. When
  a segment is quarantined for corruption its data is by definition unusable —
  either delete it, or cap what quarantine may retain. Silently parking 230 GB
  of dead bytes on the boot volume is what escalated this to a kernel panic.
- **Sanity-check segment size at write time.** A `link_lists.bin` that is
  orders of magnitude larger than `data_level0.bin` is never legitimate.
  Refuse the write and quarantine instead of letting it run for three weeks.
- **Measure allocated blocks, not apparent size.** Any size check must use
  block usage (`du`, `stat -f %b`), or sparse files defeat it. This was
  confirmed the hard way during recovery — see Failure 1. `find -size`,
  `ls -l`, and mtime are all unreliable indicators for this file.
- **Never rely on truncation to reclaim.** On a full APFS volume `: > file`
  and `truncate` both fail with `ENOSPC` (Failure 2). Remediation must `rm`
  the file. Emergency tooling that truncates will fail exactly when it is
  needed most.
- **Reclaim the largest file, not the directory.** All the space was in
  `link_lists.bin`; removing it alone preserved 301 MB of salvageable segment
  data at no cost in reclaimed space.
- **Free-space floor.** Refuse to write a segment when the target volume is
  below a safety threshold; failing the palace is strictly better than
  panicking the host.
- **Alerting must be sub-hourly, not daily.** A daily disk check would *not*
  have saved this host, and one already existed: the nightly backup writes a
  `health: … disk_free_gb=` line, and it read a healthy 169 GB roughly 23 hours
  before the first panic. At 7-10 GB/h the machine was dead before the next
  sample. Any check on this failure needs a cadence of ~15 minutes.
- **Bound the segment from its own header, not from a threshold.** A legitimate
  `link_lists.bin` cannot exceed
  `cur_element_count * (4 + (maxM*4+4) * (maxlevel+1))`, and every term is read
  from `header.bin` — so the bound is exact rather than tuned. For the segment
  that killed this host the ceiling was 58.5 MiB against 230.3 GiB actual, an
  overshoot of ~4000×. Implemented in `mempalace-tools/palace_disk_guard.py`
  (`--watch`, `--kill`, `--reap`); a 256 MiB absolute floor keeps it from
  tripping on a rebuild whose header briefly lags its link lists.
- **Don't kill a palace writer mid-persist.** `persistDirty` is not atomic and a
  torn `link_lists.bin` is the seed of the whole failure. Reconsider anything
  that terminates a writer abruptly, including the mcp_server stdin-EOF
  self-exit (68a4855).
- **Get the palace off the boot volume.** None of the three kernel panics
  happen if the palace does not share a volume with swap. Moving
  `~/.mempalace` to `/Volumes/xData` turns this class of bug from host-fatal
  into palace-fatal.

## Detection one-liner

```bash
find ~/.mempalace/palace -name 'link_lists.bin' -exec du -h {} \;
```

Anything above a few MB is suspect. **If `du` and `ls` disagree, trust `du`** —
`du` reports allocated blocks, which is what fills the disk.

Or just ask the guard, which computes each segment's exact ceiling from its own
`header.bin` and checks both volumes plus writability:

```bash
python3 ~/code/mini-utils/mempalace-tools/palace_disk_guard.py
```

A caveat on the incident's own numbers: the write-up records a 258 MB apparent
size against 231 GiB of blocks. That is not physically constructible —
`st_blocks` cannot exceed the file length — so treat it as a reading taken
against the wrong path (three directories shared the UUID prefix by then), not
as a property of this failure.

## Incident timeline (host `xcarbo-dev`)

| When | Event |
|---|---|
| 2026-07-05 18:33 | Segment created — `link_lists.bin` birth date, **not** the onset |
| 2026-07-23 00:40 | backup health: 110,879 drawers, **174 GB free** |
| 2026-07-24 00:40 | backup health: 117,786 drawers, **168 GB free** |
| 2026-07-25 00:40 | backup health: 124,393 drawers, **169 GB free** — still healthy |
| 2026-07-26 ~01-23 | Runaway onset. 169 GB consumed in <23 h (~7-10 GB/h) |
| 2026-07-26 23:51 | Hard crash #1 — watchdog panic, no clean shutdown |
| 2026-07-27 07:51 | Hard crash #2 |
| 2026-07-27 07:56:47 | Integrity gate quarantines segment; space **not** reclaimed |
| 2026-07-27 ~08:1x | Reindex attempt drives load up; free space falls 232→104 MiB |
| 2026-07-27 08:24 | Hard crash #3 (hard power cycle) |
| 2026-07-27 08:26 | Truncate attempts fail (Failures 1 and 2) |
| 2026-07-27 08:28 | `rm` of `link_lists.bin` — 231 GiB reclaimed, host stable |

Detect the hard crashes with `last reboot`: a `reboot` entry with no matching
`shutdown time` is an unclean stop. All three above have no matching entry;
every prior reboot back to 2026-06-04 does.
