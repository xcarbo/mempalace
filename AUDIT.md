# Test-suite audit — mempalace

Date: 2026-07-29
Worktree: `/Users/xdev/.local/state/herdr-spawn/test-audit-260729-112824/worktree`
Branch: `test/audit-coverage-260729` (from `88e4485`)
Interpreter used for every number below: pyenv CPython 3.12.8 — see §0, this matters.

Every claim here is backed by a command and its output. Where I could not verify
something, I say so.

---

## 0. First: the documented test command does not work on a clean checkout

This outranks everything else in the brief, because it invalidates the premise
that "the full suite" is a thing anyone can currently run.

`CLAUDE.md` and the task both give the command as `uv run pytest tests/ …`. On a
fresh worktree that is exactly what I ran. It does not fail — it **dies**:

```
$ env -u MEMPALACE_RERANK_URL -u MEMPALACE_RERANK_MODEL uv run pytest tests/ -q --ignore=tests/benchmarks
Using CPython 3.12.11
Creating virtual environment at: .venv
........................................................................ [  1%]
.............................................Fatal Python error: Bus error

Thread 0x00000001ed179d80 (most recent call first):
  File ".../mempalace/backends/chroma.py", line 1607 in checkpoint_wal
  File ".../mempalace/backends/chroma.py", line 2703 in close
  File ".../tests/test_backends.py", line 313 in test_chroma_lexical_search_ids_roundtrip_through_get
```

Rate, measured over 15 consecutive runs of that one file:

```
SIGBUS in 14/15 runs of tests/test_backends.py
```

### It is not a test bug

Twenty lines, no pytest, production API only
(`scratchpad/repro_sigbus.py`): open a `ChromaBackend`, add a drawer,
`close()`, repeat. It dies on the **second** palace, inside `close()`:

```
iter 0: closing...
iter 0: closed ok
iter 1: closing...
Fatal Python error: Bus error
  File ".../mempalace/backends/chroma.py", line 1607 in checkpoint_wal
  File ".../mempalace/backends/chroma.py", line 2703 in close
```

Narrowed by replacing `checkpoint_wal` with variants and reading the exit code
(138 = SIGBUS):

| what `close()` does to `chroma.sqlite3` after chromadb's client closed | exit |
|---|---|
| nothing (`checkpoint_wal` a no-op)                                     | 0    |
| `sqlite3.connect(...)` and **nothing else**                            | 0    |
| `connect` + `SELECT 1`                                                 | 138  |
| `connect` + `PRAGMA quick_check`                                       | 138  |
| `connect` + `PRAGMA wal_checkpoint(PASSIVE / FULL / RESTART / TRUNCATE)`| 138 |

So it is not the `TRUNCATE` checkpoint specifically. It is **any statement
executed by Python's `sqlite3` against a database chromadb's Rust core just
closed**, from the second palace onward in a process. `mempalace/backends/chroma.py:1607`
is simply the first place the codebase does that — and `checkpoint_wal()` runs on
every `close()` **and** every `release()`.

### Why the user's machine says 3,575 passed

The interpreter. Same `uv.lock`, same chromadb 1.5.7, same code:

| interpreter | sqlite | result |
|---|---|---|
| uv-managed CPython 3.12.11 (`~/.local/share/uv/python/…`) | 3.50.4 | **SIGBUS** |
| pyenv CPython 3.12.8 (`~/.pyenv/versions/3.12.8`)         | 3.51.0 | clean  |

`.python-version` in the repo root is:

```
$ od -c .python-version
0000000    3   .   1   2  \n   3   .   1   3   .   7  \n
```

Two lines, and the first is the unpinned `3.12`. `uv` prefers its own managed
builds, downloads **3.12.11**, and that is the one that crashes. The pre-existing
`/Volumes/xData/codeXD/mempalace/.venv` was built on pyenv 3.12.8, which is why
the main checkout is green and a fresh worktree is not.

CI never covers it either: `.github/workflows/ci.yml:14` runs `["3.9", "3.11", "3.13"]`
via `actions/setup-python` — no 3.12, no uv.

**Consequence.** "Nobody ran the full suite" is generous. On a clean checkout,
following the documented instructions, nobody *can*. And because a SIGBUS prints
no pytest summary at all, the failure mode is a wall of dots followed by a
traceback — which is easy to read as an interrupted run rather than a total one.

**Everything else in this report was measured on pyenv 3.12.8**, where the suite
does complete.

---

## 1. Ground truth — what I could and could not confirm

| claim | verdict |
|---|---|
| 3,575 passed, 34 skipped, ~84s | **Confirmed.** `3575 passed, 34 skipped in 85.69s` |
| Coverage 82.37% against an 85% gate | **Wrong, and worse.** Actual: **80.43%** |
| 34 skips, unexamined | Confirmed; examined in §4 |

```
$ … uv run pytest tests/ --ignore=tests/benchmarks --cov=mempalace --cov-report=term-missing
TOTAL                                     21270   4163    80%
FAIL Required test coverage of 85.0% not reached. Total coverage: 80.43%
3575 passed, 34 skipped in 91.26s
```

Coverage has fallen ~1.9 points below the remembered figure, not held steady.

### The 85% gate is not the gate

`pyproject.toml:203` sets `fail_under = 85`. CI passes `--cov-fail-under=80` on
the command line (`.github/workflows/ci.yml:22, :39, :50`), and a CLI flag
overrides the config file. So:

- locally, `pytest --cov` **fails** at 80.43%
- in CI, the same run **passes**, with 0.43 points of headroom

That gap is exactly why an 85% target has been unmet for a long time without
anything going red. Pick one number and put it in one place.

---

## 2. The mocking problem

### How much of it there is

131 distinct `patch("…")` targets across 883 call sites
(`scratchpad/patch_audit.py`, AST-classifies each target against the module it
names):

```
TOTAL patch sites: 883 | unique targets: 131

  296  imported-into-module (BRITTLE)
  234  defined-in-module public
  207  defined-in-module private
   82  external (stdlib/3rd-party)
   62  module-level constant/assign
    2  MISSING attribute (would AttributeError)
```

**296 sites — one third of all patching — are the same shape as the 23 that
broke.** They patch a symbol that the target module *imported*, not one it
*defines*. Move that import into a function (as `27a3437` did), rename it
upstream, or switch to `import x; x.y()`, and every one raises `AttributeError`
with no behaviour change whatsoever.

By file:

```
    67  tests/test_layers.py
    45  tests/test_cli.py
    37  tests/test_hooks_cli.py
    33  tests/test_repair.py
    28  tests/test_corpus_origin_integration.py
    25  tests/test_format_miner.py
    20  tests/test_searcher.py
    17  tests/test_llm_client.py
```

Worst individual targets:

| sites | target | first sites |
|---|---|---|
| 72 | `mempalace.cli.MempalaceConfig` | `tests/test_cli.py:96, :106, :121` |
| 38 | `mempalace.layers.MempalaceConfig` | `tests/test_layers.py:86, :105, :120` |
| 29 | `mempalace.layers._get_collection` | `tests/test_layers.py:106, :121, :137` |
| 26 | `mempalace.repair.ChromaBackend` | `tests/test_repair.py:180, :192, :209` |
| 14 | `mempalace.searcher.get_collection` | `tests/test_searcher.py:185, :195, :209` |

A further **207 sites patch a module-private helper** (`_extract_via_markitdown`,
`_maybe_run_mine_after_init`, `_save_diary_direct`, …). Those survive an import
move but not a rename — they make every underscore-prefixed function in the
codebase a public API with 207 callers.

### What they should assert instead

The pattern is always the same: the test is asserting *how* the code gets its
dependency instead of *what it does with it*.

- **`patch("mempalace.<mod>.MempalaceConfig")` (123 sites across cli/layers/hooks_cli).**
  Every one of these exists to point the code at a scratch palace. `MempalaceConfig`
  already accepts `config_dir=` — the `config` fixture in `conftest.py:198` builds one.
  Pass the real object (via `monkeypatch.setattr(mod, "_config", config)`, the way
  `tests/test_mcp_server.py:78` already does) and assert on the palace's contents.
  That is immune to where the import lives and actually proves the palace was used.
- **`patch("mempalace.repair.ChromaBackend")` (26 sites).** These assert that repair
  called a method. Build a real chroma palace in `tmp_path` — the suite already has
  `tests/_chroma_palace_helper.py` for exactly this — run repair, and assert the
  palace afterwards. The current tests would pass against a repair function that
  called every method in the right order and wrote nothing.
- **`patch("mempalace.llm_client.urlopen")` (17 sites).** This one is *legitimate in
  intent* — the network is a boundary the code owns — but patched at the wrong
  layer. `llm_client` should take an injectable transport, or the tests should
  patch `urllib.request.urlopen` (8 sites already do). Patching the re-exported
  name means moving `from urllib.request import urlopen` breaks 17 tests.

### Two tests that assert nothing, hidden by `create=True`

`mock.patch(..., create=True)` disables the very `AttributeError` that caught the
23. There are exactly two, and both are dead:

- **`tests/test_repair.py:18`** — `@patch("mempalace.repair.MempalaceConfig", create=True)`.
  `repair._get_palace_path` imports `MempalaceConfig` *inside the function*
  (`mempalace/repair.py:171`), so the mock is installed on a name nothing reads.
  The body then does `with patch.dict("sys.modules", {})` (a no-op) and asserts
  `isinstance(result, str)` — true of the fallback too. It passed regardless of what
  the function returned. **Fixed** in this branch.
- **`tests/test_normalize.py:1465`** — patches `mempalace.normalize.spellcheck_user_text`,
  which does not exist, around a call that passes `spellcheck=False`. Doubly inert.
  **Fixed** (patch removed; the real assertions kept).

One more of the same family that `create=True` did not flag, because it has an
assertion — it is just an assertion about a mock:

- **`tests/test_repair.py:27`** (`test_get_palace_path_fallback`) patched
  `_get_palace_path` **itself**, called the mock, and asserted on the mock's own
  return value. No production code executed. **Fixed.**

A blanket scan for assertion-free tests turns up 74 (`scratchpad`, AST), but 71 of
those are legitimate "must not raise" smoke tests, and 24 are benchmarks. The three
above are the real ones.

---

## 3. Coverage gaps that matter, ranked by risk

Not by percentage. The order is "what does an uncovered line cost if it is wrong".

### R1 — `write_log.py` is 100% covered and was 40% wired (fixed)

```
mempalace/write_log.py       49      0   100%
```

100%, and every test in `tests/test_write_log.py` called `log_write()` directly.
Nothing tested that the *write path* calls it:

```
$ grep -rn "log_write\|write_log" tests/ | grep -v "^tests/test_write_log.py"
(nothing)
```

Commit `88e4485` added 84 lines of wiring to `mcp_server.py` and `palace.py` and
zero tests for it. I wrote behaviour tests that drive `tool_add_drawer` and assert
the record that lands. **The first one failed**, and found a real bug:

> `tool_add_drawer`'s single-document branch (`mempalace/mcp_server.py:2653`)
> returned without calling `log_write`. Only the *chunked* branch logged. Since
> `chunk_size` defaults to 800 characters, that is the ordinary case: **every
> normal drawer add was invisible to the flight recorder.** Fixed in this branch,
> one call, mirroring the chunked branch.

This is the cleanest possible illustration of the brief's point: a coverage number
of 100% on a module whose integration was half-dead.

### R2 — three of six documented `error_class` values cannot be emitted

`mempalace/write_log.py:31-38` documents six stable error classes and says
"alerting keys off these". Tracing every call site:

- **`not_found`** — unreachable. Both places a drawer can be missing
  (`mcp_server.py:3267` in update, `:2727` in delete) are *early returns*, not
  exceptions, so they never reach `log_write_failure`.
- **`validation`** — mostly unreachable. `tool_add_drawer` catches its own
  `sanitize_name`/`sanitize_content` `ValueError` at `mcp_server.py:2582` and returns
  **before** the `try` that wraps `log_write_failure`. Same for the three
  `sanitize_*` guards in `tool_update_drawer` (`:3306, :3316, :3326`).
- **`delete_drawer`** — the op is named in the module docstring's list and
  `tool_delete_drawer` never calls `log_write` at all. Deletion is the one operation
  that can lose a memory outright, and it is the one operation the flight recorder
  cannot see.

I did **not** change these. They are design decisions (where to instrument), not
a self-contradiction inside one function the way R1 was. Flagged for the owner.

### R3 — `locks.py` at 72% is the lowest-covered data-integrity module

```
mempalace/locks.py          303     84    72%
```

The palace lock is what stops two writers driving HNSW inserts into one palace —
the failure that produces the `link_lists.bin` runaway. Uncovered and load-bearing:

- **`locks.py:356-357`** — `_gc_one_lock_file`'s "someone replaced this file under
  us" guard. If it were wrong, the residue sweep would unlink a **live** holder's
  lock and let a second writer in. Test added.
- **`locks.py:414-416`** — `maybe_gc_stale_locks` swallowing a sweep failure. This
  runs on the acquire path of *every* palace write; if it could propagate, an
  unreadable locks directory becomes a total write outage. Test added.
- **`locks.py:481-483`** — `list_locks`' `held_for_seconds`. The field a human reads
  during an outage to answer "has this been stuck for hours?", and the only branch
  that needs a genuinely held lock, which no existing test sets up. Test added.
- **`palace.py:1202-1203, 1206-1209`** — the retry protocol for "the residue GC
  unlinked the inode between our `open()` and our `flock()`". Not covered. I did
  not test these two; see §7.

### R4 — the contention event was executed but never asserted

`palace.py:1103` (`log_write` inside `_palace_contention_error`) *is* covered —
`tests/test_locks.py:105` runs through it. Nothing asserted the record. Delete the
call and coverage does not move and no test fails. Test added.

### R5 — check-and-set was untested on the drawer shape it exists for

`if_unchanged` exists to protect singletons that are read-modify-written: a
project's roadmap, the wings-registry. Those are long, so they are stored
**chunked** — one row per chunk, reassembled on read. All four existing CAS tests
(`tests/test_mcp_server.py:2489-2548`) use the short single-row
`drawer_proj_backend_aaa`.

If the reassembly that publishes `content_sha256` ever diverged from the one
`tool_update_drawer` compares against, **every guarded roadmap write would be
refused forever** and the existing tests would all stay green. Four tests added,
including that a shrinking rewrite deletes its orphaned tail chunks (a regression
there splices old text onto new — corruption that reads as a plausible drawer).

Related: `tool_add_drawer`'s docstring (`mcp_server.py:2566-2569`) still claims
`tool_get_drawer`/`tool_delete_drawer` report "not found" on the chunked path.
They do not — `_logical_drawer_record` (`:2390`) falls back to the chunk group.
Stale docstring; the new test pins the true behaviour.

### What I deliberately did not chase

`mempalace/backends/milvus.py` (22%, 623 uncovered lines) is the single biggest
number in the report and the lowest priority: `pymilvus` is not installed, its
tests skip, and it is not the default backend. Same for `qdrant` (68%) and
`pgvector` (74%). Raising those would move the percentage the most and the risk
the least.

---

## 4. The 34 skips

Every one is a real, still-true environmental gate. **None is hiding a gap in the
default configuration**, and none should be deleted. Grouped:

Full list from `pytest -rs` (counts are exact and sum to 34):

| n | reason | still true? | verdict |
|---|---|---|---|
| 15 | `set MEMPALACE_PGVECTOR_LIVE_DSN (scratch DB) to run` (`test_live_pgvector_conformance.py:83…336`) | yes — needs a live Postgres | keep |
| 1 | `set MEMPALACE_PGVECTOR_LIVE_URL …` (`test_pgvector_backend.py:661`) | yes | keep |
| 1 | `set MEMPALACE_QDRANT_LIVE_URL …` (`test_qdrant_backend.py:519`) | yes | keep |
| 11 | `could not import 'pymilvus'` (`test_milvus_backend.py:27` ×10, `:317` ×1) | yes — optional extra | keep |
| 3 | `this SQLite build refuses direct FTS5 shadow-table writes` (`test_miner_fts5_validation.py:97`) | **build-dependent** | see below |
| 1 | `could not import 'striprtf.striprtf'` (`test_format_miner.py:587`) | yes — optional extra | keep |
| 1 | `real Microsoft markitdown not installed` (`test_format_miner.py:612`) | yes — optional extra | keep |
| 1 | `tmp_path is not under home, cannot build ~-relative path` (`test_hooks_cli.py:1513`) | yes, on macOS | keep |

Three observations worth acting on:

- **The live-backend skips are 17 of 34** — half the skip count is one feature
  nobody runs locally. That is fine, but it means "34 skipped" carries less
  information than it looks like it does.
- **None of the ~12 `os.name == "nt"` markers are in this 34.** They are false on
  macOS, so those tests run. The Windows-only paths they guard (msvcrt locking,
  the `_gc_one_lock_file` Windows branch at `locks.py:370-383`) are covered only by
  the Windows CI job, and are invisible to any local run.
- **`test_miner_fts5_validation.py:81-97` is the one skip that can silently widen.**
  It skips when the SQLite build refuses direct FTS5 shadow-table writes, i.e. when
  the test cannot *fabricate* the corruption it wants to test. That is a legitimate
  guard, but it is keyed to the interpreter's SQLite — the same variable that
  produces §0's SIGBUS. On a different build this becomes a silent no-op instead of
  a regression test for a documented outage. It deserves a comment saying so; I did
  not add one (not my call which build to target).

The two `pytest.skip` calls for symlink-creation failure
(`test_exporter.py:152`, `test_format_miner.py:45`) did not fire here and are
correct defensive gates for restricted environments.

---

## 5. Ambient-state traps — the biggest finding in this report

### 5a. CRITICAL: with one env var set, the suite reads and writes the user's real palace

`MempalaceConfig.palace_path` (`mempalace/config.py:406`) reads
`MEMPALACE_PALACE_PATH` / `MEMPAL_PALACE_PATH` **and returns it in preference to
the config file**. `conftest.py` redirected `HOME` but never scrubbed this. So the
`config` fixture that every palace test passes around is silently overridden.

Measured. I pointed the variable at an empty scratch directory and ran the whole
suite:

```
$ MEMPALACE_PALACE_PATH=<empty scratch dir> … uv run pytest tests/ -q --ignore=tests/benchmarks
1 failed, 3574 passed, 34 skipped in 82.84s

===== WHAT LANDED IN THE FAKE LIVE PALACE =====
  chroma.sqlite3
  chroma.sqlite3-wal
  chroma.sqlite3-shm
  mempalace_embedder.json
  palace_format.json
  .collection_type_fixed
  .blob_seq_ids_migrated
  .wal_enabled
  c6267fa2-…/data_level0.bin
  c6267fa2-…/link_lists.bin
  c6267fa2-…/header.bin
  c6267fa2-…/length.bin
  c6267fa2-…/index_metadata.pickle
file count: 13
```

A complete, live ChromaDB palace — HNSW segment, WAL and all. **One test failed
out of 3,575.** Everything else reported success while operating on the wrong
database.

Which tests, exactly (pytest plugin, `scratchpad/palace_touch_plugin.py`, snapshots
the directory after every test):

```
===== TESTS THAT WROTE INTO THE CONFIGURED PALACE =====
   tests/test_antigravity_hooks_shell.py::test_wake_hook_never_emits_decision_field
   tests/test_antigravity_hooks_shell.py::test_wake_hook_state_files_are_namespaced_antigravity
   tests/test_asof.py::TestSnapshot::test_counts_respect_cutoff
   tests/test_asof.py::TestSnapshot::test_latest_sorted_desc
   tests/test_clean_lone_surrogates.py::TestToolsAcceptSurrogates::test_add_drawer_content
   tests/test_clean_lone_surrogates.py::TestToolsAcceptSurrogates::test_add_drawer_metadata
   tests/test_clean_lone_surrogates.py::TestToolsAcceptSurrogates::test_search_query
   tests/test_clean_lone_surrogates.py::TestToolsAcceptSurrogates::test_update_drawer
   tests/test_clean_lone_surrogates.py::TestToolsAcceptSurrogates::test_diary_write
   tests/test_cli_api.py::test_full_crud_round_trip
   tests/test_daemon.py::test_systemexit_in_job_does_not_kill_worker
   tests/test_daemon.py::test_health_rejects_missing_and_wrong_token
  total: 12
```

`test_add_drawer_content` files a drawer. `test_full_crud_round_trip` creates,
updates and **deletes**. Pointed at `~/.mempalace/palace` that is 140,000+ drawers
of the user's actual memory, being written by a test run that reports itself green.

`sandbox.env` exports exactly this variable. The distance between "safe" and
"writing junk drawers into your real memory" is one `source`.

Reported as required, not fixed quietly — but it is fixed, loudly:
`tests/conftest.py` now has an autouse `_no_ambient_config` fixture whose docstring
is this finding, and `tests/test_config.py` pins the precedence that causes it.

### 5b. The general shape: 53 variables read, 3 guarded

```
$ grep -rhoE '"(MEMPALACE|MEMPAL)_[A-Z0-9_]+"' mempalace/ | sort -u | wc -l
53
```

Before this branch, `conftest.py` neutralised three: `MEMPALACE_RETRIEVAL_LOG`,
`MEMPALACE_RERANK_URL`, `MEMPALACE_RERANK_MODEL`. The other fifty were free to
reconfigure the system under test. Proven, not theorised:

```
$ MEMPALACE_PALACE_PATH=<scratch> uv run pytest tests/test_config.py -q
2 failed, 120 passed

$ MEMPALACE_BACKEND=sqlite_exact uv run pytest tests/test_config.py tests/test_backends.py -q
3 failed, 216 passed
```

The rest of the dangerous set, by mechanism:

- `MEMPALACE_EMBEDDING_MODEL` / `_DEVICE` / `_THREADS` — different vectors, so every
  ranking assertion moves. Identical mechanism to the rerank leak.
- `MEMPALACE_WRITE_WATCHDOG_SECONDS` — the watchdog calls `os._exit`
  (`backends/chroma.py:1655`). A low exported value kills the test process mid-run.
- `MEMPALACE_MCP_READ_ONLY` — every write tool starts refusing.
- `MEMPALACE_OUTBOX_URL` / `_SECRET` — tests would emit to a real endpoint.
- `MEMPALACE_MAX_CHUNKS_PER_FILE`, `MEMPALACE_TOPIC_TUNNEL_MIN_COUNT` — change ingest
  shape and with it mining assertions.

`conftest.py` now scrubs 28 of them (the list is in `_AMBIENT_VARS_TO_SCRUB`).
Deliberately **not** scrubbed: `MEMPALACE_*_LIVE_URL` / `_LIVE_DSN`, which exist to
be set from the shell.

### 5c. Non-env ambient state

- **Network.** `tests/conftest.py:39-51` deliberately points chromadb's ONNX model
  cache back at the *real* user's `~/.cache/chroma/onnx_models/all-MiniLM-L6-v2` to
  avoid re-downloading 79 MB. Correct optimisation, but it means: on a machine
  without that cache, the first suite run downloads a model over the network. A
  "local-first, no external dependency" project has a network fetch on the cold-start
  test path. Worth a note in CONTRIBUTING at minimum.
- **The interpreter itself.** §0. The most expensive ambient dependency in the repo,
  and the least visible.
- **The real `HOME` on teardown.** `conftest.py:165-179` restores the original HOME
  at session end. Fine, but eleven test files re-`monkeypatch.setenv("HOME")`
  themselves (10 of them in `test_palace_locks.py` alone) — that is a shared fixture
  waiting to be written.

---

## 6. The process — what would actually have caught this

The 23-test breakage needed exactly one thing: *somebody running the whole suite
before calling it green*. Everything fancier is a distraction. Two artifacts, both
in this branch:

- **`scripts/full-suite.sh`** — strips the ambient variables, runs the whole suite
  serially, then ruff. Crucially it **checks the exit code for a signal**: a SIGBUS
  exits 138 and prints no pytest summary, so "I saw no FAILED lines" is not evidence
  of success. The script says so in words.
- **`scripts/pre-push`** — installs with
  `ln -sf ../../scripts/pre-push .git/hooks/pre-push`, runs `full-suite.sh`, blocks
  the push on failure. Pre-**push**, not pre-commit, deliberately: 90 seconds is too
  slow to pay per commit and cheap to pay before publishing. `--no-verify` remains
  the escape hatch, and using it is a visible choice.

Three more, cheap, that I recommend but did not do (they are the owner's call):

1. **Pin the interpreter.** `.python-version` currently holds `3.12` on line 1 and a
   stray `3.13.7` on line 2. As long as it names an unpinned minor, `uv` will
   provision whichever managed build it likes, and §0 shows that decides whether the
   suite runs at all. Either pin an exact version known to work, or add a startup
   assertion that fails loudly on a known-bad `sqlite3.sqlite_version`.
2. **Make the coverage gate one number.** 85 in `pyproject.toml`, 80 on the CI
   command line. Delete one.
3. **Install `pytest-randomly`.** It is not installed, so test order is deterministic
   file order and **order-dependence in this suite is currently unmeasured** — the
   brief's suggested `-p no:randomly` comparison has nothing to compare against. See
   §7 for the one order probe I did run.

---

## 7. Other findings

- **Order-dependence, partially measured.** `pytest-randomly` is absent, so I probed
  by hand: `test_backends.py::test_chroma_lexical_search_ids_roundtrip_through_get`
  passes alone and crashes after its file-mates — but that is §0's native crash, not
  a Python-level order dependency. Under the working interpreter I found no ordering
  failure, but with no randomiser installed that is weak evidence, not a clean bill.
- **Duplicated setup.** `_seed` is defined in 5 test files, `_run_hook` in 5,
  `_seed_drawers` in 4, `_collection` in 4, `_patch_mcp_server` in 3, `_set_home` in
  2. `monkeypatch.setenv("HOME", …)` appears in 11 files. These are candidates for
  `conftest.py`, and `_patch_mcp_server` in particular is the seam that would let the
  123 `MempalaceConfig` patches become real-config injection (§2).
- **Runtime.** 85s serial for 3,575 tests is healthy; no hot spot large enough to be
  worth attacking before the correctness items above.
- **Regression locks for the documented incidents** — all five have one:
  FTS5 malformed index (`tests/test_miner_fts5_validation.py`), orphaned lock holder
  (`tests/test_locks.py`, `tests/test_mine_lock_lifecycle.py`), `link_lists.bin`
  runaway (`tests/test_link_lists_runaway.py`), unexpanded `~` in `get_collection`
  (`tests/test_palace.py`, shipped with `12b75e0`), taxonomy counting chunk rows
  (`tests/test_miner.py`, shipped with `ea8db8f`). Nothing to add.
- **What I did not test.** `palace.py:1202-1209` — the lock-acquire retry when the
  residue GC unlinks the inode mid-acquire. Reaching it deterministically needs a
  patched seam inside the acquire loop, and I judged that a worse trade than leaving
  it uncovered: the test would have been exactly the kind of implementation-shaped
  test this audit is about. It stays a known gap.

---

## 8. What changed on this branch, and the honest numbers

| | before | after |
|---|---|---|
| suite | 3,575 passed, 34 skipped, 85.69s | **3,591 passed, 34 skipped, 83.59s** |
| coverage | 80.43% | **80.46%** |
| `ruff check` / `ruff format --check` | clean | clean |
| `locks.py` | 72% | 75% |
| `write_log.py` | 100% (and half-unwired) | 100% (and wired, with tests) |

**Coverage moved 0.03 points. That is the expected result and it is fine.**
Sixteen tests against 21,271 statements cannot move a percentage, and moving the
percentage was never the goal — `backends/milvus.py` alone has 623 uncovered lines
that could be bought cheaply and would buy nothing. What the sixteen bought:

- one production bug found and fixed (§3 R1), which no amount of `milvus` coverage
  would have surfaced;
- the write path's flight recorder is now provably connected, not merely present;
- check-and-set is proven on the drawer shape it was written for;
- a contended lock now has an asserted event, not just an executed line;
- the residue sweep's two "do not delete a live lock" guards are held down;
- three tests that ran no production code now do.

If the number matters more than the risk, the way to move it is milvus/qdrant/
pgvector, and that would be worth roughly nothing.

### Files touched

| file | change |
|---|---|
| `mempalace/mcp_server.py` | **production fix**: single-doc `tool_add_drawer` now logs |
| `tests/conftest.py` | autouse `_no_ambient_config` — scrubs 28 MEMPALACE_* vars |
| `tests/test_write_log.py` | +7 wiring tests (add / update / CAS conflict / opt-out / fail-soft) |
| `tests/test_mcp_server.py` | +4 chunked check-and-set tests |
| `tests/test_locks.py` | +5 contention-event and sweep-safety tests |
| `tests/test_config.py` | +1 env-vs-file precedence pin |
| `tests/test_repair.py` | 2 vacuous tests replaced with behaviour tests |
| `tests/test_normalize.py` | 1 inert patch removed |
| `scripts/full-suite.sh`, `scripts/pre-push` | new — §6 |

### Still open, deliberately

1. **The SIGBUS (§0).** Root-caused and reproduced, not fixed — the fix is either an
   interpreter pin or a change to how `checkpoint_wal` reaches a chromadb-owned
   database, and both are the owner's call, not a test-suite change.
2. **The three unreachable `error_class` values (§3 R2)**, including deletes being
   entirely absent from the write log. Instrumentation design.
3. **`palace.py:1202-1209`** (§7) — left uncovered on purpose rather than tested with
   an implementation-shaped test.
4. **The 296 brittle patch sites (§2).** Named, quantified and prioritised; converting
   them is a large mechanical change that wants its own branch and its own review.
