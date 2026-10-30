# CLAUDE.md

## The Mission

Memory is identity. When an AI forgets everything between conversations, it cannot build real understanding — of you, your work, your people, your life.

MemPalace exists to solve this. It is a memory system — not a search engine, not a RAG pipeline, not a vector database wrapper. It treats every word you have shared as sacred, stores it verbatim, and makes it instantly available. Your data never leaves your machine. We never summarize. We never paraphrase. We return your exact words.

100% recall is the design requirement — the target every search path is measured against. Anything less means forgetting, and forgetting means starting over.

The name comes from the ancient "method of loci" — the memory palace technique used for thousands of years to organize and recall vast amounts of information by placing it in imagined rooms of an imagined building. We were also inspired by the Zettelkasten method (created by German sociologist Niklas Luhmann) — small cross-referenced index cards that point to each other. We apply both ideas to AI memory:

- **Wings** for broad categories (people, projects, topics)
- **Rooms** for time-based groupings (days, sessions)
- **Drawers** for full verbatim content (your exact words)
- **AAAK compression** for the index layer — a compact symbolic format (via `dialect.py`) that lets an LLM scan thousands of entries instantly and know exactly which drawer to open

## Design Principles

These are non-negotiable. Every PR, every feature, every refactor must honor them.

- **Verbatim always** — Never summarize, paraphrase, or lossy-compress user data. The system searches the index and returns the original words. If a user said it, we store exactly what they said. This is the foundational promise.
- **Incremental only** — Append-only ingest after initial build. Never destroy existing data to rebuild. A crash mid-operation must leave the existing palace untouched.
- **Entity-first** — Everything is keyed by real names with disambiguation by DOB, ID, or context. People matter more than topics.
- **Local-first, zero external API by default** — All extraction, chunking, embedding, and LLM-assisted refinement happens on the user's machine by default, using locally-hosted runtimes (Ollama, LM Studio, llama.cpp, vLLM, unsloth studio, etc.). External providers (Anthropic, OpenAI, Google) are supported via BYOK but are never required and never enabled silently. The system never sends user content to a service the user has not explicitly configured. "Local LLM" is not an external API — Ollama and equivalents running on localhost are part of the user's machine. External BYOK is always a deliberate user choice, never a default and never a silent fallback.
- **Performance budgets** — Hooks under 500ms. Startup injection under 100ms. Memory should feel instant.
- **Privacy by architecture** — The system physically cannot send your data because it never leaves your machine. No telemetry, no phone-home, no external service dependencies for core operations.
- **Background everything** — Filing, indexing, timestamps, and pipeline work happen via hooks in the background. Nothing interrupts the user's conversation. Zero tokens spent on bookkeeping in the chat window.

## Contributing

We welcome bug fixes, performance improvements, new language support, better entity disambiguation, documentation, and test coverage.

We do not accept summarization of user content, cloud storage/sync features, telemetry or analytics, features requiring API keys for core memory, or shortcuts that bypass verbatim storage.

## Setup

```bash
uv sync --extra dev   # recommended; or: pip install -e ".[dev]"
```

## Commands

```bash
# Run tests
uv run pytest tests/ -v --ignore=tests/benchmarks

# Run tests with coverage
uv run pytest tests/ -v --ignore=tests/benchmarks --cov=mempalace --cov-report=term-missing

# Lint
uv run ruff check .

# Format
uv run ruff format .

# Format check (CI mode)
uv run ruff format --check .
```

## Project Structure

```
mempalace/
├── mcp_server.py        # MCP server — all read/write tools
├── cli.py               # CLI dispatcher
├── config.py            # Configuration + input validation
├── miner.py             # Project file miner
├── convo_miner.py       # Conversation transcript miner
├── searcher.py          # Semantic search (hybrid BM25 + vector)
├── knowledge_graph.py   # Temporal entity-relationship graph (SQLite)
├── palace.py            # Shared palace operations
├── palace_graph.py      # Room traversal + cross-wing tunnels
├── backends/            # Pluggable storage backends (ChromaDB default)
│   ├── base.py          # Abstract interface — implement this for new backends
│   └── chroma.py        # ChromaDB implementation
├── dialect.py           # AAAK compression dialect
├── normalize.py         # Transcript format detection + normalization
├── entity_detector.py   # Auto-detect people/projects from content
├── entity_registry.py   # Entity storage and disambiguation
├── layers.py            # L0-L3 memory wake-up stack
├── onboarding.py        # Interactive first-run setup
├── repair.py            # Palace repair and consistency checks
├── dedup.py             # Deduplication
├── migrate.py           # ChromaDB version migration
├── spellcheck.py        # Auto-correct user messages
├── exporter.py          # Palace data export
├── hooks_cli.py         # Hook management CLI
├── query_sanitizer.py   # Prompt contamination prevention
├── split_mega_files.py  # Split concatenated transcript files
└── version.py           # Single source of truth for version

hooks/                   # Claude Code hook scripts
├── mempal_save_hook.sh        # Stop: triggers diary save
└── mempal_precompact_hook.sh  # PreCompact: saves state before compression
```

## Conventions

- **Python style**: snake_case for functions/variables, PascalCase for classes
- **Linter**: ruff with E/F/W rules
- **Formatter**: ruff format, double quotes
- **Commits**: conventional commits (`fix:`, `feat:`, `test:`, `docs:`, `ci:`)
- **Tests**: `tests/test_*.py`, fixtures in `tests/conftest.py`
- **Coverage**: 85% threshold (80% on Windows due to ChromaDB file lock cleanup)

## Architecture

```
User → CLI / MCP Server → Storage Backend (ChromaDB default, pluggable)
                        → SQLite (knowledge graph)

Palace structure:
  WING (person/project)
    └── ROOM (day/topic)
          └── DRAWER (verbatim text chunk)

Index layer (AAAK):
  Compressed pointers → DRAWER locations
  Scanned by LLM to find relevant drawers without reading all content

Knowledge Graph:
  ENTITY → PREDICATE → ENTITY (with valid_from / valid_to dates)
```

## Fork Deltas (this fork only — read before any upstream merge)

`xdev-patches` diverges from `MemPalace/mempalace` in a few places where the divergence is
deliberate and **silent-revert-shaped**: a clean merge can restore upstream's value, nothing
raises, and the suite stays green. Two of the 3.7.1 merge's worst moments were this shape.

The register is `tests/test_fork_deltas.py` — every delta has a test that fails loudly. Run it
first after any upstream merge:

```bash
uv run pytest tests/test_fork_deltas.py -v
```

| Delta | Ours | Upstream | Why a silent revert hurts |
|---|---|---|---|
| `.mcp.json` | must NOT exist | ships it (`2f72c91`) | Re-arms the MCP server retired 2026-06-16 — every session in this repo regains the `mcp__mempalace__*` **write** surface against the live palace |
| `NORMALIZE_VERSION` (`palace.py`) | `3` | `2` | At 2 the v3 re-mine no-ops and the 6,066-drawer oversized backlog stays frozen, reporting success |
| `_open_search_collection` (`searcher.py`) | `read_only=True` | — | Reverted once already, by stage 5 of the 3.7.1 merge. Without it every search during a mine raises `MineAlreadyRunning` |
| `_main_worktree_root` (`hooks_cli.py`) | take-ours | — | Worktree sessions file into a wing literally named `worktree` |
| backend palace lock (`backends/sqlite_exact.py`, `backends/chroma.py`) | `short_writer_palace_lock()` (waits 10 s) | bare `mine_palace_lock()` | Bare binds wait=0: a session hook ending in the same second as another loses its diary checkpoint as "Backend open failed". The suite pins the wait to 0, so it stays green |
| `ID_RECIPE` (`ids.py`) | `"v3"` | same | Pinned, not diverged — a change on either side re-derives every drawer id |
| `mine_palace_lock` release (`palace.py`) | calls `_cleanup_mine_lock_file` | leaves the file | Every `memp` call then finds the previous call's dead holder and logs a "lock-residue GC: reclaiming" line on stderr — noise in every agent's tool output and in hook.log, one per process forever |
| exchange stale check (`palace.py` mined-set) | no `convo_chunker_version` check | missing stamp = stale (3.11) | Any pick of upstream's `palace/mined.py` makes the next `mine --mode convos` re-mine every transcript still on disk. #2591 was taken forward-only (5878576) |
| chromadb import (`backends/chroma.py`, `mcp_server.py`, `repair.py`) | lazy, first use only | module-level in all three | Every `memp` call, hook and write pays 0.3 to 0.55 s loading a library no fleet palace uses (all `sqlite_exact`). Restoring any one of the three imports brings the whole cost back |
| `get(limit, offset)` paging (`backends/sqlite_exact.py` `_rows`) | rowid seek via an index-only anchor; metadata-only reads skip `document` | SQL `OFFSET` on the full-row read | Same rows either way, so nothing fails: a one-file hook mine goes back to 30 minutes on the Mini palace, holding the lock and dropping other sessions' diary saves |
| convo mine lock (`convo_miner.mine_convos`) | taken after the read-only prefetch | wraps the whole mine | Same drawers filed, so nothing fails: the full-palace prefetch runs under the lock again and hook diary saves time out behind it |
| read paths besides `_open_search_collection` (`searcher.search`, the closets open in `search_memories`, `miner.status`) | `read_only=True` via `palace.get_collection_read_only` | writable opens | While any mine runs, plain `memp search` and `memp status` raise `MineAlreadyRunning` and `memp search --json` waits 10 s |
| daemon writer lease (`daemon.run_server`, `service.run_diary_write` / `run_mcp_tool`) | none: each job takes the lock for its own write; a busy lock is a refusal and the job re-queues unspent | `mine_palace_lock` held for the daemon's lifetime | With the daemon on, every write outside it (`memp add-drawer`, crons, `:4109` write routes) fails with `MineAlreadyRunning` |
| MCP read tools (`mcp_server._get_collection(read_only=True)`: get-drawer, list-drawers, list-wings, list-rooms, taxonomy, check-duplicate, graph tools, diary-read) | read-only open, not cached | writable open for every non-MCP-server caller | While any mine runs, every `memp get-drawer` / `list-drawers` waits 10 s and fails "palace busy, write not filed" |
| hook daemon wait (`hooks_cli._save_diary_direct`, `_mine_sync`, `hook_stop`) | 3 s, then the job is handed off and the Stop marker advances | 30 s diary / 60 s precompact wait; a timeout reports nothing filed | A diary queued behind a mine freezes the agent for the whole 30 s hook budget, and the next Stop files the same messages a second time while the daemon still writes the first |
| failed-job retention (`daemon.QueueStore.prune_terminal`) | failed jobs kept until `memp daemon ack` (at most `FAILED_RETENTION_DAYS`, 30) | pruned after `JOB_RETENTION_DAYS` (7) | A failed batch's payload is the only copy of a parked roadmap it could not file; it is deleted unseen after 7 days |
| Stop diary window key (`hooks_cli._diary_window_key`, `QueueStore.enqueue(reuse_succeeded=)`, `daemon.stored_job_for_key`) | one dedupe key per unconfirmed save window (1 h at most); the daemon hands back a queued, running or succeeded job under it, and a direct-routed retry checks the queue for it first | no key for diaries; dedupe matches queued/running only | A diary submit that times out after the daemon stored the job leaves the marker, and the next Stop files the same checkpoint again |
| drawer write lock (`mcp_server._drawer_write_lock`, `tool_update_drawer`, `tool_delete_drawer`) | palace lock, then one lock per drawer, across the read, `if_unchanged` check, revision and write | no lock (and no check-and-set) | Same result on a quiet palace, so nothing fails: two writers both pass the check, the second replaces the first's text, and `drawer_revisions.jsonl` holds only the older text, so a roadmap rewrite is lost with no copy |

Two rules:

- **When a delta test goes red after a merge, restore OUR value.** Never re-point the
  assertion at upstream's.
- **A new delta lands with a test in that file, in the same commit.** Every case above
  survived a full green suite before it had one.

Watch the ~113 theirs-only files a merge brings in untouched: they never conflict, so a
conflict-driven review gives them zero attention. `.mcp.json` was one, and it changed
behaviour by existing rather than by being called.

## Deploying to the servers

Every palace server (the Mini plus one Hetzner box per client) is seen and updated with
`scripts/fleet`, run on the Mini. The server list is private and stays out of this public repo:
`~/.config/mempalace/fleet.servers.json` (`FLEET_CONFIG` overrides); `scripts/fleet.servers.example.json`
shows the shape.

```bash
scripts/fleet status                      # one row per server: commit, drift, versions, drawers, API, backup
scripts/fleet deploy <box> --dry-run      # every step, nothing changed
scripts/fleet deploy <box>                # xdev-patches HEAD; --ref <sha>, --api also ships mempalace-api
scripts/fleet rollback <box>              # flip back to the previous venv (run again to flip forward)
bun test scripts/fleet.test.ts
```

- Only commits on `xdev-patches` ship, from a clean checkout, as a `git archive`. Never rsync a tree to a box.
- A box keeps `~/opt/mempalace/.venv`, `~/opt/mempalace-src` and `~/opt/mempalace-api` as symlinks to
  `<name>.<sha>` dirs; a deploy builds a new one beside them and swaps the link by rename, so a running
  hook never sees a half-installed venv. Venvs are built with uv on the box's own Python 3.14.6 /
  SQLite 3.53.4, never a uv-managed Python, with dependencies pinned to the live venv (`--fresh-deps` drops that).
- fleet touches code and venvs only, never `~/.mempalace`. The Mini is report-only (editable install).
- fleet restarts `mempalace-daemon.service` after the API on any box where it is enabled.

## Write queue (daemon + batches, this fork)

- `mempalace daemon` is a **shared write queue**, not the palace owner (fork delta 16). It runs as
  `com.mempalace.daemon` on the Mini and `mempalace-daemon.service` on each box (unit files in
  claude-mech `machines/<machine>/`). It imports mempalace once at start: after a commit on the Mini,
  `launchctl kickstart -k gui/$(id -u)/com.mempalace.daemon`.
- `memp submit <batch.json|->` puts a batch in the on-disk queue and prints `queued <job-id>` at
  once, daemon up or not; `memp job <id>` shows queued / running / succeeded / failed.
  `memp daemon wait <id> --timeout <s>` waits for it: exit 0 succeeded, 1 failed, 3 still
  queued or running when the time ran out (it prints the state).
- A batch: `{"wing": "<w>", "steps": [{"id": "f1", "tool": "add-drawer", "args": {...}},
  {"tool": "kg-add", "args": {"subject": "action:${f1.drawer_id}", ...}},
  {"tool": "update-drawer", "args": {..., "if_unchanged": "<sha>"}, "on_conflict": "park"}]}`.
  Steps run in order under one palace lock (a busy palace re-queues the whole batch, nothing
  written). `${id.field}` reads an earlier step's result. A refused check-and-set marked `park` is
  filed verbatim as an open follow-up in the batch's wing. Deletes are not batchable.
- A failed job, batch or diary, writes a `Daemon job failed:` line to `~/.mempalace/hook_state/hook.log`.
  `memp daemon status`, the Stop hook's checkpoint line and the Python SessionStart hook report
  "N failed jobs" until `memp daemon ack`; `memp daemon jobs --failed` lists them. Retention keeps
  those jobs (and their payloads) until then, at most 30 days. A failed batch names
  `failed_step` and lists `done`; a park that could not be filed fails it with `unfiled_content`.
- Hooks wait 3 s for their own job, then hand it off (`Diary checkpoint queued: <id>`).
- Hooks hand their diary saves and transcript ingests to the daemon when it is up with
  `"write_routing": {"hooks": "prefer"}` in `~/.mempalace/config.json`, and write directly when not.
- State and the deploy log are in `~/opt/.fleet/` on each box.

## Key Files for Common Tasks

- **Adding an MCP tool**: `mempalace/mcp_server.py` — add handler function + TOOLS dict entry
- **Changing search**: `mempalace/searcher.py`
- **Modifying mining**: `mempalace/miner.py` (project files) or `mempalace/convo_miner.py` (transcripts)
- **Adding a storage backend**: subclass `mempalace/backends/base.py`, register in `backends/__init__.py`
- **Input validation**: `mempalace/config.py` — `sanitize_name()` / `sanitize_content()`
- **Tests**: mirror source structure in `tests/test_<module>.py`
