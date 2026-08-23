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
| `ID_RECIPE` (`ids.py`) | `"v3"` | same | Pinned, not diverged — a change on either side re-derives every drawer id |

Two rules:

- **When a delta test goes red after a merge, restore OUR value.** Never re-point the
  assertion at upstream's.
- **A new delta lands with a test in that file, in the same commit.** Every case above
  survived a full green suite before it had one.

Watch the ~113 theirs-only files a merge brings in untouched: they never conflict, so a
conflict-driven review gives them zero attention. `.mcp.json` was one, and it changed
behaviour by existing rather than by being called.

## Key Files for Common Tasks

- **Adding an MCP tool**: `mempalace/mcp_server.py` — add handler function + TOOLS dict entry
- **Changing search**: `mempalace/searcher.py`
- **Modifying mining**: `mempalace/miner.py` (project files) or `mempalace/convo_miner.py` (transcripts)
- **Adding a storage backend**: subclass `mempalace/backends/base.py`, register in `backends/__init__.py`
- **Input validation**: `mempalace/config.py` — `sanitize_name()` / `sanitize_content()`
- **Tests**: mirror source structure in `tests/test_<module>.py`
