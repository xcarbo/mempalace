# MemPalace `memp` CLI — flat MCP-parity command surface

**Date:** 2026-06-16
**Status:** Design — approved direction, pending spec review
**Branch:** `xdev-patches` (fork patch)
**Author:** Chris + Claude

## Motivation

pi.dev (and other bash-first agents) cannot use the MemPalace MCP server — they
drive tools by shelling out to a CLI, not by speaking JSON-RPC over stdio. Today
the `mempalace` CLI exposes only ingest/search/maintenance commands (`mine`,
`search`, `wake-up`, `status`, `sync`, `repair`, …). The ~33 read/write tools the
MCP server provides — palace browsing, drawer CRUD, knowledge-graph, tunnels,
diary — have **no CLI equivalent**.

Second, independent motivation: the MCP server is a resident per-session stdio
process and has been repeatedly killed by macOS memory-pressure jetsam (see
roadmap "MCP-disconnect root cause"). A per-invocation CLI has **no resident
process to kill** — inherently more robust on a memory-constrained Mini.

**End goal:** this CLI fully replaces the MemPalace MCP server on the Mini (and the
laptop later). Every operation an agent does via MCP today must be reachable from
bash.

## Goal & non-goals

**Goal:** Expose the full MCP tool surface as **flat** top-level `memp` commands
that behave byte-for-byte like the MCP `tools/call` path, default to
machine-parseable JSON for the new commands, and are pleasant for an agent to
drive from bash.

**Non-goals (this build):**
- Decommissioning the MCP server registration in Claude Code — documented
  follow-on (see "Migration / decommission"), gated on parity proof + pi.dev
  validation. We build the replacement now; flip the switch later.
- Renaming the Claude Code skills to `memp-*` — a separate effort sharing only the
  brand, tracked as its own follow-up. Not part of this CLI.
- Any change to MCP server behavior, tool semantics, or storage.
- Re-implementing tool logic. The CLI reuses existing handlers verbatim.

## Decisions (locked)

- **Binary name `memp`.** Added as a second console-script alias pointing at the
  same `mempalace.cli:main` entry. `mempalace` is **kept** — the session hooks and
  `machine.md` call it by name. Both invoke the same code; `memp` is the short form
  agents and humans type.
- **Flat command surface.** `memp <tool>` — no `api`/`tool` middle group. Tools are
  top-level subcommands.
- **Output defaults:**
  - **New** MCP-tool commands (`get-drawer`, `list-wings`, `add-drawer`, `kg-*`,
    tunnels, diary, traverse, …) → **JSON by default** (they have no legacy human
    format).
  - The **4 inherited** commands that collide with existing CLI commands —
    `search`, `status`, `sync`, `mine` — keep their **upstream human output** by
    default.
  - `--json` (or env `MEMP_JSON=1`) forces JSON on **any** command, including the
    4 inherited ones (routes them to the MCP handler). `--pretty` indents JSON;
    plain default is compact.
- **Excluded:** `mempalace_reconnect` — reconnects the MCP server's persistent
  backend handle; a fresh-process CLI call makes it a no-op. Hidden, documented.

## Origination note (verified 2026-06-16)

The human-format output of `search`/`status` is **100% upstream**, not a local
patch. `mempalace/cli.py` is byte-identical to `upstream/develop`
(`git diff upstream/develop HEAD -- mempalace/cli.py` and the merge-base diff are
both empty). `cmd_search` doesn't format anything — it calls `searcher.search(...)`
which prints inside upstream code. Therefore `--json` must **not** reformat the
upstream print path; it routes to the MCP handler (`tool_search`, `tool_status`,
…) which already returns a JSON dict. Upstream code stays untouched; `--json` is
purely additive.

## Approach: flat commands, data-driven dispatch over `handle_request`

The MCP server is already fully data-driven:

```
TOOLS = { "mempalace_<name>": {"description": ..., "input_schema": {...}, "handler": fn}, ... }
handle_request({"method":"tools/call","params":{"name","arguments"}}) -> JSON-RPC envelope
```

`handle_request` performs arg-whitelisting to schema properties, int/number
coercion, alias mapping (e.g. diary `content`→`entry`), handler invocation, and
structured error formatting.

`cli_api.py` reuses **all** of that as an alternate transport:

1. **Generate flat subcommands from `TOOLS`** at parser-build time:
   - one subcommand per tool, named by stripping the `mempalace_` prefix and
     kebab-casing (`mempalace_get_drawer` → `get-drawer`).
   - skip the documented exclusions and any name that **collides** with an existing
     `mempalace` command (`search`, `status`, `sync`, `mine`); those four are
     handled by the interception path below instead of being re-registered.
   - one `--<prop>` flag per `input_schema.properties` entry; `required` array →
     `required=True`; `type: boolean` → `store_true`; `integer`/`number` → typed;
     `array`/`object` → JSON string (parsed before dispatch); `description` → help.
   - Auto-generation means **new upstream tools appear in the CLI for free.**
2. **Collect flags → `arguments` dict** (omit unset flags so schema defaults and
   the server's required-param diagnostics still apply).
3. **Dispatch:** synthesize
   `{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":<full>,"arguments":<args>}}`
   and call `handle_request(...)`.
4. **Render the envelope:**
   - success → unwrap `result.content[0].text` (already a JSON string from the
     server), print to **stdout**. Compact by default; `--pretty` indents. Exit 0.
   - error → print `{"error":<message>,"code":<code>}` to **stderr**, exit 1.

Because dispatch goes through the identical `handle_request`, the CLI's behavior
(validation, coercion, aliases, error text) is identical to MCP by construction.

### The 4 collider commands (`search`/`status`/`sync`/`mine`)

These already exist as upstream subcommands. We do **not** re-register or modify
their upstream handlers. Instead:

- A **global** `--json` flag (one `add_argument` on the top-level parser) and the
  `MEMP_JSON` env var.
- In `cli.py`'s dispatch (the point where it picks the `cmd_*` function), an
  **interception**: if `--json`/`MEMP_JSON` is set **and** the command is one of
  the four, route to `cli_api.run_tool("mempalace_<name>", args)` instead of the
  upstream `cmd_*`. Otherwise call the upstream handler unchanged.
- This keeps the fork's edits to upstream `cli.py` to a handful of isolated lines
  (one global flag + one interception stanza + one `register_flat(sub)` call);
  all real logic lives in `cli_api.py`, which upstream never touches.

### I/O contract (built for an agent)

- **stdout:** result as JSON (unwrapped payload, not the MCP `content` envelope).
  New commands: JSON default. Colliders: human default, JSON under `--json`.
  Compact unless `--pretty`.
- **stderr:** errors as `{"error":"...","code":N}`.
- **exit code:** 0 success, 1 tool/usage error.
- **stdin for large content:** any string flag accepts `-` to read the value from
  stdin (e.g. `memp add-drawer --wing w --room r --content -`). Avoids
  shell-escaping large verbatim drawer bodies.
- **JSON-valued flags:** `array`/`object` properties (e.g. `kg-add` triples) take a
  JSON string parsed before dispatch; `-` (stdin) also supported.

### Discoverability (parity with MCP `tools/list`)

- `memp list-tools` (or `memp --list`) prints the registry — every tool's name,
  description, and input schema — as JSON. The CLI analogue of MCP `tools/list`, so
  pi.dev can introspect the surface programmatically.
- `memp <tool> --help` shows that tool's flags (derived from the schema).

### Performance

Each invocation is a cold start. Cheap commands (`list-*`, `get-drawer`) must not
pay the embedder/HNSW warmup that only `search`/`mine` need. Requirement: importing
`cli_api` (and transitively `mcp_server`) must **not** eagerly warm the embedder;
warmup happens lazily inside the handlers that need it. Verify during
implementation — the MCP server has module-level warmup paths gated by env; the
CLI path must not trip them.

## File-level design

- **New: `mempalace/cli_api.py`** — single responsibility: be the bash transport
  for the MCP tool registry. Contains:
  - `register_flat(subparsers, existing_names)` — loop `TOOLS`, add a flat
    subcommand per non-colliding, non-excluded tool, with schema-derived flags +
    `--json`/`--pretty` and a `func` bound to the generic dispatcher.
  - `run_tool(tool_name, args) -> int` — flags→arguments, `handle_request`
    dispatch, envelope render, exit code. Used by both new commands and the
    collider interception.
  - `list_tools(args)` and `main(argv)` helpers.
- **Edit: `mempalace/cli.py`** — minimal, isolated:
  - one `register_flat(sub, EXISTING)` call where subparsers are built;
  - one global `--json`/`--pretty` `add_argument`;
  - one interception stanza in dispatch for the 4 colliders.
- **Edit: `pyproject.toml`** — add `memp = "mempalace.cli:main"` to
  `[project.scripts]` (one line; keep existing `mempalace`).

## Error handling

- Unknown tool / unknown flag / missing required / bad value → surfaced via the
  server's existing JSON-RPC error (`-32601`/`-32602`), rendered as
  `{"error","code"}` on stderr, exit 1.
- Handler exceptions → server's `_internal_tool_error` envelope → same rendering.
- Malformed `--<prop>` JSON (array/object flags) → CLI usage error on stderr, exit
  1, before dispatch.

## Testing

`tests/test_cli_api.py`, mirroring existing layout, against a temp palace fixture
(reuse `tests/conftest.py` patterns):

- **Parity guard:** every key in `TOOLS` (minus documented exclusions) is reachable
  — either as a generated flat command or via the collider interception — and
  `list-tools` reports the full registry. Locks parity so an upstream-added tool the
  CLI fails to surface breaks CI.
- **Round-trip:** `add-drawer` → `get-drawer` → `list-drawers` → `update-drawer` →
  `delete-drawer` on a temp palace; assert JSON shape and exit codes.
- **Read paths:** `list-wings`, `list-rooms` emit valid JSON, exit 0.
- **Collider behavior:** `search`/`status` default to human output; with `--json`
  (and with `MEMP_JSON=1`) route to the MCP handler and emit JSON. Upstream
  human path unchanged.
- **Contract:** compact-vs-`--pretty`; error → stderr JSON + exit 1; `-`/stdin
  content path; JSON-valued flag parse + malformed-JSON rejection.
- **No-eager-warmup:** importing `cli_api` does not trigger embedder load.

## Migration / decommission (follow-on, not this build)

1. Build + land the `memp` CLI (this spec).
2. Prove parity: parity-guard test green; manual spot-check of the tools pi.dev
   uses.
3. Repoint pi.dev to `memp …`.
4. Once validated in daily use on the Mini, remove the `mempalace` MCP server
   registration from Claude Code settings (Mini), then later the laptop. File the
   decommission as its own follow-up; record the rollback (re-add the MCP server)
   in case of regressions.
5. (Separate) Rename Claude Code skills to `memp-*` for brand consistency.

## Open considerations (resolve in plan, not blocking)

- Exact kebab/JSON-flag conventions for nested-object tools (`kg-add`,
  `create-tunnel`) — confirm against their schemas during implementation.
- Whether `--human`/`--pretty` should be no-ops or pretty-print for the new
  JSON-native commands (lean: `--pretty` indents; no `--human` for new commands).
