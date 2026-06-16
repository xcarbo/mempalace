# MemPalace `api` CLI — MCP-parity command surface

**Date:** 2026-06-16
**Status:** Design — approved direction, pending spec review
**Branch:** `xdev-patches` (fork patch)
**Author:** Chris + Claude

## Motivation

pi.dev (and other bash-first agents) cannot use the MemPalace MCP server — they
drive tools by shelling out to a CLI, not by speaking JSON-RPC over stdio. Today
the `mempalace` CLI exposes only ingest/search/maintenance commands (`mine`,
`search`, `wake-up`, `status`, `sync`, `repair`, …). The ~33 read/write tools
that the MCP server provides — palace browsing, drawer CRUD, knowledge-graph,
tunnels, diary — have **no CLI equivalent**.

A second, independent motivation: the MCP server is a resident per-session stdio
process and has been repeatedly killed by macOS memory-pressure jetsam (see
roadmap "MCP-disconnect root cause"). A per-invocation CLI has **no resident
process to kill** — it is inherently more robust on a memory-constrained Mini.

**End goal:** the `api` CLI fully replaces the MemPalace MCP server on the Mini
(and the laptop later). Every operation an agent does via MCP today must be
reachable via `mempalace api …`.

## Goal & non-goals

**Goal:** Add `mempalace api <tool> [flags]` — one subcommand per MCP tool —
that behaves byte-for-byte like the MCP `tools/call` path, emits machine-parseable
JSON, and is suitable for an agent to drive from bash.

**Non-goals (this build):**
- Decommissioning the MCP server registration in Claude Code. That is a documented
  follow-on (see "Migration / decommission"), gated on parity proof + pi.dev
  validation. We build the replacement now; we flip the switch later.
- Any change to MCP server behavior, tool semantics, or storage.
- Re-implementing tool logic. The CLI reuses existing handlers verbatim.

## Approach (chosen: A — subcommand group on the existing binary)

`mempalace api <tool>` — a new top-level subcommand group on the existing
`mempalace` console script. All command logic lives in a **new isolated module**
`mempalace/cli_api.py`; `cli.py` gains only a one-stanza delegation. This keeps
the fork's merge surface against upstream `cli.py` to a few lines, while all the
new code sits in a file upstream never touches.

Rejected alternatives:
- **B — separate `mempalace-cli`/`mp` binary:** zero `cli.py` change, but a second
  entry point and a `pyproject.toml` edit (a common upstream-sync conflict point).
- **C — thin client over the `:4109` HTTP API:** that API is read-only; full
  parity needs writes.

## Architecture

### Core idea: data-driven dispatch over `handle_request`

The MCP server is already fully data-driven:

```
TOOLS = { "mempalace_<name>": {"description": ..., "input_schema": {...}, "handler": fn}, ... }
handle_request({"method": "tools/call", "params": {"name", "arguments"}}) -> JSON-RPC envelope
```

`handle_request` performs arg-whitelisting (to schema properties), int/number
type coercion, alias mapping (e.g. diary `content`→`entry`), handler invocation,
and structured error formatting.

`cli_api.py` reuses **all** of that by acting as an alternate transport:

1. **Build argparse from `TOOLS`** (at startup, by iterating the registry):
   - one subcommand per tool, named by stripping the `mempalace_` prefix and
     kebab-casing (`mempalace_get_drawer` → `get-drawer`).
   - one `--<prop>` flag per `input_schema.properties` entry; `required` array →
     argparse `required=True`; `type: boolean` → `store_true`; `integer`/`number`
     → typed; `array`/`object` → accept a JSON string (parsed before dispatch);
     `description` → flag help text.
   - This auto-generation means **new upstream tools appear in the CLI for free**.
2. **Collect flags → `arguments` dict** (omit unset flags so schema defaults and
   the server's required-param diagnostics still apply).
3. **Dispatch:** synthesize
   `{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":<full>,"arguments":<args>}}`
   and call `handle_request(...)`.
4. **Render the envelope:**
   - success → unwrap `result.content[0].text` (already a JSON string from the
     server), print to **stdout**. Compact by default; `--pretty` passes the
     server's indented form through. Exit 0.
   - error → print `{"error": <message>, "code": <code>}` to **stderr**, exit 1.

Because dispatch goes through the identical `handle_request`, the CLI's behavior
(validation, coercion, aliases, error text) is identical to MCP by construction.

### I/O contract (built for an agent)

- **stdout:** the tool's result as JSON (the unwrapped payload, not the MCP
  `content` envelope). Compact by default; `--pretty` for indented.
- **stderr:** errors as `{"error": "...", "code": N}`.
- **exit code:** 0 success, 1 tool/usage error.
- **stdin for large content:** any string flag accepts `-` to read the value from
  stdin (e.g. `mempalace api add-drawer --wing w --room r --content -`). Avoids
  shell-escaping large verbatim text — important for an agent piping drawer bodies.
- **JSON-valued flags:** `array`/`object` properties (e.g. `kg-add` triples) take
  a JSON string, parsed before dispatch; `-` (stdin) also supported.

### Discoverability (parity with MCP `tools/list`)

- `mempalace api --list` (alias `list-tools`) prints the registry — every tool's
  name, description, and input schema — as JSON. This is the CLI analogue of MCP
  `tools/list`, so pi.dev can introspect the surface programmatically.
- `mempalace api <tool> --help` shows that tool's flags (from the schema).

### Excluded tools

- `mempalace_reconnect` — reconnects the MCP server's persistent backend handle.
  A CLI invocation is a fresh process, so it is a no-op; **hidden** from the CLI
  (documented). All other tools are exposed.

### Performance

Each invocation is a cold start. Cheap commands (`list-*`, `get-drawer`) must not
pay the embedder/HNSW warmup that only `search`/`mine` need. Requirement: importing
`cli_api` (and transitively `mcp_server`) must **not** eagerly warm the embedder;
warmup happens lazily inside the handlers that need it. This must be verified
during implementation (the MCP server has module-level warmup paths gated by env;
the CLI path must not trip them).

## File-level design

- **New: `mempalace/cli_api.py`** — registry→argparse builder, flag→arguments
  mapper, `handle_request` dispatch, envelope renderer, stdin/JSON-flag handling,
  `--list`/`--pretty`. Single responsibility: be the bash transport for the MCP
  tool registry. Public entry: `main(argv) -> int`.
- **Edit: `mempalace/cli.py`** — one `sub.add_parser("api", ...)` stanza with
  `add_help=False` (so per-tool `--help` is handled inside `cli_api`), plus a
  dispatch line: `if args.command == "api": return cli_api.main(remaining_argv)`.
  Uses `parse_known_args`/`REMAINDER` so the `api` group owns its own arg parsing.

## Error handling

- Unknown tool / unknown flag / missing required / bad value → surfaced via the
  server's existing JSON-RPC error (`-32601`/`-32602`), rendered as
  `{"error","code"}` on stderr, exit 1.
- Handler exceptions → server's `_internal_tool_error` envelope → same rendering.
- Malformed `--<prop>` JSON (for array/object flags) → CLI usage error on stderr,
  exit 1, before dispatch.

## Testing

`tests/test_cli_api.py`, mirroring existing test layout, against a temp palace
fixture (reuse `tests/conftest.py` patterns):

- **Parity guard:** every key in `TOOLS` (except the documented exclusions) has a
  generated subcommand, and `--list` reports the full registry. Locks parity so an
  upstream-added tool that the CLI somehow fails to surface fails CI.
- **Round-trip:** `add-drawer` → `get-drawer` → `list-drawers` → `update-drawer`
  → `delete-drawer` on a temp palace; assert JSON shape and exit codes.
- **Read paths:** `list-wings`, `list-rooms`, `status` emit valid JSON, exit 0.
- **Contract:** compact-vs-`--pretty`; error → stderr JSON + exit 1; `-`/stdin
  content path; JSON-valued flag parse + malformed-JSON rejection.
- **No-eager-warmup:** importing `cli_api` does not trigger embedder load
  (assert via the warmup hook/env the server uses).

## Migration / decommission (follow-on, not this build)

1. Build + land the `api` CLI (this spec).
2. Prove parity: parity-guard test green; manual spot-check of the tools pi.dev
   uses.
3. Repoint pi.dev to `mempalace api …`.
4. Once validated in daily use on the Mini, remove the `mempalace` MCP server
   registration from Claude Code settings (Mini), then later the laptop. File the
   decommission as its own follow-up; record the rollback (re-add the MCP server)
   in case of regressions.

## Open considerations (resolve in plan, not blocking)

- Exact kebab/JSON-flag conventions for nested-object tools (`kg-add`,
  `create-tunnel`) — confirm against their schemas during implementation.
- Whether to also alias the already-existing top-level commands (`search`,
  `status`) under `api` for a uniform namespace (planned: yes — uniform surface
  for the agent, both route to the same handlers).
