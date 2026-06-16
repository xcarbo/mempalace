"""Bash transport for the MemPalace MCP tool registry.

``memp <tool>`` flat commands are auto-generated from ``mcp_server.TOOLS`` and
dispatched through the same ``handle_request`` path as the MCP server, so the
CLI is a byte-identical drop-in replacement. Heavy imports (``mcp_server`` pulls
in chromadb, ~0.36s) are deferred into the functions, so importing this module
is cheap and core ``mempalace`` commands stay fast.

stdout note: ``mcp_server`` redirects stdout->stderr at import (issue #225) to
keep the MCP JSON-RPC channel clean. The CLI is not an MCP stdio server, so we
restore the real stdout (``_restore_real_stdout``) before emitting results.
"""

import json
import os
import sys

# Tools that make no sense as a one-shot CLI process (MCP-transport-only).
EXCLUDED = {"mempalace_reconnect"}

# Existing top-level commands that overlap an MCP tool. These keep their upstream
# (human) behavior by default; --json / MEMP_JSON routes them to the MCP handler.
COLLIDERS_JSON = {"search", "status"}


class _UsageError(Exception):
    """A CLI-level argument error (bad JSON flag, etc.) — exit 1, JSON to stderr."""


def tool_command_name(tool_name):
    """``mempalace_get_drawer`` -> ``get-drawer``."""
    base = tool_name[len("mempalace_") :] if tool_name.startswith("mempalace_") else tool_name
    return base.replace("_", "-")


def _resolve_stdin(value):
    """A flag value of ``-`` means: read the value from stdin."""
    if value == "-":
        return sys.stdin.read()
    return value


def _format_payload(payload, pretty=False):
    """Render a tool's result payload as JSON (compact by default)."""
    if pretty:
        return json.dumps(payload, indent=2, ensure_ascii=False)
    return json.dumps(payload, ensure_ascii=False)


def _format_error(error):
    """Render an error dict as a one-line JSON object."""
    return json.dumps(
        {"error": error.get("message"), "code": error.get("code")}, ensure_ascii=False
    )


def dispatch_tool(tool_name, arguments):
    """Call a tool through ``handle_request`` and normalize the response.

    Returns ``{"ok": bool, "payload": <obj>|None, "error": {"message","code"}|None}``.
    Does NOT print — this is the pure, testable core.
    """
    from .mcp_server import handle_request

    resp = handle_request(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        }
    )
    if not isinstance(resp, dict):
        return {
            "ok": False,
            "payload": None,
            "error": {"message": "no response from handler", "code": -32603},
        }
    if "error" in resp:
        err = resp["error"] or {}
        return {
            "ok": False,
            "payload": None,
            "error": {
                "message": err.get("message", "unknown error"),
                "code": err.get("code", -32603),
            },
        }
    try:
        text = resp["result"]["content"][0]["text"]
    except (KeyError, IndexError, TypeError):
        return {
            "ok": False,
            "payload": None,
            "error": {"message": "malformed handler result", "code": -32603},
        }
    try:
        payload = json.loads(text)
    except (ValueError, TypeError):
        payload = text  # non-JSON text payload (rare)
    return {"ok": True, "payload": payload, "error": None}


def _restore_real_stdout():
    """Undo ``mcp_server``'s import-time stdout->stderr redirect.

    No-op if ``mcp_server`` was never imported. For a normal one-shot CLI process
    this correctly restores the terminal stdout; the redirect only ever mattered
    for the MCP stdio server.
    """
    mod = sys.modules.get("mempalace.mcp_server")
    if mod is not None and hasattr(mod, "_restore_stdout"):
        try:
            mod._restore_stdout()
        except Exception:  # fail-soft: never let a stdout-restore quirk crash the CLI
            pass


def _run_tool(tool_name, arguments, pretty=False):
    """Dispatch + emit one tool call. Returns a process exit code (0 ok, 1 err)."""
    result = dispatch_tool(tool_name, arguments)
    _restore_real_stdout()
    if not result["ok"]:
        print(_format_error(result["error"]), file=sys.stderr)
        return 1
    payload = result["payload"]
    if isinstance(payload, str):
        print(payload)
    else:
        print(_format_payload(payload, pretty))
    return 0


def _coerce_json_value(raw, prop_name):
    """Parse a JSON-valued flag (array/object props). Raises ``_UsageError``."""
    try:
        return json.loads(raw)
    except (ValueError, TypeError) as e:
        raise _UsageError(f"--{prop_name.replace('_', '-')} must be valid JSON: {e}")


def _args_for_tool(tool_name, args):
    """Build the MCP ``arguments`` dict from an argparse Namespace via the schema.

    Only schema properties that were actually set (non-None) are included, so the
    server's defaults and required-param diagnostics still apply. String props
    support ``-`` (stdin); array/object props are parsed as JSON.
    """
    from .mcp_server import TOOLS

    props = TOOLS[tool_name].get("input_schema", {}).get("properties", {})
    out = {}
    for name, info in props.items():
        val = getattr(args, name, None)
        if val is None:
            continue
        ptype = info.get("type")
        if ptype in ("array", "object"):
            out[name] = _coerce_json_value(_resolve_stdin(val), name)
        elif ptype == "string":
            out[name] = _resolve_stdin(val)
        else:
            out[name] = val
    return out


def run_command(args):
    """Dispatch a generated api subcommand (``args._api_tool`` set)."""
    tool_name = args._api_tool
    try:
        arguments = _args_for_tool(tool_name, args)
    except _UsageError as e:
        _restore_real_stdout()
        print(_format_error({"message": str(e), "code": -32602}), file=sys.stderr)
        return 1
    return _run_tool(tool_name, arguments, pretty=getattr(args, "pretty", False))


def wants_json(args):
    """True if the caller asked for JSON via ``--json`` or ``MEMP_JSON``."""
    if getattr(args, "json", False):
        return True
    return os.environ.get("MEMP_JSON", "").lower() in {"1", "true", "yes", "on"}


def build_tool_list():
    """The CLI analogue of MCP ``tools/list`` — registry as a list of dicts."""
    from .mcp_server import TOOLS

    return [
        {
            "command": tool_command_name(name),
            "tool": name,
            "description": spec.get("description", ""),
            "input_schema": spec.get("input_schema", {}),
        }
        for name, spec in TOOLS.items()
        if name not in EXCLUDED
    ]


def print_tool_list(pretty=False):
    """Emit the tool registry as JSON to stdout. Returns exit code 0."""
    _restore_real_stdout()
    print(_format_payload(build_tool_list(), pretty))
    return 0


# ── argparse builders + collider routing ────────────────────────────────────

# Map a collider command to the arguments for its MCP tool. Only search/status
# are json-routed (sync/mine ingest arg-mapping is out of scope; left upstream).
_COLLIDER_ARG_MAP = {
    "search": lambda a: {
        k: v
        for k, v in {
            "query": getattr(a, "query", None),
            "wing": getattr(a, "wing", None),
            "room": getattr(a, "room", None),
            "limit": getattr(a, "results", None),
        }.items()
        if v is not None
    },
    "status": lambda a: {},
}


def _add_schema_flags(parser, input_schema):
    """Add one ``--flag`` per schema property (dest = the schema property name)."""
    import argparse

    props = input_schema.get("properties", {})
    required = set(input_schema.get("required", []))
    for name, info in props.items():
        flag = "--" + name.replace("_", "-")
        ptype = info.get("type")
        help_text = info.get("description", "")
        is_req = name in required
        if ptype == "boolean":
            parser.add_argument(
                flag, dest=name, action=argparse.BooleanOptionalAction, default=None, help=help_text
            )
        elif ptype == "integer":
            parser.add_argument(
                flag, dest=name, type=int, default=None, required=is_req, help=help_text
            )
        elif ptype == "number":
            parser.add_argument(
                flag, dest=name, type=float, default=None, required=is_req, help=help_text
            )
        else:  # string / array / object — array & object are parsed as JSON at dispatch
            parser.add_argument(flag, dest=name, default=None, required=is_req, help=help_text)


def register_flat(subparsers, existing_names):
    """Add one flat subcommand per non-excluded, non-colliding tool, plus list-tools."""
    from .mcp_server import TOOLS

    for tool_name, spec in TOOLS.items():
        if tool_name in EXCLUDED:
            continue
        cmd = tool_command_name(tool_name)
        if cmd in existing_names:
            continue  # collider — handled by add_json_flags + interception
        desc = spec.get("description", "")
        p = subparsers.add_parser(cmd, help=desc[:80], description=desc)
        _add_schema_flags(p, spec.get("input_schema", {}))
        p.add_argument("--json", dest="json", action="store_true", help="(default) emit JSON")
        p.add_argument("--pretty", action="store_true", help="indent JSON output")
        p.set_defaults(_api_tool=tool_name)

    lt = subparsers.add_parser(
        "list-tools", help="List every tool (name, description, schema) as JSON"
    )
    lt.add_argument("--pretty", action="store_true", help="indent JSON output")
    lt.set_defaults(_api_list=True)


def add_json_flags(subparsers, collider_names):
    """Add ``--json`` / ``--pretty`` to existing collider subparsers (idempotent)."""
    import argparse

    for name in collider_names:
        p = subparsers.choices.get(name)
        if p is None:
            continue
        try:
            p.add_argument(
                "--json",
                dest="json",
                action="store_true",
                help="emit JSON via the MCP handler instead of human text",
            )
            p.add_argument("--pretty", action="store_true", help="indent JSON output")
        except argparse.ArgumentError:
            pass  # already added


def run_collider(args):
    """Dispatch a collider command (search/status) to its MCP tool as JSON."""
    cmd = args.command
    tool_name = "mempalace_" + cmd
    arguments = _COLLIDER_ARG_MAP[cmd](args)
    return _run_tool(tool_name, arguments, pretty=getattr(args, "pretty", False))
