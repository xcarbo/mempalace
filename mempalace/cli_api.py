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
