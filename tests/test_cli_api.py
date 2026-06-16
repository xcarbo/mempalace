"""Tests for the memp api CLI transport (mempalace.cli_api).

Note: ``mcp_server`` redirects stdout->stderr at import (issue #225) to protect
the MCP JSON-RPC channel. So tests assert on the *pure* layers (``dispatch_tool``
returns Python objects; ``_format_*`` return strings) and use a subprocess for
the one real-stdout end-to-end check, rather than fighting the fd redirect in
process.
"""

import io
import json

import pytest

from mempalace import cli_api


def _patch_mcp_server(monkeypatch, config, kg):
    from mempalace import mcp_server

    monkeypatch.setattr(mcp_server, "_config", config)
    monkeypatch.setattr(mcp_server, "_get_kg", lambda *a, **kw: kg)


def test_tool_command_name_strips_prefix_and_kebabs():
    assert cli_api.tool_command_name("mempalace_get_drawer") == "get-drawer"
    assert cli_api.tool_command_name("mempalace_list_wings") == "list-wings"
    assert cli_api.tool_command_name("mempalace_kg_add") == "kg-add"


def test_resolve_stdin_reads_dash(monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO("piped body"))
    assert cli_api._resolve_stdin("-") == "piped body"
    assert cli_api._resolve_stdin("literal") == "literal"


def test_format_payload_compact_vs_pretty():
    assert cli_api._format_payload({"a": 1}, pretty=False) == '{"a": 1}'
    pretty = cli_api._format_payload({"a": 1}, pretty=True)
    assert "\n" in pretty and "  " in pretty


def test_format_error():
    out = cli_api._format_error({"message": "boom", "code": -32602})
    assert json.loads(out) == {"error": "boom", "code": -32602}


def test_dispatch_tool_list_wings_ok(monkeypatch, config, kg):
    _patch_mcp_server(monkeypatch, config, kg)
    result = cli_api.dispatch_tool("mempalace_list_wings", {})
    assert result["ok"] is True
    assert result["error"] is None
    assert result["payload"] is not None


def test_dispatch_tool_unknown_tool_error(monkeypatch, config, kg):
    _patch_mcp_server(monkeypatch, config, kg)
    result = cli_api.dispatch_tool("mempalace_does_not_exist", {})
    assert result["ok"] is False
    assert result["error"]["code"] == -32601
