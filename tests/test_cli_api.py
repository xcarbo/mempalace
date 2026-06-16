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


# ── Task 3: arg-mapping, run_command, tool list, wants_json ──────────────────


def test_coerce_json_value_parses_and_rejects():
    assert cli_api._coerce_json_value("[1, 2]", "triples") == [1, 2]
    with pytest.raises(cli_api._UsageError):
        cli_api._coerce_json_value("not json", "triples")


def test_args_for_tool_drops_none_and_keeps_set(monkeypatch, config, kg):
    import argparse

    _patch_mcp_server(monkeypatch, config, kg)
    ns = argparse.Namespace(wing="wing_x", room=None, limit=10, offset=None)
    args = cli_api._args_for_tool("mempalace_list_drawers", ns)
    assert args == {"wing": "wing_x", "limit": 10}


def test_args_for_tool_reads_stdin_for_string(monkeypatch, config, kg):
    import argparse

    _patch_mcp_server(monkeypatch, config, kg)
    monkeypatch.setattr("sys.stdin", io.StringIO("from stdin"))
    ns = argparse.Namespace(wing="w", room="r", content="-", source_file=None, added_by=None)
    args = cli_api._args_for_tool("mempalace_add_drawer", ns)
    assert args["content"] == "from stdin"


def test_run_command_maps_args_and_dispatches(monkeypatch, config, kg):
    import argparse

    _patch_mcp_server(monkeypatch, config, kg)
    captured = {}

    def fake(tool, arguments, pretty=False):
        captured.update(tool=tool, arguments=arguments, pretty=pretty)
        return 0

    monkeypatch.setattr(cli_api, "_run_tool", fake)
    ns = argparse.Namespace(
        _api_tool="mempalace_list_drawers", pretty=True, wing="w", room=None, limit=10, offset=None
    )
    rc = cli_api.run_command(ns)
    assert rc == 0
    assert captured["tool"] == "mempalace_list_drawers"
    assert captured["arguments"] == {"wing": "w", "limit": 10}
    assert captured["pretty"] is True


def test_build_tool_list_excludes_reconnect():
    tools = cli_api.build_tool_list()
    names = {t["tool"] for t in tools}
    assert "mempalace_get_drawer" in names
    assert "mempalace_reconnect" not in names
    sample = next(t for t in tools if t["tool"] == "mempalace_get_drawer")
    assert sample["command"] == "get-drawer"
    assert "input_schema" in sample


def test_wants_json_flag_and_env(monkeypatch):
    import argparse

    assert cli_api.wants_json(argparse.Namespace(json=True)) is True
    assert cli_api.wants_json(argparse.Namespace(json=False)) is False
    monkeypatch.setenv("MEMP_JSON", "1")
    assert cli_api.wants_json(argparse.Namespace(json=False)) is True


# ── Task 4: argparse builders + collider routing ────────────────────────────


def _build_api_parser():
    """A parser with only the api subcommands registered (no core commands)."""
    import argparse

    parser = argparse.ArgumentParser(prog="memp")
    sub = parser.add_subparsers(dest="command")
    cli_api.register_flat(sub, existing_names=set())
    return parser, sub


def test_register_flat_creates_subcommand_per_tool():
    from mempalace.mcp_server import TOOLS

    _, sub = _build_api_parser()
    expected = {cli_api.tool_command_name(n) for n in TOOLS if n not in cli_api.EXCLUDED}
    expected.add("list-tools")
    assert expected.issubset(set(sub.choices))
    assert "reconnect" not in sub.choices  # excluded


def test_register_flat_skips_existing_names():
    import argparse

    parser = argparse.ArgumentParser(prog="memp")
    sub = parser.add_subparsers(dest="command")
    cli_api.register_flat(sub, existing_names={"search", "status", "sync", "mine"})
    assert "search" not in sub.choices  # collider not re-registered
    assert "get-drawer" in sub.choices


def test_generated_command_parses_schema_flags():
    parser, _ = _build_api_parser()
    ns = parser.parse_args(["get-drawer", "--drawer-id", "abc123"])
    assert ns._api_tool == "mempalace_get_drawer"
    assert ns.drawer_id == "abc123"


def test_add_json_flags_lets_collider_accept_json():
    import argparse

    parser = argparse.ArgumentParser(prog="memp")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("search").add_argument("query")
    cli_api.add_json_flags(sub, {"search"})
    ns = parser.parse_args(["search", "q", "--json"])
    assert ns.json is True


def test_run_collider_search_maps_results_to_limit(monkeypatch):
    import argparse

    captured = {}

    def fake_run_tool(tool_name, arguments, pretty=False):
        captured["tool"] = tool_name
        captured["args"] = arguments
        return 0

    monkeypatch.setattr(cli_api, "_run_tool", fake_run_tool)
    ns = argparse.Namespace(
        command="search", query="hello", wing="w", room=None, results=7, json=True, pretty=False
    )
    rc = cli_api.run_collider(ns)
    assert rc == 0
    assert captured["tool"] == "mempalace_search"
    assert captured["args"] == {"query": "hello", "wing": "w", "limit": 7}


# ── Task 5: cli.py integration (lazy pre-scan + dispatch routing) ────────────


def _run_cli(monkeypatch, argv):
    monkeypatch.setattr("sys.argv", ["memp"] + argv)
    from mempalace import cli

    try:
        cli.main()
        return 0
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 0


def test_cli_routes_unknown_command_to_run_command(monkeypatch):
    from mempalace import cli_api

    captured = {}

    def fake_run_command(args):
        captured["tool"] = args._api_tool
        captured["drawer_id"] = getattr(args, "drawer_id", None)
        return 0

    monkeypatch.setattr(cli_api, "run_command", fake_run_command)
    rc = _run_cli(monkeypatch, ["get-drawer", "--drawer-id", "X"])
    assert rc == 0
    assert captured["tool"] == "mempalace_get_drawer"
    assert captured["drawer_id"] == "X"


def test_cli_routes_list_tools(monkeypatch):
    from mempalace import cli_api

    called = {}
    monkeypatch.setattr(
        cli_api, "print_tool_list", lambda pretty=False: (called.setdefault("yes", True), 0)[1]
    )
    rc = _run_cli(monkeypatch, ["list-tools"])
    assert rc == 0 and called.get("yes")


def test_cli_collider_json_routes_to_run_collider(monkeypatch):
    from mempalace import cli, cli_api

    routed = {}
    monkeypatch.setattr(
        cli_api, "run_collider", lambda args: (routed.setdefault("json", True), 0)[1]
    )
    monkeypatch.setattr(cli, "cmd_search", lambda args: routed.setdefault("human", True))
    _run_cli(monkeypatch, ["search", "q", "--json"])
    assert routed.get("json") and not routed.get("human")


def test_cli_search_without_json_calls_upstream(monkeypatch):
    from mempalace import cli, cli_api

    routed = {}
    monkeypatch.setattr(
        cli_api, "run_collider", lambda args: (routed.setdefault("json", True), 0)[1]
    )
    monkeypatch.setattr(cli, "cmd_search", lambda args: routed.setdefault("human", True))
    _run_cli(monkeypatch, ["search", "q"])
    assert routed.get("human") and not routed.get("json")


def test_cli_core_command_does_not_import_mcp_server(monkeypatch):
    import sys as _sys

    from mempalace import cli

    monkeypatch.delitem(_sys.modules, "mempalace.mcp_server", raising=False)
    monkeypatch.delitem(_sys.modules, "mempalace.cli_api", raising=False)
    monkeypatch.setattr(cli, "cmd_search", lambda args: None)
    monkeypatch.setattr("sys.argv", ["memp", "search", "anything"])
    try:
        cli.main()
    except SystemExit:
        pass
    assert "mempalace.mcp_server" not in _sys.modules
