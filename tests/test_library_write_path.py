"""A library import must not inherit the MCP server's read-only stance.

`mcp_server._parse_args()` runs at IMPORT time against the importing process's
`sys.argv`. Nothing in an agent wrapper passes `--transport`, so it defaulted to
"stdio" and every library caller looked like an un-promoted stdio MCP server —
which opens sqlite_exact read-only.

The result was an eight-day silent outage (2026-08-16 to 08-24): night-watcher's
nightly digest, the mempalace-monitor briefing and palace-gnome's briefing and
verdicts all failed with "attempt to write a readonly database", every night,
while `memp add-drawer` from a shell kept working and hid it.
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys

import pytest


# Written flush-left on purpose: an indented heredoc has to survive dedent AND
# the formatter, and it did not.
_CHILD_SCRIPT = """
import json, sys
import mempalace.mcp_server as m

if m._SERVING_AS_MCP:
    print(json.dumps({"error": "import armed the MCP server guard"}))
    sys.exit(1)

res = m.tool_add_drawer(
    wing="probe", room="events", content="# Probe -- a library caller writing a drawer."
)
if not res.get("success"):
    print(json.dumps(res))
    sys.exit(1)

got = m.tool_get_drawer(drawer_id=res["drawer_id"])
print(json.dumps({"ok": "library caller writing" in (got.get("content") or "")}))
"""


@pytest.fixture
def mcp():
    return importlib.import_module("mempalace.mcp_server")


def test_import_does_not_arm_the_server_guard(mcp):
    # The flag is the whole fix: importing this module is not serving MCP.
    assert mcp._SERVING_AS_MCP is False, (
        "importing mempalace.mcp_server armed the MCP server's read-only guard; "
        "every library writer (the gnome wrappers, the agent tools) will fail "
        "with 'attempt to write a readonly database'"
    )


def test_main_arms_it(mcp, monkeypatch):
    # And the guard must still exist for a real server, or the thing it protects
    # against (a stdio server opening sqlite_exact through the schema-init path
    # while a daemon owns the palace) comes back.
    monkeypatch.setattr(mcp, "_SERVING_AS_MCP", False)
    monkeypatch.setattr(mcp, "_install_shutdown_signal_handlers", lambda: None)
    monkeypatch.setattr(mcp, "_run_stdio_loop", lambda: None)
    monkeypatch.setattr(mcp, "_run_http_loop", lambda: None)
    monkeypatch.setattr(mcp._args, "transport", "stdio", raising=False)

    mcp.main()
    assert mcp._SERVING_AS_MCP is True


def test_library_caller_can_write(tmp_path):
    """End to end: import, add a drawer, read it back. No MCP anywhere.

    Runs in a SUBPROCESS on purpose. The palace path and backend are resolved at
    module import, so exercising this in-process needs an importlib.reload, and
    that rebinds module state the rest of the suite is already holding — it left
    24 unrelated tests pointing at a deleted temp palace. A child process is the
    honest way to ask "what does a fresh library caller get".
    """
    palace = tmp_path / "palace"
    script = _CHILD_SCRIPT
    env = {
        **os.environ,
        "MEMPALACE_PALACE_PATH": str(palace),
        "MEMPALACE_BACKEND": "sqlite_exact",
        "MEMPALACE_RETRIEVAL_LOG": "0",
    }
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, env=env, timeout=300
    )
    assert proc.returncode == 0, (
        f"a plain library caller could not write to the palace: "
        f"{proc.stdout.strip()[-400:]} {proc.stderr.strip()[-400:]}"
    )
    # Scan BOTH streams. mempalace routes its own prints to stderr because for a
    # real MCP server stdout is the JSON-RPC channel and must stay clean — so a
    # child that imports it finds its output there, not on stdout. Take the last
    # line that parses as JSON, past the lock-GC and embedder-init noise.
    payloads = []
    for line in (proc.stdout + "\n" + proc.stderr).splitlines():
        try:
            payloads.append(json.loads(line))
        except ValueError:
            continue
    assert payloads, f"child printed no JSON result. err={proc.stderr[-500:]!r}"
    assert payloads[-1].get("ok") is True, payloads[-1]
