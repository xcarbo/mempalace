"""Contract tests for the Codex plugin marketplace metadata."""

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
MARKETPLACE_PATH = REPO_ROOT / ".agents" / "plugins" / "marketplace.json"
MANIFEST_PATH = REPO_ROOT / ".codex-plugin" / "plugin.json"
MCP_PATH = REPO_ROOT / ".mcp.json"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_marketplace_entry_uses_supported_codex_schema():
    marketplace = _read_json(MARKETPLACE_PATH)
    plugin = marketplace["plugins"][0]

    assert plugin["name"] == "mempalace"
    assert plugin["source"] == {"source": "local", "path": "./"}
    assert plugin["policy"] == {
        "installation": "AVAILABLE",
        "authentication": "ON_INSTALL",
    }


def test_plugin_manifest_references_supported_components():
    manifest = _read_json(MANIFEST_PATH)

    assert manifest["mcpServers"] == "./.mcp.json"
    assert "hooks" not in manifest


def test_mcp_config_is_absent_fork_delta():
    """FORK DELTA — inverted from upstream. ``.mcp.json`` must NOT exist here.

    Upstream ``2f72c91`` ships a project-scoped ``.mcp.json`` at the repo root
    registering the ``mempalace-mcp`` server, and upstream's version of this
    test asserts it is present. This machine retired the MemPalace MCP server
    on 2026-06-16 — everything goes through the ``memp`` CLI, which dispatches
    the same handlers byte-identically — so the file was removed in ``3119884``.

    The file is dangerous by existing rather than by being called: ``mempalace-mcp``
    resolves on PATH via the pyenv shim, so every Claude Code session opened in this
    repo silently regains the full ``mcp__mempalace__*`` WRITE surface (add_drawer,
    update_drawer, mine, kg_add) against the live palace. Three server processes
    were running when this was first found.

    It is also a theirs-only file that never conflicts, so a conflict-driven merge
    review gives it zero attention. This assertion is the loud guard: if a future
    upstream merge resurrects it, this test fails instead of the suite going quietly
    green with the MCP surface re-armed. See tests/test_fork_deltas.py.
    """
    assert not MCP_PATH.exists(), (
        f"{MCP_PATH} is back. An upstream merge resurrected the retired MemPalace "
        "MCP server — every Claude Code session in this repo now has the full "
        "mcp__mempalace__* write surface against the live palace. Delete the file "
        "(see 3119884); do NOT re-point this assertion at upstream's."
    )
