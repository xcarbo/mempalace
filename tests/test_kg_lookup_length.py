"""KG tools that only look up a stored value must not cap its length.

Upstream #2911 (d15d0580). The live graph holds entities longer than
MAX_NAME_LENGTH (written before the cap, or by paths that skip it). With the
cap on lookups, ``memp kg-query`` / ``kg-invalidate`` / ``kg-supersede`` /
``kg-timeline`` refused them, so those facts could be neither read nor ended
through the tools. New names are still capped.
"""

from mempalace.config import MAX_NAME_LENGTH


def _patch_mcp_server(monkeypatch, config, kg):
    from mempalace import mcp_server

    monkeypatch.setattr(mcp_server, "_config", config)
    monkeypatch.setattr(mcp_server, "_get_kg", lambda *a, **kw: kg)


LONG = "x" * (MAX_NAME_LENGTH + 2)


def _notes(kg, as_of):
    return [
        f["object"]
        for f in kg.query_entity("Bot", as_of=as_of, direction="outgoing")
        if f["predicate"] == "notes"
    ]


def test_invalidate_long_object(monkeypatch, config, palace_path, kg):
    _patch_mcp_server(monkeypatch, config, kg)
    from mempalace.mcp_server import tool_kg_invalidate

    kg.add_triple("Bot", "notes", LONG, valid_from="2026-05-01")
    result = tool_kg_invalidate(subject="Bot", predicate="notes", object=LONG, ended="2026-06-01")
    assert result["success"] is True, result
    assert _notes(kg, "2026-07-01") == []


def test_supersede_long_old_object(monkeypatch, config, palace_path, kg):
    _patch_mcp_server(monkeypatch, config, kg)
    from mempalace.mcp_server import tool_kg_supersede

    kg.add_triple("Bot", "notes", LONG, valid_from="2026-05-01")
    result = tool_kg_supersede(
        subject="Bot", predicate="notes", old_object=LONG, new_object="short", at="2026-06-02"
    )
    assert result["success"] is True, result
    assert _notes(kg, "2026-06-02") == ["short"]
    assert _notes(kg, "2026-05-15") == [LONG]


def test_query_long_entity(monkeypatch, config, palace_path, kg):
    _patch_mcp_server(monkeypatch, config, kg)
    from mempalace.mcp_server import tool_kg_query

    kg.add_triple("Bot", "notes", LONG, valid_from="2026-05-01")
    result = tool_kg_query(entity=LONG)
    assert "error" not in result, result
    assert result["count"] == 1


def test_timeline_long_entity(monkeypatch, config, palace_path, kg):
    _patch_mcp_server(monkeypatch, config, kg)
    from mempalace.mcp_server import tool_kg_timeline

    kg.add_triple("Bot", "notes", LONG, valid_from="2026-05-01")
    result = tool_kg_timeline(entity=LONG)
    assert "error" not in result, result
    assert len(result["timeline"]) == 1


def test_add_still_caps_new_names(monkeypatch, config, palace_path, kg):
    _patch_mcp_server(monkeypatch, config, kg)
    from mempalace.mcp_server import tool_kg_add, tool_kg_supersede

    result = tool_kg_add(subject="Bot", predicate="notes", object=LONG)
    assert result["success"] is False
    assert "exceeds maximum length" in result["error"]

    kg.add_triple("Bot", "notes", "old", valid_from="2026-05-01")
    result = tool_kg_supersede(subject="Bot", predicate="notes", old_object="old", new_object=LONG)
    assert result["success"] is False
    assert "exceeds maximum length" in result["error"]


def test_add_still_refuses_bare_prefix(monkeypatch, config, palace_path, kg):
    """Our fork's bare-prefix guard stays on the write path."""
    _patch_mcp_server(monkeypatch, config, kg)
    from mempalace.mcp_server import tool_kg_add

    result = tool_kg_add(subject="action:", predicate="belongs_to", object="wing_x")
    assert result["success"] is False


def test_lookup_still_refuses_empty_and_nul(monkeypatch, config, palace_path, kg):
    _patch_mcp_server(monkeypatch, config, kg)
    from mempalace.mcp_server import tool_kg_query

    assert "error" in tool_kg_query(entity="   ")
    assert "error" in tool_kg_query(entity="a\x00b")
