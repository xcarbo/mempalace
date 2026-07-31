"""The sqlite fast path must agree with the canonical chromadb readers.

``fastread`` bypasses chromadb entirely to answer `memp get-drawer`, so its
only justification is that it returns what the chromadb path would have
returned. These tests build a real palace through the normal write path, then
assert both readers agree on every value — and that the fast path declines
(rather than guessing) whenever it meets something it does not understand.
"""

import json
import sqlite3

import pytest

from mempalace import fastread


def _payloads(palace, collection, drawer_id):
    """Return ``(fast, slow)`` payloads for one drawer id."""
    import mempalace.mcp_server as m

    fast = fastread.get_drawer(palace, collection, drawer_id)
    col = m._get_collection()
    record = m._logical_drawer_record(col, drawer_id)
    slow = m._drawer_payload(record) if record is not None else None
    return fast, slow


def _assert_agrees(fast, slow):
    """Equal on every value; key order inside metadata is explicitly not part
    of the contract (see the module docstring)."""
    assert fast is not None, "fast path declined a drawer the slow path found"
    assert slow is not None
    assert fast["content"] == slow["content"]
    assert fast["content_sha256"] == slow["content_sha256"]
    assert fast["wing"] == slow["wing"]
    assert fast["room"] == slow["room"]
    assert fast["metadata"] == slow["metadata"]
    assert fast.get("chunks") == slow.get("chunks")
    assert fast.get("chunk_ids") == slow.get("chunk_ids")
    assert json.loads(json.dumps(fast)) == json.loads(json.dumps(slow))


@pytest.fixture
def seeded(monkeypatch, config, palace_path):
    """A palace holding one short drawer and one long (chunked) drawer.

    Written through ``tool_add_drawer`` rather than hand-built rows, so the
    fast path is read against exactly what the real write path produces —
    including how chunk ids and ``parent_drawer_id`` are actually stamped.
    """
    import mempalace.mcp_server as m

    monkeypatch.setattr(m, "_config", config)

    short = m.tool_add_drawer(wing="ops", room="notes", content="a short drawer that won't chunk")
    long_content = "\n".join(f"paragraph {i} of a drawer long enough to chunk" for i in range(200))
    long = m.tool_add_drawer(wing="ops", room="notes", content=long_content)
    assert "drawer_id" in short and "drawer_id" in long, (short, long)

    yield {
        "palace": palace_path,
        "collection": config.collection_name,
        "short_id": short["drawer_id"],
        "long_id": long["drawer_id"],
        "long_content": long_content,
    }


def test_fast_path_matches_slow_path_for_a_single_drawer(seeded):
    fast, slow = _payloads(seeded["palace"], seeded["collection"], seeded["short_id"])
    _assert_agrees(fast, slow)
    assert "chunks" not in fast


def test_fast_path_matches_slow_path_for_a_chunked_drawer(seeded):
    fast, slow = _payloads(seeded["palace"], seeded["collection"], seeded["long_id"])
    _assert_agrees(fast, slow)
    assert fast["chunks"] > 1, "fixture did not actually chunk; the test proves nothing"
    assert fast["content"] == seeded["long_content"]
    assert fast["chunk_ids"] == sorted(fast["chunk_ids"])


def test_fast_path_stitches_chunks_in_index_order(seeded):
    """Reassembly must follow chunk_index, not whatever order sqlite returns."""
    fast = fastread.get_drawer(seeded["palace"], seeded["collection"], seeded["long_id"])
    assert fast["content"] == seeded["long_content"]
    assert fast["metadata"]["chunk_index"] == 0, "metadata must come from the first chunk"


def test_fast_path_declines_a_missing_drawer(seeded):
    """Declining, not reporting a miss: only the canonical path may say 'not found'."""
    assert fastread.get_drawer(seeded["palace"], seeded["collection"], "drawer_nope") is None


def test_fast_path_declines_an_unknown_collection(seeded):
    assert fastread.get_drawer(seeded["palace"], "no_such_collection", seeded["short_id"]) is None


def test_fast_path_declines_a_missing_palace(tmp_path):
    assert fastread.get_drawer(str(tmp_path / "nothing"), "mempalace_drawers", "x") is None


def test_fast_path_does_not_leak_the_reserved_document_key(seeded):
    fast = fastread.get_drawer(seeded["palace"], seeded["collection"], seeded["short_id"])
    assert "chroma:document" not in fast["metadata"]
    assert fast["content"]


def test_fast_path_scopes_to_the_requested_collection(seeded):
    """Ids are unique per collection only; a cross-collection read must not hit."""
    conn = sqlite3.connect(f"{seeded['palace']}/chroma.sqlite3")
    try:
        others = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM collections WHERE name <> ?", (seeded["collection"],)
            )
        ]
    finally:
        conn.close()
    for name in others:
        assert fastread.get_drawer(seeded["palace"], name, seeded["short_id"]) is None


def test_fast_path_declines_array_metadata(seeded, monkeypatch):
    """List-valued metadata would be silently dropped, so bail to the slow path."""
    monkeypatch.setattr(fastread, "_has_array_metadata", lambda *a, **k: True)
    assert fastread.get_drawer(seeded["palace"], seeded["collection"], seeded["short_id"]) is None


def test_fast_path_reads_a_cleanly_closed_wal_palace(seeded):
    """A `mode=ro` open of a WAL db with no -shm fails; the reader must cope.

    This is the normal resting state of the palace between writes, so a reader
    that only tried `mode=ro` would decline almost every real call.
    """
    import os

    for suffix in ("-shm", "-wal"):
        path = f"{seeded['palace']}/chroma.sqlite3{suffix}"
        if os.path.exists(path):
            os.remove(path)
    fast = fastread.get_drawer(seeded["palace"], seeded["collection"], seeded["short_id"])
    assert fast is not None
    assert fast["content"]


# ── CLI shape gate ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "argv",
    [
        ["search", "--query", "x"],  # not get-drawer
        ["get-drawer"],  # no id
        ["get-drawer", "--drawer-id"],  # dangling flag
        ["--palace", "/tmp/p", "get-drawer", "--drawer-id", "d"],  # global option
        ["get-drawer", "--drawer-id", "d", "--unknown-flag"],  # unrecognised flag
    ],
)
def test_cli_fast_path_declines_shapes_it_does_not_own(argv):
    assert fastread.try_cli_fast_path(argv) is None


@pytest.fixture
def cli_seeded(seeded, config, monkeypatch):
    """``try_cli_fast_path`` builds its own config, so point that at the temp palace."""
    import mempalace.config

    monkeypatch.setattr(mempalace.config, "MempalaceConfig", lambda *a, **k: config)
    return seeded


def test_cli_fast_path_emits_the_same_json_the_api_path_would(cli_seeded, capsys):
    rc = fastread.try_cli_fast_path(["get-drawer", "--drawer-id", cli_seeded["short_id"]])
    assert rc == 0
    emitted = json.loads(capsys.readouterr().out)
    _, slow = _payloads(cli_seeded["palace"], cli_seeded["collection"], cli_seeded["short_id"])
    assert emitted == slow


def test_cli_fast_path_honours_pretty(cli_seeded, capsys):
    rc = fastread.try_cli_fast_path(
        ["get-drawer", "--drawer-id", cli_seeded["short_id"], "--pretty"]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "\n  " in out, "--pretty must indent"
    assert json.loads(out)["drawer_id"] == cli_seeded["short_id"]
