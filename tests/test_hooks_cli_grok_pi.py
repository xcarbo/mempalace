"""Grok and pi harness support in the palace hooks.

The fixtures under ``tests/fixtures/{grok,pi}/`` are real session files with
every free-text string and id scrubbed; keys, nesting and enum-like values are
untouched, so they pin the on-disk shapes these harnesses actually write.

Grok fires the Claude Code hooks through its compat layer. Its payload carries
``session_id`` and ``transcript_path``, but the path names ``updates.jsonl``
(the ACP event stream) rather than the conversation, so every Grok Stop counted
0 exchanges and SessionEnd mined raw protocol JSON.
"""

import json
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from mempalace.hooks_cli import (
    SUPPORTED_HARNESSES,
    _count_human_messages,
    _diary_agent_for_harness,
    _extract_recent_messages,
    _parse_harness_input,
    _wing_from_transcript_path,
    hook_precompact,
    hook_session_end,
    hook_stop,
)
from mempalace.normalize import _try_grok_jsonl, _try_pi_jsonl, normalize

FIXTURES = Path(__file__).parent / "fixtures"
GROK_CHAT = FIXTURES / "grok" / "chat_history.jsonl"
GROK_UPDATES = FIXTURES / "grok" / "updates.jsonl"
PI_SESSION = FIXTURES / "pi" / "session.jsonl"

SESSION_ID = "01a04e92-4f54-7e43-9408-a281cff55845"


def _grok_session_dir(root: Path, cwd: str = "/Volumes/xData/codeXD/cc") -> Path:
    """Lay the fixtures out the way Grok does on disk."""
    from urllib.parse import quote

    session = root / ".grok" / "sessions" / quote(cwd, safe="") / SESSION_ID
    session.mkdir(parents=True)
    shutil.copy(GROK_CHAT, session / "chat_history.jsonl")
    shutil.copy(GROK_UPDATES, session / "updates.jsonl")
    return session


def _compat_payload(session: Path, **extra) -> dict:
    """The payload Grok's Claude-compat layer sends: snake_case Claude keys
    alongside Grok's own camelCase envelope."""
    return {
        "session_id": SESSION_ID,
        "transcript_path": str(session / "updates.jsonl"),
        "hook_event_name": "Stop",
        "hookEventName": "stop",
        "sessionId": SESSION_ID,
        "cwd": "/Volumes/xData/codeXD/cc",
        "stopHookActive": False,
        **extra,
    }


# --- harness registration -------------------------------------------------


def test_grok_and_pi_are_supported_harnesses():
    assert {"grok", "pi"} <= SUPPORTED_HARNESSES


@pytest.mark.parametrize("harness", ["grok", "pi"])
def test_diary_agent_name_is_the_harness_name(harness):
    assert _diary_agent_for_harness(harness) == harness


# --- grok payload parsing -------------------------------------------------


def test_grok_swaps_event_stream_for_chat_history(tmp_path):
    session = _grok_session_dir(tmp_path)
    parsed = _parse_harness_input(_compat_payload(session), "grok")
    assert parsed["session_id"] == SESSION_ID
    assert parsed["transcript_path"] == str(session / "chat_history.jsonl")
    assert parsed["skip"] is False


def test_grok_native_camelcase_payload_derives_the_transcript(tmp_path):
    session = _grok_session_dir(tmp_path)
    native = {
        "hookEventName": "stop",
        "sessionId": SESSION_ID,
        "cwd": "/Volumes/xData/codeXD/cc",
        "stopHookActive": True,
    }
    with patch.dict("os.environ", {"GROK_HOME": str(tmp_path / ".grok")}):
        parsed = _parse_harness_input(native, "grok")
    assert parsed["session_id"] == SESSION_ID
    assert parsed["stop_hook_active"] is True
    assert parsed["transcript_path"] == str(session / "chat_history.jsonl")


def test_grok_subagent_payload_is_marked_skip(tmp_path):
    session = _grok_session_dir(tmp_path)
    parsed = _parse_harness_input(_compat_payload(session, subagentType="explore"), "grok")
    assert parsed["skip"] is True


def test_claude_code_payload_is_untouched_by_the_grok_swap(tmp_path):
    """A Claude transcript that happens to be named updates.jsonl stays put."""
    path = str(tmp_path / "updates.jsonl")
    parsed = _parse_harness_input({"session_id": "s", "transcript_path": path}, "claude-code")
    assert parsed["transcript_path"] == path
    assert parsed["skip"] is False


# --- counting and recent messages ----------------------------------------


def test_event_stream_counts_zero_which_is_the_bug():
    assert _count_human_messages(str(GROK_UPDATES)) == 0


def test_grok_chat_history_counts_only_typed_prompts():
    """5 ``type: user`` lines in the fixture: 1 preamble, 2 injected system
    reminders, 2 real prompts (the ones carrying ``prompt_index``)."""
    lines = [json.loads(line) for line in GROK_CHAT.read_text().splitlines()]
    assert sum(1 for entry in lines if entry.get("type") == "user") == 5
    assert _count_human_messages(str(GROK_CHAT)) == 2


def test_grok_recent_messages_are_the_typed_prompts():
    assert _extract_recent_messages(str(GROK_CHAT)) == ["human prompt 1", "human prompt 2"]


def test_pi_session_counts_and_extracts():
    assert _count_human_messages(str(PI_SESSION)) == 1
    assert _extract_recent_messages(str(PI_SESSION)) == ["human prompt 1"]


# --- wing derivation -------------------------------------------------------


def test_grok_wing_comes_from_the_urlencoded_cwd(tmp_path):
    session = _grok_session_dir(tmp_path, cwd="/Volumes/xData/codeXD/hunt-1")
    assert _wing_from_transcript_path(str(session / "chat_history.jsonl")) == "hunt-1"


def test_pi_wing_comes_from_the_session_header_cwd():
    assert _wing_from_transcript_path(str(PI_SESSION)) == "demo-project"


# --- normalize -------------------------------------------------------------


def test_try_grok_jsonl_keeps_prompts_and_replies_only():
    result = _try_grok_jsonl(GROK_CHAT.read_text())
    assert result is not None
    assert "> human prompt 1" in result
    assert "> human prompt 2" in result
    assert "assistant reply 1" in result
    assert "system-reminder" not in result
    assert "scrubbed" not in result


def test_try_grok_jsonl_rejects_the_event_stream():
    assert _try_grok_jsonl(GROK_UPDATES.read_text()) is None


def test_try_grok_jsonl_rejects_claude_code_and_pi():
    claude = "\n".join(
        json.dumps({"type": role, "message": {"role": role, "content": f"{role} text"}})
        for role in ("user", "assistant")
    )
    assert _try_grok_jsonl(claude) is None
    assert _try_grok_jsonl(PI_SESSION.read_text()) is None


def test_normalize_pipeline_picks_grok(tmp_path):
    target = tmp_path / "chat_history.jsonl"
    shutil.copy(GROK_CHAT, target)
    result = normalize(str(target))
    assert "> human prompt 1" in result
    assert '"type"' not in result


def test_pi_normalizer_already_covers_real_sessions():
    result = _try_pi_jsonl(PI_SESSION.read_text())
    assert result is not None
    assert "> human prompt 1" in result


# --- hooks end to end ------------------------------------------------------


def _run(hook_fn, data, harness, state_dir):
    import io
    from unittest.mock import PropertyMock

    buf = io.StringIO()
    with (
        patch("mempalace.hooks_cli._output", side_effect=lambda d: buf.write(json.dumps(d))),
        patch("mempalace.hooks_cli.STATE_DIR", state_dir),
        patch("mempalace.hooks_cli._palace_root_exists", return_value=True),
        patch("mempalace.hooks_cli.MempalaceConfig") as config,
    ):
        type(config.return_value).hooks_auto_save = PropertyMock(return_value=True)
        type(config.return_value).hooks_silent_save = PropertyMock(return_value=True)
        type(config.return_value).hook_desktop_toast = PropertyMock(return_value=False)
        hook_fn(data, harness)
    return json.loads(buf.getvalue() or "{}")


def test_grok_stop_logs_real_exchange_count(tmp_path):
    session = _grok_session_dir(tmp_path)
    state = tmp_path / "state"
    state.mkdir()
    assert _run(hook_stop, _compat_payload(session), "grok", state) == {}
    log = (state / "hook.log").read_text()
    assert f"Session {SESSION_ID}: 2 exchanges" in log


@pytest.mark.parametrize("hook_fn", [hook_stop, hook_session_end, hook_precompact])
def test_grok_subagent_payload_never_saves_or_mines(tmp_path, hook_fn):
    session = _grok_session_dir(tmp_path)
    state = tmp_path / "state"
    state.mkdir()
    with (
        patch("mempalace.hooks_cli._ingest_transcript") as ingest,
        patch("mempalace.hooks_cli._save_diary_direct") as save,
    ):
        result = _run(hook_fn, _compat_payload(session, subagentType="explore"), "grok", state)
    assert result == {}
    ingest.assert_not_called()
    save.assert_not_called()


def test_grok_session_end_ingests_chat_history_not_the_event_stream(tmp_path):
    session = _grok_session_dir(tmp_path)
    state = tmp_path / "state"
    state.mkdir()
    with (
        patch("mempalace.hooks_cli._ingest_transcript") as ingest,
        patch("mempalace.hooks_cli._save_diary_direct") as save,
        patch("mempalace.hooks_cli._maybe_auto_ingest"),
    ):
        _run(hook_session_end, _compat_payload(session), "grok", state)
    ingested = [call.args[0] for call in ingest.call_args_list]
    assert ingested == [str(session / "chat_history.jsonl")]
    assert save.call_args.kwargs["agent_name"] == "grok"
    assert save.call_args.kwargs["wing"] == "cc"
