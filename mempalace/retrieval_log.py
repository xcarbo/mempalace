"""Local retrieval log — the palace's flight recorder.

Appends one JSON line per read operation (search, get_drawer) to
``~/.mempalace/telemetry/retrieval-YYYY-MM.jsonl``. The log is the raw
signal for retrieval observability: which drawers actually get surfaced
and opened, powering recall regression checks and future salience-based
curation. Without it the palace is blind to its own usage.

Strictly local — this file never leaves the machine and nothing here
phones home (the project's no-telemetry principle refers to external
reporting; this is the user's own data about their own palace, stored
beside it). Month-stamped filenames give natural rotation; old months
can be archived or deleted freely.

Fail-soft by contract: a logging failure must never break a read path,
so every failure mode (unwritable dir, full disk, bad field) degrades to
a debug log line. Appends are single ``write()`` calls of one line under
the POSIX pipe-buffer size, so concurrent writers (CLI, MCP, HTTP API)
interleave whole records.

Opt out with ``MEMPALACE_RETRIEVAL_LOG=0`` (or ``false``/``off``/``no``).
"""

import json
import logging
import os
from datetime import datetime, timezone

logger = logging.getLogger("mempalace_mcp")

DISABLE_ENV = "MEMPALACE_RETRIEVAL_LOG"

_FALSEY = ("0", "false", "off", "no")


def retrieval_log_enabled() -> bool:
    """True unless the user opted out via MEMPALACE_RETRIEVAL_LOG."""
    return os.environ.get(DISABLE_ENV, "1").strip().lower() not in _FALSEY


def retrieval_log_dir() -> str:
    """Directory holding the month-stamped JSONL logs."""
    return os.path.join(os.path.expanduser("~"), ".mempalace", "telemetry")


def retrieval_log_path(now: "datetime | None" = None) -> str:
    """Current month's log file path."""
    now = now or datetime.now(timezone.utc)
    return os.path.join(retrieval_log_dir(), f"retrieval-{now:%Y-%m}.jsonl")


def log_retrieval(tool: str, **fields) -> None:
    """Append one retrieval event; never raises.

    ``tool`` names the read operation ("search", "get_drawer"); ``fields``
    are flat JSON-serializable details. Non-serializable values are
    stringified rather than dropped.
    """
    if not retrieval_log_enabled():
        return
    try:
        now = datetime.now(timezone.utc)
        record = {
            "ts": now.isoformat(timespec="seconds"),
            "tool": tool,
            "pid": os.getpid(),
        }
        record.update(fields)
        os.makedirs(retrieval_log_dir(), exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, default=str)
        with open(retrieval_log_path(now), "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        logger.debug("retrieval log append failed (non-fatal)", exc_info=True)
