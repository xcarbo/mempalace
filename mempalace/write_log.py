"""Local write log — the flight recorder for the WRITE path.

Appends one JSON line per write attempt (add/update/delete drawer, and every
failed palace-lock acquire) to ``~/.mempalace/telemetry/writes-YYYY-MM.jsonl``.

Why this exists
---------------
:mod:`mempalace.retrieval_log` records reads. Nothing recorded writes. Save
failures did get written down — as free text in ``~/.mempalace/hook_state/hook.log``
— but that file is unstructured, unbounded, and its timestamps carry no date, so
"did anything fail to save yesterday?" was not an answerable question. A failure
that is logged but unanswerable is, operationally, a failure that is invisible.

This log makes the write path answerable. Each record is a fact with a real UTC
timestamp: what was attempted, whether it succeeded, how long it took, and when
it failed, why — as a stable ``error_class`` rather than a prose message that
regexes must chase.

Same contract as the retrieval log
----------------------------------
Strictly local; nothing here phones home. Month-stamped filenames rotate
naturally. Fail-soft by construction: a logging failure must never break a
write, so every failure mode degrades to a debug line. Appends are single
``write()`` calls of one line under the POSIX pipe-buffer size, so concurrent
writers (CLI, MCP, HTTP API, hooks) interleave whole records.

Opt out with ``MEMPALACE_WRITE_LOG=0`` (or ``false``/``off``/``no``).

Error classes
-------------
Stable identifiers, safe to alert on. Add to this list rather than renaming:

``lock_contention``   another process held the palace write lock
``cas_conflict``      ``if_unchanged`` digest did not match; write refused
``not_found``         target drawer does not exist
``validation``        input rejected (bad wing/room/content)
``backend``           storage backend raised (chroma, sqlite, FTS5)
``unknown``           anything else; the message is kept verbatim
"""

import json
import logging
import os
from datetime import datetime, timezone

logger = logging.getLogger("mempalace_mcp")

DISABLE_ENV = "MEMPALACE_WRITE_LOG"

_FALSEY = ("0", "false", "off", "no")

# Stable error-class identifiers. Alerting keys off these, so they must not be
# renamed once shipped.
ERR_LOCK_CONTENTION = "lock_contention"
ERR_CAS_CONFLICT = "cas_conflict"
ERR_NOT_FOUND = "not_found"
ERR_VALIDATION = "validation"
ERR_BACKEND = "backend"
ERR_UNKNOWN = "unknown"


def write_log_enabled() -> bool:
    """True unless the user opted out via MEMPALACE_WRITE_LOG."""
    return os.environ.get(DISABLE_ENV, "1").strip().lower() not in _FALSEY


def write_log_dir() -> str:
    """Directory holding the month-stamped JSONL logs."""
    return os.path.join(os.path.expanduser("~"), ".mempalace", "telemetry")


def write_log_path(now: "datetime | None" = None) -> str:
    """Current month's log file path."""
    now = now or datetime.now(timezone.utc)
    return os.path.join(write_log_dir(), f"writes-{now:%Y-%m}.jsonl")


def classify_error(exc: BaseException) -> str:
    """Map an exception to a stable error_class.

    Deliberately conservative: an unrecognised exception is ``unknown`` with
    its message preserved, never silently folded into a neighbouring class.
    """
    name = type(exc).__name__
    msg = str(exc).lower()
    if name == "MineAlreadyRunning" or "is held by" in msg:
        return ERR_LOCK_CONTENTION
    if "if_unchanged" in msg or "if-unchanged" in msg or "conflict" in msg:
        return ERR_CAS_CONFLICT
    if "not found" in msg or "no such drawer" in msg:
        return ERR_NOT_FOUND
    if name in ("ValueError", "TypeError") or "invalid" in msg or "sanitiz" in msg:
        return ERR_VALIDATION
    if "malformed" in msg or "fts5" in msg or "database" in msg or "chroma" in msg:
        return ERR_BACKEND
    return ERR_UNKNOWN


def log_write(op: str, *, ok: bool, **fields) -> None:
    """Append one write event; never raises.

    ``op`` names the operation ("add_drawer", "update_drawer", "delete_drawer",
    "lock_acquire"); ``ok`` is the outcome; ``fields`` are flat
    JSON-serializable details. Non-serializable values are stringified rather
    than dropped.
    """
    if not write_log_enabled():
        return
    try:
        now = datetime.now(timezone.utc)
        record = {
            "ts": now.isoformat(timespec="seconds"),
            "op": op,
            "ok": bool(ok),
            "pid": os.getpid(),
        }
        record.update(fields)
        os.makedirs(write_log_dir(), exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, default=str)
        with open(write_log_path(now), "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        logger.debug("write log append failed (non-fatal)", exc_info=True)


def log_write_failure(op: str, exc: BaseException, **fields) -> None:
    """Convenience: record a failed write with a classified error."""
    log_write(
        op,
        ok=False,
        error_class=classify_error(exc),
        error=str(exc)[:500],
        **fields,
    )
