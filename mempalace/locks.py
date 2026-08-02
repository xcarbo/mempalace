"""
locks.py — lock-file holder records, contention diagnostics, and residue GC.

The mine locks in ``palace.py`` (per-file ``mine_lock`` and per-palace
``mine_palace_lock``) rendezvous on files under ``~/.mempalace/locks``. This
module owns everything about those files that is *not* the locking itself:

* the holder record written into a lock file at acquire time (JSON with pid,
  ppid, argv, and an ISO acquire timestamp — so a later contender can say
  exactly who holds the lock and for how long),
* rich contention diagnostics (``describe_holder``) used in
  ``MineAlreadyRunning`` messages and the ``memp locks`` listing,
* the race-safe residue GC that removes lock files which are unheld AND
  (empty OR recorded-holder dead) — the 2026-07-10 outage left ~2,200 such
  0-byte residues behind,
* the data source for the ``memp locks`` CLI subcommand.

Kept free of any mempalace imports so ``memp locks`` stays fast (no chromadb
import) and ``palace.py`` can import it without cycles.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("mempalace_mcp")

# Byte 0 of every lock file is reserved as the OS lock sentinel: Windows
# msvcrt.locking locks exactly that byte, and reading it while locked blocks.
# The holder record therefore lives from byte 1 onward on all platforms.
LOCK_SENTINEL_BYTES = 1

# Cap the live `ps` command line embedded in diagnostics so a holder with a
# pathological argv cannot balloon the error message.
_PS_COMMAND_MAX_CHARS = 160

# Slack allowed when comparing a live process's elapsed runtime against how long
# the lock has been held (see ``pid_is_recycled``). Both numbers are truncated to
# whole seconds and can be read from slightly different clocks, so only a gap
# wider than this counts as proof that the PID was recycled.
_RECYCLED_PID_MARGIN_SECONDS = 60.0

ORPHAN_GUIDANCE = (
    "if this PID is an orphan (e.g. dead SSH parent), kill it — the flock releases on process exit."
)


def locks_dir() -> str:
    """Directory holding all mine lock files. Computed per call so tests
    that monkeypatch HOME see the redirected path."""
    return os.path.join(os.path.expanduser("~"), ".mempalace", "locks")


# ---------------------------------------------------------------------------
# Holder record (written at acquire time, read by contenders)
# ---------------------------------------------------------------------------


def write_holder_record(lock_file) -> None:
    """Record this process's identity in the lock-file body. Best-effort.

    Writes a JSON object from byte 1 onward; byte 0 is the lock sentinel and
    must not be touched after acquire (truncating it on Windows can interact
    badly with the active byte-range lock). A failure to write must never
    block lock acquisition — diagnostics are strictly best-effort.
    """
    try:
        record = {
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "argv": list(sys.argv[:6]),
            "acquired_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        body = json.dumps(record, ensure_ascii=False).encode("utf-8")
        lock_file.seek(LOCK_SENTINEL_BYTES)
        lock_file.truncate(LOCK_SENTINEL_BYTES + len(body))
        lock_file.write(body)
        lock_file.flush()
    except (OSError, UnicodeError, ValueError):
        pass


def parse_holder_record(text: str) -> Optional[dict]:
    """Parse a lock-file body into a holder record, or None if unreadable.

    Handles both the current JSON format and the legacy pre-2026-07 format
    (``"<pid> <argv...>"`` plain text). Legacy records carry no acquire
    timestamp — callers fall back to the lock file's mtime for age.
    """
    text = text.strip().lstrip("\x00").strip()
    if not text:
        return None
    if text.startswith("{"):
        try:
            record = json.loads(text)
        except ValueError:
            return None
        if isinstance(record, dict) and isinstance(record.get("pid"), int):
            return record
        return None
    parts = text.split(maxsplit=1)
    if parts and parts[0].isdigit():
        record: dict = {"pid": int(parts[0])}
        if len(parts) > 1 and parts[1].strip():
            record["argv"] = parts[1].strip().split()
        return record
    return None


def read_holder_record(lock_file) -> Optional[dict]:
    """Read the holder record from an open lock file, best-effort."""
    try:
        lock_file.seek(LOCK_SENTINEL_BYTES)
        content = lock_file.read()
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="replace")
    except OSError:
        return None
    return parse_holder_record(content)


# ---------------------------------------------------------------------------
# Process probes (no psutil — it is a dev-only dependency)
# ---------------------------------------------------------------------------


def pid_alive(pid) -> Optional[bool]:
    """Whether ``pid`` is a live process; None when unknowable."""
    if not isinstance(pid, int) or pid <= 0:
        return None
    if os.name == "nt":
        # os.kill(pid, 0) on Windows would TERMINATE the process (the signal
        # becomes the exit code) — probe via OpenProcess instead.
        try:
            import ctypes

            synchronize = 0x00100000
            handle = ctypes.windll.kernel32.OpenProcess(synchronize, False, pid)
            if handle:
                ctypes.windll.kernel32.CloseHandle(handle)
                return True
            return False
        except Exception:
            return None
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Signal not permitted, but the process exists.
        return True
    except OSError:
        return None


def pid_details(pid) -> dict:
    """Live facts about ``pid``: {'alive', 'ppid', 'command'} via ps."""
    details: dict = {"alive": pid_alive(pid), "ppid": None, "command": None}
    if details["alive"] is not True or os.name == "nt":
        return details
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "ppid=,command="],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return details
    line = result.stdout.strip().splitlines()[0].strip() if result.stdout.strip() else ""
    if result.returncode == 0 and line:
        parts = line.split(None, 1)
        if parts and parts[0].isdigit():
            details["ppid"] = int(parts[0])
        if len(parts) > 1:
            details["command"] = parts[1].strip()[:_PS_COMMAND_MAX_CHARS]
    return details


def pid_elapsed_seconds(pid) -> Optional[float]:
    """How long ``pid`` has been running, or None when unknowable.

    Reads ``ps -o etime=`` (``[[dd-]hh:]mm:ss``) rather than ``lstart``:
    elapsed time is locale-independent and needs no timezone parsing.
    """
    if pid_alive(pid) is not True or os.name == "nt":
        return None
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "etime="],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    raw = result.stdout.strip()
    if result.returncode != 0 or not raw:
        return None
    days = 0
    if "-" in raw:
        day_part, _, raw = raw.partition("-")
        try:
            days = int(day_part)
        except ValueError:
            return None
    parts = raw.split(":")
    if not 1 <= len(parts) <= 3:
        return None
    try:
        values = [int(p) for p in parts]
    except ValueError:
        return None
    while len(values) < 3:
        values.insert(0, 0)
    hours, minutes, seconds = values
    return float(days * 86400 + hours * 3600 + minutes * 60 + seconds)


def pid_is_recycled(pid, record: Optional[dict]) -> bool:
    """Whether a *live* ``pid`` provably is NOT the process that took the lock.

    A recycled PID is the one way a dead holder can read as alive: the OS
    hands the number to an unrelated process and a residual lock file starts
    naming a live stranger. The test is a start-time comparison — a process
    that has been running for LESS time than the lock has been held cannot be
    the one that acquired it.

    Deliberately one-directional and conservative. It returns True only on a
    positive contradiction; every unknown (legacy record with no
    ``acquired_at``, unreadable ``ps``, Windows) returns False, i.e. "treat as
    the real holder". A generous margin absorbs clock skew and the second-level
    truncation in both ``etime`` and the recorded timestamp — breaking a live
    lock is far worse than leaving a stale file behind.

    Note this never decides on its own that a lock may be broken: callers reach
    it only after a non-blocking flock probe has already proved the file unheld.
    """
    if not isinstance(record, dict) or not record.get("acquired_at"):
        return False
    if pid_alive(record.get("pid")) is not True:
        return False
    held = held_duration_seconds("", record)
    if held is None:
        return False
    elapsed = pid_elapsed_seconds(pid)
    if elapsed is None:
        return False
    return elapsed + _RECYCLED_PID_MARGIN_SECONDS < held


# ---------------------------------------------------------------------------
# Contention diagnostics
# ---------------------------------------------------------------------------


def _format_age(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h {minutes:02d}m"
    days, hours = divmod(hours, 24)
    return f"{days}d {hours}h"


def held_duration_seconds(lock_path: str, record: Optional[dict]) -> Optional[float]:
    """How long the lock has been held: recorded acquire time, else file mtime."""
    if record and record.get("acquired_at"):
        try:
            acquired = datetime.fromisoformat(record["acquired_at"])
            if acquired.tzinfo is None:
                acquired = acquired.replace(tzinfo=timezone.utc)
            return max(0.0, time.time() - acquired.timestamp())
        except (ValueError, TypeError, OverflowError, OSError):
            pass
    try:
        return max(0.0, time.time() - os.stat(lock_path).st_mtime)
    except OSError:
        return None


def describe_holder(lock_path: str, record: Optional[dict]) -> str:
    """Rich one-line holder diagnosis for lock-contention errors.

    Includes the recorded pid + argv, whether that pid is alive, its live
    command line and parent pid (via ps), and how long the lock has been
    held. Every field is best-effort; missing facts are simply omitted.
    """
    held_secs = held_duration_seconds(lock_path, record)
    if not record:
        base = "another writer (identity not recorded)"
    else:
        pid = record["pid"]
        base = f"PID {pid}"
        argv = record.get("argv")
        if argv:
            base += f" ({' '.join(str(a) for a in argv)})"
        details = pid_details(pid)
        if details["alive"] is True:
            base += ", alive"
            if details["command"]:
                base += f" (ps: {details['command']})"
            ppid = details["ppid"] if details["ppid"] is not None else record.get("ppid")
            if ppid is not None:
                base += f", parent PID {ppid}"
        elif details["alive"] is False:
            base += ", NOT alive (recorded holder is dead; the flock must be "
            base += "held by another process, or the record is stale)"
            if record.get("ppid") is not None:
                base += f", recorded parent PID {record['ppid']}"
        elif record.get("ppid") is not None:
            base += f", recorded parent PID {record['ppid']}"
    if held_secs is not None:
        base += f", lock held for {_format_age(held_secs)}"
        if record and record.get("acquired_at"):
            base += f" (acquired {record['acquired_at']})"
    return base


# ---------------------------------------------------------------------------
# Non-blocking probe primitives (mirror palace.py's byte-0 lock convention)
# ---------------------------------------------------------------------------


def _probe_lock(lock_file) -> bool:
    """Try a non-blocking exclusive lock; True means the file was unheld."""
    lock_file.seek(0)
    if os.name == "nt":
        import msvcrt

        try:
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False
    import fcntl

    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:
        return False


def _release_lock(lock_file) -> None:
    lock_file.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
        return
    import fcntl

    fcntl.flock(lock_file, fcntl.LOCK_UN)


def _lock_file_is_current(lock_file, lock_path: str) -> bool:
    """Whether ``lock_file`` is still the inode reached by ``lock_path``."""
    if os.name == "nt":
        return True
    try:
        path_stat = os.stat(lock_path)
        file_stat = os.fstat(lock_file.fileno())
    except OSError:
        return False
    return (path_stat.st_dev, path_stat.st_ino) == (file_stat.st_dev, file_stat.st_ino)


# ---------------------------------------------------------------------------
# Residue GC
# ---------------------------------------------------------------------------


def gc_stale_locks(lock_dir: Optional[str] = None) -> int:
    """Remove residual lock files that are unheld AND (empty OR dead-holder).

    Race-safe by construction: a file is only unlinked while this process
    holds a successful non-blocking flock on that same fd, and only after
    verifying the fd still matches the pathname's inode (waiters that raced
    us re-check currency and retry — see ``_mine_lock_file_is_current``).
    A lock held by any live process always fails the probe and is never
    touched. Returns the number of files removed; logs one summary line.
    """
    lock_dir = lock_dir or locks_dir()
    try:
        names = sorted(os.listdir(lock_dir))
    except OSError:
        return 0
    removed = 0
    scanned = 0
    for name in names:
        if not name.endswith(".lock"):
            continue
        scanned += 1
        path = os.path.join(lock_dir, name)
        if _gc_one_lock_file(path):
            removed += 1
    if removed:
        logger.info(
            "lock-residue GC: removed %d of %d lock file(s) in %s", removed, scanned, lock_dir
        )
    return removed


def _gc_one_lock_file(path: str) -> bool:
    """GC a single lock file if it is provably a residue. True if removed."""
    try:
        fd = os.open(path, os.O_RDWR)
    except OSError:
        return False
    lock_file = os.fdopen(fd, "r+b")
    acquired = False
    try:
        try:
            acquired = _probe_lock(lock_file)
        except OSError:
            return False
        if not acquired:
            return False  # live holder — never touch
        if not _lock_file_is_current(lock_file, path):
            return False  # someone replaced/unlinked it under us
        try:
            size = os.fstat(lock_file.fileno()).st_size
        except OSError:
            return False
        record = None
        reason = "empty (0-byte residue)"
        stale = size <= LOCK_SENTINEL_BYTES
        if not stale:
            record = read_holder_record(lock_file)
            pid = record.get("pid") if record else None
            if pid is not None and pid_alive(pid) is False:
                stale, reason = True, f"recorded holder PID {pid} is dead"
            elif pid is not None and pid_is_recycled(pid, record):
                stale, reason = (
                    True,
                    f"recorded holder PID {pid} was recycled (the live process "
                    "started after the lock was taken)",
                )
        if not stale:
            return False
        # A broken lock is an event worth reading in a log after the fact — the
        # 2026-08-02 investigation turned on knowing whether a residue file had
        # been reclaimed or was still sitting there. Emitted before the unlink so
        # it survives a failure to remove.
        logger.info(
            "lock-residue GC: reclaiming %s — %s%s",
            path,
            reason,
            f"; recorded argv: {' '.join(str(a) for a in record.get('argv') or [])}"
            if record and record.get("argv")
            else "",
        )

        if os.name == "nt":
            # Windows generally cannot unlink an open locked file: release
            # and close first, then remove best-effort (os.remove fails if
            # another process reopened it in the gap, leaving it in place).
            with contextlib.suppress(OSError):
                _release_lock(lock_file)
            acquired = False
            lock_file.close()
            lock_file = None
            try:
                os.remove(path)
                return True
            except OSError:
                return False

        try:
            os.remove(path)
            return True
        except FileNotFoundError:
            return False
        except OSError:
            logger.debug("lock-residue GC unlink failed for %s", path, exc_info=True)
            return False
    finally:
        if lock_file is not None:
            if acquired:
                with contextlib.suppress(OSError):
                    _release_lock(lock_file)
            lock_file.close()


# One sweep per process is enough: residue accumulates across crashed
# processes, not within one. Keyed by pid so a forked child sweeps again.
_gc_done_for_pid: Optional[int] = None


def maybe_gc_stale_locks(lock_dir: Optional[str] = None) -> int:
    """Run ``gc_stale_locks`` at most once per process. Never raises."""
    global _gc_done_for_pid
    if _gc_done_for_pid == os.getpid():
        return 0
    _gc_done_for_pid = os.getpid()
    try:
        return gc_stale_locks(lock_dir)
    except Exception:
        logger.debug("lock-residue GC failed", exc_info=True)
        return 0


# ---------------------------------------------------------------------------
# `memp locks` listing
# ---------------------------------------------------------------------------


def list_locks(lock_dir: Optional[str] = None) -> list[dict]:
    """Inventory every file in the locks dir with holder diagnostics.

    Read-only: the held-probe briefly takes and releases a non-blocking
    flock but never unlinks anything.
    """
    lock_dir = lock_dir or locks_dir()
    try:
        names = sorted(os.listdir(lock_dir))
    except OSError:
        return []
    entries = []
    for name in names:
        path = os.path.join(lock_dir, name)
        try:
            st = os.stat(path)
        except OSError:
            continue
        entry: dict = {
            "name": name,
            "size": st.st_size,
            "mtime": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
            "held": None,
            "pid": None,
            "ppid": None,
            "argv": None,
            "acquired_at": None,
            "pid_alive": None,
            "held_for_seconds": None,
        }
        record = None
        try:
            fd = os.open(path, os.O_RDWR)
        except OSError:
            entries.append(entry)
            continue
        lock_file = os.fdopen(fd, "r+b")
        try:
            try:
                unheld = _probe_lock(lock_file)
            except OSError:
                unheld = None
            if unheld is not None:
                entry["held"] = not unheld
            record = read_holder_record(lock_file)
            if unheld:
                with contextlib.suppress(OSError):
                    _release_lock(lock_file)
        finally:
            lock_file.close()
        if record:
            entry["pid"] = record.get("pid")
            entry["ppid"] = record.get("ppid")
            entry["argv"] = record.get("argv")
            entry["acquired_at"] = record.get("acquired_at")
            entry["pid_alive"] = pid_alive(record.get("pid"))
        if entry["held"]:
            duration = held_duration_seconds(path, record)
            if duration is not None:
                entry["held_for_seconds"] = int(duration)
        entries.append(entry)
    return entries
