"""Helpers for the Python child processes MemPalace starts.

Hooks, ``daemon start`` and ``serve`` start ``python -m mempalace...`` children
in whatever directory the caller is in. For Claude Code hooks that is the
session's working directory. ``-m`` puts that directory first on
``sys.path``, so a folder holding a ``mempalace/`` package (any checkout of
this repository, a worktree included) or a ``sqlite3.py`` would be imported
instead of the installed code, and would run against the live palace.

``safe_child_env`` keeps the working directory off ``sys.path`` through
``PYTHONSAFEPATH`` (Python 3.11+). The argv is left as ``-m ...`` so process
listings and the hook's per-target PID slots stay as they were.
"""

from __future__ import annotations

import os


def safe_child_env(env: dict | None = None) -> dict:
    """A copy of ``env`` (default: this process's) with the cwd kept off sys.path."""
    child = dict(os.environ if env is None else env)
    child["PYTHONSAFEPATH"] = "1"
    return child
