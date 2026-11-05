"""Palace access levels: who may read a drawer, stamped by the folder it came from.

A drawer carries an integer ``access`` in its metadata. A reader at level N sees
drawers with ``access >= N``. Missing ``access`` reads as team.

==========  ========  ==================================
Level       access    Who reads it
==========  ========  ==================================
owners      1         the owners
team        2         owners plus the team
extended    3         the above plus whoever joins later
==========  ========  ==================================

**Where a level comes from.** One file, ``~/.mempalace/levels.json`` (or the path
in ``MEMPALACE_LEVELS_FILE``)::

    {"default": "team",
     "owners_folders": ["/home/xdev/code/llcsetup", "/home/xdev/code/fpmdataroom"]}

A folder is owners when it is one of those folders or inside one; anything else
gets ``default``. A git worktree resolves to its main repo, and a herdr-spawn
worktree also through its ``meta.json`` ``repo``.

**One session, one level.** A process's level is the more private (lower) of
its folder's level and an inherited ``MEMPALACE_LEVEL``. It reads up to that
level and stamps that level on what it writes, so a team-folder session can
never read owners drawers and then save what it learned into team drawers.

**Off unless configured.** With no levels file and no ``MEMPALACE_LEVEL`` the
filter and the stamping are both off and every result is exactly what it was
before levels existed. The Mini and the other boxes run that way.

The filter is a guard rail for processes that run as the palace's unix user,
not a lock: such a process can open the SQLite file directly. The lock for
people is a door that runs this code with the caller's level (the team MCP).
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

OWNERS = 1
TEAM = 2
EXTENDED = 3

LEVEL_NAMES = {"owners": OWNERS, "team": TEAM, "extended": EXTENDED}
LEVEL_LABELS = {v: k for k, v in LEVEL_NAMES.items()}

# The most private level that exists. A reader at this level sees everything,
# so no filter is applied (keeps owners reads on the unfiltered fast path).
# Adding a more private level (e.g. 0, "just me") means lowering this.
LOWEST_LEVEL = OWNERS
# A drawer with no ``access`` reads as this level.
MISSING_ACCESS_LEVEL = TEAM

ACCESS_KEY = "access"
ACCESS_SRC_KEY = "access_src"
OVERRIDE_SRC = "override"

LEVEL_ENV = "MEMPALACE_LEVEL"
LEVELS_FILE_ENV = "MEMPALACE_LEVELS_FILE"


class LevelError(ValueError):
    """A level name, number or levels file that cannot be understood."""


class LevelDenied(PermissionError):
    """The caller's level does not allow this operation."""


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_level(value: Any) -> int:
    """Return the integer level for ``owners``/``team``/``extended`` or 1/2/3.

    Raises :class:`LevelError` for anything else. Never guesses: a typo in
    ``MEMPALACE_LEVEL`` must fail loudly, not silently read as some level.
    """
    if isinstance(value, bool):
        raise LevelError(f"not a level: {value!r}")
    if isinstance(value, int):
        if value in LEVEL_LABELS:
            return value
        raise LevelError(f"unknown level number {value!r} (known: {sorted(LEVEL_LABELS)})")
    if isinstance(value, str):
        text = value.strip().lower()
        if text in LEVEL_NAMES:
            return LEVEL_NAMES[text]
        if text.isdigit():
            return parse_level(int(text))
    raise LevelError(f"not a level: {value!r} (use owners, team or extended)")


def level_label(level: int) -> str:
    return LEVEL_LABELS.get(level, str(level))


def stored_access(meta: Optional[dict]) -> int:
    """The level a stored drawer's metadata grants. Missing reads as team.

    A malformed stored value reads as the most private level, so a damaged
    stamp hides a drawer rather than opening it.
    """
    if not meta:
        return MISSING_ACCESS_LEVEL
    raw = meta.get(ACCESS_KEY)
    if raw is None:
        return MISSING_ACCESS_LEVEL
    try:
        return parse_level(raw)
    except LevelError:
        return LOWEST_LEVEL


# ---------------------------------------------------------------------------
# The levels file
# ---------------------------------------------------------------------------


def levels_file_path() -> Path:
    override = os.environ.get(LEVELS_FILE_ENV, "").strip()
    if override:
        return Path(os.path.expanduser(override))
    return Path(os.path.expanduser("~")) / ".mempalace" / "levels.json"


class LevelsConfig:
    """Parsed levels file. ``broken`` marks a file that exists but cannot be read.

    A broken file makes every folder owners: writes stamp owners (too private,
    reversible with ``memp level set``) instead of stamping an owners folder's
    work as team (a leak, not reversible once read).
    """

    def __init__(
        self,
        default: int,
        owners_folders: list[str],
        broken: str = "",
        owners_wings: Optional[list[str]] = None,
    ):
        self.default = default
        self.owners_folders = owners_folders
        self.broken = broken
        # Wings that are owners as a whole (used by ``backfill-levels`` only;
        # live writes stamp by folder). Default: each owners folder's name.
        self.owners_wings = owners_wings if owners_wings is not None else []

    def folder_level(self, path: str) -> int:
        if self.broken:
            return OWNERS
        if not path:
            return self.default
        for candidate in _path_forms(path):
            for folder in self.owners_folders:
                if _is_within(candidate, folder):
                    return OWNERS
        return self.default


_config_cache: dict = {"key": None, "value": None}


def load_levels_config() -> Optional[LevelsConfig]:
    """The levels file, or ``None`` when there is none. Re-read when it changes."""
    path = levels_file_path()
    try:
        st = path.stat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        return LevelsConfig(OWNERS, [], broken=f"cannot stat {path}: {exc}")
    key = (str(path), st.st_mtime_ns, st.st_size)
    if _config_cache["key"] == key:
        return _config_cache["value"]
    value = _parse_levels_file(path)
    if value.broken:
        logger.warning(
            "levels file %s unusable (%s): every folder reads as owners", path, value.broken
        )
    _config_cache["key"] = key
    _config_cache["value"] = value
    return value


def _parse_levels_file(path: Path) -> LevelsConfig:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return LevelsConfig(OWNERS, [], broken=f"unreadable: {exc}")
    if not isinstance(data, dict):
        return LevelsConfig(OWNERS, [], broken="not a JSON object")
    try:
        default = parse_level(data.get("default", "team"))
    except LevelError as exc:
        return LevelsConfig(OWNERS, [], broken=f"default: {exc}")
    folders = data.get("owners_folders", [])
    if not isinstance(folders, list) or not all(isinstance(f, str) and f.strip() for f in folders):
        return LevelsConfig(OWNERS, [], broken="owners_folders must be a list of paths")
    expanded: list[str] = []
    for folder in folders:
        for form in _path_forms(folder):
            if form not in expanded:
                expanded.append(form)
    wings = data.get("owners_wings")
    if wings is None:
        wings = [_norm(f).rsplit("/", 1)[-1].lower() for f in folders]
    elif not isinstance(wings, list) or not all(isinstance(w, str) and w.strip() for w in wings):
        return LevelsConfig(OWNERS, [], broken="owners_wings must be a list of wing names")
    return LevelsConfig(default, expanded, owners_wings=[w.strip() for w in wings])


def _norm(path: str) -> str:
    return os.path.normpath(os.path.expanduser(path)).replace("\\", "/").rstrip("/") or "/"


def _path_forms(path: str) -> list[str]:
    """The path as written and with symlinks resolved (``~/code`` is often a link)."""
    forms = [_norm(path)]
    try:
        real = _norm(os.path.realpath(os.path.expanduser(path)))
    except OSError:
        real = forms[0]
    if real not in forms:
        forms.append(real)
    return forms


def _is_within(path: str, folder: str) -> bool:
    return path == folder or path.startswith(folder.rstrip("/") + "/")


# ---------------------------------------------------------------------------
# Folder resolution (worktrees)
# ---------------------------------------------------------------------------


def _main_worktree_root(path: str) -> Optional[str]:
    """Main repo root for a linked git worktree at or above ``path``, else None."""
    current = Path(path)
    for _ in range(40):
        gitfile = current / ".git"
        try:
            if gitfile.is_dir():
                return None
            if gitfile.is_file():
                text = gitfile.read_text(encoding="utf-8", errors="replace").strip()
                match = re.match(r"gitdir:\s*(.+)$", text, re.MULTILINE)
                if not match:
                    return None
                gitdir = Path(match.group(1).strip())
                if not gitdir.is_absolute():
                    gitdir = current / gitdir
                parts = gitdir.parts
                if len(parts) < 4 or parts[-2] != "worktrees" or parts[-3] != ".git":
                    return None
                return str(Path(*parts[:-3]))
        except OSError:
            return None
        if current.parent == current:
            return None
        current = current.parent
    return None


def _herdr_spawn_repo(path: str) -> Optional[str]:
    """The ``repo`` a herdr-spawn worktree was cut from (``<run>/meta.json``)."""
    current = Path(path)
    for _ in range(40):
        if current.name == "worktree":
            meta = current.parent / "meta.json"
            try:
                if meta.is_file():
                    data = json.loads(meta.read_text(encoding="utf-8"))
                    repo = data.get("repo") if isinstance(data, dict) else None
                    if isinstance(repo, str) and repo.strip():
                        return repo.strip()
            except (OSError, ValueError):
                return None
        if current.parent == current:
            return None
        current = current.parent
    return None


def folder_level(path: Optional[str], config: Optional[LevelsConfig] = None) -> int:
    """Level of a folder: owners if it (or its main repo) is in an owners folder.

    Every form of the path is checked: as given, its main worktree root and its
    herdr-spawn ``repo``. The most private answer wins, so a worktree of an
    owners repo can never stamp team.
    """
    config = config if config is not None else load_levels_config()
    if config is None:
        return MISSING_ACCESS_LEVEL
    if not path:
        return config.folder_level("")
    candidates = [path]
    for resolver in (_main_worktree_root, _herdr_spawn_repo):
        try:
            other = resolver(path)
        except Exception:
            other = None
        if other:
            candidates.append(other)
    return min(config.folder_level(c) for c in candidates)


# ---------------------------------------------------------------------------
# Caller level
# ---------------------------------------------------------------------------

# Per-job override (daemon jobs, the HTTP MCP per request). Unset = the process level.
_LEVEL_OVERRIDE: contextvars.ContextVar[Optional[int]] = contextvars.ContextVar(
    "mempalace_level_override", default=None
)


def env_level() -> Optional[int]:
    raw = os.environ.get(LEVEL_ENV, "").strip()
    if not raw:
        return None
    return parse_level(raw)


def levels_active() -> bool:
    """True when levels are configured here: a levels file or ``MEMPALACE_LEVEL``.

    Off means today's behaviour exactly: no filter, no stamping.
    """
    if _LEVEL_OVERRIDE.get() is not None:
        return True
    if os.environ.get(LEVEL_ENV, "").strip():
        return True
    return load_levels_config() is not None


def resolve_level(cwd: Optional[str] = None, env: Optional[str] = None) -> int:
    """Level for a session in ``cwd`` with ``MEMPALACE_LEVEL`` = ``env``.

    The more private of the two wins. ``env`` of ``None`` reads the process
    environment; pass ``""`` for "no inherited level".
    """
    if env is None:
        inherited = env_level()
    else:
        inherited = parse_level(env) if env.strip() else None
    level = folder_level(cwd if cwd is not None else _safe_cwd())
    if inherited is not None:
        level = min(level, inherited)
    return level


def _safe_cwd() -> str:
    try:
        return os.getcwd()
    except OSError:
        return ""


def current_level() -> int:
    """The caller's level: the per-job override if one is set, else the process's."""
    override = _LEVEL_OVERRIDE.get()
    if override is not None:
        return override
    if not levels_active():
        return MISSING_ACCESS_LEVEL
    return resolve_level()


@contextlib.contextmanager
def caller_level(level: Optional[int]):
    """Run a block at ``level`` (a daemon job, one HTTP request). ``None`` = no change."""
    if level is None:
        yield
        return
    token = _LEVEL_OVERRIDE.set(parse_level(level))
    try:
        yield
    finally:
        _LEVEL_OVERRIDE.reset(token)


def submit_level() -> Optional[int]:
    """The level a queued write carries, or ``None`` when levels are off here."""
    if not levels_active():
        return None
    return current_level()


JOB_LEVEL_KEY = "access_level"


def attach_level(payload: dict) -> dict:
    """Put the submitter's level on a queued job (the daemon runs it at that level).

    A level already on the payload can only be made more private, never more
    open. Levels off here: the payload is left as it is.
    """
    own = submit_level()
    if own is None:
        return payload
    given = payload.get(JOB_LEVEL_KEY)
    if given is not None:
        try:
            own = min(own, parse_level(given))
        except LevelError:
            pass
    payload[JOB_LEVEL_KEY] = own
    return payload


def job_level(payload: dict) -> Optional[int]:
    """The level a daemon job runs at. A job with no level runs as owners when
    levels are on: what it reads it also stamps owners, so nothing leaks."""
    given = payload.get(JOB_LEVEL_KEY) if isinstance(payload, dict) else None
    if given is not None:
        try:
            return parse_level(given)
        except LevelError:
            return LOWEST_LEVEL
    if levels_active():
        return LOWEST_LEVEL
    return None


def filter_needed(level: Optional[int] = None) -> bool:
    """True when reads must hide drawers from this caller."""
    if not levels_active():
        return False
    level = current_level() if level is None else level
    return level > LOWEST_LEVEL


def require_owners(what: str) -> None:
    """Admin tools (repair, migrate, raw dumps) need an owners-level caller."""
    if filter_needed():
        raise LevelDenied(
            f"{what} needs an owners-level session (this one is {level_label(current_level())})"
        )


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


def access_where(level: Optional[int] = None) -> Optional[dict]:
    """The metadata filter that keeps only drawers ``level`` may read, or None.

    For team (and below) a missing ``access`` must stay visible, so the clause
    excludes the more private levels rather than requiring ``access >= N``
    (``$gte`` would drop every unstamped drawer). Above team, missing reads as
    team and so is itself hidden, and ``$gte`` is exact.
    """
    level = current_level() if level is None else level
    if not filter_needed(level):
        return None
    if level <= MISSING_ACCESS_LEVEL:
        return {ACCESS_KEY: {"$nin": list(range(0, level))}}
    return {ACCESS_KEY: {"$gte": level}}


def and_where(where: Optional[dict], extra: Optional[dict]) -> Optional[dict]:
    if not extra:
        return where
    if not where:
        return dict(extra)
    return {"$and": [where, extra]}


def visible(meta: Optional[dict], level: Optional[int] = None) -> bool:
    """Python-side check for paths that never pass through a ``where``."""
    level = current_level() if level is None else level
    if not filter_needed(level):
        return True
    return stored_access(meta) >= level


def stamp_value(given: Any = None, level: Optional[int] = None) -> int:
    """The ``access`` a write stores: the writer's level, or a more private given one."""
    level = current_level() if level is None else level
    if given is None:
        return level
    try:
        return min(parse_level(given), level)
    except LevelError:
        return level


# ---------------------------------------------------------------------------
# Collection wrapper
# ---------------------------------------------------------------------------


def _as_list(value):
    if value is None:
        return None
    if isinstance(value, (str, dict)):
        return [value]
    return list(value)


class LevelFilteredCollection:
    """Wrap a palace collection so every read and write honours the caller's level.

    The level is read on each call, not at wrap time, so one cached collection
    serves a daemon whose jobs run at different levels.

    Reads AND the access clause into every ``where`` and drop hidden ids from
    ``get(ids=...)``. Writes stamp the caller's level and refuse to touch a
    drawer the caller cannot see: to it that drawer does not exist. With levels
    off every call passes straight through.
    """

    _levels_wrapper = True

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    @property
    def inner(self):
        return self._inner

    @property
    def upsert_merges_metadata(self):
        return getattr(self._inner, "upsert_merges_metadata", False)

    @property
    def distance_metric(self) -> str:
        return self._inner.distance_metric

    # --- pass-through identity / maintenance ---------------------------------

    def get_stored_embedder_identity(self):
        return self._inner.get_stored_embedder_identity()

    def set_embedder_identity(self, identity) -> None:
        return self._inner.set_embedder_identity(identity)

    def effective_embedder_identity(self):
        return self._inner.effective_embedder_identity()

    def maintenance_state(self) -> dict:
        return self._inner.maintenance_state()

    def run_maintenance(self, kind: str):
        return self._inner.run_maintenance(kind)

    def close(self) -> None:
        return self._inner.close()

    def health(self):
        return self._inner.health()

    # --- reads -----------------------------------------------------------------

    def query(self, **kwargs):
        clause = access_where()
        if clause:
            kwargs["where"] = and_where(kwargs.get("where"), clause)
        return self._inner.query(**kwargs)

    def get(self, **kwargs):
        clause = access_where()
        if clause:
            kwargs["where"] = and_where(kwargs.get("where"), clause)
        return self._inner.get(**kwargs)

    def lexical_search(self, *, query: str, n_results: int = 10, where: Optional[dict] = None):
        return self._inner.lexical_search(
            query=query, n_results=n_results, where=and_where(where, access_where())
        )

    def facet_counts(self, field: str, where: Optional[dict] = None, limit: int = 1000):
        return self._inner.facet_counts(field, where=and_where(where, access_where()), limit=limit)

    def get_all_metadata(self, where: Optional[dict] = None) -> list[dict]:
        return self._inner.get_all_metadata(where=and_where(where, access_where()))

    def count(self) -> int:
        clause = access_where()
        if not clause:
            return self._inner.count()
        return self._visible_count(clause)

    def estimated_count(self) -> int:
        clause = access_where()
        if not clause:
            return self._inner.estimated_count()
        return self._visible_count(clause)

    def _visible_count(self, clause: dict) -> int:
        total = 0
        offset = 0
        page = 10000
        while True:
            got = self._inner.get(where=clause, include=[], limit=page, offset=offset)
            n = len(got["ids"])
            total += n
            if n < page:
                return total
            offset += page

    # --- writes ----------------------------------------------------------------

    def _existing_meta(self, ids: list[str]) -> dict[str, dict]:
        if not ids:
            return {}
        got = self._inner.get(ids=list(ids), include=["metadatas"])
        metas = got.get("metadatas") or [None] * len(got["ids"])
        return {doc_id: (meta or {}) for doc_id, meta in zip(got["ids"], metas)}

    def _plan_writes(self, ids, metadatas, *, content_changes: bool):
        """Return (keep_index_list, new_metadatas) for a write at the caller's level."""
        level = current_level()
        existing = self._existing_meta(ids)
        keep: list[int] = []
        new_metas: list[dict] = []
        skipped = 0
        for idx, doc_id in enumerate(ids):
            old = existing.get(doc_id)
            if old is not None and stored_access(old) < level:
                skipped += 1
                continue
            given = dict((metadatas[idx] if metadatas is not None else None) or {})
            if old is not None and old.get(ACCESS_SRC_KEY) == OVERRIDE_SRC:
                # An override survives re-mining and re-stamping, unless the
                # writer explicitly asks for something more private.
                access = stored_access(old)
                if ACCESS_KEY in given:
                    access = min(access, stamp_value(given[ACCESS_KEY], level))
                given[ACCESS_KEY] = access
                given[ACCESS_SRC_KEY] = OVERRIDE_SRC
            elif old is not None and not content_changes:
                # Metadata-only move: no new words, so the level stays put
                # (only ever made more private by an explicit value).
                access = stored_access(old)
                if ACCESS_KEY in given:
                    access = min(access, stamp_value(given[ACCESS_KEY], level))
                given[ACCESS_KEY] = access
            else:
                given[ACCESS_KEY] = stamp_value(given.get(ACCESS_KEY), level)
            keep.append(idx)
            new_metas.append(given)
        if skipped:
            logger.info("levels: skipped %d write(s) to drawers hidden at level %s", skipped, level)
        return keep, new_metas

    @staticmethod
    def _pick(values, keep):
        if values is None:
            return None
        return [values[i] for i in keep]

    def _stamped_insert(self, method: str, *, documents, ids, metadatas=None, embeddings=None):
        inner = getattr(self._inner, method)
        if not levels_active():
            return inner(documents=documents, ids=ids, metadatas=metadatas, embeddings=embeddings)
        ids = _as_list(ids)
        documents = _as_list(documents)
        metadatas = _as_list(metadatas)
        keep, new_metas = self._plan_writes(ids, metadatas, content_changes=True)
        if not keep:
            return None
        return inner(
            documents=self._pick(documents, keep),
            ids=self._pick(ids, keep),
            metadatas=new_metas,
            embeddings=self._pick(embeddings, keep),
        )

    def add(self, *, documents, ids, metadatas=None, embeddings=None):
        return self._stamped_insert(
            "add", documents=documents, ids=ids, metadatas=metadatas, embeddings=embeddings
        )

    def upsert(self, *, documents, ids, metadatas=None, embeddings=None):
        return self._stamped_insert(
            "upsert", documents=documents, ids=ids, metadatas=metadatas, embeddings=embeddings
        )

    def update(self, *, ids, documents=None, metadatas=None, embeddings=None):
        if not levels_active():
            return self._inner.update(
                ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings
            )
        ids = _as_list(ids)
        documents = _as_list(documents)
        metadatas = _as_list(metadatas)
        keep, new_metas = self._plan_writes(
            ids, metadatas, content_changes=documents is not None or embeddings is not None
        )
        if not keep:
            return None
        return self._inner.update(
            ids=self._pick(ids, keep),
            documents=self._pick(documents, keep),
            metadatas=new_metas,
            embeddings=self._pick(embeddings, keep),
        )

    def delete(self, *, ids=None, where=None):
        clause = access_where()
        if not clause:
            return self._inner.delete(ids=ids, where=where)
        if ids is not None:
            ids = _as_list(ids)
            seen = self.get(ids=ids, where=where, include=[])["ids"]
            if not seen:
                return None
            return self._inner.delete(ids=list(seen))
        if where is None:
            raise LevelDenied("delete without ids or where is not allowed below owners level")
        return self._inner.delete(where=and_where(where, clause))

    # --- level set (bypasses stamping on purpose) ------------------------------

    def set_access(self, ids: Iterable[str], level: int) -> int:
        """Write ``access`` + ``access_src: override`` on ``ids`` the caller can see."""
        ids = [i for i in ids]
        if not ids:
            return 0
        visible_ids = self.get(ids=ids, include=[])["ids"]
        if not visible_ids:
            return 0
        self._inner.update(
            ids=list(visible_ids),
            metadatas=[{ACCESS_KEY: level, ACCESS_SRC_KEY: OVERRIDE_SRC}] * len(visible_ids),
        )
        return len(visible_ids)


def wrap_collection(collection):
    """Wrap ``collection`` in the level filter (idempotent; ``None`` passes through)."""
    if collection is None or getattr(collection, "_levels_wrapper", False):
        return collection
    return LevelFilteredCollection(collection)


def unwrap_collection(collection):
    while getattr(collection, "_levels_wrapper", False):
        collection = collection.inner
    return collection


# ---------------------------------------------------------------------------
# Override log
# ---------------------------------------------------------------------------


def override_log_path() -> Path:
    return Path(os.path.expanduser("~")) / ".mempalace" / "hook_state" / "level-overrides.jsonl"


def log_override(record: dict) -> None:
    """One JSON line per ``level set``: who, what, when, from, to. Never content."""
    path = override_log_path()
    line = dict(record)
    line.setdefault("ts", datetime.now(timezone.utc).isoformat())
    line.setdefault("user", os.environ.get("USER") or os.environ.get("LOGNAME") or "")
    line.setdefault("pid", os.getpid())
    line.setdefault("cwd", _safe_cwd())
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(line, sort_keys=True) + "\n")
    except OSError as exc:
        print(f"warning: could not write level override log {path}: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Side files keyed by wing (tunnels, hallways)
# ---------------------------------------------------------------------------


def hidden_wings(col=None) -> set[str]:
    """Wings in which the caller can see no drawer at all. Empty when unfiltered.

    Side files (explicit tunnels, hallways) name wings, not drawers; a record
    touching one of these wings is hidden, so an owners-only wing's name never
    reaches a team caller through them.
    """
    level = current_level()
    if not filter_needed(level):
        return set()
    if col is None:
        from .config import MempalaceConfig
        from .palace import get_collection_read_only

        cfg = MempalaceConfig()
        col = get_collection_read_only(cfg.palace_path, collection_name=cfg.collection_name)
    inner = unwrap_collection(col)
    candidates: set[str] = set()
    if level <= MISSING_ACCESS_LEVEL:
        got = inner.get(where={ACCESS_KEY: {"$lt": level}}, include=["metadatas"])
        metas = got.get("metadatas") or []
    else:
        metas = [m for m in inner.get_all_metadata() if stored_access(m) < level]
    for meta in metas:
        wing = (meta or {}).get("wing")
        if wing:
            candidates.add(wing)
    filtered = wrap_collection(inner)
    out = set()
    for wing in candidates:
        if not filtered.get(where={"wing": wing}, limit=1, include=[])["ids"]:
            out.add(wing)
    return out


# ---------------------------------------------------------------------------
# Raw SQL readers of chroma.sqlite3
# ---------------------------------------------------------------------------


def chroma_access_sql(row_id_expr: str, level: Optional[int] = None) -> tuple[str, list]:
    """``AND ...`` clause for raw reads of ``chroma.sqlite3`` (``embedding_metadata``).

    ``row_id_expr`` is the SQL expression for the ``embeddings.id`` of the row.
    A stored non-integer ``access`` reads as owners (fail closed), a missing one
    as team. Returns ``("", [])`` when no filter is needed.
    """
    level = current_level() if level is None else level
    if not filter_needed(level):
        return "", []
    sql = (
        " AND COALESCE((SELECT COALESCE(am.int_value, ?) FROM embedding_metadata am"
        f" WHERE am.id = {row_id_expr} AND am.key = '{ACCESS_KEY}' LIMIT 1), ?) >= ?"
    )
    return sql, [LOWEST_LEVEL, MISSING_ACCESS_LEVEL, level]


# ---------------------------------------------------------------------------
# Drawer lookups that must not themselves be filtered
# ---------------------------------------------------------------------------

DRAWER_ID_RE = re.compile(r"drawer_[A-Za-z0-9][A-Za-z0-9_.\-]*")


def drawer_access_lookup(ids: list[str], palace_path: Optional[str] = None) -> dict[str, int]:
    """Stored level of each drawer id, read past the filter. Unknown ids are absent.

    An id that names no row is tried as a logical parent (``parent_drawer_id``),
    whose level is its most private chunk's. Callers treat an absent id as the
    most private level (fail closed).
    """
    if not ids:
        return {}
    from .config import MempalaceConfig
    from .palace import get_collection_read_only

    if palace_path is None:
        cfg = MempalaceConfig()
        palace_path = cfg.palace_path
        collection_name = cfg.collection_name
    else:
        collection_name = None
    kwargs = {"collection_name": collection_name} if collection_name else {}
    col = unwrap_collection(get_collection_read_only(palace_path, **kwargs))
    out: dict[str, int] = {}
    got = col.get(ids=list(ids), include=["metadatas"])
    for doc_id, meta in zip(got["ids"], got.get("metadatas") or []):
        out[doc_id] = stored_access(meta)
    for doc_id in ids:
        if doc_id in out:
            continue
        chunks = col.get(where={"parent_drawer_id": doc_id}, include=["metadatas"])
        metas = chunks.get("metadatas") or []
        if metas:
            out[doc_id] = min(stored_access(m) for m in metas)
    return out


# ---------------------------------------------------------------------------
# ``python -m mempalace.levels resolve [--cwd PATH]`` for shell hooks
# ---------------------------------------------------------------------------


def _main(argv: Optional[list[str]] = None) -> int:
    """Print the level number for a folder (stdlib only, ~30 ms; for hooks).

    Prints nothing and exits 0 when levels are off. A bad ``MEMPALACE_LEVEL``
    or argument exits 2 with the reason on stderr, so the caller pins nothing
    and every palace call in that session fails loudly instead of guessing.
    """
    import argparse

    parser = argparse.ArgumentParser(prog="python -m mempalace.levels")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_resolve = sub.add_parser("resolve", help="print the session level for a folder")
    p_resolve.add_argument("--cwd", default=None)
    args = parser.parse_args(argv)
    try:
        if not levels_active():
            return 0
        print(resolve_level(args.cwd))
        return 0
    except LevelError as exc:
        print(f"mempalace.levels: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(_main())
