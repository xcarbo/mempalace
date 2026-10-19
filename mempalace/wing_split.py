"""Split a machine-level transcript wing into one wing per source project.

Hook-ingested transcripts used to land in one flat ``sessions`` wing. On a real
palace that wing is hundreds of projects, so no room set means anything and a
wing-scoped search returns everything. This module re-keys those drawers by the
project the transcript came from. Ported from upstream v3.11.0 (#2576, #2669).

Two steps:

* **plan** — group the wing's drawers by *project key* (the encoded project
  directory of a Claude Code / pi transcript path, the URL-encoded cwd of a Grok
  session, or the ``cwd`` of a Codex rollout), resolve each key to a target wing
  and write ``<plans-dir>/split-<wing>.json`` for review.
* **apply** — rewrite ``wing`` metadata for every drawer in the plan, in
  batches. Content, room, embedding, FTS row and drawer id never change.

Fork deltas from upstream (pinned in ``tests/test_fork_deltas.py``):

* targets come from the hook's own derivation (``hooks_cli``), not upstream's
  suffix-match-else-snake_case, so a split transcript lands in the wing a new
  transcript of the same project would, and a ``.palace-wing`` pin is honoured;
* Grok and pi session paths carry a project key too;
* the plan directory is overridable, so a dry run can write outside the palace;
* hallway records are NOT dropped (see :func:`apply_split`).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import unquote

from .config import MempalaceConfig, normalize_wing_name, sanitize_name

SPLIT_SCHEMA_VERSION = 1
_PAGE_SIZE = 2000
_UPDATE_BATCH = 500
# Source files kept per project key to try for a readable cwd.
_SAMPLES_PER_KEY = 5

_CLAUDE_PROJECTS_RE = re.compile(r"[/\\]\.claude[/\\]projects[/\\]([^/\\]+)[/\\]")
_CODEX_SESSIONS_RE = re.compile(r"[/\\]\.codex[/\\]sessions[/\\]")
_GROK_SESSIONS_RE = re.compile(r"[/\\]\.grok[/\\]sessions[/\\](%2[Ff][^/\\]*)[/\\]")
_PI_SESSIONS_RE = re.compile(r"[/\\]\.pi[/\\]agent[/\\]sessions[/\\]--([^/\\]+?)--[/\\]")
# Derivations that mean "no project": the drawers stay where they are.
_NO_PROJECT_WINGS = {"sessions", "worktree"}


# ── project keys ─────────────────────────────────────────────────────────────


def _codex_cwd(path: str) -> Optional[str]:
    """``cwd`` from a Codex rollout's first lines, or ``None`` when unreadable."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for _ in range(3):
                line = f.readline()
                if not line:
                    break
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                payload = obj.get("payload") if isinstance(obj, dict) else None
                for holder in (obj, payload):
                    if isinstance(holder, dict) and isinstance(holder.get("cwd"), str):
                        return holder["cwd"]
    except OSError:
        return None
    return None


def project_key(source_file: Optional[str]) -> Optional[str]:
    """The project identifier a transcript path carries, or ``None``.

    Claude Code encodes the working directory in the path
    (``.claude/projects/-Users-me-dev-mempalace/<session>.jsonl``, subagent
    transcripts nest below it); the encoded segment is the key. pi
    (``.pi/agent/sessions/--Users-me-dev-mempalace--/``) and Grok
    (``.grok/sessions/%2FUsers%2Fme%2Fdev%2Fmempalace/``) carry the same path
    in their own encoding and are normalised to the Claude form. Codex
    rollouts carry no project in the path; the key is the basename of the
    ``cwd`` in the file when the file is on this machine.
    """
    if not source_file:
        return None
    match = _CLAUDE_PROJECTS_RE.search(source_file)
    if match:
        return match.group(1)
    match = _PI_SESSIONS_RE.search(source_file)
    if match:
        return "-" + match.group(1)
    match = _GROK_SESSIONS_RE.search(source_file)
    if match:
        cwd = unquote(match.group(1)).replace("\\", "/")
        return cwd.replace("/", "-") if cwd.strip("/") else None
    if _CODEX_SESSIONS_RE.search(source_file):
        cwd = _codex_cwd(source_file)
        if cwd:
            base = cwd.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
            return base or None
    return None


def _key_dir(source_file: str) -> Optional[str]:
    """The per-project directory a Claude / pi / Grok transcript lives under."""
    for pattern in (_CLAUDE_PROJECTS_RE, _PI_SESSIONS_RE, _GROK_SESSIONS_RE):
        match = pattern.search(source_file)
        if match:
            return source_file[: match.end() - 1]
    return None


def _sibling_transcripts(source_file: str) -> list[str]:
    """A few other transcripts of the same project, for when the sampled file is gone."""
    directory = _key_dir(source_file)
    if not directory:
        return []
    found: list[str] = []
    try:
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.name.endswith(".jsonl") and entry.is_file():
                    found.append(entry.path)
                    if len(found) >= _SAMPLES_PER_KEY:
                        break
    except OSError:
        return []
    return found


def resolve_target(key: str, samples: Iterable[str] = ()) -> tuple[str, str]:
    """``(target_wing, how)`` for a project key, using the hook's derivation.

    ``how`` is ``"home"`` (the home directory: the machine's workstation wing),
    ``"cwd"`` (a readable sample transcript recorded its working directory, so
    worktrees and ``.palace-wing`` pins resolve exactly as for a new
    transcript) or ``"path"`` (the encoded project folder alone). A derivation
    that names no project returns ``("sessions", "none")``.
    """
    from . import hooks_cli

    home = str(Path.home()).replace("\\", "/").rstrip("/")
    if key == home.replace("/", "-"):
        return hooks_cli._workstation_wing(), "home"
    samples = list(samples)
    if samples:
        samples += [p for p in _sibling_transcripts(samples[0]) if p not in samples]
    for sample in samples:
        first = next(hooks_cli._jsonl_cwds(sample), None)
        if first is None:
            continue
        if first.replace("\\", "/").rstrip("/").lower() == home.lower():
            return hooks_cli._workstation_wing(), "home"
        wing = hooks_cli._wing_from_jsonl_cwd(sample)
        if wing and wing not in _NO_PROJECT_WINGS:
            return wing, "cwd"
    wing = hooks_cli._wing_from_transcript_path(f"/.claude/projects/-{key.lstrip('-')}/x.jsonl")
    if wing in _NO_PROJECT_WINGS:
        return "sessions", "none"
    return wing, "path"


# ── plan ─────────────────────────────────────────────────────────────────────


def _iter_wing_rows(col, wing: str):
    offset = 0
    while True:
        batch = col.get(
            where={"wing": wing}, limit=_PAGE_SIZE, offset=offset, include=["metadatas"]
        )
        ids = list(batch.get("ids") or [])
        if not ids:
            return
        metas = batch.get("metadatas") or [None] * len(ids)
        for row_id, meta in zip(ids, metas):
            meta = meta or {}
            if meta.get("wing") == wing:
                yield row_id, meta
        offset += len(ids)
        if len(ids) < _PAGE_SIZE:
            return


def existing_wings(col) -> set[str]:
    """Every wing that holds at least one drawer (one paged metadata scan)."""
    wings: set[str] = set()
    offset = 0
    while True:
        batch = col.get(limit=_PAGE_SIZE, offset=offset, include=["metadatas"])
        ids = list(batch.get("ids") or [])
        if not ids:
            break
        for meta in batch.get("metadatas") or []:
            name = (meta or {}).get("wing")
            if name:
                wings.add(str(name))
        offset += len(ids)
        if len(ids) < _PAGE_SIZE:
            break
    return wings


def snap_to_existing(name: str, known: Iterable[str]) -> Optional[str]:
    """The existing wing whose name ``name`` ends with, at a segment boundary.

    Longest match wins: ``volumes-xdata-codexd-gamedev-pangaea`` snaps to
    ``gamedev-pangaea`` over ``pangaea``. Used only for keys whose transcripts
    are gone, so the encoded path is all there is to go on.
    """
    lowered = name.lower().strip("-")
    best: Optional[str] = None
    best_len = 0
    for wing in known:
        norm = normalize_wing_name(wing)
        for variant in {norm, norm.replace("_", "-")}:
            if variant and (lowered == variant or lowered.endswith("-" + variant)):
                if len(variant) > best_len:
                    best, best_len = wing, len(variant)
    return best


_WORKTREE_RE = re.compile(
    r"^(?P<repo>.*?)-+\.?(?:claude-|codex-|herdr-)?worktrees?-(?P<rest>.+)$", re.I
)


def collapse_worktree_key(key: str, known: Iterable[str]) -> Optional[str]:
    """The existing wing a deleted worktree checkout belonged to, or ``None``.

    ``<repo>--worktrees-<name>`` and ``.worktrees/<name>`` encode the repo in
    front of the marker; ``~/.herdr/worktrees/<repo>-<branch>`` encodes it
    after. Only a match against an existing wing counts, so a branch name can
    never become a wing.
    """
    # The home directory's own name is a container, never a project.
    skip = _NO_PROJECT_WINGS | {Path.home().name}
    known = [w for w in known if w not in skip]
    match = _WORKTREE_RE.match(key.strip())
    if not match:
        return None

    def by_repo() -> Optional[str]:
        repo = match.group("repo").strip("-")
        if not repo:
            return None
        snapped = snap_to_existing(resolve_target(repo, [])[0], known)
        return snapped if snapped not in skip else None

    def by_branch_prefix() -> Optional[str]:
        rest = match.group("rest").lower()
        best: Optional[str] = None
        for wing in known:
            name = wing.lower()
            if (rest == name or rest.startswith(name + "-")) and (
                best is None or len(wing) > len(best)
            ):
                best = wing
        return best

    # ``~/.herdr/worktrees/<repo>-<branch>`` names the repo after the marker;
    # everywhere else it is in front.
    if "herdr-worktrees" in key.lower():
        return by_branch_prefix() or by_repo()
    return by_repo() or by_branch_prefix()


def filter_plan(plan: dict, allowed: Iterable[str]) -> dict:
    """Drop every project whose target is not in ``allowed``; those drawers stay."""
    allowed = set(allowed)
    kept = {k: v for k, v in plan["projects"].items() if v["target"] in allowed}
    dropped = sum(int(v["drawers"]) for k, v in plan["projects"].items() if k not in kept)
    return {**plan, "projects": kept, "unresolved": plan["unresolved"] + dropped}


def plan_split(col, wing: str, existing_wings: Iterable[str] = ()) -> dict:
    """Group the wing's drawers by project key and resolve targets.

    Returns ``{"wing", "planned_at", "projects": {key: {"target", "how",
    "drawers", "existing"}}, "unresolved": n}``; drawers whose path carries no
    project, or whose project derives no wing, stay where they are and are only
    counted. ``existing`` says whether the target wing already holds drawers.
    """
    known = {w for w in existing_wings if w != wing}
    counts: dict[str, int] = defaultdict(int)
    samples: dict[str, list[str]] = defaultdict(list)
    unresolved = 0
    for _, meta in _iter_wing_rows(col, wing):
        source_file = meta.get("source_file")
        key = project_key(source_file)
        if key is None:
            unresolved += 1
            continue
        counts[key] += 1
        if len(samples[key]) < _SAMPLES_PER_KEY and source_file not in samples[key]:
            samples[key].append(source_file)
    projects = {}
    for key in sorted(counts, key=lambda k: -counts[k]):
        target, how = resolve_target(key, samples[key])
        if how in ("path", "none"):
            snapped = collapse_worktree_key(key, known) or (
                snap_to_existing(target, known) if how == "path" else None
            )
            if snapped and snapped not in _NO_PROJECT_WINGS:
                target, how = snapped, "existing"
        if target == wing or how == "none":
            unresolved += counts[key]
            continue
        projects[key] = {
            "target": target,
            "how": how,
            "drawers": counts[key],
            "existing": target in known,
        }
    return {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "wing": wing,
        "planned_at": datetime.now(timezone.utc).isoformat(),
        "projects": projects,
        "unresolved": unresolved,
    }


def plans_dir(config: MempalaceConfig, override: Optional[str] = None) -> str:
    """Where plan and pending files live: ``override`` or ``<palace>/wings``."""
    if override:
        return os.path.abspath(os.path.expanduser(override))
    return os.path.join(config.palace_path, "wings")


def split_plan_path(config: MempalaceConfig, wing: str, directory: Optional[str] = None) -> str:
    return os.path.join(plans_dir(config, directory), f"split-{sanitize_name(wing, 'wing')}.json")


def save_split_plan(config: MempalaceConfig, plan: dict, directory: Optional[str] = None) -> str:
    path = split_plan_path(config, plan["wing"], directory)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)
    return path


def load_split_plan(config: MempalaceConfig, wing: str, directory: Optional[str] = None) -> dict:
    with open(split_plan_path(config, wing, directory), encoding="utf-8") as f:
        plan = json.load(f)
    if not isinstance(plan, dict):
        raise ValueError("split plan is not an object")
    # ``apply_split`` re-keys the wing the plan names, so a copied or edited
    # file naming another wing would move the wrong drawers.
    if str(plan.get("wing") or "") != wing:
        raise ValueError(
            f"plan is for wing {plan.get('wing')!r}, not {wing!r}; re-run without --yes"
        )
    projects = plan.get("projects")
    if not isinstance(projects, dict) or not projects:
        raise ValueError("split plan has no projects")
    for key, entry in projects.items():
        target = entry.get("target") if isinstance(entry, dict) else None
        if not target:
            raise ValueError(f"project {key!r} has no target wing")
        entry["target"] = sanitize_name(str(target), "wing")
    return plan


def plan_targets(plan: dict) -> dict[str, int]:
    """``{target_wing: drawers}`` the plan would produce."""
    out: dict[str, int] = defaultdict(int)
    for entry in plan["projects"].values():
        out[entry["target"]] += int(entry.get("drawers") or 0)
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


# ── apply ────────────────────────────────────────────────────────────────────


def _rekey_collection(col, wing: str, targets: dict[str, str], stamp: str, progress=None):
    """Move every record of ``wing`` whose source file maps to a target.

    Returns ``(moved, per_target, skipped)``. Ids are materialized first:
    rewriting ``wing`` while paging a wing-scoped query would shift the pages
    under the cursor. The update carries only ``wing`` and ``last_modified``;
    the backend merges it into the stored metadata and leaves the document,
    embedding and FTS row alone, and the drawer id is never touched.
    """
    per_target: dict[str, int] = defaultdict(int)
    moved = skipped = 0
    batch_ids: list[str] = []
    batch_metas: list[dict] = []

    def flush():
        nonlocal moved
        if not batch_ids:
            return
        col.update(ids=list(batch_ids), metadatas=list(batch_metas))
        moved += len(batch_ids)
        batch_ids.clear()
        batch_metas.clear()
        if progress:
            progress(moved)

    rows = [(row_id, meta.get("source_file")) for row_id, meta in _iter_wing_rows(col, wing)]
    keys: dict[Optional[str], Optional[str]] = {}
    for row_id, source_file in rows:
        if source_file not in keys:
            keys[source_file] = project_key(source_file)
        target = targets.get(keys[source_file] or "")
        if not target or target == wing:
            skipped += 1
            continue
        batch_ids.append(row_id)
        batch_metas.append({"wing": target, "last_modified": stamp})
        per_target[target] += 1
        if len(batch_ids) >= _UPDATE_BATCH:
            flush()
    flush()
    return moved, per_target, skipped


def split_pending_path(config: MempalaceConfig, wing: str, directory: Optional[str] = None) -> str:
    """Marker that a split of ``wing`` started and has not finished every phase."""
    return os.path.join(
        plans_dir(config, directory), f"split-{sanitize_name(wing, 'wing')}.pending"
    )


def split_plan_fingerprint(plan: dict) -> str:
    """Digest of the wing and each project's target — the decisions apply_split follows."""
    targets = {str(key): str(entry["target"]) for key, entry in sorted(plan["projects"].items())}
    payload = json.dumps(
        {"wing": str(plan["wing"]), "targets": targets},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_split_pending(
    config: MempalaceConfig, wing: str, plan: dict, directory: Optional[str] = None
) -> str:
    """Record that a split started, pinning the plan targets being applied."""
    path = split_pending_path(config, wing, directory)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "plan_sha256": split_plan_fingerprint(plan),
    }
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)
    return path


def assert_split_pending_matches_plan(
    config: MempalaceConfig, wing: str, plan: dict, directory: Optional[str] = None
) -> None:
    """Refuse a retry whose saved plan targets no longer match the pinned split."""
    path = split_pending_path(config, wing, directory)
    with open(path, encoding="utf-8") as f:
        raw = f.read().strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = None
    pinned = data.get("plan_sha256") if isinstance(data, dict) else None
    current = split_plan_fingerprint(plan)
    if not pinned or pinned != current:
        raise ValueError(
            f"split plan for {wing!r} changed since this split started "
            f"(pinned {pinned or 'missing'}, current {current}); "
            "restore the original plan targets before re-running --yes, "
            "or remove the pending marker only after deciding how to finish"
        )


def apply_split(col, plan: dict, progress=None, closets_col=None) -> dict:
    """Rewrite ``wing`` for every drawer whose project key is in the plan.

    Returns ``{"moved", "per_target", "skipped", "closets_moved"}``. Drawers
    whose key is missing from the plan (or resolves back to the source wing)
    are skipped. ``closets_col`` — the AAAK index layer, keyed by the same
    ``source_file`` and filtered by the same ``wing`` at search time — is
    re-keyed by the same rule, so a wing-scoped search keeps its index boosts.

    Interruption: rows move in batches, each batch one backend write, and a
    row is only ever wholly in its old wing or wholly in its new one. The loop
    reads the rows *currently* in the source wing, so running the same plan
    again after a crash or Ctrl-C moves exactly the remainder. A completed
    split re-run changes nothing. Nothing is deleted and no content is
    rewritten at any point.

    Fork delta: upstream also drops the source wing's hallway records here. We
    keep them: ``hallways.json`` is shared across the palace and 99.8% of its
    records are ``sessions`` wing, so the drop would empty it, and this fork
    has no ``hallways --rebuild`` to recompute them.
    """
    wing = plan["wing"]
    targets = {key: entry["target"] for key, entry in plan["projects"].items()}
    stamp = datetime.now(timezone.utc).isoformat()
    moved, per_target, skipped = _rekey_collection(col, wing, targets, stamp, progress)
    closets_moved = 0
    if closets_col is not None:
        closets_moved, _, _ = _rekey_collection(closets_col, wing, targets, stamp)
    return {
        "moved": moved,
        "per_target": dict(sorted(per_target.items(), key=lambda kv: -kv[1])),
        "skipped": skipped,
        "closets_moved": closets_moved,
    }
