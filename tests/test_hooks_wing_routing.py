"""Fork wing-routing invariants (xdev-patches).

Pins the wing names the hook flow produces so upstream merges can never
silently rename wings in an existing palace. The production palace carries
~80k drawers, including legacy path-encoded wings with leading dashes and
mixed case (e.g. ``-Volumes-codeXD-vlad-ozweb``); a renamed wing means new
drawers split away from the existing wing — the exact failure class upstream
PR #1852 (``_safe_wing_slug``, commit aac947a, on upstream/develop but NOT in
v3.5.0) would introduce if it ever routed our derivation paths.

Invariant: the expected literals below pin the wing names the current
derivation produces; any change to them renames wings in the production
palace and needs a matching migrate-wings pass. Since 2026-07-07 the
derivation emits BARE hyphenated names (``cc``, ``hunt-1``) instead of the
historical ``wing_`` + underscore slugs, folding hook checkpoints into the
same wings deliberate writes use.

Fork patches covered:
- bf6278c — ``.palace-wing`` in a repo root pins the wing verbatim, before
  any leaf-slug derivation.
- config.py ``_SAFE_NAME_RE`` — accepts leading-dash wing names so legacy
  path-encoded wings stay writable.
"""

import json
from pathlib import Path

import pytest

from mempalace.config import sanitize_name
from mempalace.convo_miner import _resolve_wing
from mempalace.hooks_cli import _wing_from_transcript_path

LEGACY_WING = "-Volumes-codeXD-vlad-ozweb"


def _write_transcript(tmp_path: Path, cwd: Path) -> str:
    """Create a minimal Claude Code JSONL transcript recording ``cwd``."""
    transcript = tmp_path / "session.jsonl"
    transcript.write_text(
        json.dumps({"type": "user", "cwd": str(cwd), "content": "hi"}) + "\n",
        encoding="utf-8",
    )
    return str(transcript)


def _project_with_palace_wing(tmp_path: Path, leaf: str, pinned: str) -> Path:
    project = tmp_path / leaf
    project.mkdir(parents=True)
    (project / ".palace-wing").write_text(pinned, encoding="utf-8")
    return project


# --- .palace-wing override (fork patch bf6278c) ---


def test_palace_wing_pins_wing_verbatim_regardless_of_cwd_name(tmp_path):
    """A ``.palace-wing`` file wins over leaf-slug derivation, verbatim."""
    project = _project_with_palace_wing(tmp_path, "upgrade-v350", "mempalace\n")
    transcript = _write_transcript(tmp_path, project)
    assert _wing_from_transcript_path(transcript) == "mempalace"


@pytest.mark.parametrize(
    "pinned",
    [
        "torq-terminal",  # dashes preserved (utilities/torq → torq-terminal)
        LEGACY_WING,  # leading dash + mixed case + dashes preserved
        "MixedCase-Wing",  # no lowercasing
    ],
)
def test_palace_wing_pinned_name_is_byte_identical(tmp_path, pinned):
    """Pinned wing names pass through untouched — no slugging, no case change.

    This is the load-bearing guard against ``_safe_wing_slug``-style
    normalization: a pinned legacy wing like ``-Volumes-codeXD-vlad-ozweb``
    must never become ``volumes_codexd_vlad_ozweb``.
    """
    project = _project_with_palace_wing(tmp_path, "whatever-dir", pinned + "\n")
    transcript = _write_transcript(tmp_path, project)
    assert _wing_from_transcript_path(transcript) == pinned


def test_palace_wing_first_line_wins(tmp_path):
    project = _project_with_palace_wing(tmp_path, "someproj", "mempalace\nsecond line is ignored\n")
    transcript = _write_transcript(tmp_path, project)
    assert _wing_from_transcript_path(transcript) == "mempalace"


def test_palace_wing_empty_file_falls_back_to_leaf_slug(tmp_path):
    project = _project_with_palace_wing(tmp_path, "My-Proj", "")
    transcript = _write_transcript(tmp_path, project)
    assert _wing_from_transcript_path(transcript) == "my-proj"


# --- legacy path-encoded wings stay byte-identical end to end ---


def test_sanitize_name_accepts_legacy_leading_dash_wing():
    """Fork config patch: leading-dash path-encoded wings are valid names.

    Upstream's ``sanitize_name`` rejects a leading dash; our ``_SAFE_NAME_RE``
    accepts it so writes to existing legacy wings keep landing in them.
    """
    assert sanitize_name(LEGACY_WING) == LEGACY_WING


def test_convo_miner_explicit_wing_passes_through_verbatim():
    """An explicit wing always wins in ``_resolve_wing`` — byte-identical.

    ``memp ... --wing -Volumes-codeXD-vlad-ozweb`` (and the hook's
    ``_ingest_transcript`` spawn) must never have the wing re-slugged en
    route to the miner.
    """
    assert _resolve_wing(Path("/tmp/anything.jsonl"), LEGACY_WING) == LEGACY_WING


# --- cwd leaf derivation: bare hyphenated wing convention (2026-07-07) ---
#
# Deliberate convention change: hook derivation now emits the BARE
# hyphenated leaf (lowercase, spaces→hyphens, everything else verbatim) —
# the same names deliberate writes and pinned roadmaps use — instead of the
# historical ``wing_`` + dash→underscore slug that splintered ~40 wings.
# The old splinter wings are folded back via the migrate-wings pass.
# Upstream ``_safe_wing_slug``-style normalization must still never route
# these paths (the '+' tripwire below).


@pytest.mark.parametrize(
    ("leaf", "expected"),
    [
        ("myproj", "myproj"),
        ("MyProj", "myproj"),
        ("vlad-ozweb", "vlad-ozweb"),
        ("my.app", "my.app"),
        # Tripwire: upstream _safe_wing_slug (aac947a) would emit
        # "foo_bar" here. The fork maps invalid runs to hyphens instead
        # (_bare_wing_slug, v3.6.0 merge). The pre-merge verbatim "foo+bar"
        # was never writable — sanitize_name rejects '+' — so no existing
        # wing can carry that name and this pin change renames nothing.
        ("foo+bar", "foo-bar"),
    ],
)
def test_cwd_leaf_derivation_matches_pre_merge_fork(tmp_path, leaf, expected):
    project = tmp_path / leaf
    project.mkdir()
    transcript = _write_transcript(tmp_path, project)
    assert _wing_from_transcript_path(transcript) == expected


# --- transcript-path fallbacks (no cwd in JSONL): pre-merge parity ---


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        # Encoded projects folder for a legacy path-encoded dir. The encoded
        # fallback has never emitted the verbatim leading-dash form; legacy
        # wings are reached via explicit --wing or .palace-wing (above).
        (
            "/Users/xdev/.claude/projects/-Volumes-codeXD-vlad-ozweb/s.jsonl",
            "volumes-codexd-vlad-ozweb",
        ),
        (
            "/Users/xdev/.claude/projects/-Users-xdev-code-mempalace/s.jsonl",
            "mempalace",
        ),
        ("/x/y-Projects-FooBar/s.jsonl", "foobar"),
        ("/some/random/path.jsonl", "sessions"),
    ],
)
def test_transcript_path_fallbacks_match_pre_merge_fork(path, expected):
    assert _wing_from_transcript_path(path) == expected
