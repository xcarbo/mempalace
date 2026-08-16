"""Static guard: read_only=True must always be paired with create=False.

``palace.get_collection`` defaults ``create=True``, and
``sqlite_exact._connect`` rejects ``create and read_only`` outright:

    ValueError: sqlite_exact read-only connections cannot create a palace

So a call site that adds ``read_only=True`` without also passing
``create=False`` does not degrade — it raises on every call. That is not
theoretical: it shipped. The 3.7.1 cutover added ``read_only=True`` to
mempalace-api's collection open and to ``exporter.export_palace`` without
``create=False``, which 500'd every ``/drawers`` and ``/wings`` endpoint on the
live read API while ``/search`` and ``/health`` kept working (they reach the
palace by other routes) — so the cutover smoke test missed it entirely, and the
session-start hook's pinned-drawer load was failing silently.

An AST check rather than a runtime one, because the failing sites are spread
across CLI, hooks, service and export paths that no single test exercises.
"""

import ast
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
PACKAGE = REPO_ROOT / "mempalace"


def _kwarg(call: ast.Call, name: str):
    for kw in call.keywords:
        if kw.arg == name:
            return kw
    return None


def _is_true(node) -> bool:
    return isinstance(node, ast.Constant) and node.value is True


def _is_false(node) -> bool:
    return isinstance(node, ast.Constant) and node.value is False


def _collection_call_sites():
    """Yield (path, lineno, call) for every *_get_collection(...) call."""
    for path in sorted(PACKAGE.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:  # pragma: no cover - a parse error is its own failure
            pytest.fail(f"{path} does not parse: {exc}")
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name and name.endswith("get_collection"):
                yield path, node.lineno, node


def test_read_only_call_sites_also_pass_create_false():
    offenders = []
    for path, lineno, call in _collection_call_sites():
        read_only = _kwarg(call, "read_only")
        if read_only is None or not _is_true(read_only.value):
            continue
        create = _kwarg(call, "create")
        if create is None:
            offenders.append(
                f"{path.relative_to(REPO_ROOT)}:{lineno} — read_only=True, create omitted (defaults True)"
            )
        elif not _is_false(create.value):
            offenders.append(
                f"{path.relative_to(REPO_ROOT)}:{lineno} — read_only=True with a non-False create"
            )

    assert not offenders, (
        "read_only=True requires create=False; sqlite_exact raises "
        "ValueError('read-only connections cannot create a palace') otherwise:\n  "
        + "\n  ".join(offenders)
    )


def test_the_guard_can_actually_see_call_sites():
    """Guard the guard: if the AST walk stops matching, the test above passes vacuously."""
    sites = list(_collection_call_sites())
    assert len(sites) > 10, f"expected many get_collection call sites, found {len(sites)}"
    read_only_sites = [
        (p, ln)
        for p, ln, c in sites
        if (_kwarg(c, "read_only") and _is_true(_kwarg(c, "read_only").value))
    ]
    assert read_only_sites, (
        "no read_only=True call sites found — the 3.7.1 lease fix would be missing"
    )
