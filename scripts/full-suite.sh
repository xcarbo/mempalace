#!/usr/bin/env bash
#
# full-suite.sh — the only run that may be called "green".
#
# Why this exists
# ---------------
# On 2026-07-29 commit 27a3437 moved two imports in mempalace/cli.py into
# function scope. It broke 23 test sites that patched them as module
# attributes. It shipped because the change was verified against a 705-test
# subset; the full suite stops at the first of them. A subset that passes says
# nothing about a suite that does not.
#
# It also guards a second failure mode a subset cannot see: the suite can die
# from a native crash (SIGBUS/SIGSEGV inside chromadb's SQLite) rather than a
# test failure. pytest prints no summary line in that case, so "no FAILED lines
# in the output" is not evidence of success. Only the exit code is.
#
# What it does
# ------------
#   1. Strips the ambient MEMPALACE_* variables the user's shell exports.
#   2. Runs the whole suite, serially.
#   3. Names the signal explicitly when the interpreter dies.
#   4. Runs ruff check + ruff format --check.
#
# Usage:  ./scripts/full-suite.sh            (suite + lint)
#         ./scripts/full-suite.sh --cov      (also coverage, with the gate)

set -uo pipefail

cd "$(dirname "$0")/.." || exit 1

RUN=(env -u MEMPALACE_RERANK_URL -u MEMPALACE_RERANK_MODEL
     -u MEMPALACE_PALACE_PATH -u MEMPAL_PALACE_PATH
     -u MEMPALACE_BACKEND -u MEMPALACE_BACKEND_EXPLICIT
     -u MEMPALACE_WRITE_LOG -u MEMPALACE_WRITE_WATCHDOG_SECONDS
     uv run pytest tests/ --ignore=tests/benchmarks)

if [ "${1:-}" = "--cov" ]; then
    "${RUN[@]}" -q --cov=mempalace --cov-report=term-missing
else
    "${RUN[@]}" -q
fi
status=$?

if [ "$status" -gt 128 ]; then
    signal=$((status - 128))
    echo ""
    echo "FAILED: the test process was killed by signal ${signal} (exit ${status})."
    echo "This is a native crash, not a test failure — pytest never printed a"
    echo "summary, so any counts above are partial. Do not read this as green."
    exit "$status"
fi

if [ "$status" -ne 0 ]; then
    echo ""
    echo "FAILED: pytest exited ${status}."
    exit "$status"
fi

echo ""
echo "--- ruff check ---"
uv run ruff check . || exit 1
echo "--- ruff format --check ---"
uv run ruff format --check . || exit 1

echo ""
echo "Full suite + lint clean. This run may be called green."
