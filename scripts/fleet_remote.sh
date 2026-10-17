# shellcheck shell=bash
# shellcheck disable=SC2153 # CMD, KEEP, ID, ... arrive in the header scripts/fleet prepends
# fleet_remote.sh: the box half of scripts/fleet. Never run by hand.
#
# scripts/fleet pipes this file to `ssh <box> bash -s`, preceded by a header of
# variable assignments (CMD, VENV_LINK, SRC_LINK, ...). Results go to stdout as
# key=value lines; progress goes to stderr. A non-zero exit means the step
# failed and nothing after it should run.
#
# Layout it maintains (every link is swapped by rename(2), so it is atomic):
#   ~/opt/mempalace/.venv       -> .venv.<id>           (uv venv, box-built Python)
#   ~/opt/mempalace-src         -> mempalace-src.<id>   (git archive, .deployed-from inside)
#   ~/opt/mempalace-api         -> mempalace-api.<id>   (only with --api)
#   ~/opt/.fleet/               state: <component>.prev, <component>.history, deploys.log, lock
# It touches code and venvs only. It never reads or writes ~/.mempalace except
# through read-only `memp status` and the read API.

set -euo pipefail

emit() { printf '%s=%s\n' "$1" "$2"; }
say() { printf '  %s\n' "$*" >&2; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

STATE_DIR="${STATE_DIR:-$HOME/opt/.fleet}"
SYS_PY="${SYS_PY:-/usr/bin/python3}"
UV="${UV:-$HOME/.local/bin/uv}"

# ── links ────────────────────────────────────────────────────────────────

# link_id <link> <prefix>: the id a link points at, "dir" for a pre-fleet real
# directory, "missing" when absent.
link_id() {
  local link=$1 prefix=$2 target
  if [ -L "$link" ]; then
    target=$(readlink "$link")
    printf '%s' "${target#"$prefix"}"
  elif [ -d "$link" ]; then
    printf 'dir'
  else
    printf 'missing'
  fi
}

# exchange <a> <b>: swap two paths in one renameat2(RENAME_EXCHANGE) call.
# Used once per link, to turn a pre-fleet real directory into a symlink with
# no instant where the path is missing.
exchange() {
  "$SYS_PY" - "$1" "$2" <<'PY'
import ctypes, os, sys
libc = ctypes.CDLL(None, use_errno=True)
AT_FDCWD, RENAME_EXCHANGE = -100, 2
if libc.renameat2(AT_FDCWD, os.fsencode(sys.argv[1]), AT_FDCWD, os.fsencode(sys.argv[2]), RENAME_EXCHANGE) != 0:
    sys.exit("renameat2 exchange failed: " + os.strerror(ctypes.get_errno()))
PY
}

# point_link <link> <prefix> <id> <legacy-id>: make <link> point at <prefix><id>,
# a sibling. A pre-fleet real directory is kept as <prefix><legacy-id>.
point_link() {
  local link=$1 prefix=$2 id=$3 legacy=$4
  local dir new
  dir=$(dirname "$link")
  new="$dir/.$(basename "$link").fleet-new"
  [ -d "$dir/$prefix$id" ] || die "missing $dir/$prefix$id"
  rm -f "$new"
  ln -s "$prefix$id" "$new"
  if [ -L "$link" ] || [ ! -e "$link" ]; then
    mv -T "$new" "$link"
  elif [ -d "$link" ]; then
    [ -n "$legacy" ] || die "$link is a real directory and no legacy id was given"
    [ ! -e "$dir/$prefix$legacy" ] || die "$dir/$prefix$legacy already exists"
    exchange "$new" "$link"
    mv -T "$new" "$dir/$prefix$legacy"
    say "kept the pre-fleet $(basename "$link") as $prefix$legacy"
  else
    die "$link exists and is neither a link nor a directory"
  fi
}

# ── state ────────────────────────────────────────────────────────────────

record_swap() { # <component> <from-id> <to-id>
  mkdir -p "$STATE_DIR"
  printf '%s\n' "$2" >"$STATE_DIR/$1.prev"
  printf '%s\n' "$3" >>"$STATE_DIR/$1.history"
}

log_line() {
  mkdir -p "$STATE_DIR"
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >>"$STATE_DIR/deploys.log"
}

# prune <component> <parent> <prefix> <current> [<also-keep>]: delete
# <parent>/<prefix><id> dirs except the current, the previous, and the KEEP
# most recent ids in the history.
prune() {
  local comp=$1 parent=$2 prefix=$3 current=$4 also=${5:-}
  local prev keep d id
  prev=$(cat "$STATE_DIR/$comp.prev" 2>/dev/null || true)
  keep=" $current $prev $also "
  if [ -f "$STATE_DIR/$comp.history" ]; then
    keep+=" $(tac "$STATE_DIR/$comp.history" | awk '!seen[$0]++' | head -n "$KEEP" | tr '\n' ' ') "
  fi
  for d in "$parent/$prefix"*; do
    [ -d "$d" ] && [ ! -L "$d" ] || continue
    id=${d#"$parent/$prefix"}
    case "$id" in *.tmp|*.fleet-new) ;; *) case "$keep" in *" $id "*) continue ;; esac ;; esac
    say "pruning $d"
    rm -rf "$d"
  done
}

api_drawers() { # prints the drawer count from /stats, or nothing
  curl -s -m 5 "http://127.0.0.1:$API_PORT/stats" 2>/dev/null |
    "$SYS_PY" -c 'import json,sys
try: print(json.load(sys.stdin)["drawers"])
except Exception: pass' || true
}

# ── commands ─────────────────────────────────────────────────────────────

cmd_probe() {
  local svc py
  emit host "$(hostname)"
  emit core_id "$(link_id "$VENV_LINK" .venv.)"
  emit src_id "$(link_id "$SRC_LINK" mempalace-src.)"
  emit deployed_from "$(head -1 "$SRC_LINK/.deployed-from" 2>/dev/null || true)"
  emit core_prev "$(cat "$STATE_DIR/core.prev" 2>/dev/null || true)"
  emit version "$("$VENV_LINK/bin/mempalace" --version 2>&1 | head -1 || true)"
  py="$VENV_LINK/bin/python"
  emit pyinfo "$("$py" -c 'import sys, sqlite3; print(sys.version.split()[0], sqlite3.sqlite_version)' 2>&1 | head -1 || true)"
  emit pyhome "$(sed -n 's/^home = //p' "$VENV_LINK/pyvenv.cfg" 2>/dev/null || true)"
  emit api_active "$(systemctl --user is-active "$API_UNIT" 2>/dev/null || true)"
  emit api_drawers "$(api_drawers)"
  emit api_id "$(link_id "$API_LINK" mempalace-api.)"
  emit api_prev "$(cat "$STATE_DIR/api.prev" 2>/dev/null || true)"
  emit api_deployed_from "$(head -1 "$API_LINK/.deployed-from" 2>/dev/null || true)"
  for f in $API_HASH_FILES; do
    emit "api_sha256:$f" "$(sha256sum <"$API_LINK/$f" 2>/dev/null | cut -d' ' -f1 || true)"
  done
  if [ -n "${BACKUP_TIMER:-}" ]; then
    svc=$(systemctl --user show "$BACKUP_TIMER" -p Unit --value 2>/dev/null || true)
    emit backup_unit "$svc"
    emit backup_last_trigger "$(systemctl --user show "$BACKUP_TIMER" -p LastTriggerUSec --value 2>/dev/null || true)"
    if [ -n "$svc" ]; then
      emit backup_result "$(systemctl --user show "$svc" -p Result --value 2>/dev/null || true)"
      emit backup_exit "$(systemctl --user show "$svc" -p ExecMainStatus --value 2>/dev/null || true)"
      emit backup_state "$(systemctl --user show "$svc" -p ActiveState --value 2>/dev/null || true)"
    fi
  fi
  if [ -d "$STATE_DIR/deploy.lock" ]; then emit lock "$(cat "$STATE_DIR/deploy.lock/owner" 2>/dev/null || echo held)"; fi
}

cmd_lock() {
  mkdir -p "$STATE_DIR"
  if ! mkdir "$STATE_DIR/deploy.lock" 2>/dev/null; then
    local age
    age=$(($(date +%s) - $(stat -c %Y "$STATE_DIR/deploy.lock")))
    if [ "$age" -lt 1800 ]; then
      die "another fleet run holds $STATE_DIR/deploy.lock ($(cat "$STATE_DIR/deploy.lock/owner" 2>/dev/null)), ${age}s old"
    fi
    say "taking over a stale lock (${age}s old)"
  fi
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$OWNER" >"$STATE_DIR/deploy.lock/owner"
}

cmd_unlock() { rm -rf "$STATE_DIR/deploy.lock"; }

cmd_stage_core() {
  local parent src venv cons=()
  parent=$(dirname "$VENV_LINK")
  src="$(dirname "$SRC_LINK")/mempalace-src.$ID"
  venv="$parent/.venv.$ID"
  [ "$(link_id "$VENV_LINK" .venv.)" != "$ID" ] || die "$ID is already live"

  if [ ! -f "$src/.fleet-complete" ]; then
    say "extracting source into $src"
    rm -rf "$src.tmp" "$src"
    mkdir -p "$src.tmp"
    tar -xzf "$TARBALL" -C "$src.tmp"
    printf '%s %s %s\n' "$ID" "$BRANCH" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"$src.tmp/.deployed-from"
    touch "$src.tmp/.fleet-complete"
    mv -T "$src.tmp" "$src"
  else
    printf '%s %s %s\n' "$ID" "$BRANCH" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"$src/.deployed-from"
  fi
  rm -f "$TARBALL"

  if [ -f "$venv/.fleet-complete" ]; then
    say "reusing the complete venv $venv"
  else
    rm -rf "$venv"
    say "creating $venv on $PYBIN"
    "$UV" venv --quiet --relocatable --no-python-downloads --python "$PYBIN" "$venv"
    if [ "${FRESH_DEPS:-0}" != 1 ] && [ -x "$VENV_LINK/bin/python" ]; then
      # Pin every dependency to what the live venv runs, so a code deploy is
      # only a code change. --fresh-deps drops this.
      "$UV" pip freeze --python "$VENV_LINK/bin/python" 2>/dev/null |
        grep -viE '^mempalace([ =@]|$)' >"$venv/fleet-constraints.txt" || true
      cons=(--constraint "$venv/fleet-constraints.txt")
    fi
    say "installing $src (+ $EXTRA_PKGS)"
    # shellcheck disable=SC2086 # EXTRA_PKGS is a space-separated list
    "$UV" pip install --quiet --no-python-downloads --python "$venv/bin/python" "${cons[@]}" "$src" $EXTRA_PKGS >&2
  fi

  smoke_core "$venv"
  touch "$venv/.fleet-complete"
}

smoke_core() {
  local venv=$1 got out drawers
  say "smoke test: $venv"
  got=$("$venv/bin/python" -c 'import sys, sqlite3, mempalace, fastapi, uvicorn; print(sys.version.split()[0], sqlite3.sqlite_version)') ||
    die "the new venv cannot import mempalace, fastapi and uvicorn"
  [ "$got" = "$EXPECT_PY $EXPECT_SQLITE" ] || die "venv runs Python/SQLite '$got', expected '$EXPECT_PY $EXPECT_SQLITE'"
  emit smoke_pyinfo "$got"
  emit smoke_version "$("$venv/bin/mempalace" --version 2>&1 | head -1)"
  out=$("$venv/bin/memp" status 2>&1) || die "memp status failed: $(printf '%s' "$out" | tail -3)"
  drawers=$(printf '%s\n' "$out" | sed -n 's/.*-- \([0-9][0-9]*\) drawers.*/\1/p' | head -1)
  [ -n "$drawers" ] || die "memp status printed no drawer count: $(printf '%s' "$out" | head -3)"
  emit smoke_drawers "$drawers"
  (cd "$API_LINK" && "$venv/bin/python" -c 'import mempalace_api') >/dev/null 2>&1 ||
    die "the live mempalace-api code does not import under the new venv"
}

cmd_stage_api() {
  local dir py
  dir="$(dirname "$API_LINK")/mempalace-api.$API_ID"
  [ "$(link_id "$API_LINK" mempalace-api.)" != "$API_ID" ] || die "api $API_ID is already live"
  say "extracting mempalace-api into $dir"
  rm -rf "$dir.tmp" "$dir"
  mkdir -p "$dir.tmp"
  tar -xzf "$API_TARBALL" -C "$dir.tmp"
  rm -f "$API_TARBALL"
  printf '%s %s %s\n' "$API_ID" "$API_BRANCH" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"$dir.tmp/.deployed-from"
  mv -T "$dir.tmp" "$dir"
  # The venv the api will run under: the one being staged, else the live one.
  py="$(dirname "$VENV_LINK")/.venv.${ID:-x}/bin/python"
  [ -x "$py" ] || py="$VENV_LINK/bin/python"
  (cd "$dir" && "$py" -c 'import mempalace_api') >/dev/null 2>&1 || die "new mempalace-api does not import"
  emit api_staged "$API_ID"
}

cmd_preswap() { # refuse while a backup is reading the palace through the venv
  local svc state
  [ -n "${BACKUP_TIMER:-}" ] || return 0
  svc=$(systemctl --user show "$BACKUP_TIMER" -p Unit --value 2>/dev/null || true)
  [ -n "$svc" ] || return 0
  state=$(systemctl --user show "$svc" -p ActiveState --value 2>/dev/null || true)
  case "$state" in active | activating | deactivating) die "backup $svc is $state; deploy again when it ends" ;; esac
}

cmd_swap_core() {
  local cur src_cur legacy="" d
  cur=$(link_id "$VENV_LINK" .venv.)
  src_cur=$(link_id "$SRC_LINK" mempalace-src.)
  if [ "$cur" = dir ] || [ "$src_cur" = dir ]; then
    # First fleet deploy: the pre-fleet dirs are kept under the id they ran.
    legacy=$(head -1 "$SRC_LINK/.deployed-from" 2>/dev/null | cut -d' ' -f1 || true)
    legacy=${legacy:-pre-fleet}
    [ "$legacy" != "$ID" ] || die "$ID is already what the pre-fleet install runs"
    for d in "$(dirname "$VENV_LINK")/.venv.$legacy" "$(dirname "$SRC_LINK")/mempalace-src.$legacy"; do
      [ ! -e "$d" ] || die "$d already exists; cannot keep the pre-fleet install under that name"
    done
    [ "$cur" != dir ] || cur=$legacy
  fi
  # Check both targets before touching either link.
  [ -f "$(dirname "$VENV_LINK")/.venv.$ID/.fleet-complete" ] || die ".venv.$ID is not a complete, smoke-tested venv"
  [ -f "$(dirname "$SRC_LINK")/mempalace-src.$ID/.fleet-complete" ] || die "mempalace-src.$ID is incomplete"
  point_link "$SRC_LINK" mempalace-src. "$ID" "$legacy"
  point_link "$VENV_LINK" .venv. "$ID" "$legacy"
  record_swap core "$cur" "$ID"
  emit core_from "$cur"
  emit core_to "$ID"
}

cmd_swap_api() {
  local cur legacy=""
  cur=$(link_id "$API_LINK" mempalace-api.)
  if [ "$cur" = dir ]; then legacy=pre-fleet; cur=$legacy; fi
  point_link "$API_LINK" mempalace-api. "$API_ID" "$legacy"
  record_swap api "$cur" "$API_ID"
  emit api_from "$cur"
  emit api_to "$API_ID"
}

# rollback <component>: flip back to the recorded previous id. Run twice, it
# flips forward again.
cmd_rollback() {
  local comp=$COMPONENT cur prev
  prev=$(cat "$STATE_DIR/$comp.prev" 2>/dev/null || true)
  [ -n "$prev" ] || die "no previous $comp recorded in $STATE_DIR/$comp.prev"
  if [ "$comp" = core ]; then
    cur=$(link_id "$VENV_LINK" .venv.)
    [ -f "$(dirname "$VENV_LINK")/.venv.$prev/bin/python" ] || die "previous venv .venv.$prev is gone"
    point_link "$SRC_LINK" mempalace-src. "$prev" ""
    point_link "$VENV_LINK" .venv. "$prev" ""
  else
    cur=$(link_id "$API_LINK" mempalace-api.)
    point_link "$API_LINK" mempalace-api. "$prev" ""
  fi
  record_swap "$comp" "$cur" "$prev"
  emit "${comp}_from" "$cur"
  emit "${comp}_to" "$prev"
}

cmd_restart_api() {
  local n=""
  systemctl --user restart "$API_UNIT"
  for _ in $(seq 1 60); do
    n=$(api_drawers)
    [ -n "$n" ] && break
    sleep 1
  done
  [ -n "$n" ] || die "$API_UNIT gave no /stats answer within 60s"
  emit api_drawers "$n"
}

cmd_finish() { # log + prune
  log_line "$LOG_MSG"
  local cur
  cur=$(link_id "$VENV_LINK" .venv.)
  prune core "$(dirname "$VENV_LINK")" .venv. "$cur"
  prune core "$(dirname "$SRC_LINK")" mempalace-src. "$(link_id "$SRC_LINK" mempalace-src.)"
  if [ -f "$STATE_DIR/api.history" ]; then
    prune api "$(dirname "$API_LINK")" mempalace-api. "$(link_id "$API_LINK" mempalace-api.)"
  fi
  rm -f "$STATE_DIR"/incoming/*.tar.gz 2>/dev/null || true
}

case "$CMD" in
  probe) cmd_probe ;;
  lock) cmd_lock ;;
  unlock) cmd_unlock ;;
  stage_core) cmd_stage_core ;;
  stage_api) cmd_stage_api ;;
  preswap) cmd_preswap ;;
  swap_core) cmd_swap_core ;;
  swap_api) cmd_swap_api ;;
  rollback) cmd_rollback ;;
  restart_api) cmd_restart_api ;;
  finish) cmd_finish ;;
  *) die "unknown command '$CMD'" ;;
esac
