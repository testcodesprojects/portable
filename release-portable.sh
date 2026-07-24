#!/usr/bin/env bash
# release-portable.sh — ONE command to publish adv_sTiles -> stiles-portable.
#
# It: (1) previews the sync, (2) applies it (adv solver sources -> portable),
# (3) runs portable's MinGW cross-syntax portability probe as a gate, and
# (4) commits + pushes to the portable repo (which triggers CI to build the
# real per-platform artifacts on manylinux / macOS-arm / ubuntu-arm).
#
#   ./release-portable.sh          # sync + probe + show diff + CONFIRM + commit + push
#   ./release-portable.sh --dry    # PREVIEW only — show what would sync, change nothing
#   ./release-portable.sh --yes    # skip the confirm prompt (non-interactive / scripted)
#   ./release-portable.sh --build  # also run a full local `make` before pushing (slower; CI does this too)
#   ./release-portable.sh -m "..." # custom commit message
#
# The confirm prompt is your review gate: a sync can carry behavior/perf drift
# (e.g. adv's api.cpp/collect.cpp work), so glance at the diff before pushing.
# CI is the real build authority — the local probe here is a fast pre-flight.
set -euo pipefail

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYNC="$SELF/sync-portable.sh"
[ -x "$SYNC" ] || { echo "ERROR: sync-portable.sh not found/executable next to this script ($SYNC)"; exit 1; }

# Resolve adv (source) and portable (target) the same sibling-relative way as
# sync-portable.sh, so this works whether run from adv_sTiles/ or stiles-portable/.
ADV="${STILES_ADV_DIR:-$SELF/../adv_sTiles}"
POR="${STILES_PORTABLE_DIR:-$SELF/../stiles-portable}"
[ -d "$ADV/tools" ] || { echo "ERROR: adv tree not at $ADV/tools (set STILES_ADV_DIR)"; exit 1; }
[ -d "$POR/tools" ] || { echo "ERROR: portable tree not at $POR/tools (set STILES_PORTABLE_DIR)"; exit 1; }
ADV="$(cd "$ADV" && pwd)"; POR="$(cd "$POR" && pwd)"

DRY=0; YES=0; BUILD=0; MSG=""
while [ $# -gt 0 ]; do
  case "$1" in
    --dry|-n)  DRY=1 ;;
    --yes|-y)  YES=1 ;;
    --build)   BUILD=1 ;;
    -m)        shift; MSG="${1:-}" ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "usage: $0 [--dry] [--yes] [--build] [-m msg]"; exit 2 ;;
  esac
  shift
done

echo "======================================================================"
echo " release-portable :  $ADV/tools  ->  $POR/tools"
echo "======================================================================"

# -- 1. preview -------------------------------------------------------------
echo; echo "==> [1/4] preview (what would sync)"
"$SYNC"
n=$("$SYNC" 2>/dev/null | grep -c '^>f' || true)
echo "    $n file(s) differ."

if [ "$DRY" = 1 ]; then echo; echo "(--dry) preview only -- nothing changed."; exit 0; fi
if [ "$n" -eq 0 ] && [ -z "$(git -C "$POR" status --porcelain)" ]; then
  echo; echo "Nothing to release -- portable already matches adv and its tree is clean."; exit 0
fi

# -- 2. apply ---------------------------------------------------------------
echo; echo "==> [2/4] applying sync"
"$SYNC" --apply >/dev/null
echo "    done."

# -- 3. portability gate (fast) + optional full build -----------------------
echo; echo "==> [3/4] portability probe (make cross-syntax)"
( cd "$POR" && make cross-syntax >/dev/null 2>&1 ) || true
fails=$(grep -cE '^  FAIL ' "$POR/cross-syntax.log" 2>/dev/null || echo 0)
if [ "${fails:-0}" -gt 0 ]; then
  echo "    ABORT: $fails file(s) FAILED the portability probe. See $POR/cross-syntax.log"
  echo "    (sync is applied locally but NOT committed -- fix in adv, guard the construct, re-run.)"
  exit 1
fi
echo "    clean."
if [ "$BUILD" = 1 ]; then
  echo "    full build (--build) ..."
  ( cd "$POR" && make >/dev/null 2>&1 ) && echo "    build ok." || { echo "    ABORT: full build failed."; exit 1; }
fi

# -- 4. publish (commit + push) ---------------------------------------------
echo; echo "==> [4/4] publish to portable git"
cd "$POR"
git add -A
if git diff --cached --quiet; then echo "    nothing staged -- already committed."; exit 0; fi
echo "    staged changes:"
git --no-pager diff --cached --stat | sed 's/^/      /'
: "${MSG:=sync from adv_sTiles ($(date +%Y-%m-%d))}"

if [ "$YES" != 1 ]; then
  if [ -t 0 ]; then
    read -r -p "    Commit + push to portable (triggers CI)? [y/N] " ans
  else
    echo "    non-interactive and --yes not given: NOT pushing."
    echo "    Changes are staged in $POR. Commit yourself, or re-run with --yes."
    exit 0
  fi
  case "$ans" in y|Y|yes|YES) ;; *) echo "    aborted -- changes staged but not committed."; exit 0 ;; esac
fi

git commit -m "$MSG" >/dev/null
git push
echo "    pushed. Watch CI in the portable repo's Actions tab."
