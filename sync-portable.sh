#!/usr/bin/env bash
# sync-portable.sh — regenerate stiles-portable's solver sources from adv_sTiles.
#
# adv_sTiles is the SINGLE SOURCE OF TRUTH. This copies the (flag-guarded) sTiles
# solver sources into the portable git repo so you never hand-copy again. mode-4
# code is compiled out in portable because it never defines STILES_MFRONT (the
# #ifdef STILES_MFRONT guards go inactive), so mfront/ itself is excluded here.
#
# What it does NOT touch (so portable stays self-contained & portable-owned):
#   - vendored deps (suitesparse, libxsmm, metis/GKlib/scotch, *_inline) — portable
#     keeps its own committed copies; these don't change during sTiles development
#   - build artifacts (*.o *.d *.a *.so) — rebuilt on each side
#   - mfront/ — mode 4 lives only in adv
#   - portable overlay: tools/module.mk (the `syntax:` MinGW-probe target) and
#     tools/process/Makefile (portable's STILES_SCOTCH_COMPRESS toggle)
#   - top-level Makefile / make.inc / README / .github — outside tools/, never synced
#
# Content-based (-c): only files whose CONTENT differs are shown/copied, so the
# dry run is an honest change list and an apply causes minimal git churn.
# No --delete: this never removes files from portable (review stale files by hand).
#
# Usage:
#   ./sync-portable.sh            # DRY RUN (default): prints exactly what would change
#   ./sync-portable.sh --apply    # actually copy
#
# After --apply, in the portable repo:
#   git diff            # review every change
#   make cross-syntax   # MinGW portability probe on the newly-synced code
#   make && ./bench/... # build + bench (first sync also pulls adv's api.cpp/collect.cpp
#                       #   INLA-affinity + semisparse-scatter improvements — real behavior change)
#   git add -A && git commit && git push
set -euo pipefail

# Resolve source (adv) and target (portable). Works whether this script lives in
# adv_sTiles/ or stiles-portable/ — they are siblings under .../ideas/. Override
# either with STILES_ADV_DIR / STILES_PORTABLE_DIR.
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ADV="${STILES_ADV_DIR:-$SELF/../adv_sTiles}"
POR="${STILES_PORTABLE_DIR:-$SELF/../stiles-portable}"

[ -d "$ADV/tools" ] || { echo "ERROR: adv tree not at $ADV/tools (set STILES_ADV_DIR)"; exit 1; }
[ -d "$POR/tools" ] || { echo "ERROR: portable tree not at $POR/tools (set STILES_PORTABLE_DIR)"; exit 1; }
ADV="$(cd "$ADV" && pwd)"; POR="$(cd "$POR" && pwd)"   # normalize (strip ../)
[ "$ADV" = "$POR" ] && { echo "ERROR: adv and portable resolve to the same dir ($ADV)"; exit 1; }

APPLY=0
case "${1:-}" in
  --apply) APPLY=1 ;;
  ""|--dry-run|-n) APPLY=0 ;;
  *) echo "usage: $0 [--apply]"; exit 2 ;;
esac
DRY="-n"; [ "$APPLY" = 1 ] && DRY=""

echo "sync: $ADV/tools/  ->  $POR/tools/"
[ "$APPLY" = 1 ] && echo "MODE: APPLY (writing)" || echo "MODE: DRY RUN (no changes; pass --apply to write)"
echo

rsync -rlpt -c --itemize-changes $DRY \
  --exclude='.git/' \
  --exclude='*.o' --exclude='*.d' --exclude='*.a' --exclude='*.so' --exclude='*.dylib' \
  --exclude='*.bak' --exclude='*_bak' --exclude='*.orig' --exclude='*.pre_*' --exclude='*~' \
  --exclude='mfront/' \
  --exclude='suitesparse/' \
  --exclude='libxsmm/' \
  --exclude='ordering/GKlib/' \
  --exclude='ordering/metis-5.1.0/' \
  --exclude='ordering/metis_inline/' \
  --exclude='ordering/scotch/' \
  --exclude='ordering/external/' \
  --exclude='ordering/fill-in/' \
  --exclude='libs/' \
  --exclude='**/build/' --exclude='**/*_local/' \
  --exclude='/module.mk' \
  --exclude='/process/Makefile' \
  "$ADV/tools/" "$POR/tools/"

echo
if [ "$APPLY" = 1 ]; then
  echo "DONE. Now in the portable repo ($POR):"
  echo "  git -C '$POR' status   # review"
  echo "  cd '$POR' && make cross-syntax && make   # portability probe + build"
else
  echo "(dry run) itemized lines above: '>f' = would copy a file, 'cd' = create dir. Nothing was written."
  echo "Re-run with --apply to sync for real."
fi
