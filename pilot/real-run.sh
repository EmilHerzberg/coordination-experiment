#!/usr/bin/env bash
# real-run.sh — start or RESUME the real 840-event run for one level (D32-aware).
#
#   bash real-run.sh <1|2> [--checkpoint]
#
#   --checkpoint   PILOT5_EVENT_LIMIT=50: only the first 50 events of the
#                  seed-shuffled order (D32). Run again WITHOUT the flag to
#                  resume the remaining events; nothing on disk is re-run.
#
# Environment overrides (all optional):
#   CAP_USD        per-SESSION cost cap. Default $8 with --checkpoint, $60 without.
#   CONCURRENCY    ladders in flight per process. Default 6.
#   DRY_RUN=1      print the assembled command and exit WITHOUT starting anything.
#                  This is the ONLY way to test this wrapper: on 2026-09-06 a
#                  PATH shim for `node` was bypassed by the Windows exe lookup
#                  and a real paid session started (55 calls, $0.04, no ladder
#                  completed, orphans purged on the next resume — see D32).
#
# What the wrapper pins, so a typo cannot start the wrong experiment:
#   * the packs file is always evidence_packs_sample.json (the pinned 840);
#     an unset PILOT5_PACKS would silently run on SYNTHETIC evidence;
#   * PILOT5_REPEATS=1 (plan §3, D31/IV-1);
#   * --fresh is refused here: it destroys the level. Type the raw command if
#     you really mean it.
set -euo pipefail
cd "$(dirname "$0")"

LEVEL="${1:-}"
if [[ "$LEVEL" != "1" && "$LEVEL" != "2" ]]; then
  echo "usage: bash real-run.sh <1|2> [--checkpoint]" >&2
  exit 2
fi
shift
LIMIT=""
for arg in "$@"; do
  case "$arg" in
    --checkpoint) LIMIT=50 ;;
    --fresh) echo "refusing --fresh: it throws the whole level away. Use the raw run-pilot5.ts command if you mean it." >&2; exit 2 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

PACKS=evidence_packs_sample.json
if [[ ! -f "../../pilot5/$PACKS" ]]; then
  echo "missing pilot5/$PACKS — build it first:" >&2
  echo "  python scripts/20_build_evidence_packs.py --sample event_sample.txt --out scaffold/pilot5/$PACKS" >&2
  exit 2
fi
if [[ ! -f "../../.env" ]]; then
  echo "missing thesis/scaffold/.env (OPENROUTER_API_KEY=...)" >&2
  exit 2
fi

if [[ -n "$LIMIT" ]]; then CAP="${CAP_USD:-8}"; else CAP="${CAP_USD:-60}"; fi
CONC="${CONCURRENCY:-6}"

echo "real run — level $LEVEL | packs $PACKS | repeats 1 | concurrency $CONC | session cap \$$CAP | event limit ${LIMIT:-none (full 840)}"
echo "status log: ../../pilot5/L${LEVEL}_run_status.txt   (re-run this same command to resume)"

# `env` so the optional limit can be passed as an argument word rather than a
# shell assignment prefix (an expanded prefix would be read as a command name).
CMD=(env PILOT5_PACKS="$PACKS" PILOT5_LEVEL="$LEVEL" PILOT5_REPEATS=1
  PILOT5_CONCURRENCY="$CONC" PILOT5_CAP_USD="$CAP" ${LIMIT:+PILOT5_EVENT_LIMIT=$LIMIT}
  node --experimental-transform-types --import ./register.mjs run-pilot5.ts full)
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1 — would exec:"; printf '  %q' "${CMD[@]}"; echo
  exit 0
fi
exec "${CMD[@]}"
