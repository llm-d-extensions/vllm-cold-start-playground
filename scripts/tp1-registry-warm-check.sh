#!/usr/bin/env bash
# Check the subtraction that the ladder summary rests on.
#
# analysis/registry_ladder.py reports a "second boot" column as
# (first boot - the model registry resolve phase), and claims that equals what
# the same arm costs once the modelinfos cache is populated. That is an argument,
# not a measurement: the phase is serial and on the critical path, so removing it
# should leave the rest untouched. Arguments about startup timing have been wrong
# in this repo before, so measure it.
#
# Two arms, the ends of the ladder, three cycles each, interleaved cold/warm:
# boot with --cold-registry, then boot the same arm without it, and compare
# (cold total - cold registry phase) against the warm total. If the subtraction
# is sound the two agree inside the arm's own spread.
#
#   scripts/tp1-registry-warm-check.sh -n lionel-cold-start
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NS=""
POD="vllm-coldstart"
MODEL="Qwen/Qwen3-32B"
PREFIX="regw"
MML=8192
CYCLES=3

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--namespace) NS="$2"; shift 2 ;;
    --pod) POD="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --prefix) PREFIX="$2"; shift 2 ;;
    --cycles) CYCLES="$2"; shift 2 ;;
    -h|--help) sed -n "2,18p" "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

# Same two arms as the ladder's ends, so the comparison is against numbers that
# ladder already reports rather than a third configuration.
ARMS=(
  "s1-baseline|VLLM_WORKER_MULTIPROC_METHOD=spawn|--load-format auto"
  "s5-fsttuned|VLLM_WORKER_MULTIPROC_METHOD=fork PYTHONPYCACHEPREFIX=/cache/pycache CS_FST=1 CS_FST_MAX_THREADS=8 CS_FST_BBUF_KB=32768|--load-format fastsafetensors"
)

echo "=== registry cold/warm paired check: model=$MODEL cycles=$CYCLES ==="
echo "started $(date -u +%FT%TZ)"
FAILED=()
for ((c=1; c<=CYCLES; c++)); do
  for row in "${ARMS[@]}"; do
    IFS='|' read -r arm envs vargs <<<"$row"
    for state in cold warm; do
      RUN_ID="$PREFIX-$arm-$state-c$c"
      # cold clears modelinfos; warm leaves whatever the cold run just wrote, so
      # the pair runs back to back on identical cache state otherwise.
      COLDFLAG=()
      [[ "$state" == cold ]] && COLDFLAG=(--cold-registry)
      echo
      echo "########## $RUN_ID   $(date -u +%FT%TZ)"
      ENVFLAGS=(--env CUDA_VISIBLE_DEVICES=0)
      for kv in $envs; do ENVFLAGS+=(--env "$kv"); done
      ok=0
      for attempt in 1 2; do
        # shellcheck disable=SC2086 -- deliberate word split
        if "$REPO/scripts/run-experiment.sh" ${NS:+-n "$NS"} --pod "$POD" --no-apply \
            --run-id "$RUN_ID" --repeat 1 "${COLDFLAG[@]+${COLDFLAG[@]}}" \
            "${ENVFLAGS[@]}" -- --model "$MODEL" --max-model-len "$MML" $vargs; then
          ok=1; break
        fi
        [[ "$attempt" == 1 ]] && { echo "!! $RUN_ID attempt 1 failed, retrying in 30s"; sleep 30; }
      done
      [[ "$ok" == 1 ]] || FAILED+=("$RUN_ID")
    done
  done
done

echo
echo "=== done $(date -u +%FT%TZ) ==="
[[ ${#FAILED[@]} -gt 0 ]] && echo "failed: ${FAILED[*]}"
echo "compare with: python3 analysis/registry_warm_check.py runs/$PREFIX-*"
