#!/usr/bin/env bash
# Does the capture-prologue hoist still pay on the *current* loader?
#
# WHY THIS EXISTS. The -4.0s hoist (docs/concurrent-cudagraph-capture.md 4b) was
# measured on the fastsafetensors+cs_fst arm, back when weight load was 13.3s and
# the deck's TP=1 total was 43.2s. The adopted loader is now --load-format
# instanttensor (runs/lt1-it-*, 36.87s median), which moves weight load to 7.70s
# and touches nothing in the capture phase -- so the hoist *should* compose, but
# "should" is a projection, and the slides were quoting it as one. This measures
# it instead.
#
# Two arms, same pod, same registry-warm second-boot conditions as runs/lt1-it-*:
#
#   base   --load-format instanttensor, upstream capture path: torch's
#          synchronize()+empty_cache() prologue inside all 3366 captures.
#   hoist  the same, with the prologue moved out of the loop -- one call after
#          capture_model (CS_CGEMPTY=once) and no per-capture device sync
#          (CS_CGSYNC=none). The `nofreeonce` arm of scripts/cudagraph-pool-matrix.sh,
#          which is the shape an upstream patch would take.
#
# Cycle-major and interleaved per docs/experiments.md; cycle 0 is a warm-up and is
# discarded, so 4 cycles give a median of 3 per arm. Memory is the other half of
# the claim -- score it with analysis/gpu_steady_memory.py (NVML, end of run), not
# with vLLM's "took X GiB", which is measured inside capture_model and cannot see
# a free that happens after the loop.
#
#   CS_NS=my-ns scripts/hoist-instanttensor.sh
#   CS_NS=my-ns scripts/hoist-instanttensor.sh --arm hoist --cycles 1
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NS="${CS_NS:-}"; POD="${CS_POD:-vllm-coldstart}"; MODEL="Qwen/Qwen3-32B"
PREFIX="hi1"; MML=8192; CYCLES=4; TP=1; ONLY=()
LP="${CS_LP:-/cache/lp}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --arm) ONLY+=("$2"); shift 2 ;;
    --cycles) CYCLES="$2"; shift 2 ;;
    --prefix) PREFIX="$2"; shift 2 ;;
    --tp) TP="$2"; shift 2 ;;
    -n|--namespace) NS="$2"; shift 2 ;;
    --pod) POD="$2"; shift 2 ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

COMMON=(--env VLLM_WORKER_MULTIPROC_METHOD=fork
        --env PYTHONPYCACHEPREFIX=/cache/pycache
        --env "PYTHONPATH=$LP")

wanted() {
  [[ ${#ONLY[@]} -eq 0 ]] && return 0
  local a; for a in "${ONLY[@]}"; do [[ "$1" == "$a" ]] && return 0; done; return 1
}

run_arm() {
  local arm="$1" rid="$2" ev=()
  case "$arm" in
    base)  ev=() ;;
    hoist) ev=(--env CS_CGSYNC=none --env CS_CGEMPTY=once) ;;
    *) echo "unknown arm: $arm" >&2; return 2 ;;
  esac
  "$REPO/scripts/run-experiment.sh" ${NS:+-n "$NS"} --pod "$POD" --no-apply \
    --run-id "$rid" --repeat 1 "${COMMON[@]}" "${ev[@]+${ev[@]}}" \
    -- --model "$MODEL" --tensor-parallel-size "$TP" --max-model-len "$MML" \
       --load-format instanttensor
}

ARMS=(base hoist)
FAILED=()
echo "=== TP=$TP instanttensor x capture-prologue hoist: $CYCLES cycle(s), pod=$POD ns=${NS:-<current>} ==="
echo "started $(date -u +%FT%TZ)"
for ((c = 0; c < CYCLES; c++)); do
  for arm in "${ARMS[@]}"; do
    wanted "$arm" || continue
    rid="$PREFIX-$arm-c$c"
    echo; echo "########## $rid   $(date -u +%FT%TZ)"
    if ! run_arm "$arm" "$rid"; then
      echo "!! $rid attempt 1 failed; waiting for the GPUs to drain, then retrying"
      for _ in $(seq 1 30); do
        used=$(kubectl ${NS:+-n "$NS"} exec "$POD" -- nvidia-smi \
                 --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null |
               paste -sd+ | bc 2>/dev/null || echo 0)
        [[ "${used:-0}" -lt 2000 ]] && break
        sleep 10
      done
      if ! run_arm "$arm" "$rid"; then
        echo "!! $rid FAILED twice"; FAILED+=("$rid")
      fi
    fi
  done
done
echo; echo "=== done $(date -u +%FT%TZ) ==="
[[ ${#FAILED[@]} -gt 0 ]] && { echo "failed: ${FAILED[*]}"; exit 1; }
exit 0
