#!/usr/bin/env bash
# Re-run the README's TP=1 cold-start ladder with the model registry accounted.
#
# WHY THIS EXISTS
#
# Every run behind the published TP=1 table (runs/lad2-*, 90.65s -> 43.30s) was
# warm in one place nobody was measuring. Resolving an architecture to a model
# class asks whether it is a text-generation model, is multimodal, supports LoRA
# -- and vLLM answers by importing the class in a throwaway interpreter, because
# importing it in the API server would initialise CUDA there:
#
#   _SUBPROCESS_COMMAND = [sys.executable, "-m", "vllm.model_executor.models.registry"]
#
# That child pays a full `import vllm`, and so a full `import torch`, for a
# handful of booleans. It is cached in
# $VLLM_CACHE_ROOT/modelinfos/<module>-<class>.json, keyed on a hash of the model
# module's bytes, so it is paid once per (vLLM build, model module) per cache
# volume and is invisible on every boot after the first.
#
# Isolated, no GPU, median of 5 interleaved trials
# (scripts/registry-cache-probe.py, reports/registry-cache-cold-vs-warm-*.txt):
#
#   cold modelinfos cache  14.95s   (min 14.896 / max 15.018 -- ±0.06s)
#   warm modelinfos cache   0.0002s (a file-bytes hash plus a ~870 B JSON read)
#
# The published 43.30s is therefore not a first-boot number: a pod with a fresh
# VLLM_CACHE_ROOT pays ~15s more. It was never wrong about what it measured, it
# was measuring a second boot. Worse, `--cold-compile` clears all of
# VLLM_CACHE_ROOT, so any run meaning to measure a cold torch.compile was also
# silently paying the registry -- which is why coldstart-run.sh now has
# --cold-registry as a separate lever.
#
# WHAT THIS LADDER ADDS
#
# Same five arms as the published table, same model, same --max-model-len 8192,
# same interleaved cycle-major schedule with cycle 0 discarded -- but every run
# is --cold-registry, so each arm's total is a real first boot. Because the
# report now has a "model registry resolve" phase, ONE registry-cold ladder
# yields both numbers per arm: the honest first-boot total, and (total minus that
# phase) the second-boot figure comparable to the published table. No separate
# registry-warm arm is needed, which is what keeps this at 20 boots.
#
# The interesting interaction is arm 1 vs arm 2. The registry subprocess is a
# sixth interpreter doing `import vllm`, so PYTHONPYCACHEPREFIX should shrink the
# registry phase too -- meaning step 2's win is LARGER on a first boot than the
# published -22.31s. Reading the registry phase across arms measures that instead
# of assuming it.
#
# CUDA_VISIBLE_DEVICES=0 on every arm: the published ladder ran on the 1-GPU pod
# (manifests/pod-exec.yaml) and this runs on the 2-GPU one, so the engine is
# given the same single-device view. Uniform across arms, so intra-ladder deltas
# are unaffected either way.
#
#   scripts/tp1-registry-ladder.sh -n lionel-cold-start
#   scripts/tp1-registry-ladder.sh -n lionel-cold-start --arm s1-baseline --cycles 1
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NS=""
POD="vllm-coldstart"
MODEL="Qwen/Qwen3-32B"
PREFIX="reg1"
MML=8192
CYCLES=4          # c0 is a discarded warm-up, c1..c3 are the measured median-of-3
ONLY=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--namespace) NS="$2"; shift 2 ;;
    --pod) POD="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --prefix) PREFIX="$2"; shift 2 ;;
    --max-model-len) MML="$2"; shift 2 ;;
    --cycles) CYCLES="$2"; shift 2 ;;
    --arm) ONLY+=("$2"); shift 2 ;;
    -h|--help) sed -n "2,56p" "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

# arm | env overrides (space separated K=V) | vllm serve args
# Cumulative: each row adds one change to the row above, so its delta is its
# lever. Mirrors the README table steps 1-5 exactly.
ARMS=(
  "s1-baseline|VLLM_WORKER_MULTIPROC_METHOD=spawn|--load-format auto"
  "s2-pycache|VLLM_WORKER_MULTIPROC_METHOD=spawn PYTHONPYCACHEPREFIX=/cache/pycache|--load-format auto"
  "s3-fork|VLLM_WORKER_MULTIPROC_METHOD=fork PYTHONPYCACHEPREFIX=/cache/pycache|--load-format auto"
  "s4-fst|VLLM_WORKER_MULTIPROC_METHOD=fork PYTHONPYCACHEPREFIX=/cache/pycache|--load-format fastsafetensors"
  "s5-fsttuned|VLLM_WORKER_MULTIPROC_METHOD=fork PYTHONPYCACHEPREFIX=/cache/pycache CS_FST=1 CS_FST_MAX_THREADS=8 CS_FST_BBUF_KB=32768|--load-format fastsafetensors"
)

wanted() {
  [[ ${#ONLY[@]} -eq 0 ]] && return 0
  local a
  for a in "${ONLY[@]}"; do [[ "$1" == "$a" ]] && return 0; done
  return 1
}

echo "=== TP=1 registry-cold ladder: model=$MODEL cycles=$CYCLES (c0 discarded) ==="
echo "pod=$POD ns=${NS:-<current>} max-model-len=$MML"
echo "started $(date -u +%FT%TZ)"
FAILED=()
# Cycle-major, not arm-major: the arms of a cycle run back to back, so node
# drift over the ~30 minutes lands on every arm equally instead of on whichever
# one happened to run first. This is the schedule the published table used.
for ((c=0; c<CYCLES; c++)); do
  echo
  echo "==================== cycle c$c $( ((c==0)) && echo '(warm-up, discarded)') ===================="
  for row in "${ARMS[@]}"; do
    IFS='|' read -r arm envs vargs <<<"$row"
    wanted "$arm" || continue
    RUN_ID="$PREFIX-$arm-c$c"
    echo
    echo "########## $RUN_ID   $(date -u +%FT%TZ)"
    echo "# env : $envs CUDA_VISIBLE_DEVICES=0"
    echo "# args: $vargs --max-model-len $MML   (--cold-registry)"
    ENVFLAGS=(--env CUDA_VISIBLE_DEVICES=0)
    for kv in $envs; do ENVFLAGS+=(--env "$kv"); done
    # One retry, because a lost arm costs a third of that arm's median while a
    # retry costs 90 seconds. The usual cause is coldstart-run.sh's exclusivity
    # guard (exit 3) tripping on a sibling that is on its way out -- a teardown
    # that has not finished releasing the GPU. Waiting for the device to drain
    # and going again is exactly what a human would do. Two failures in a row is
    # a real problem, so it is recorded and the ladder moves on.
    ok=0
    for attempt in 1 2; do
      # shellcheck disable=SC2086 -- $vargs is a deliberate word-split arg list
      if "$REPO/scripts/run-experiment.sh" ${NS:+-n "$NS"} --pod "$POD" --no-apply \
          --run-id "$RUN_ID" --repeat 1 --cold-registry "${ENVFLAGS[@]}" \
          -- --model "$MODEL" --max-model-len "$MML" $vargs; then
        ok=1; break
      fi
      if [[ "$attempt" == 1 ]]; then
        echo "!! $RUN_ID attempt 1 failed; waiting for the GPUs to drain, then retrying"
        for _ in $(seq 1 30); do
          used="$(kubectl ${NS:+-n "$NS"} exec "$POD" -c vllm -- \
                  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
                  2>/dev/null | paste -sd+ - | bc 2>/dev/null || echo unknown)"
          [[ "$used" == 0 ]] && { echo "   GPUs idle, retrying"; break; }
          sleep 2
        done
      fi
    done
    if [[ "$ok" != 1 ]]; then
      echo "!! $RUN_ID FAILED twice (continuing; the ladder is still readable without it)"
      FAILED+=("$RUN_ID")
    fi
  done
done

echo
echo "=== ladder done $(date -u +%FT%TZ) ==="
[[ ${#FAILED[@]} -gt 0 ]] && echo "failed arms: ${FAILED[*]}"
echo "summarise with: python3 analysis/registry_ladder.py runs/$PREFIX-*"
