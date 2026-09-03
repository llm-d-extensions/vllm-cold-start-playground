#!/usr/bin/env bash
# Does instanttensor's win survive at TP=2 -- and does its distributed mode do
# what the RFC implies but never explains?
#
# WHY THIS EXISTS. instanttensor is the fastest loader measured at TP=1 in this
# pod (reports/loader-comparison-32b-part2.txt: 9.61s weight load vs 10.1s for
# runai-to-GPU and 12.2s for the published fastsafetensors arm). But TP=1 is the
# case where its headline feature is switched OFF. vLLM's iterator passes a real
# process group only when world_size > 1:
#
#   weight_utils.py:1121-1124
#     process_group = world_group.device_group if world_group.world_size > 1 else None
#
# and instanttensor's own code shows what that turns on (_impl.py:544-546, 626-656):
# reads are STRIPED across ranks and reunified over NCCL --
#
#   io_depth = max(512 // world_size, 3)   # aio read + cudaMemcpyAsync + ncclAllGather
#   concurrency = max(32 // world_size, 1) # CUFILE path
#   meta_read_start = len(files)//world_size*rank + ...   # striped metadata too
#
# so at TP=2 each rank should pull ~half the bytes off GPFS and get the other
# half over NVLink at ~400 GiB/s. This pod is storage-bound (the floor
# measurement puts asynchronous O_DIRECT at 11.36 GiB/s), so if the striping
# works, TP=2 weight load should be roughly HALF of TP=1 -- and it should beat
# fastsafetensors, which also shards but pays the same per-rank read because each
# rank reads its own files rather than a stripe of all of them.
#
# The three arms are the same three loaders at TP=2, with the published step 1-3
# levers on all of them and --max-model-len 8192 so they are comparable to the
# existing tp2m8k-* runs:
#
#   fsttuned    fastsafetensors + cs_fst tuned -- the published TP=2 best.
#               Anchor: runs/tp2m8k-s5b-fsttuned weight load 7.71s.
#   runaicuda   runai_streamer + {"distributed":true,"concurrency":8}
#   it          instanttensor
#   itdbg       instanttensor with INSTANTTENSOR_DEBUG=1, so its own log states
#               the backend it chose, the buffer it sized, and what the metadata
#               all_gather cost. One cycle is enough; this arm is for the log,
#               not the number.
#
# CUDA_VISIBLE_DEVICES is deliberately NOT set: TP=2 needs both GPUs, and the
# TP=1 ladders pin GPU 0 only so a second engine can share the pod.
#
#   CS_NS=my-ns scripts/loader-tp2.sh
#   CS_NS=my-ns scripts/loader-tp2.sh --arm it --cycles 1
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NS="${CS_NS:-}"; POD="${CS_POD:-vllm-coldstart}"; MODEL="Qwen/Qwen3-32B"
PREFIX="lt2"; MML=8192; CYCLES=4; TP=2; ONLY=()
LP="${CS_LP:-/cache/lp}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --arm) ONLY+=("$2"); shift 2 ;;
    --cycles) CYCLES="$2"; shift 2 ;;
    --prefix) PREFIX="$2"; shift 2 ;;
    --tp) TP="$2"; shift 2 ;;
    -n|--namespace) NS="$2"; shift 2 ;;
    --pod) POD="$2"; shift 2 ;;
    -h|--help) sed -n '2,48p' "$0"; exit 0 ;;
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
  local arm="$1" rid="$2"
  local extra=() ev=()
  case "$arm" in
    fsttuned)  extra=(--load-format fastsafetensors)
               ev=(--env CS_FST=1 --env CS_FST_MAX_THREADS=8 --env CS_FST_BBUF_KB=32768) ;;
    runaicuda) extra=(--load-format runai_streamer
                      --model-loader-extra-config '{"distributed": true, "concurrency": 8}') ;;
    it)        extra=(--load-format instanttensor) ;;
    itdbg)     extra=(--load-format instanttensor); ev=(--env INSTANTTENSOR_DEBUG=1) ;;
    *) echo "unknown arm: $arm" >&2; return 2 ;;
  esac
  "$REPO/scripts/run-experiment.sh" ${NS:+-n "$NS"} --pod "$POD" --no-apply \
    --run-id "$rid" --repeat 1 "${COMMON[@]}" "${ev[@]+${ev[@]}}" \
    -- --model "$MODEL" --tensor-parallel-size "$TP" --max-model-len "$MML" "${extra[@]}"
}

# itdbg is opt-in only (--arm itdbg): INSTANTTENSOR_DEBUG=1 changes the arm, so it
# must never join a comparison run by default.
ARMS=(fsttuned runaicuda it)
for a in "${ONLY[@]+${ONLY[@]}}"; do [[ "$a" == itdbg ]] && ARMS+=(itdbg); done
FAILED=()
echo "=== TP=$TP loader comparison: $CYCLES cycle(s), pod=$POD ns=${NS:-<current>} ==="
echo "started $(date -u +%FT%TZ)"
# Cycle-major, so a node-level drift hits every arm rather than whichever arm ran
# while it was happening. Cycle 0 is discarded by analysis/loader_ladder.py.
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
[[ ${#FAILED[@]} -gt 0 ]] && echo "failed: ${FAILED[*]}"
echo "summarise with: python3 analysis/loader_ladder.py runs/$PREFIX-*"
