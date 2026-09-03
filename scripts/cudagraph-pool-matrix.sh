#!/usr/bin/env bash
# Price the two preconditions of concurrent CUDA graph capture on the real model.
#
# scripts/cudagraph-concurrent-capture.py shows on bare torch that concurrent
# capture works but needs (a) one mempool per capturing thread and (b) no
# `torch.cuda.synchronize()` inside the capture window. This matrix asks what
# each of those costs Qwen3-32B, using coldstart/cs_cgcapture.py -- still
# strictly serial capture, so the answer is not confounded by threading.
#
#   base    upstream: one shared graph pool, torch's device sync per capture
#   pool1   probe engaged, still one pool -- control for the probe's own code path
#   pool2   two pools round-robin
#   pool4   four pools round-robin
#   nosync  one pool, device sync inside graph.__enter__ suppressed
#
# --set prologue swaps the arms for the follow-up question the first matrix
# raised: nosync saved nothing, because cudaFree inside empty_cache() is
# itself device-synchronizing. So price the prologue as a whole.
#
#   base    the same control, re-run interleaved with these arms
#   noempty empty_cache() suppressed, device sync left alone
#   nofree  both suppressed -- the whole graph.__enter__ prologue, minus
#           torch._C._host_emptyCache() which is not patchable from Python
#
# --set hoist then asks whether the time saving survives keeping the memory:
#
#   emptyonce   one synchronize()+empty_cache() after capture_model instead of
#               3366 inside it -- the shape an upstream patch would take
#   nofreeonce  the same hoist, plus dropping the per-capture device sync: the
#               whole prologue moved out of the loop rather than deleted. This is
#               the arm to quote.
#
# Note vLLM's "took X GiB" cannot score the hoist arms: it is measured inside
# capture_model (model_runner.py:871,901), so a free that happens after the loop
# is invisible to it. Use analysis/gpu_steady_memory.py, which reads NVML at the
# end of the run instead.
#
# Interleaved (all arms once, then again) per docs/experiments.md, so drift in
# node state hits every arm equally. Read the answer out of vLLM's own
# "Graph capturing finished in N secs, took X GiB" line and the report's
# cudagraph.* spans; runs/<id>/trace also carries cgcapture.pool instants that
# prove the arm engaged.
#
#   scripts/cudagraph-pool-matrix.sh -n lionel-cold-start-cc [--passes 3]
#   scripts/cudagraph-pool-matrix.sh -n lionel-cold-start-cc --set prologue --prefix cp
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NS=""
PASSES=3
PREFIX="cc"
SET="pools"
MODEL_ARGS=(--model Qwen/Qwen3-32B --load-format fastsafetensors --max-model-len 8192)

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--namespace) NS="$2"; shift 2 ;;
    --passes) PASSES="$2"; shift 2 ;;
    --prefix) PREFIX="$2"; shift 2 ;;
    --set) SET="$2"; shift 2 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$NS" ]] || { echo "give -n <namespace>" >&2; exit 2; }

# arm name -> extra env for that arm. CS_FST=1 is on everywhere: it shortens
# weight load and touches nothing in the capture phase.
case "$SET" in
  pools)    declare -a ARMS=(base pool1 pool2 pool4 nosync) ;;
  prologue) declare -a ARMS=(base noempty nofree) ;;
  hoist)    declare -a ARMS=(base emptyonce) ;;
  hoist2)   declare -a ARMS=(base nofreeonce) ;;
  *) echo "unknown --set: $SET (pools|prologue|hoist|hoist2)" >&2; exit 2 ;;
esac
env_for() {
  case "$1" in
    base)    echo "" ;;
    pool1)   echo "CS_CGPOOLS=1" ;;
    pool2)   echo "CS_CGPOOLS=2" ;;
    pool4)   echo "CS_CGPOOLS=4" ;;
    nosync)  echo "CS_CGSYNC=none" ;;
    noempty) echo "CS_CGEMPTY=0" ;;
    nofree)  echo "CS_CGSYNC=none CS_CGEMPTY=0" ;;
    emptyonce) echo "CS_CGEMPTY=once" ;;
    nofreeonce) echo "CS_CGSYNC=none CS_CGEMPTY=once" ;;
  esac
}

for pass in $(seq 1 "$PASSES"); do
  for arm in "${ARMS[@]}"; do
    id="$PREFIX-$arm-c$pass"
    # One --env per variable: run-experiment.sh validates each as K=V, and the
    # nofree arm sets two.
    extra=()
    for kv in $(env_for "$arm"); do extra+=(--env "$kv"); done
    echo "=================== $id ==================="
    "$REPO/scripts/run-experiment.sh" -n "$NS" --no-apply --run-id "$id" --repeat 1 \
      --env CS_FST=1 ${extra[@]+"${extra[@]}"} -- "${MODEL_ARGS[@]}" \
      || echo "!! $id FAILED (kept going)"
  done
done

echo
echo "== graph pool memory, as vLLM reports it =="
for pass in $(seq 1 "$PASSES"); do
  for arm in "${ARMS[@]}"; do
    id="$PREFIX-$arm-c$pass"
    line="$(grep -h "Graph capturing finished" "$REPO/runs/$id/vllm.log" 2>/dev/null | tail -1)"
    printf '%-16s %s\n' "$id" "${line#*model_runner.py:*] }"
  done
done
