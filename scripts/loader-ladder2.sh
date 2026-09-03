#!/usr/bin/env bash
# End-to-end boot A/B for the three load formats the first comparison could not
# reach: tensorizer, instanttensor and modelexpress.
#
# reports/loader-comparison-32b.txt ruled all three out on availability -- each
# raises ImportError against the pod's stock site-packages. That is a statement
# about the image, not about the loaders, so this ladder installs them into an
# isolated prefix (/cache/lp, `pip install --no-deps --target`) and reaches them
# with PYTHONPATH per arm. Nothing is added to the base interpreter, so every
# published run in runs/ still describes the environment it was measured in.
#
# Arms (all carry the published step 1-3 levers, so load format is the only
# difference, and all are --max-model-len 8192):
#
#   runaicuda   --load-format runai_streamer + {"distributed":true,"concurrency":8}
#               -- the winner of the first comparison, re-run here as a
#                  same-session anchor rather than trusted across sessions
#   it          --load-format instanttensor
#   mx          --load-format modelexpress, nothing else set
#   mxms        --load-format modelexpress + MX_MODEL_URI=<local snapshot>
#   tz          --load-format tensorizer, against an artifact serialized by
#               scripts/tensorize-artifact.sh
#
# tz is not comparable to the rest without saying so: it reads a different file
# (a 61 GiB vllm-tensorized .tensors) that has to be produced by a separate
# offline pass, and that pass is itself a full model load. It is in the ladder to
# price the boot, not to suggest it is free.
#
#   CS_NS=my-ns scripts/loader-ladder2.sh
#   CS_NS=my-ns scripts/loader-ladder2.sh --arm it --cycles 1
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NS="${CS_NS:-}"; POD="${CS_POD:-vllm-coldstart}"; MODEL="Qwen/Qwen3-32B"
PREFIX="lf2"; MML=8192; CYCLES=4; ONLY=()
LP="${CS_LP:-/cache/lp}"
SNAP="${CS_SNAP:-/cache/hf/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137}"
TZDIR="${CS_TZDIR:-/cache/tz/vllm/Qwen/Qwen3-32B/v1}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --arm) ONLY+=("$2"); shift 2 ;;
    --cycles) CYCLES="$2"; shift 2 ;;
    --prefix) PREFIX="$2"; shift 2 ;;
    -n|--namespace) NS="$2"; shift 2 ;;
    --pod) POD="$2"; shift 2 ;;
    -h|--help) sed -n '2,33p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

COMMON=(--env CUDA_VISIBLE_DEVICES=0
        --env VLLM_WORKER_MULTIPROC_METHOD=fork
        --env PYTHONPYCACHEPREFIX=/cache/pycache
        --env "PYTHONPATH=$LP")

wanted() {
  [[ ${#ONLY[@]} -eq 0 ]] && return 0
  local a; for a in "${ONLY[@]}"; do [[ "$1" == "$a" ]] && return 0; done; return 1
}

# One function per arm rather than a table: the extra-config JSON and the
# per-arm env do not survive the word-splitting a "env|args" string table needs,
# and a silently dropped --model-loader-extra-config is exactly the kind of
# failure that produces a plausible, wrong number.
run_arm() {
  local arm="$1" rid="$2"; shift 2
  local extra=() ev=()
  case "$arm" in
    runaicuda) extra=(--load-format runai_streamer
                      --model-loader-extra-config '{"distributed": true, "concurrency": 8}') ;;
    it)        extra=(--load-format instanttensor) ;;
    mx)        extra=(--load-format modelexpress) ;;
    mxms)      extra=(--load-format modelexpress); ev=(--env "MX_MODEL_URI=$SNAP") ;;
    tz)        extra=(--load-format tensorizer
                      --model-loader-extra-config "{\"tensorizer_dir\": \"$TZDIR\"}") ;;
    *) echo "unknown arm $arm" >&2; return 2 ;;
  esac
  "$REPO/scripts/run-experiment.sh" ${NS:+-n "$NS"} --pod "$POD" --no-apply \
    --run-id "$rid" --repeat 1 "${COMMON[@]}" "${ev[@]+${ev[@]}}" \
    -- --model "$MODEL" --max-model-len "$MML" "${extra[@]}"
}

ARMS=(runaicuda it mx mxms tz)
echo "=== loader ladder 2: $MODEL cycles=$CYCLES (c0 discarded) ==="
echo "pod=$POD ns=${NS:-<current>} prefix=$PREFIX lp=$LP"
FAILED=()
for ((c=0; c<CYCLES; c++)); do
  echo; echo "==================== cycle c$c $( ((c==0)) && echo '(warm-up, discarded)') ===================="
  for arm in "${ARMS[@]}"; do
    wanted "$arm" || continue
    RUN_ID="$PREFIX-$arm-c$c"
    echo; echo "########## $RUN_ID   $(date -u +%FT%TZ)"
    ok=0
    for attempt in 1 2; do
      run_arm "$arm" "$RUN_ID" && { ok=1; break; }
      if [[ "$attempt" == 1 ]]; then
        echo "!! $RUN_ID attempt 1 failed; waiting for the GPU to drain, then retrying"
        for _ in $(seq 1 30); do
          u="$(kubectl ${NS:+-n "$NS"} exec "$POD" -c vllm -- \
               nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
               2>/dev/null | paste -sd+ - | bc 2>/dev/null || echo unknown)"
          [[ "$u" == 0 ]] && { echo "   GPU idle, retrying"; break; }
          sleep 2
        done
      fi
    done
    [[ "$ok" == 1 ]] || { echo "!! $RUN_ID FAILED twice"; FAILED+=("$RUN_ID"); }
  done
done
echo; echo "=== done $(date -u +%FT%TZ) ==="
[[ ${#FAILED[@]} -gt 0 ]] && echo "failed: ${FAILED[*]}"
echo "summarise with: python3 analysis/loader_ladder.py runs/$PREFIX-*"
