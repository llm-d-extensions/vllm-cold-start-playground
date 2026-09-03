#!/usr/bin/env bash
# End-to-end boot A/B: the published step-5 loader vs vLLM's other viable loaders.
#
# Answers "the weight phase is still 13.3s -- can another load format do better?"
# with real boots rather than a loader microbenchmark. Three arms, all carrying
# the published step 1-3 levers so the load format is the only difference:
#
#   fst        --load-format fastsafetensors + the cs_fst patch  (published step 5)
#   runaicpu   --load-format runai_streamer, stock
#   runaicuda  --load-format runai_streamer + {"distributed": true, "concurrency": 8}
#
# `distributed: true` at TP=1 is not a mistake: weight_utils.py:1002-1013 uses that
# flag, and nothing else, to decide whether the streamer writes to the GPU or to
# host RAM. See reports/loader-comparison-32b.txt.
#
#   CS_NS=my-ns scripts/loader-ladder.sh
# Cycle-major, cycle 0 discarded, mirroring scripts/tp1-registry-ladder.sh.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NS="${CS_NS:-}"; POD="${CS_POD:-vllm-coldstart}"; MODEL="Qwen/Qwen3-32B"; MML=8192
PREFIX="lf1"; CYCLES=4
FORKENV=(--env CUDA_VISIBLE_DEVICES=0 --env VLLM_WORKER_MULTIPROC_METHOD=fork --env PYTHONPYCACHEPREFIX=/cache/pycache)

run_arm() {
  local id="$1"; shift
  local -a envf=() vargs=()
  while [[ "$1" != "--" ]]; do envf+=(--env "$1"); shift; done
  shift; vargs=("$@")
  echo; echo "########## $id   $(date -u +%FT%TZ)"
  echo "# env : ${envf[*]}"; echo "# args: ${vargs[*]}"
  for attempt in 1 2; do
    "$REPO/scripts/run-experiment.sh" ${NS:+-n "$NS"} --pod "$POD" --no-apply \
      --run-id "$id" --repeat 1 "${FORKENV[@]}" "${envf[@]+${envf[@]}}" \
      -- --model "$MODEL" --max-model-len "$MML" "${vargs[@]}" && return 0
    echo "!! $id attempt $attempt failed"
    [[ $attempt == 1 ]] && for _ in $(seq 1 30); do
      u="$(kubectl ${NS:+-n "$NS"} exec "$POD" -c vllm -- nvidia-smi --query-gpu=memory.used \
           --format=csv,noheader,nounits 2>/dev/null | paste -sd+ - | bc 2>/dev/null || echo x)"
      [[ "$u" == 0 ]] && break; sleep 2
    done
  done
  echo "!! $id FAILED twice"; return 1
}

for ((c=0;c<CYCLES;c++)); do
  echo; echo "==================== cycle c$c $( ((c==0)) && echo '(warm-up, discarded)') ===================="
  run_arm "$PREFIX-fst-c$c"      CS_FST=1 CS_FST_MAX_THREADS=8 CS_FST_BBUF_KB=32768 -- --load-format fastsafetensors
  run_arm "$PREFIX-runaicuda-c$c" NOOP=1 -- --load-format runai_streamer \
      --model-loader-extra-config '{"distributed": true, "concurrency": 8}'
  run_arm "$PREFIX-runaicpu-c$c"  NOOP=1 -- --load-format runai_streamer
done
echo; echo "=== loader ladder done $(date -u +%FT%TZ) ==="
