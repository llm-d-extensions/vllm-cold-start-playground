#!/usr/bin/env bash
# Re-run the README's cold-start ladder at TP=2, arm for arm, to find out which
# of the TP=1 wins survive when there are two workers instead of one.
#
# The TP=1 ladder (README table, reports/32b-step*.txt) went 90.65s -> 43.30s.
# Three of its four levers have a reason to behave differently at TP>1, so none
# of them can be assumed:
#
#   PYTHONPYCACHEPREFIX  should scale UP: the import cost it removes is paid once
#                        per process, and TP=2 has one more process.
#   fork vs forkserver   README:211 predicts forkserver is "parity" at TP=1
#                        ("one child means nothing to amortise") but that "At
#                        TP>1 one preload serves N workers, which is where it
#                        should pay". Both arms run here so that prediction is
#                        measured rather than repeated.
#   cs_fst nogds         should go to ZERO by construction: weight_utils.py:1057
#                        computes `nogds = pg.size() > 1`, so at TP=2 vLLM
#                        already passes nogds=True and the patch's headline lever
#                        has nothing left to fix. Step 5 is therefore split into
#                        s5a (nogds alone) and s5b (nogds + max_threads +
#                        bbuf_size_kb) to bill the two halves separately.
#
# The pod's own env is the step-3 arm already (VLLM_WORKER_MULTIPROC_METHOD=fork,
# no PYTHONPYCACHEPREFIX), so every arm below states its overrides explicitly
# with --env rather than trusting the default -- including the ones that put a
# variable *back* to the stock value.
#
# Runs against an already-deployed 2-GPU pod (--no-apply), --repeat 3 per arm,
# median reported, same as the TP=1 ladder. Budget ~2 min per repeat.
#
# MEASURED (Qwen3-32B, 2xH100 NVLink, runs/tp2-*, median of 3):
#   s1  94.7s  s2  58.9s (-35.8)  s3  59.6s (+0.7)  s3b 57.2s (noise)
#   s4  52.7s (-6.2 vs s2)  s5a 52.9s (+0.2)  s5b 53.0s (+0.1)
# So: pycache survives and GROWS (imports 54.4 -> 24.6s, -29.8s vs -21.0s at
# TP=1); fastsafetensors survives at ~40% (weight load 14.3 -> 8.81s, because
# each rank already reads its own shard in parallel); nogds is dead as predicted
# (-0.14s); and BOTH fork and forkserver die on one veto -- CUDA is already
# initialised in the launching process at TP=2 (in 21/21 runs cuda.lazy_init
# fires 3.4-6.0s before engine.core_proc_manager, inside
# config.create_engine_config), so
# _maybe_force_spawn() reverts fork to spawn and the forkserver probe declines
# with served=0. The README:211 prediction is therefore NOT confirmed: it cannot
# be tested until that veto is gone. See README "What survives at TP=2".
#
# MEASURED at matched context length (runs/tp2m8k-*, --max-model-len 8192, med of 3):
#   s1-baseline 108s (110/95.6/108)   s5b 51.8s (51.7/54.6/51.8)
#   capture     10.0s (noisy arm)     capture 8.74s (8.44/8.82/8.74)
#   weight      15.1s                 weight  7.71s
# Against TP=1 with the same probe at 8192 (runs/cc-base-c*): capture 11.1s,
# weight 15.8s. So capture is -21% at TP=2 (not -50%: the per-size launch
# overhead does not divide by rank count) and weight load halves. Capture barely
# depends on context length -- 8.74s at 8192 vs ~8.7s at 40960 -- so the original
# mismatch turned out not to affect that comparison.
#
# --max-model-len: the seven arms above were run WITHOUT it, so they took vLLM's
# default 40960 for this model, while the published TP=1 ladder used 8192. Every
# TP=2-vs-TP=2 delta in this file is unaffected (all arms share one command), and
# so are the cross-TP import and weight-load comparisons (neither depends on
# context length). But `torch.compile` and `cudagraph capture` do depend on it,
# so comparing those two phases across TP needs both sides at the same value --
# hence the flag, and the tp2m8k runs.
#
# A TP=1 re-anchor at 40960 is not possible on this node, which is worth knowing
# on its own: Qwen3-32B on ONE H100 leaves 7.92 GiB for KV where the full 40960
# context needs 10.0 GiB, so the engine refuses to start (estimated max 32432).
# Full-context 32B at TP=1 is not a cold-start question, it is a capacity one.
#
#   scripts/tp2-ladder.sh -n lionel-cold-start
#   scripts/tp2-ladder.sh -n lionel-cold-start --arm s5b --repeat 1
#   scripts/tp2-ladder.sh -n lionel-cold-start --prefix tp2m8k --max-model-len 8192 \
#       --arm s1-baseline --arm s5b-fsttuned
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NS=""
POD="vllm-coldstart"
MODEL="Qwen/Qwen3-32B"
REPEAT=3
TP=2
PREFIX="tp2"
MML=""
ONLY=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--namespace) NS="$2"; shift 2 ;;
    --pod) POD="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --repeat) REPEAT="$2"; shift 2 ;;
    --tp) TP="$2"; shift 2 ;;
    --prefix) PREFIX="$2"; shift 2 ;;
    --max-model-len) MML="$2"; shift 2 ;;
    --arm) ONLY+=("$2"); shift 2 ;;
    -h|--help) sed -n "2,71p" "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

# arm | env overrides (space separated K=V) | vllm serve args
# Each row is cumulative with the one above it, so a row's delta is its lever.
ARMS=(
  "s1-baseline|VLLM_WORKER_MULTIPROC_METHOD=spawn|--load-format auto"
  "s2-pycache|VLLM_WORKER_MULTIPROC_METHOD=spawn PYTHONPYCACHEPREFIX=/cache/pycache|--load-format auto"
  "s3-fork|VLLM_WORKER_MULTIPROC_METHOD=fork PYTHONPYCACHEPREFIX=/cache/pycache|--load-format auto"
  "s3b-forkserver|VLLM_WORKER_MULTIPROC_METHOD=spawn CS_FORKSERVER=1 PYTHONPYCACHEPREFIX=/cache/pycache|--load-format auto"
  "s4-fst|VLLM_WORKER_MULTIPROC_METHOD=fork PYTHONPYCACHEPREFIX=/cache/pycache|--load-format fastsafetensors"
  "s5a-nogds|VLLM_WORKER_MULTIPROC_METHOD=fork PYTHONPYCACHEPREFIX=/cache/pycache CS_FST=1|--load-format fastsafetensors"
  "s5b-fsttuned|VLLM_WORKER_MULTIPROC_METHOD=fork PYTHONPYCACHEPREFIX=/cache/pycache CS_FST=1 CS_FST_MAX_THREADS=8 CS_FST_BBUF_KB=32768|--load-format fastsafetensors"
)

wanted() {
  [[ ${#ONLY[@]} -eq 0 ]] && return 0
  local a
  for a in "${ONLY[@]}"; do [[ "$1" == "$a" ]] && return 0; done
  return 1
}

echo "=== TP=$TP ladder: model=$MODEL repeat=$REPEAT pod=$POD ns=${NS:-<current>} ==="
echo "started $(date -u +%FT%TZ)"
FAILED=()
for row in "${ARMS[@]}"; do
  IFS='|' read -r arm envs vargs <<<"$row"
  wanted "$arm" || continue
  RUN_ID="$PREFIX-$arm"
  echo
  echo "########## $RUN_ID"
  echo "# env : $envs"
  echo "# args: --tensor-parallel-size $TP $vargs${MML:+ --max-model-len $MML}"
  [[ -n "$MML" ]] && vargs="$vargs --max-model-len $MML"
  ENVFLAGS=()
  for kv in $envs; do ENVFLAGS+=(--env "$kv"); done
  # shellcheck disable=SC2086 -- $vargs is a deliberate word-split arg list
  if ! "$REPO/scripts/run-experiment.sh" ${NS:+-n "$NS"} --pod "$POD" --no-apply \
      --run-id "$RUN_ID" --repeat "$REPEAT" "${ENVFLAGS[@]}" \
      -- --model "$MODEL" --tensor-parallel-size "$TP" $vargs; then
    echo "!! $RUN_ID FAILED (continuing; the ladder is still readable without it)"
    FAILED+=("$RUN_ID")
  fi
done

echo
echo "=== ladder done $(date -u +%FT%TZ) ==="
[[ ${#FAILED[@]} -gt 0 ]] && echo "failed arms: ${FAILED[*]}"
echo "renders: $REPO/runs/$PREFIX-*/report.txt"
echo "compare adjacent arms with: make compare BASE=runs/$PREFIX-<a> NEW=runs/$PREFIX-<b>"
