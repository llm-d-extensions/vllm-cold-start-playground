#!/bin/bash
# Live demo: fastest cold start -> park (free the GPU) -> second tenant ->
# migrate-and-wake -- at --tp 1 and --tp 2.
#
#   Boot A (best-config, warm PVC) -> baseline completion -> park A
#   (/sleep?level=1 + cuda-checkpoint, both GPUs read 0 MiB) -> boot tenant B
#   on the freed GPU(s), correctness-check B -> [tp=1: migrate A onto the
#   *other* GPU with a swap device-map, correctness-check A | tp=2: stop B,
#   restore A with an *identity* device-map onto the same two GPUs -- a real
#   swap does not fit a 2-GPU budget while B holds both cards -- and
#   correctness-check A] -> [tp=1 only, bonus: refresh baseline, /sleep?level=2
#   (reaches 0 MiB with no cuda-checkpoint at all), /wake_up (reproduces the
#   documented footgun on purpose), /reload_weights (the fix -- vLLM already
#   ships this, it is just never wired to a route), re-check].
#
# Everything is injected into the vanilla vllm/vllm-openai image via the
# repo's coldstart/ monkeypatch probe (PYTHONPATH + sitecustomize.py) --
# CS_FORKSERVER=1 for the engine start method (never VLLM_WORKER_MULTIPROC_
# METHOD=fork; forkserver per the user's explicit instruction), CS_FST=1 for
# fastsafetensors tuning, CS_DEV_ROUTES=1 for /checkpoint_prepare,
# /checkpoint_restore and /reload_weights (coldstart/cs_dev_routes.py) --
# nothing here is baked into a custom image.
#
# Usage:
#   ./scripts/demo-cold-start.sh --tp 1
#   ./scripts/demo-cold-start.sh --tp 2
#
# Run via `kubectl exec` into the existing 2-GPU pod (manifests/pod-exec-2gpu.yaml,
# PVC coldstart-cache warm). All output is tee'd to
# reports/demo-cold-start-tp<N>-<timestamp>.txt.
#
# cuda-checkpoint is NOT in the vLLM image; stage bin/x86_64_Linux/cuda-checkpoint
# from github.com/NVIDIA/cuda-checkpoint matching the driver version at $CC
# before running (see docs/sleep-mode.md).
set -uo pipefail

TP=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tp) TP="$2"; shift 2 ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
[[ "$TP" == "1" || "$TP" == "2" ]] || { echo "--tp must be 1 or 2" >&2; exit 2; }

CC=${CC:-/tmp/cuda-checkpoint}
MODEL=${MODEL:-Qwen/Qwen3-32B}
PROBE_SRC=${CS_PROBE_SRC:-/opt/coldstart-src}
PROBE_DIR=/tmp/coldstart-demo
PORT_A=8010
PORT_B=8011
LOG_A=/tmp/demo-a.log
LOG_B=/tmp/demo-b.log

[[ -x "$CC" ]] || { echo "no cuda-checkpoint at $CC -- stage it first (see docs/sleep-mode.md)"; exit 2; }
[[ -f "$PROBE_SRC/sitecustomize.py" ]] || { echo "no probe at $PROBE_SRC"; exit 2; }

ts(){ date +%H:%M:%S; }
mem(){ nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '; }
apps(){ nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader | sed 's/^/      /'; }
gen(){ curl -s "localhost:$1/v1/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"prompt\":\"The capital of France is\",\"max_tokens\":32,\"temperature\":0}" \
  | python3 -c 'import sys,json; print(repr(json.load(sys.stdin)["choices"][0]["text"]))' 2>/dev/null; }
elapsed(){ python3 -c "import time;print(f'{time.time()-$1:.2f}s')"; }
verdict(){ # $1=label $2=got $3=base
  if [[ -n "$2" && "$2" == "$3" ]]; then echo "[$(ts)] $1: IDENTICAL ($2)"
  else echo "[$(ts)] $1: DIFFERS  got=$2  base=$3"; fi
}

# worker_pids: TP=1 -> just EngineCore (UniProcExecutor: the single worker runs
# inside EngineCore, so its own pid is the CUDA-context holder). TP=2 ->
# MultiprocExecutor's per-rank WorkerProc children, discovered via vLLM's own
# "Worker_TP{rank}" process-title convention (multiproc_executor.py) and
# cross-checked against nvidia-smi's compute-apps list, since vllm.log carries
# no "WorkerProc pid=" line the way it carries "EngineCore pid=".
worker_pids(){ # $1=log
  if [[ "$TP" == "1" ]]; then
    grep -oE 'EngineCore pid=[0-9]+' "$1" | head -1 | grep -oE '[0-9]+'
  else
    ps -e -o pid,cmd | grep -E 'Worker_TP[0-9]+' | grep -v grep | awk '{print $1}'
  fi
}

boot(){ # $1=port $2=log $3=cuda-visible-devices ("" = both)
  local port=$1 log=$2 cvd=$3
  rm -rf "$PROBE_DIR"; mkdir -p "$PROBE_DIR"
  cp "$PROBE_SRC"/*.py "$PROBE_DIR/"
  python3 -m compileall -q "$PROBE_DIR" >/dev/null 2>&1 || true
  local env=(
    PYTHONPYCACHEPREFIX=/cache/pycache
    PYTHONPATH="$PROBE_DIR"
    CS_FORKSERVER=1
    CS_FST=1 CS_FST_MAX_THREADS=8 CS_FST_BBUF_KB=32768
    CS_DEV_ROUTES=1
    VLLM_SERVER_DEV_MODE=1
  )
  [[ -n "$cvd" ]] && env+=(CUDA_VISIBLE_DEVICES="$cvd")
  env "${env[@]}" nohup vllm serve --model "$MODEL" --load-format fastsafetensors \
    --max-model-len 8192 --enable-sleep-mode --tensor-parallel-size "$TP" \
    --port "$port" > "$log" 2>&1 &
  for _ in $(seq 1 900); do curl -sf "localhost:$port/health" >/dev/null 2>&1 && return 0; sleep 1; done
  echo "[$(ts)] $log FAILED to become ready -- tail:"; tail -40 "$log"; return 1
}

teardown(){ pkill -f "vllm serve.*--port $1" 2>/dev/null; sleep 3; }

REPORT="reports/demo-cold-start-tp${TP}-$(date -u +%Y%m%d-%H%M%S).txt"
mkdir -p reports
exec > >(tee "$REPORT") 2>&1

echo "=== demo-cold-start --tp $TP  $(date -u +%FT%TZ) ==="
U0=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i 0)
U1=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i 1)
echo "GPU0=$U0"
echo "GPU1=$U1"
pkill -f 'vllm serve' 2>/dev/null; sleep 3
echo "[$(ts)] idle: $(mem)"

# ---------------------------------------------------------------------------
echo; echo "## step 1: boot A (best config, warm PVC, forkserver, fastsafetensors)"
S=$(date +%s.%N)
boot "$PORT_A" "$LOG_A" "" || exit 1
echo "[$(ts)] A ready in $(elapsed "$S")"
PIDS_A=$(worker_pids "$LOG_A")
echo "[$(ts)] A worker pid(s): $PIDS_A"
[[ -n "$PIDS_A" ]] || { echo "[$(ts)] could not discover A's worker pid(s) -- stopping"; exit 1; }
echo "[$(ts)] A mem: $(mem)"; apps

echo; echo "## step 2: baseline completion on A"
BASE=$(gen "$PORT_A"); echo "[$(ts)] BASE: $BASE"

echo; echo "## step 3: park A (sleep level=1 + cuda-checkpoint -> both GPUs 0 MiB)"
S=$(date +%s.%N)
curl -s -X POST "localhost:$PORT_A/sleep?level=1" -o /dev/null
echo "[$(ts)] /sleep?level=1 in $(elapsed "$S")   mem: $(mem)"
if [[ "$TP" == "2" ]]; then
  S=$(date +%s.%N)
  curl -s -X POST "localhost:$PORT_A/checkpoint_prepare" -o /dev/null
  # Not "tear down NCCL": on vLLM 0.28.0 this releases only the FlashInfer
  # all-reduce workspace and FlashInfer all2all (cuda_communicator.py:588), so
  # with stock backends it frees nothing and the checkpoint below then wedges
  # the driver. See docs/sleep-mode.md.
  echo "[$(ts)] /checkpoint_prepare (FlashInfer workspaces only) in $(elapsed "$S")"
fi
S=$(date +%s.%N)
for p in $PIDS_A; do "$CC" --action lock --pid "$p" --timeout 120000; done
for p in $PIDS_A; do "$CC" --action checkpoint --pid "$p"; done
echo "[$(ts)] cuda-checkpoint lock+checkpoint (all pids) in $(elapsed "$S")"
for p in $PIDS_A; do echo "      pid=$p state=$("$CC" --get-state --pid "$p")"; done
echo "[$(ts)] parked: $(mem)  -- entire GPU(s) free"

# ---------------------------------------------------------------------------
echo; echo "## step 4: second tenant B boots on the now-free GPU(s)"
S=$(date +%s.%N)
if [[ "$TP" == "1" ]]; then
  boot "$PORT_B" "$LOG_B" "0" || exit 1
else
  boot "$PORT_B" "$LOG_B" "" || exit 1
fi
echo "[$(ts)] B ready in $(elapsed "$S")"
echo "[$(ts)] B mem: $(mem)"; apps
OUT_B=$(gen "$PORT_B"); verdict "B vs BASE" "$OUT_B" "$BASE"

# ---------------------------------------------------------------------------
if [[ "$TP" == "1" ]]; then
  echo; echo "## step 5: migrate A onto the OTHER physical GPU (swap device-map) + wake"
  E=$PIDS_A
  S=$(date +%s.%N)
  if "$CC" --action restore --pid "$E" --device-map "$U0=$U1,$U1=$U0"; then
    echo "[$(ts)] restore (swap) in $(elapsed "$S")"
  else
    echo "[$(ts)] restore FAILED -- stopping"; teardown "$PORT_B"; teardown "$PORT_A"; exit 1
  fi
  "$CC" --action unlock --pid "$E"
  echo "[$(ts)] restored: $(mem)"; apps
  S=$(date +%s.%N)
  curl -s -X POST "localhost:$PORT_A/wake_up" -o /dev/null
  echo "[$(ts)] /wake_up in $(elapsed "$S")   mem: $(mem)"; apps
  OUT_A=$(gen "$PORT_A"); verdict "A vs BASE (after migrate)" "$OUT_A" "$BASE"

  echo; echo "## step 6: bonus -- sleep level=2 (discard weights) vs reload_weights (the fix)"
  BASE2=$(gen "$PORT_A"); echo "[$(ts)] refreshed baseline: $BASE2"
  S=$(date +%s.%N)
  curl -s -X POST "localhost:$PORT_A/sleep?level=2" -o /dev/null
  T_SLEEP2=$(elapsed "$S")
  echo "[$(ts)] /sleep?level=2 in $T_SLEEP2   mem: $(mem)  -- no cuda-checkpoint used here"
  S=$(date +%s.%N)
  curl -s -X POST "localhost:$PORT_A/wake_up" -o /dev/null
  T_WAKE2=$(elapsed "$S")
  echo "[$(ts)] /wake_up in $T_WAKE2   mem: $(mem)"
  OUT_BROKEN=$(gen "$PORT_A"); verdict "A after level=2 wake_up (expect DIFFERS -- the documented footgun)" "$OUT_BROKEN" "$BASE2"
  S=$(date +%s.%N)
  curl -s -X POST "localhost:$PORT_A/reload_weights" -o /dev/null
  T_RELOAD=$(elapsed "$S")
  echo "[$(ts)] /reload_weights in $T_RELOAD"
  if curl -sf "localhost:$PORT_A/health" >/dev/null 2>&1; then
    OUT_FIXED=$(gen "$PORT_A"); verdict "A after reload_weights (expect IDENTICAL -- the fix)" "$OUT_FIXED" "$BASE2"
  else
    echo "[$(ts)] A after reload_weights: ENGINE CRASHED -- /health no longer responds."
    echo "      reload_weights() apparently corrupted non-weight state (a buffer left on"
    echo "      the meta device); see EngineDeadError / NotImplementedError in $LOG_A around"
    echo "      this timestamp. reload_weights is NOT a safe live fix once CUDA graphs are"
    echo "      captured -- report this honestly rather than the intended 'fixed' verdict."
  fi
  echo "[$(ts)] level=2 total (sleep $T_SLEEP2 + wake $T_WAKE2 + reload $T_RELOAD) vs level=1 total (park step3 + migrate/wake step5) -- see report totals above"

  echo; echo "## step 7: tear down B then A"
  teardown "$PORT_B"; teardown "$PORT_A"
else
  echo; echo "## step 5: stop B (no genuine swap fits a 2-GPU budget while B holds both cards)"
  teardown "$PORT_B"
  echo "[$(ts)] after stopping B: $(mem)"

  echo; echo "## step 6: restore A with an IDENTITY device-map (same physical GPUs) + wake"
  S=$(date +%s.%N)
  ok=1
  for p in $PIDS_A; do "$CC" --action restore --pid "$p" --device-map "$U0=$U0,$U1=$U1" || ok=0; done
  if [[ "$ok" == "1" ]]; then
    echo "[$(ts)] restore (identity, all pids) in $(elapsed "$S")"
  else
    echo "[$(ts)] restore FAILED -- stopping"; teardown "$PORT_A"; exit 1
  fi
  for p in $PIDS_A; do "$CC" --action unlock --pid "$p"; done
  echo "[$(ts)] restored: $(mem)"; apps
  S=$(date +%s.%N)
  curl -s -X POST "localhost:$PORT_A/checkpoint_restore" -o /dev/null
  echo "[$(ts)] /checkpoint_restore (re-attach FlashInfer workspaces) in $(elapsed "$S")"
  S=$(date +%s.%N)
  curl -s -X POST "localhost:$PORT_A/wake_up" -o /dev/null
  echo "[$(ts)] /wake_up in $(elapsed "$S")   mem: $(mem)"; apps
  OUT_A=$(gen "$PORT_A"); verdict "A vs BASE (after checkpoint/restore/wake round trip)" "$OUT_A" "$BASE"
  echo "[$(ts)] NOTE: this proves the TP>1 checkpoint/restore/wake round trip (previously"
  echo "        flagged untested in docs/sleep-mode.md) -- it is NOT a physical migration,"
  echo "        since no genuine swap fits a 2-GPU budget when a TP=2 tenant needs both cards."

  echo; echo "## step 7: tear down A"
  teardown "$PORT_A"
fi

echo; echo "[$(ts)] idle: $(mem)"
echo "=== done. full transcript: $REPORT ==="
