#!/bin/bash
# TP=2 park probe: can a warmed TP>1 engine reach 0 MiB and come back with its
# CUDA graphs intact?
#
# scripts/cuda-checkpoint-ipc-matrix.py established the mechanism without vLLM:
# `cuda-checkpoint --action checkpoint` hangs on any process holding a live NCCL
# communicator or torch symmetric memory, releasing that state fixes it, and the
# NCCL comm cannot be released while a CUDA graph that captured a NCCL op is
# alive (ncclCommAbort never returned in 30s; 0.19s once the graph was dropped).
# Keeping those graphs is the whole point of sleep level 1, so the two goals
# collide -- unless the all-reduce that lands inside the graphs goes through a
# backend with an address-stable detach.
#
# Exactly one such backend exists in this build. flashinfer calls it "Stable-VA
# checkpointing" (`flashinfer/comm/allreduce.py:204`): unmap the physical
# backing, keep the virtual address, so a graph that baked the VA in still
# replays. vLLM wires it up through `CudaCommunicator.checkpoint_prepare`, whose
# own comment says "Only FlashInfer all-reduce and FlashInfer all2all are
# supported for now" -- and the default TP dispatch chain on this node is
# ['CUSTOM', 'SYMM_MEM', 'PYNCCL'], none of which has such a path. Hence the
# two arms:
#
#   --arm default    stock backends. Expected to hang, and bounded here so it
#                    costs 60s instead of the 29 minutes the first attempt took
#                    (reports/demo-cold-start-tp2-20260903-110308.txt).
#   --arm flashinfer VLLM_ALLREDUCE_USE_FLASHINFER=1, custom AR and torch symm
#                    mem off, so the graph-captured all-reduce dispatches to the
#                    one backend whose workspace can be detached and re-attached
#                    at the same address.
#
# Every cuda-checkpoint call is wall-clock bounded: a hang is a result to record,
# never a reason to wedge the pod. On a hang the whole tree is SIGKILLed and the
# GPUs are re-read, because the recovery path is itself a finding.
#
# Usage, inside the 2-GPU pod (manifests/pod-exec-2gpu.yaml), cuda-checkpoint
# staged at $CC:
#   ./tp2-park-probe.sh --arm flashinfer
set -uo pipefail

ARM=flashinfer
CCTIMEOUT=90
while [[ $# -gt 0 ]]; do
  case "$1" in
    --arm) ARM="$2"; shift 2 ;;
    --cc-timeout) CCTIMEOUT="$2"; shift 2 ;;
    -h|--help) sed -n '2,36p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
[[ "$ARM" == "default" || "$ARM" == "flashinfer" ]] || { echo "--arm must be default|flashinfer" >&2; exit 2; }

CC=${CC:-/tmp/cuda-checkpoint}
MODEL=${MODEL:-Qwen/Qwen3-32B}
PROBE_SRC=${CS_PROBE_SRC:-/opt/coldstart-src}
PROBE_DIR=/tmp/coldstart-tp2probe
PORT=8020
LOG=/tmp/tp2probe-$ARM.log

[[ -x "$CC" ]] || { echo "no cuda-checkpoint at $CC"; exit 2; }
[[ -f "$PROBE_SRC/sitecustomize.py" ]] || { echo "no probe at $PROBE_SRC"; exit 2; }

ts(){ date +%H:%M:%S; }
mem(){ nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '; }
elapsed(){ python3 -c "import time;print(f'{time.time()-$1:.2f}s')"; }
gen(){ curl -s --max-time 60 "localhost:$PORT/v1/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"prompt\":\"The capital of France is\",\"max_tokens\":32,\"temperature\":0}" \
  | python3 -c 'import sys,json; print(repr(json.load(sys.stdin)["choices"][0]["text"]))' 2>/dev/null; }

# Bounded cuda-checkpoint. Prints ok/HUNG/rc and never blocks past $CCTIMEOUT.
ccdo(){ # $1=action $2=pid [extra...]
  local action=$1 pid=$2; shift 2
  local s rc; s=$(date +%s.%N)
  # Capture the status directly: after a failed `if` with no else branch, `$?`
  # is the status of the compound command (0), not of the condition, which would
  # silently report every timeout as rc=0.
  timeout -k 5 "$CCTIMEOUT" "$CC" --action "$action" --pid "$pid" "$@" >/tmp/ccout 2>&1
  rc=$?
  if [[ $rc -eq 0 ]]; then
    echo "      $action pid=$pid: ok in $(elapsed "$s")"; return 0
  fi
  if [[ $rc -eq 124 || $rc -eq 137 ]]; then
    echo "      $action pid=$pid: HUNG -- no return in ${CCTIMEOUT}s after $(elapsed "$s") (client killed)"
  else
    echo "      $action pid=$pid: FAILED rc=$rc in $(elapsed "$s")  $(head -c 200 /tmp/ccout)"
  fi
  return 1
}
ccstate(){ timeout -k 2 15 "$CC" --get-state --pid "$1" 2>&1 || echo "BLOCKED"; }

# Kill the engine and every process still holding a GPU. The workers rename
# themselves to "VLLM::Worker_TP0" via setproctitle, so a `pkill -f 'vllm serve'`
# misses them entirely and leaves memory stranded; and a `pkill -f 'VLLM::'`
# would match this script's own command line. Go by pid, and verify.
reap(){
  pkill -f "vllm serve.*--port $PORT" 2>/dev/null
  sleep 5
  local p
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    kill -9 "$p" 2>/dev/null
  done
  sleep 6
  local left; left=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr '\n' ' ')
  [[ -n "${left// /}" ]] && echo "      WARNING: still holding GPU: $left"
}

REPORT="reports/tp2-park-$ARM-$(date -u +%Y%m%d-%H%M%S).txt"
mkdir -p reports
exec > >(tee "$REPORT") 2>&1

echo "=== tp2-park-probe --arm $ARM  $(date -u +%FT%TZ) ==="
echo "driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader -i 0)"
U0=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i 0)
U1=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i 1)
pkill -f 'vllm serve' 2>/dev/null; sleep 3
echo "[$(ts)] idle: $(mem)"

rm -rf "$PROBE_DIR"; mkdir -p "$PROBE_DIR"
cp "$PROBE_SRC"/*.py "$PROBE_DIR/"
python3 -m compileall -q "$PROBE_DIR" >/dev/null 2>&1 || true

ENV=(
  PYTHONPYCACHEPREFIX=/cache/pycache
  PYTHONPATH="$PROBE_DIR"
  CS_FORKSERVER=1
  CS_FST=1 CS_FST_MAX_THREADS=8 CS_FST_BBUF_KB=32768
  CS_DEV_ROUTES=1
  VLLM_SERVER_DEV_MODE=1
)
SERVE_EXTRA=()
if [[ "$ARM" == "flashinfer" ]]; then
  # Route the graph-captured all-reduce to the only backend with a stable-VA
  # detach, and remove the two that have none.
  ENV+=(VLLM_ALLREDUCE_USE_FLASHINFER=1 VLLM_ALLREDUCE_USE_SYMM_MEM=0)
  SERVE_EXTRA+=(--disable-custom-all-reduce)
fi

echo; echo "## step 1: boot TP=2 (arm=$ARM)"
S=$(date +%s.%N)
env "${ENV[@]}" nohup vllm serve --model "$MODEL" --load-format fastsafetensors \
  --max-model-len 8192 --enable-sleep-mode --tensor-parallel-size 2 \
  "${SERVE_EXTRA[@]}" --port "$PORT" > "$LOG" 2>&1 &
for _ in $(seq 1 900); do curl -sf "localhost:$PORT/health" >/dev/null 2>&1 && break; sleep 1; done
curl -sf "localhost:$PORT/health" >/dev/null 2>&1 || { echo "FAILED to become ready"; tail -40 "$LOG"; exit 1; }
echo "[$(ts)] ready in $(elapsed "$S")   mem: $(mem)"
echo "[$(ts)] all-reduce backends actually selected:"
grep -o "Using \[.*\] all-reduce backends" "$LOG" | head -2 | sed 's/^/      /'
grep -o "Initialized FlashInfer Allreduce.*" "$LOG" | head -2 | sed 's/^/      /'

PIDS=$(ps -e -o pid,cmd | grep -E 'Worker_TP[0-9]+' | grep -v grep | awk '{print $1}')
echo "[$(ts)] worker pids: $(echo $PIDS | tr '\n' ' ')"
[[ -n "$PIDS" ]] || { echo "no worker pids"; exit 1; }

echo; echo "## step 2: baseline completion"
BASE=$(gen); echo "[$(ts)] BASE: $BASE"
[[ -n "$BASE" ]] || { echo "no baseline completion -- stopping"; pkill -f "vllm serve.*--port $PORT"; exit 1; }

echo; echo "## step 3: sleep level=1 (graphs deliberately kept)"
S=$(date +%s.%N)
curl -s --max-time 300 -X POST "localhost:$PORT/sleep?level=1" -o /dev/null
echo "[$(ts)] /sleep?level=1 in $(elapsed "$S")   mem: $(mem)"

echo; echo "## step 4: /checkpoint_prepare (upstream stable-VA detach)"
S=$(date +%s.%N)
curl -s --max-time 300 -X POST "localhost:$PORT/checkpoint_prepare" -o /dev/null
echo "[$(ts)] /checkpoint_prepare in $(elapsed "$S")   mem: $(mem)"

echo; echo "## step 5: cuda-checkpoint both ranks (bounded at ${CCTIMEOUT}s each)"
OK=1
for p in $PIDS; do ccdo lock "$p" --timeout 30000 || OK=0; done
if [[ "$OK" == "1" ]]; then
  for p in $PIDS; do ccdo checkpoint "$p" || { OK=0; break; }; done
fi
for p in $PIDS; do echo "      pid=$p state=$(ccstate "$p")"; done
echo "[$(ts)] after checkpoint attempt: $(mem)"

if [[ "$OK" != "1" ]]; then
  echo
  echo "[$(ts)] VERDICT arm=$ARM: checkpoint did NOT complete."
  echo "        This is the same wedge as reports/demo-cold-start-tp2-*.txt, now"
  echo "        bounded. Root cause and the isolating matrix:"
  echo "        scripts/cuda-checkpoint-ipc-matrix.py / docs/sleep-mode.md."
  echo "[$(ts)] recovering: SIGKILL the tree"
  reap
  echo "[$(ts)] after SIGKILL: $(mem)   (0 MiB here means the wedge was"
  echo "        process-level, not unrecoverable driver state)"
  echo "=== done. transcript: $REPORT ==="
  exit 0
fi

echo; echo "## step 6: restore + unlock + rebuild communicators + wake"
for p in $PIDS; do ccdo restore "$p" --device-map "$U0=$U0,$U1=$U1" || OK=0; done
for p in $PIDS; do ccdo unlock "$p" || OK=0; done
echo "[$(ts)] restored: $(mem)"
if [[ "$OK" == "1" ]]; then
  S=$(date +%s.%N)
  curl -s --max-time 300 -X POST "localhost:$PORT/checkpoint_restore" -o /dev/null
  echo "[$(ts)] /checkpoint_restore in $(elapsed "$S")"
  S=$(date +%s.%N)
  curl -s --max-time 300 -X POST "localhost:$PORT/wake_up" -o /dev/null
  echo "[$(ts)] /wake_up in $(elapsed "$S")   mem: $(mem)"
fi

echo; echo "## step 7: correctness after the round trip"
if curl -sf --max-time 30 "localhost:$PORT/health" >/dev/null 2>&1; then
  GOT=$(gen)
  if [[ -n "$GOT" && "$GOT" == "$BASE" ]]; then
    echo "[$(ts)] A vs BASE: IDENTICAL ($GOT)"
    echo "[$(ts)] VERDICT arm=$ARM: TP=2 park to 0 MiB and back, graphs intact."
  else
    echo "[$(ts)] A vs BASE: DIFFERS  got=$GOT  base=$BASE"
    echo "[$(ts)] VERDICT arm=$ARM: checkpoint/restore completed but the engine"
    echo "        no longer computes the same tokens -- the graphs did not"
    echo "        survive. Report this, do not call it a success."
  fi
else
  echo "[$(ts)] /health does not respond after the round trip -- engine dead."
  echo "[$(ts)] VERDICT arm=$ARM: checkpoint/restore completed, engine did not."
  tail -30 "$LOG" | sed 's/^/      /'
fi

echo; echo "## step 8: teardown"
reap
echo "[$(ts)] idle: $(mem)"
echo "=== done. transcript: $REPORT ==="
