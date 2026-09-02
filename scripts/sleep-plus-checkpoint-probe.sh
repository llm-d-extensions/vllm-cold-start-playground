#!/usr/bin/env bash
# Full composition probe: vLLM sleep mode frees the weights + KV cache (69.70 GiB)
# via the CUDA VMM API, then cuda-checkpoint releases the ~3-4 GiB residual (CUDA
# context, modules, graph pool) so the card reads 0 MiB -- with the captured CUDA
# graphs still replayable afterwards.  Measured results in docs/sleep-mode.md.
#
# Needs, inside the pod: /tmp/cuda-checkpoint (NVIDIA/cuda-checkpoint, bin/x86_64_Linux,
# must match the driver major version) and ptrace_scope=0 for same-uid ptrace.
# Checkpoint the *EngineCore* pid: at TP=1 it is the only process holding CUDA.
#
# sleep mode (frees 69.70 GiB) + cuda-checkpoint (frees the residual 4.17 GiB)
# => zero GPU held, process alive, graphs intact?
set -uo pipefail
CC=/tmp/cuda-checkpoint; PORT=8100; LOG=/tmp/compose-vllm.log
export VLLM_SERVER_DEV_MODE=1
mem() { nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader; }
say() { echo "[$(date -u +%H:%M:%S)] $*"; }
now() { python3 -c 'import time;print(f"{time.monotonic():.3f}")'; }
el()  { python3 -c "print(f'{$2-$1:.2f}s')"; }
gen() { curl -s -m 120 localhost:$PORT/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-32B","prompt":"The capital of France is","max_tokens":32,"temperature":0,"seed":1}' \
  | python3 -c 'import sys,json;d=json.load(sys.stdin);print(repr(d["choices"][0]["text"]) if "choices" in d else "ERR "+json.dumps(d)[:200])'; }

nohup vllm serve --model Qwen/Qwen3-32B --load-format fastsafetensors --max-model-len 8192 \
  --enable-sleep-mode --port $PORT > $LOG 2>&1 &
for _ in $(seq 1 600); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' -m 2 localhost:$PORT/health || true)" = 200 ] && break; sleep 1; done
API=$(pgrep -f "vllm serve --model Qwen/Qwen3-32B" | head -1)
EC=$(grep -oE "EngineCore pid=[0-9]+" $LOG | head -1 | cut -d= -f2)
say "READY  APIServer=$API EngineCore=$EC  gpu=$(mem)"
say "cuda state: APIServer=$($CC --get-state --pid $API 2>&1) EngineCore=$($CC --get-state --pid $EC 2>&1)"
BASE=$(gen); say "baseline OUT: $BASE"

say "--- step 1: sleep level 1"
curl -s -o /dev/null -X POST "localhost:$PORT/sleep?level=1" -m 900
say "GPU after sleep:      $(mem)"

say "--- step 2: cuda-checkpoint the EngineCore"
t=$(now); $CC --action lock --pid $EC --timeout 60000; say "lock rc=$? $(el $t $(now))"
t=$(now); $CC --action checkpoint --pid $EC;           say "checkpoint rc=$? $(el $t $(now)) state=$($CC --get-state --pid $EC)"
say "GPU AFTER CHECKPOINT: $(mem)"
sleep 5
say "GPU +5s:              $(mem)"

say "--- step 3: restore"
t=$(now); $CC --action restore --pid $EC; say "restore rc=$? $(el $t $(now))"
t=$(now); $CC --action unlock  --pid $EC; say "unlock rc=$? $(el $t $(now)) state=$($CC --get-state --pid $EC)"
say "GPU after restore:    $(mem)"

say "--- step 4: wake_up"
t=$(now); curl -s -o /dev/null -X POST "localhost:$PORT/wake_up" -m 900; say "wake_up $(el $t $(now))"
say "GPU after wake:       $(mem)"
AFTER=$(gen); say "OUT: $AFTER"
[ "$AFTER" = "$BASE" ] && say "CORRECTNESS: IDENTICAL" || say "CORRECTNESS: DIFFERS"
say "--- vllm timings"; grep -E "fall asleep|to wake up|Sleep mode freed|EngineCore.*died|dead|Traceback" $LOG | tail -12
pkill -f "vllm serve --model Qwen/Qwen3-32B"; sleep 6; pkill -9 -f "vllm serve" 2>/dev/null
say "done gpu=$(mem)"
