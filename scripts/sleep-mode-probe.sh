#!/usr/bin/env bash
# Measure vLLM sleep mode: does freeing weights + KV cache via the CUDA VMM API
# release the GPU, and do the captured CUDA graphs still replay afterwards?
#
# Run *inside* the experiment pod (see docs/sleep-mode.md for the results):
#   kubectl -n <ns> cp scripts/sleep-mode-probe.sh vllm-coldstart:/tmp/ -c vllm
#   kubectl -n <ns> exec vllm-coldstart -c vllm -- bash /tmp/sleep-mode-probe.sh
#
# The load-bearing assertion is not a timing: it is that the generation after
# wake_up is byte-identical to the one before sleep. The probe request is a short
# prefill (a PIECEWISE graph) plus 32 greedy decode steps (32 FULL graphs), so an
# identical string means every device pointer baked into those graphs still
# resolves correctly after their physical pages were released and re-created.
#
# vLLM's own log lines are the authoritative timings ("It took N seconds to fall
# asleep" / "to wake up"); the wall-clock echoes here are only a sanity check.
set -uo pipefail

MODEL="${MODEL:-Qwen/Qwen3-32B}"
PORT="${PORT:-8100}"
LEVELS="${LEVELS:-1 1 2}"
LOG=/tmp/sleep-mode-probe-vllm.log
export VLLM_SERVER_DEV_MODE=1        # gates /sleep, /wake_up, /is_sleeping

mem() { nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader; }
say() { echo "[$(date -u +%H:%M:%S)] $*"; }
now() { python3 -c 'import time; print(f"{time.monotonic():.3f}")'; }
el()  { python3 -c "print(f'{$2-$1:.2f}s')"; }
gen() {
  curl -s -m 120 "localhost:$PORT/v1/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"prompt\":\"The capital of France is\",\"max_tokens\":32,\"temperature\":0,\"seed\":1}" \
  | python3 -c 'import sys,json; d=json.load(sys.stdin); print(repr(d["choices"][0]["text"]) if "choices" in d else "ERR "+json.dumps(d)[:300])'
}

say "GPU before launch: $(mem)"
nohup vllm serve --model "$MODEL" --load-format fastsafetensors \
  --max-model-len 8192 --enable-sleep-mode --port "$PORT" > "$LOG" 2>&1 &
VPID=$!
trap 'kill $VPID 2>/dev/null; sleep 5; kill -9 $VPID 2>/dev/null' EXIT

t0=$(now)
for _ in $(seq 1 600); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' -m 2 "localhost:$PORT/health" || true)" = 200 ] && break
  kill -0 $VPID 2>/dev/null || { say "vllm died early"; tail -40 "$LOG"; exit 1; }
  sleep 1
done
say "READY after $(el "$t0" "$(now)")"
say "GPU ready: $(mem)"
grep -E "Model loading took|GPU KV cache size|Graph capturing finished|sleep-mode backend" "$LOG"

BASE=$(gen); say "baseline OUT: $BASE"

for LEVEL in $LEVELS; do
  say "===== sleep level=$LEVEL ====="
  s0=$(now)
  curl -s -o /dev/null -X POST "localhost:$PORT/sleep?level=$LEVEL" -m 900
  say "sleep wall $(el "$s0" "$(now)")  is_sleeping=$(curl -s "localhost:$PORT/is_sleeping")"
  say "GPU ASLEEP: $(mem)"
  sleep 3; say "GPU ASLEEP +3s: $(mem)"     # second reading: util must settle to 0%

  w0=$(now)
  curl -s -o /dev/null -X POST "localhost:$PORT/wake_up" -m 900
  say "wake wall $(el "$w0" "$(now)")"
  say "GPU AWAKE: $(mem)"
  AFTER=$(gen); say "OUT: $AFTER"
  [ "$AFTER" = "$BASE" ] && say "CORRECTNESS: IDENTICAL to baseline" \
                         || say "CORRECTNESS: DIFFERS from baseline"
done

say "--- vLLM's own sleep/wake timings"
grep -E "fall asleep|to wake up|Sleep mode freed|CuMemAllocator: sleep freed" "$LOG"
