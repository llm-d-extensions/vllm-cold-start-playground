#!/bin/bash
# Migrate a fully warmed vLLM engine to a different GPU on the same node.
#
#   ready -> /sleep?level=1 -> cuda-checkpoint (0 MiB) -> another tenant takes
#   the original card -> restore --device-map onto the other card -> /wake_up
#   -> serve, and compare the completion byte-for-byte against the baseline.
#
# The question this answers is not "does restore succeed" but "do the 61.68 GiB
# of NEW allocations that /wake_up makes land on the REMAPPED card or on the
# original one". They follow the remap.
#
# Measured 2026-09-02, Qwen3-32B, TP=1, 2x H100 80GB, vLLM 0.28.0,
# driver + cuda-checkpoint 580.105.08:
#
#   step                          wall     GPU0        GPU1
#   ready                         --       74629 MiB   4 MiB
#   /sleep?level=1                22.807s  2819 MiB    4 MiB      (freed 70.13 GiB)
#   --action checkpoint           ~2s      0 MiB       0 MiB      state=checkpointed
#   another tenant takes GPU0     --       78351 MiB   0 MiB      (hog holds 76 GiB)
#   restore --device-map (swap)   2.67s    78351 MiB   2819 MiB
#   /wake_up                      1.46s    78351 MiB   72141 MiB  <- new allocs moved
#   completion                    --       byte-identical to baseline
#
# Unpark-onto-a-different-card is 2.67 + 1.46 = 4.13s.
#
# Preconditions, all three load-bearing (see cuda-checkpoint-cross-gpu.sh for
# the matrix that establishes them):
#   * the pod must see ALL the GPUs it might move between (CUDA_VISIBLE_DEVICES
#     unset, nvidia.com/gpu >= 2) -- this breaks the k8s device-plugin
#     isolation model and is the real cost of the technique;
#   * --device-map must list every visible device and be a bijection;
#   * only EngineCore holds a CUDA context; the APIServer does not, so it is
#     the EngineCore pid that gets checkpointed.
#
# Untested: TP>1, which must additionally drive Worker.checkpoint_prepare /
# checkpoint_restore (gpu_worker.py:261) to tear down and rebuild NCCL.
#
# cuda-checkpoint is NOT in the vLLM image; fetch bin/x86_64_Linux/cuda-checkpoint
# from github.com/NVIDIA/cuda-checkpoint matching the driver version.
set -uo pipefail
CC=${CC:-/tmp/cuda-checkpoint}; PORT=${PORT:-8020}; LOG=/tmp/xv.log
MODEL=${MODEL:-Qwen/Qwen3-32B}
U0=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i 0)
U1=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i 1)
ts(){ date +%H:%M:%S; }
mem(){ nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '; }
apps(){ nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader | sed 's/^/      /'; }
gen(){ curl -s localhost:$PORT/v1/completions -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"prompt\":\"The capital of France is\",\"max_tokens\":32,\"temperature\":0}" \
  | python3 -c 'import sys,json; print(repr(json.load(sys.stdin)["choices"][0]["text"]))' 2>/dev/null; }

cat > /tmp/hog.py <<'PY'
import torch, time, os
torch.cuda.set_device(0)
bufs=[]
try:
    while True: bufs.append(torch.empty(4*1024**3, dtype=torch.uint8, device="cuda:0"))
except RuntimeError: pass
torch.cuda.synchronize()
print(f"HOG holds {sum(b.numel() for b in bufs)/2**30:.0f} GiB on physical GPU0", flush=True)
open("/tmp/hog_ready","w").write("1")
while not os.path.exists("/tmp/hog_stop"): time.sleep(0.3)
PY

pkill -f 'vllm serve' 2>/dev/null; rm -f /tmp/hog_ready /tmp/hog_stop; sleep 3
echo "[$(ts)] GPU0=$U0"
echo "[$(ts)] GPU1=$U1"
VLLM_SERVER_DEV_MODE=1 nohup vllm serve --model "$MODEL" --load-format fastsafetensors \
  --max-model-len 8192 --enable-sleep-mode --port $PORT > $LOG 2>&1 &
for _ in $(seq 1 900); do curl -sf localhost:$PORT/health >/dev/null 2>&1 && break; sleep 1; done
A=$(grep -oE 'APIServer pid=[0-9]+'  $LOG|head -1|grep -oE '[0-9]+')
E=$(grep -oE 'EngineCore pid=[0-9]+' $LOG|head -1|grep -oE '[0-9]+')
echo "[$(ts)] READY APIServer=$A EngineCore=$E   $(mem)"
echo "  which processes hold CUDA?"; apps
BASE=$(gen); echo "[$(ts)] baseline: $BASE"

echo "[$(ts)] --- sleep level 1"
curl -s -X POST "localhost:$PORT/sleep?level=1" -o /dev/null; sleep 2
echo "[$(ts)] asleep: $(mem)"
echo "[$(ts)] --- checkpoint EngineCore"
$CC --action lock --pid "$E" --timeout 120000 && $CC --action checkpoint --pid "$E"
echo "[$(ts)] parked: $(mem)   state=$($CC --get-state --pid "$E")"

echo "[$(ts)] --- another tenant claims physical GPU0"
CUDA_VISIBLE_DEVICES=0 python3 /tmp/hog.py 2>&1 | sed 's/^/    /' &
for _ in $(seq 1 300); do [ -f /tmp/hog_ready ] && break; sleep 0.5; done
sleep 2; echo "[$(ts)] GPU0 taken: $(mem)"

echo "[$(ts)] --- restore EngineCore onto physical GPU1 (swap map)"
S=$(date +%s.%N)
if $CC --action restore --pid "$E" --device-map "$U0=$U1,$U1=$U0"; then
  echo "[$(ts)] restore rc=0 in $(python3 -c "import time;print(f'{time.time()-$S:.2f}s')")"
else
  echo "[$(ts)] restore FAILED -- stopping here"; touch /tmp/hog_stop; pkill -f 'vllm serve'; exit 1
fi
$CC --action unlock --pid "$E"
echo "[$(ts)] restored: $(mem)"; apps

echo "[$(ts)] --- wake_up: 61.68 GiB of NEW allocations. Which card do they land on?"
S=$(date +%s.%N)
curl -s -X POST "localhost:$PORT/wake_up" -o /dev/null
echo "[$(ts)] wake in $(python3 -c "import time;print(f'{time.time()-$S:.2f}s')")   $(mem)"
apps
OUT=$(gen); echo "[$(ts)] after:    $OUT"
if [ "$OUT" = "$BASE" ] && [ -n "$OUT" ]; then echo "[$(ts)] CORRECTNESS: IDENTICAL"; else echo "[$(ts)] CORRECTNESS: DIFFERS"; fi
echo "[$(ts)] --- vllm's own log lines"
grep -E "Sleep mode freed|fall asleep|to wake up" $LOG | sed 's/^/    /'
touch /tmp/hog_stop; sleep 2; pkill -f 'vllm serve'; sleep 3
echo "[$(ts)] done: $(mem)"
