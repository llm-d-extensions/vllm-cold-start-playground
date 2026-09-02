#!/usr/bin/env bash
# What would a serialized image of a parked vLLM weigh?
#
# cuda-checkpoint cannot write bytes (see cuda-checkpoint-dumpability.sh), so the
# only route to disk is a process dumper -- and then the deciding number is the
# RSS of the parked process tree at the moment the GPU reads 0 MiB. This measures
# it for both sleep levels, with the backing of each mapping, plus the
# filesystem's read rate so the image can be priced in seconds.
#
# Level 1 keeps CuMemAllocator's pinned host copy of the weights (a large
# `rw-s /dev/zero (deleted)` mapping) and is hopeless. Level 2 discards it.
# IMPORTANT: run the two levels in SEPARATE boots. vLLM retains the host backup
# buffer across sleep/wake cycles, so a level-2 sleep after a level-1 cycle still
# shows the 69 GiB mapping and reports a contaminated figure.
#
# Written up in docs/sleep-mode.md#if-a-dumper-were-available-what-would-the-image-weigh.
#
# Run inside the pod:  LEVEL=1 bash vllm-parked-image-size.sh
#                      LEVEL=2 bash vllm-parked-image-size.sh
set -uo pipefail
CC=${CC:-/tmp/cuda-checkpoint}
LEVEL=${LEVEL:-2}
PORT=${PORT:-8013}
LOG=/tmp/parked-image-$LEVEL.log
MODEL=${MODEL:-Qwen/Qwen3-32B}

r(){ awk '/VmRSS/{printf "%.2f", $2/1048576}' /proc/$1/status 2>/dev/null; }
g(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader; }

pkill -f 'vllm serve' 2>/dev/null; sleep 3
VLLM_SERVER_DEV_MODE=1 nohup vllm serve --model "$MODEL" --load-format fastsafetensors \
  --max-model-len 8192 --enable-sleep-mode --port $PORT > $LOG 2>&1 &
for _ in $(seq 1 400); do curl -sf localhost:$PORT/health >/dev/null 2>&1 && break; sleep 1; done
A=$(grep -oE 'APIServer pid=[0-9]+'  $LOG | head -1 | grep -oE '[0-9]+')
E=$(grep -oE 'EngineCore pid=[0-9]+' $LOG | head -1 | grep -oE '[0-9]+')

echo "READY            api=$(r $A) engine=$(r $E) GiB  gpu=$(g)"
curl -s -X POST "localhost:$PORT/sleep?level=$LEVEL" -o /dev/null; sleep 2
echo "level-$LEVEL asleep  api=$(r $A) engine=$(r $E) GiB  gpu=$(g)"
$CC --action lock --pid $E --timeout 60000 >/dev/null && $CC --action checkpoint --pid $E >/dev/null
echo "checkpointed     api=$(r $A) engine=$(r $E) GiB  gpu=$(g)"
echo "=== IMAGE = api + engine above.  EngineCore RSS by backing:"
python3 - "$E" <<'PY'
import sys, re, collections
pid = sys.argv[1]; cur = None; agg = collections.Counter()
for ln in open(f"/proc/{pid}/smaps"):
    m = re.match(r'^[0-9a-f]+-[0-9a-f]+ (\S+) \S+ \S+ \S+\s*(.*)$', ln)
    if m:
        cur = (m.group(2).strip() or "[anon]", m.group(1)); continue
    if ln.startswith("Rss:") and cur:
        agg[cur] += int(ln.split()[1])
for (path, perm), kb in agg.most_common(8):
    if kb > 65536:
        print(f"  {kb/1048576:7.2f} GiB  {perm}  {path}")
print(f"  {sum(agg.values())/1048576:7.2f} GiB  TOTAL")
PY
echo "=== read budget for that image:"
dd if=/dev/zero of=/tmp/pi.bin bs=1M count=4096 oflag=direct 2>&1 | tail -1
rm -f /tmp/pi.bin
$CC --action restore --pid $E >/dev/null && $CC --action unlock --pid $E >/dev/null
pkill -f 'vllm serve'; sleep 2; echo "done gpu=$(g)"
