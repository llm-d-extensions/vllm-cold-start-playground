#!/usr/bin/env bash
# Is a cuda-checkpoint'ed process actually serializable by an external dumper?
#
# Answers it by evidence rather than inference: count the process's /dev/nvidia*
# fds and nvidia-matching mappings before, during and after a checkpoint. If the
# checkpointed state holds no device fds and no device mappings, then nothing
# GPU-specific is left for a dumper to understand -- which is exactly the
# contract cuda-checkpoint is built to provide (see --get-restore-tid).
#
# Also verifies the graph still replays correctly after restore.
#
# Written up in docs/sleep-mode.md#cuda-checkpoint-has-no-serialization-by-design.
# cuda-checkpoint is NOT in the vLLM image; fetch the release binary matching the
# driver version (580.105.08 here) from github.com/NVIDIA/cuda-checkpoint and
# place it at /tmp/cuda-checkpoint.
#
# Run inside the pod:  bash cuda-checkpoint-dumpability.sh
set -uo pipefail
CC=${CC:-/tmp/cuda-checkpoint}

cat > /tmp/dumpable.py <<'PY'
import os, time, torch
N = 512*1024*1024//4
a = torch.ones(N, device="cuda"); out = torch.zeros(N, device="cuda")
s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s): out.copy_(a*2+1)
torch.cuda.current_stream().wait_stream(s)
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g): out.copy_(a*2+1)
g.replay(); torch.cuda.synchronize()
print(f"ready pid={os.getpid()} replay={out[0].item()}", flush=True)
open("/tmp/dz_ready","w").write(str(os.getpid()))
while not os.path.exists("/tmp/dz_go"): time.sleep(0.2)
g.replay(); torch.cuda.synchronize()
print(f"post-restore replay={out[0].item()} (expect 3.0)", flush=True)
PY

snap() {
  local P=$1
  echo "--- $2"
  echo "  /dev/nvidia* fds : $(ls -l /proc/$P/fd 2>/dev/null | grep -c nvidia)"
  ls -l /proc/$P/fd 2>/dev/null | grep -o '/dev/nvidia[a-z0-9-]*' | sort | uniq -c | sed 's/^/    /'
  echo "  nvidia mappings  : $(grep -c nvidia /proc/$P/maps 2>/dev/null)"
  grep nvidia /proc/$P/maps 2>/dev/null | awk '{print $6}' | sort | uniq -c | sort -rn | head -3 | sed 's/^/    /'
  echo "  RSS              : $(awk '/VmRSS/{printf "%.2f GiB", $2/1048576}' /proc/$P/status 2>/dev/null)"
  echo "  GPU reported     : $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
}

rm -f /tmp/dz_ready /tmp/dz_go /tmp/dz.out
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 /tmp/dumpable.py > /tmp/dz.out 2>&1 &
PY_PID=$!
for _ in $(seq 1 240); do [ -f /tmp/dz_ready ] && break; kill -0 $PY_PID 2>/dev/null || break; sleep 0.5; done
P=$(cat /tmp/dz_ready 2>/dev/null); [ -z "$P" ] && { cat /tmp/dz.out; exit 1; }

snap $P "BEFORE checkpoint (running)"
$CC --action lock --pid $P --timeout 30000 && $CC --action checkpoint --pid $P
snap $P "AFTER checkpoint (state=$($CC --get-state --pid $P))"
echo "  restore-tid      : $($CC --get-restore-tid --pid $P 2>&1)   <- the dumper integration seam"
$CC --action restore --pid $P && $CC --action unlock --pid $P
snap $P "AFTER restore (state=$($CC --get-state --pid $P))"
touch /tmp/dz_go; wait $PY_PID; cat /tmp/dz.out

echo
echo "=== could a dumper even run here? ==="
which criu criu-ns 2>/dev/null || echo "  criu: NOT in image"
grep CapEff /proc/self/status
python3 -c '
caps = int(open("/proc/self/status").read().split("CapEff:")[1].split()[0], 16)
for name, bit in (("CAP_SYS_ADMIN",21), ("CAP_CHECKPOINT_RESTORE",40), ("CAP_SYS_PTRACE",19)):
    print("  %-24s %s" % (name, "yes" if caps >> bit & 1 else "NO"))'
echo "  ptrace_scope=$(cat /proc/sys/kernel/yama/ptrace_scope 2>/dev/null)  (0 => same-uid ptrace ok, a far lower bar than CRIU)"
