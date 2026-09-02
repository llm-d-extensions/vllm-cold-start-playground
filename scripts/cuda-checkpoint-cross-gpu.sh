#!/bin/bash
# Can a cuda-checkpoint snapshot be restored onto a DIFFERENT GPU?
#
# Micro-probe (no model): a process holds 2 GiB and a captured CUDA graph on
# GPU0. Checkpoint it, restore it with --device-map, then replay the graph and
# run a fresh op. Establishes the three rules that govern --device-map, each by
# measurement rather than by reading the help text.
#
# Measured 2026-09-02, 2x H100 80GB, driver 580.105.08, cuda-checkpoint
# 580.105.08, torch 2.13.0+cu130:
#
#   case                                        map              rc  memory after      graph
#   all visible, identity (control)             U0=U0,U1=U1      0   0,2773  1,4 MiB    3.0 OK
#   all visible, SWAP                           U0=U1,U1=U0      0   0,4  1,2773 MiB    3.0 OK
#   only GPU0 visible, remap to GPU1            U0=U1            1   unchanged         --
#   non-permutation (both -> GPU1)              U0=U1,U1=U1      1   --                --
#
# => 1. the map must list EVERY device visible to the process, not just the
#       devices that hold contexts (this is why single-entry maps, INCLUDING
#       the identity map, return "invalid argument");
#    2. the map must be a bijection;
#    3. the target must be visible to the process -- so CUDA_VISIBLE_DEVICES=0,
#       the default Kubernetes one-GPU-per-pod shape, CANNOT do this.
#
# The remapped process is unaware it moved: same torch device index, same
# virtual address, and get_device_properties() still reports the OLD uuid.
# A failed remap is non-destructive -- a plain restore afterwards succeeds.
#
# Requires: 2 visible GPUs, CUDA_VISIBLE_DEVICES unset, and cuda-checkpoint at
# $CC (NOT in the vLLM image; fetch bin/x86_64_Linux/cuda-checkpoint from
# github.com/NVIDIA/cuda-checkpoint, matching the driver version).
set -uo pipefail
CC=${CC:-/tmp/cuda-checkpoint}
U0=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i 0)
U1=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i 1)
mem(){ nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cat > /tmp/xgpu.py <<'PY'
import os, sys, time, torch
dev = 0
torch.cuda.set_device(dev)
N = 512*1024*1024//4                       # 2 GiB of f32
a   = torch.ones(N,  device=f"cuda:{dev}") # stands in for weights
out = torch.zeros(N, device=f"cuda:{dev}")
s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s): out.copy_(a*2+1)
torch.cuda.current_stream().wait_stream(s)
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g): out.copy_(a*2+1)   # virtual addresses baked into args
g.replay(); torch.cuda.synchronize()
print(f"READY pid={os.getpid()} dev={a.device} ptr={a.data_ptr():#x} "
      f"uuid={torch.cuda.get_device_properties(dev).uuid} replay={out[0].item()}", flush=True)
open("/tmp/xg_ready","w").write(str(os.getpid()))
while not os.path.exists("/tmp/xg_go"): time.sleep(0.2)
try:
    g.replay(); torch.cuda.synchronize()
    ok = abs(out[0].item() - 3.0) < 1e-6 and bool((out == 3.0).all().item())
    print(f"POST replay out[0]={out[0].item()} all_correct={ok} "
          f"dev={a.device} ptr={a.data_ptr():#x} "
          f"uuid_now={torch.cuda.get_device_properties(0).uuid}", flush=True)
    b = a * 3
    print(f"POST fresh-op b[0]={b[0].item()} (expect 3.0)", flush=True)
except Exception as e:
    print(f"POST FAILED: {type(e).__name__}: {e}", flush=True)
PY

launch(){ # $1=CUDA_VISIBLE_DEVICES ("" for all)
  rm -f /tmp/xg_ready /tmp/xg_go /tmp/xg.out
  if [ -n "${1:-}" ]; then CUDA_VISIBLE_DEVICES=$1 python3 /tmp/xgpu.py >/tmp/xg.out 2>&1 &
  else python3 /tmp/xgpu.py >/tmp/xg.out 2>&1 & fi
  for _ in $(seq 1 240); do [ -f /tmp/xg_ready ] && break; sleep 0.5; done
  cat /tmp/xg_ready 2>/dev/null
}

# One FRESH process per attempt. Reusing one checkpointed process across attempts
# is a trap: the first success consumes it and every later row then reports
# "the operation cannot be performed in the present state".
attempt(){ # $1=label $2=device-map ("" for none) $3=CUDA_VISIBLE_DEVICES
  local label="$1" map="$2" cvd="${3:-}" P S el out rc
  P=$(launch "$cvd"); [ -z "$P" ] && { echo "=== $label: launch failed"; return; }
  echo "=== $label"
  echo "    CUDA_VISIBLE_DEVICES=${cvd:-<unset>}   before: $(mem)"
  $CC --action lock --pid "$P" --timeout 30000 >/dev/null && $CC --action checkpoint --pid "$P" >/dev/null
  echo "    parked: $(mem)   state=$($CC --get-state --pid "$P")"
  S=$(date +%s.%N)
  if [ -n "$map" ]; then out=$($CC --action restore --pid "$P" --device-map "$map" 2>&1); rc=$?
  else out=$($CC --action restore --pid "$P" 2>&1); rc=$?; fi
  el=$(python3 -c "import time;print(f'{time.time()-$S:.2f}s')")
  echo "    map='${map:-<none>}'  ->  rc=$rc in $el ${out:+-- $out}"
  echo "    after : $(mem)"
  [ "$rc" -ne 0 ] && $CC --action restore --pid "$P" >/dev/null 2>&1   # non-destructive check
  $CC --action unlock --pid "$P" >/dev/null 2>&1
  touch /tmp/xg_go; sleep 5
  grep -E "^READY|^POST" /tmp/xg.out | sed 's/^/    /'
  kill "$P" 2>/dev/null; sleep 3
}

attempt "1. all GPUs visible, identity map (control)"   "$U0=$U0,$U1=$U1" ""
attempt "2. all GPUs visible, SWAP map (GPU0 -> GPU1)"  "$U0=$U1,$U1=$U0" ""
attempt "3. k8s shape: only GPU0 visible, -> GPU1"      "$U0=$U1"         "0"
attempt "4. non-permutation: both old devices -> GPU1"  "$U0=$U1,$U1=$U1" ""

# The real use case: park off GPU0, let another tenant TAKE GPU0, come back on GPU1.
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

echo "=== 5. reallocation: parked -> GPU0 stolen by another tenant -> restore on GPU1"
rm -f /tmp/hog_ready /tmp/hog_stop
P=$(launch "")
echo "    victim pid=$P on GPU0.  before: $(mem)"
$CC --action lock --pid "$P" --timeout 30000 >/dev/null && $CC --action checkpoint --pid "$P" >/dev/null
echo "    parked: $(mem)"
CUDA_VISIBLE_DEVICES=0 python3 /tmp/hog.py 2>&1 | sed 's/^/      /' &
for _ in $(seq 1 240); do [ -f /tmp/hog_ready ] && break; sleep 0.5; done
sleep 2; echo "    GPU0 now occupied: $(mem)"
S=$(date +%s.%N)
$CC --action restore --pid "$P" --device-map "$U0=$U1,$U1=$U0" \
  && echo "    restore rc=0 in $(python3 -c "import time;print(f'{time.time()-$S:.2f}s')")" \
  || echo "    restore FAILED"
$CC --action unlock --pid "$P" >/dev/null 2>&1
echo "    after: $(mem)"
touch /tmp/xg_go; sleep 5
grep -E "^POST" /tmp/xg.out | sed 's/^/    /'
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader | sed 's/^/      /'
touch /tmp/hog_stop; kill "$P" 2>/dev/null; sleep 3
echo "final: $(mem)"
