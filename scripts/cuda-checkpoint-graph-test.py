# Isolated test: do a captured CUDA graph and an unbacked VMM address reservation
# both survive an external cuda-checkpoint of the process?  See docs/sleep-mode.md
# (§ Composing with cuda-checkpoint to reach 0 MiB) for the measured result.
#
# Driven by the outer loop in scripts/cuda-checkpoint-graph-test.sh-style usage:
# run with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, wait for the process
# to write /tmp/cc_ready, then from another shell:
#   cuda-checkpoint --action lock       --pid <pid> --timeout 30000
#   cuda-checkpoint --action checkpoint --pid <pid>   # nvidia-smi must read 0 MiB
#   cuda-checkpoint --action restore    --pid <pid>
#   cuda-checkpoint --action unlock     --pid <pid>
#   touch /tmp/cc_go
# The assertion that matters is the last line: REPLAY ... CORRECT.
# Mirrors the sleep-mode + cuda-checkpoint composition in miniature:
#   capture a graph reading `a` -> release a's PHYSICAL pages but keep its VA
#   reservation (what CuMemAllocator.sleep does) -> external cuda-checkpoint
#   destroys and recreates the CUDA context -> re-map at the same VA (wake_up)
#   -> replay the graph captured before all of that.
import os, time, torch
N = 512 * 1024 * 1024 // 4          # 512 MiB of float32
free0 = torch.cuda.mem_get_info()[0]
a   = torch.ones(N, device="cuda")   # stand-in for "weights"
out = torch.zeros(N, device="cuda")  # stays mapped: stand-in for the graph pool
ptr_a = a.data_ptr()

s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    out.copy_(a * 2 + 1)
torch.cuda.current_stream().wait_stream(s)
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    out.copy_(a * 2 + 1)
g.replay(); torch.cuda.synchronize()
print(f"CAPTURED  a@{ptr_a:#x}  replay out[0]={out[0].item()}  (expect 3.0)", flush=True)

del a
torch.cuda.empty_cache()                       # <- the "sleep": unmap+release, VA kept
free1 = torch.cuda.mem_get_info()[0]
print(f"SLEPT     driver free {free0/2**30:.2f} -> {free1/2**30:.2f} GiB", flush=True)

open("/tmp/cc_ready", "w").write(str(os.getpid()))
t = time.monotonic()
while not os.path.exists("/tmp/cc_go"):
    if time.monotonic() - t > 300: print("TIMEOUT waiting for go", flush=True); raise SystemExit(2)
    time.sleep(0.2)
print("RESUMED   after external checkpoint/restore", flush=True)

try:
    a2 = torch.ones(N, device="cuda")          # <- the "wake_up": re-map at same VA
    print(f"WOKE      a2@{a2.data_ptr():#x}  VA_MATCH={a2.data_ptr() == ptr_a}", flush=True)
    out.zero_(); torch.cuda.synchronize()
    g.replay(); torch.cuda.synchronize()
    v = out[0].item()
    print(f"REPLAY    out[0]={v}  -> {'CORRECT' if v == 3.0 else 'WRONG (expected 3.0)'}", flush=True)
except Exception as e:
    print(f"FAILED    {type(e).__name__}: {e}", flush=True)
