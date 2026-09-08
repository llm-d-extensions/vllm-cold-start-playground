#!/usr/bin/env bash
# Isolate the fd that blocks `criu dump` of a vLLM engine -- with no GPU memory
# involved at all, so the answer cannot be confused with CUDA context state.
#
# The chain of evidence this closes:
#
#   * scripts/criu-matrix.sh: cuda-checkpoint takes a small CUDA process to
#     nvidia_fds=0 and criu then dumps it.
#   * scripts/criu-vllm-probe.sh arm ckpt: on a real engine, cuda-checkpoint
#     succeeds (GPU -> 0 MiB, RSS -> 76.78 GiB) yet EngineCore keeps
#     nvidia_fds=9, and criu dies on one: "Can't dump file 9 of that type
#     [20666] (chr 195:255)" -- /dev/nvidiactl.
#   * scripts/criu-preboot.sh --probe: merely importing vLLM's *server* module
#     graph opens 11 /dev/nvidia* fds while allocating ZERO device memory.
#
# That last one is the tell. An fd that exists before any CUDA allocation is not
# a CUDA-context fd, so cuda-checkpoint has no claim on it. The candidate is
# NVML -- libnvidia-ml, which vLLM reaches through pynvml for device queries --
# and NVML handles are released only by the owning process calling
# nvmlShutdown().
#
# So: import vLLM's server graph, take no device memory, and walk through
#   (a) what the fds are,
#   (b) what cuda-checkpoint makes of a process with nvidia fds but no context,
#   (c) whether criu can dump it,
#   (d) whether pynvml.nvmlShutdown() closes them,
#   (e) whether criu can dump it *then*.
#
# If (d) drops the count to zero and (e) succeeds, the blocker is named exactly,
# and the fix is a vLLM-side one rather than anything criu or NVIDIA must ship.
#
#   python3 criu-reaper.py bash criu-nvml-isolate.sh
set -uo pipefail

# NOT named CC: that is the C-compiler variable, and exporting it into the
# environment makes FlashInfer/Inductor JIT invoke this binary as a compiler.
# Observed 2026-09-04: `CC=/usr/local/bin/cuda-checkpoint` on the exec line gave
# "cuda-checkpoint: invalid option -- 'O'" and killed the 32B boot in
# profile_run, which looks nothing like the actual cause.
CUDA_CKPT=${CUDA_CKPT:-/tmp/cuda-checkpoint}
CRIU=${CRIU:-criu}
IMG=${IMG:-/criu/img/nvml-isolate}
TMO=${TMO:-180}
RDY=/tmp/nvml.ready
STEP=/tmp/nvml.step
OUT=/tmp/nvml.out
LOG=/tmp/nvml.log

say(){ echo "[$(date -u +%H:%M:%S)] $*"; }
nvcount(){ ls -l "/proc/$1/fd" 2>/dev/null | grep -c '/dev/nvidia'; }
nvlist(){ ls -l "/proc/$1/fd" 2>/dev/null | grep '/dev/nvidia' \
          | awk '{print "      fd " $9 " -> " $11}' | sort -t' ' -k3; }

# The subject. Imports exactly what `vllm serve` imports, allocates nothing on
# the device, then takes instructions through a file so it survives a dump.
cat > /tmp/nvml_child.py <<'PY'
import os, sys, time

rdy, step, outf = sys.argv[1], sys.argv[2], sys.argv[3]
log = open(outf, "a", buffering=1)
def note(m):
    log.write(m + "\n"); sys.stdout.write(m + "\n"); sys.stdout.flush()

def nvfds():
    n = 0
    for f in os.listdir("/proc/self/fd"):
        try:
            if "/dev/nvidia" in os.readlink("/proc/self/fd/" + f): n += 1
        except OSError: pass
    return n

note(f"start nvidia_fds={nvfds()}")
import torch                                            # noqa: F401
note(f"after import torch nvidia_fds={nvfds()}")
import vllm                                              # noqa: F401
note(f"after import vllm  nvidia_fds={nvfds()}")
from vllm.entrypoints.openai import api_server           # noqa: F401
from vllm.entrypoints.cli import main as _cli            # noqa: F401
from vllm.engine.arg_utils import AsyncEngineArgs        # noqa: F401
import uvicorn, fastapi                                  # noqa: F401
note(f"after server graph nvidia_fds={nvfds()}")
note(f"cuda initialised? {torch.cuda.is_initialized()}  "
     f"device mem allocated={torch.cuda.memory_allocated() if torch.cuda.is_initialized() else 0}")

open(rdy, "w").write(str(os.getpid()))

# Wait for instructions. Polling a file, not a pipe: a restored process has to
# resume inside this loop unaided.
done_shutdown = False
while True:
    try: want = open(step).read().strip()
    except OSError: want = ""
    if want == "shutdown" and not done_shutdown:
        try:
            import pynvml
            pynvml.nvmlShutdown()
            note(f"nvmlShutdown() ok, nvidia_fds={nvfds()}")
        except Exception as e:
            note(f"nvmlShutdown() raised {type(e).__name__}: {e}")
            # Maybe vLLM holds it through its own wrapper rather than pynvml's
            # module-level singleton.
            try:
                from vllm.platforms import cuda as vcuda
                note("vllm.platforms.cuda present; attrs: " +
                     ",".join(a for a in dir(vcuda) if "nvml" in a.lower()) or "(none)")
            except Exception as e2:
                note(f"vllm.platforms.cuda import failed: {e2}")
        done_shutdown = True
        note(f"post-shutdown nvidia_fds={nvfds()}")
        open(outf + ".shutdown_done", "w").write("1")
    elif want == "exit":
        note("exiting"); break
    time.sleep(0.05)
PY

try_dump(){ # $1=pid $2=label
  # One image dir PER LABEL: the whole point of this script is comparing the
  # before/after dump logs, and a shared dir would delete the first one.
  # Separate statements on purpose: bash expands every word of a `local` line
  # BEFORE applying any of its assignments, so `d="$IMG-$label"` on the same line
  # reads $label while it is still unset and dies under `set -u`.
  local pid=$1 label=$2 rc
  local d="$IMG-$label"
  rm -rf "$d"; mkdir -p "$d"
  timeout -k 5 "$TMO" "$CRIU" dump -t "$pid" -D "$d" -v4 --log-file dump.log \
      --tcp-close --file-locks --ext-unix-sk --manage-cgroups=ignore \
      --link-remap --force-irmap --leave-running >"$d/dump.stdout" 2>&1
  rc=$?
  say "  criu dump [$label] rc=$rc  image=$(du -sh "$d" 2>/dev/null | awk '{print $1}')  log=$d/dump.log"
  if [[ $rc -ne 0 ]]; then
    grep -aE "Error \(" "$d/dump.log" | tail -4 | sed 's/^/      /'
  else
    # A success here is the finding, so show the plugin actually engaged.
    grep -aiE "cuda_plugin|Plugin \"" "$d/dump.log" | tail -4 | sed 's/^/      /'
  fi
  return $rc
}

echo "======================================================================"
echo "criu-nvml-isolate: which fd blocks the dump, with zero device memory"
echo "criu $("$CRIU" --version 2>&1|head -1)   plugins: $(ls /usr/lib/criu/ 2>/dev/null | tr '\n' ' ')"
echo "======================================================================"
rm -f "$RDY" "$STEP" "$OUT" "$OUT.shutdown_done"
: > "$STEP"

setsid python3 /tmp/nvml_child.py "$RDY" "$STEP" "$OUT" </dev/null >"$LOG" 2>&1 &
for _ in $(seq 1 600); do [[ -s "$RDY" ]] && break; sleep 0.5; done
P=$(cat "$RDY" 2>/dev/null)
[[ -z "$P" ]] && { say "child failed"; tail -20 "$LOG"; exit 1; }
say "child pid=$P"
sed 's/^/    /' "$OUT"

echo
say "(a) the fds it holds, with no device memory allocated:"
say "    count=$(nvcount $P)"
nvlist "$P"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | sed 's/^/      gpu /'

echo
say "(b) what does cuda-checkpoint make of this process?"
say "    --get-state: $(timeout 30 $CUDA_CKPT --get-state --pid $P 2>&1)"
timeout -k 5 60 "$CUDA_CKPT" --action checkpoint --pid "$P" >/tmp/ccn 2>&1
say "    --action checkpoint rc=$? -- $(head -2 /tmp/ccn | tr '\n' ' ')"
say "    nvidia_fds after: $(nvcount $P)"

echo
say "(c) can criu dump it as-is?"
try_dump "$P" "nvml-open"

echo
say "(d) ask the process to call pynvml.nvmlShutdown()"
echo shutdown > "$STEP"
for _ in $(seq 1 120); do [[ -f "$OUT.shutdown_done" ]] && break; sleep 0.5; done
sed -n '/nvmlShutdown/,$p' "$OUT" | sed 's/^/    /'
say "    nvidia_fds now: $(nvcount $P)"
nvlist "$P"

echo
say "(e) can criu dump it now?"
try_dump "$P" "nvml-closed"

echo
say "cleanup"
echo exit > "$STEP"; sleep 2
kill -9 "$P" 2>/dev/null
echo
echo "=== VERDICT ==========================================================="
echo "If (c) failed and (e) succeeded, the blocking fd is an NVML handle and"
echo "nothing about CUDA context state is involved."
echo "======================================================================="
