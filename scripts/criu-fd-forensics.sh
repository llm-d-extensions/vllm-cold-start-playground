#!/usr/bin/env bash
# Why does `criu dump` still refuse a vLLM engine that cuda-checkpoint has
# already checkpointed?
#
# scripts/criu-matrix.sh arm `ckpt` establishes the happy path on a small CUDA
# process: after `cuda-checkpoint --action checkpoint` the process holds **0**
# /dev/nvidia* fds and criu dumps it. A real vLLM engine does not behave that
# way. Measured on Qwen3-32B TP=1 (reports/criu-raw/vllm-tp1-ckpt.txt):
#
#   cc checkpoint pid=3062 (VLLM::EngineCor): ok in 22.60s
#   gpu now: 0 MiB          <- device memory really did move to the host
#   rss=75.41 GiB           <- ...1:1, exactly as advertised
#   nvidia_fds=9            <- but nine fds survived
#   Error (criu/files-ext.c:94): Can't dump file 9 of that type [20666] (chr 195:255)
#
# 195:255 is /dev/nvidiactl. So the question this script answers is narrow and
# decisive: *which* fds are they, who opened them, and can they be closed
# without tearing down the engine? That determines whether the full-engine park
# is blocked by an implementation detail or by something structural.
#
# It costs one boot. No dump is attempted -- scripts/criu-vllm-probe.sh already
# has that number.
#
#   python3 criu-reaper.py bash criu-fd-forensics.sh [--tp 1]
set -uo pipefail

TP=1
MODEL=${MODEL:-Qwen/Qwen3-32B}
MAXLEN=${MAXLEN:-8192}
PORT=${PORT:-8500}
# NOT named CC: that is the C-compiler variable, and exporting it into the
# environment makes FlashInfer/Inductor JIT invoke this binary as a compiler.
# Observed 2026-09-04: `CC=/usr/local/bin/cuda-checkpoint` on the exec line gave
# "cuda-checkpoint: invalid option -- 'O'" and killed the 32B boot in
# profile_run, which looks nothing like the actual cause.
CUDA_CKPT=${CUDA_CKPT:-/tmp/cuda-checkpoint}
BOOT_TIMEOUT=${BOOT_TIMEOUT:-900}
LOG=/tmp/criu-fdf.log

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tp) TP="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

say(){ echo "[$(date -u +%H:%M:%S)] $*"; }
health(){ curl -s -o /dev/null -w '%{http_code}' -m 3 "localhost:$PORT/health" 2>/dev/null || echo 000; }
tree_pids(){ python3 - "$1" <<'PY'
import os, sys
root=int(sys.argv[1]); kids={}
for p in os.listdir("/proc"):
    if not p.isdigit(): continue
    try:
        st=open(f"/proc/{p}/stat").read(); ppid=int(st[st.rindex(")")+2:].split()[1])
    except Exception: continue
    kids.setdefault(ppid, []).append(int(p))
out,stack=[],[root]
while stack:
    p=stack.pop(0); out.append(p); stack.extend(sorted(kids.get(p,[])))
print(" ".join(map(str,out)))
PY
}

# The interesting part. For every /dev/nvidia* fd: the device number criu will
# complain about, and -- the point of the exercise -- which library opened it,
# inferred from the maps of the process.
nvfd_detail(){ # $1=pid
  python3 - "$1" <<'PY'
import os, sys, stat
pid = sys.argv[1]
d = f"/proc/{pid}/fd"
rows = []
try: fds = sorted(os.listdir(d), key=int)
except OSError as e:
    print(f"  (cannot read {d}: {e})"); raise SystemExit(0)
for f in fds:
    try: tgt = os.readlink(os.path.join(d, f))
    except OSError: continue
    if "nvidia" not in tgt: continue
    try:
        st = os.stat(os.path.join(d, f))
        dev = f"{os.major(st.st_rdev)}:{os.minor(st.st_rdev)}"
        kind = "chr" if stat.S_ISCHR(st.st_mode) else "?"
    except OSError:
        dev, kind = "?", "?"
    # fdinfo names the driver object behind some nvidia fds
    extra = ""
    try:
        for line in open(f"/proc/{pid}/fdinfo/{f}"):
            if line.startswith(("pos:", "flags:", "mnt_id:", "ino:")): continue
            extra += line.strip() + " "
    except OSError: pass
    rows.append((int(f), tgt, kind, dev, extra.strip()))
print(f"  {len(rows)} /dev/nvidia* fd(s):")
for f, tgt, kind, dev, extra in rows:
    print(f"    fd {f:<4} -> {tgt:<28} {kind} {dev}  {extra}")
# Which nvidia user-space libraries are mapped in? That is the best available
# attribution for who is holding them.
libs = set()
try:
    for line in open(f"/proc/{pid}/maps"):
        p = line.rstrip().split(" ", 5)[-1].strip()
        b = os.path.basename(p)
        if b.startswith(("libnvidia", "libcuda", "libnvml")):
            libs.add(b)
except OSError: pass
print("  nvidia libs mapped: " + (", ".join(sorted(libs)) or "none"))
PY
}

echo "======================================================================"
echo "criu-fd-forensics  tp=$TP  model=$MODEL"
echo "question: which /dev/nvidia* fds survive cuda-checkpoint, and who owns them"
echo "======================================================================"

pkill -9 -f "vllm serve" 2>/dev/null; pkill -9 -f "VLLM::" 2>/dev/null; sleep 3

say "booting"
setsid vllm serve --model "$MODEL" --max-model-len "$MAXLEN" \
  --tensor-parallel-size "$TP" --port "$PORT" </dev/null >"$LOG" 2>&1 &
ROOT=$!
for ((i=0; i<BOOT_TIMEOUT*10; i++)); do [[ "$(health)" == 200 ]] && break; sleep 0.1; done
[[ "$(health)" == 200 ]] || { say "boot failed"; tail -30 "$LOG"; exit 1; }
say "up. tree: $(for p in $(tree_pids $ROOT); do printf '%s(%s) ' "$p" "$(tr -d '\0' </proc/$p/comm)"; done)"

PIDS=$(tree_pids "$ROOT")

echo
say "=== BEFORE cuda-checkpoint ==="
for p in $PIDS; do
  n=$(ls -l "/proc/$p/fd" 2>/dev/null | grep -c '/dev/nvidia')
  [[ "$n" == 0 ]] && continue
  echo "  pid $p ($(tr -d '\0' </proc/$p/comm)) state=$($CUDA_CKPT --get-state --pid $p 2>&1)"
  nvfd_detail "$p"
done

echo
say "=== cuda-checkpoint everything ==="
for p in $PIDS; do
  ls -l "/proc/$p/fd" 2>/dev/null | grep -q '/dev/nvidia' || continue
  timeout -k 5 120 "$CUDA_CKPT" --action lock --pid "$p" --timeout 30000 >/dev/null 2>&1
  s=$SECONDS
  timeout -k 5 300 "$CUDA_CKPT" --action checkpoint --pid "$p" >/tmp/ccf 2>&1
  say "  pid $p checkpoint rc=$? in $((SECONDS-s))s $(head -1 /tmp/ccf)"
done

echo
say "=== AFTER cuda-checkpoint ==="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | sed 's/^/  gpu /'
for p in $PIDS; do
  [[ -d /proc/$p ]] || continue
  echo "  pid $p ($(tr -d '\0' </proc/$p/comm)) state=$($CUDA_CKPT --get-state --pid $p 2>&1) rss=$(awk '/VmRSS/{printf "%.2f GiB", $2/1048576}' /proc/$p/status 2>/dev/null)"
  nvfd_detail "$p"
done

echo
say "=== can the survivors be closed from outside? ==="
# If these are NVML handles rather than CUDA-context fds, then nothing short of
# the owning process calling nvmlShutdown() releases them -- and a dumper cannot
# do that on the process's behalf. Record whether vLLM exposes a hook for it.
say "  vLLM dev endpoints (VLLM_SERVER_DEV_MODE):"
for ep in /checkpoint_prepare /sleep /wake_up /is_sleeping /reset_prefix_cache; do
  code=$(curl -s -o /tmp/epout -w '%{http_code}' -m 5 -X POST "localhost:$PORT$ep" 2>/dev/null)
  echo "    POST $ep -> $code  $(head -c 120 /tmp/epout 2>/dev/null | tr -d '\n')"
done

echo
say "cleanup"
pkill -9 -f "vllm serve" 2>/dev/null; pkill -9 -f "VLLM::" 2>/dev/null; sleep 5
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | sed 's/^/  gpu /'
