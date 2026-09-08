#!/usr/bin/env bash
# Can CRIU serialize a CUDA process -- and what does cuda-checkpoint change?
#
# scripts/cuda-checkpoint-dumpability.sh established the *shape* of the answer:
# a checkpointed process holds 0 `/dev/nvidia*` fds and no device mappings, so
# "nothing GPU-specific is left for a dumper to understand". It could not test
# the dumper, because criu is not in the vLLM image and the restricted pod runs
# with CapEff=0. This script is that test, on a deliberately small CUDA process
# (512 MiB tensors, one captured graph) so each arm costs seconds rather than a
# 32B boot.
#
# Three arms, run in sequence, each on a fresh child:
#
#   live      criu dump while CUDA is live. The claim under test is that this
#             fails, and *why* it fails is the result: /dev/nvidia* fds are
#             unsupported file types for criu, so no amount of flags helps.
#   ckpt      cuda-checkpoint --action checkpoint first, then criu dump,
#             criu restore, cuda-checkpoint --action restore, then replay the
#             captured CUDA graph and check the value. This is the composition
#             docs/sleep-mode.md priced but could not run.
#   noc       control: same python, torch imported, CUDA never touched. Prices
#             the interpreter-image idea on its own -- an image of a process
#             that has paid `import torch` and nothing else needs no
#             cuda-checkpoint at all.
#   plugin    the arm that only exists once /usr/lib/criu/cuda_plugin.so is
#             installed: one `criu dump` with **no manual cuda-checkpoint call
#             at all**, because criu's own CUDA plugin shells out to
#             cuda-checkpoint from its dump/restore hooks. Full round trip
#             (dump -> restore -> replay the captured graph) so the verdict is
#             correctness, not just an exit code. This is what `criu dump`
#             *means* on a CUDA process when the build is complete.
#   noplug    the control for `plugin`, and the real "without cuda-checkpoint"
#             arm: same criu, same plugin, but cuda-checkpoint removed from
#             $PATH for the duration. cuda_plugin.c resolves the binary by name
#             at dump time, so hiding it disables the plugin without touching
#             the install -- isolating the *binary's* contribution from the
#             plugin's. Restores $PATH even if the dump wedges.
#
# Every criu and cuda-checkpoint call is wall-clock bounded: a hang is a result
# to record, not a reason to wedge the pod.
#
# Needs, inside the pod: criu on PATH (or $CRIU), /tmp/cuda-checkpoint (or $CUDA_CKPT),
# and a privileged container -- see manifests/pod-exec-criu.yaml.
#
# Run under the subreaper (scripts/criu-reaper.py) so a dumped child does not
# linger as a zombie on pid 1 and block the restore of its own pid:
#
#   python3 criu-reaper.py bash criu-matrix.sh [--arms "live ckpt noc"]
set -uo pipefail

ARMS="live ckpt noc"
IMGROOT=${IMGROOT:-/criu/img}
# NOT named CC: that is the C-compiler variable, and exporting it into the
# environment makes FlashInfer/Inductor JIT invoke this binary as a compiler.
# Observed 2026-09-04: `CC=/usr/local/bin/cuda-checkpoint` on the exec line gave
# "cuda-checkpoint: invalid option -- 'O'" and killed the 32B boot in
# profile_run, which looks nothing like the actual cause.
CUDA_CKPT=${CUDA_CKPT:-/tmp/cuda-checkpoint}
CRIU=${CRIU:-criu}
TMO=${TMO:-180}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --arms) ARMS="$2"; shift 2 ;;
    --img-dir) IMGROOT="$2"; shift 2 ;;
    -h|--help) sed -n '2,32p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

command -v "$CRIU" >/dev/null || { echo "no criu (set \$CRIU)"; exit 2; }
[[ -x "$CUDA_CKPT" ]] || { echo "no cuda-checkpoint at $CUDA_CKPT"; exit 2; }

say()  { echo "[$(date -u +%H:%M:%S)] $*"; }
now()  { python3 -c 'import time;print(f"{time.monotonic():.3f}")'; }
el()   { python3 -c "print(f'{$2-$1:.2f}s')"; }
gpu()  { nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' '; }
rss()  { awk '/VmRSS/{printf "%.2f GiB", $2/1048576}' "/proc/$1/status" 2>/dev/null || echo n/a; }
duh()  { du -sb "$1" 2>/dev/null | awk '{printf "%.2f GiB (%d B)", $1/1073741824, $1}'; }

# ---------------------------------------------------------------------------
# The child. Writes its pid to $RDY, then blocks on $GO. `cuda=0` skips every
# CUDA call so the same file is the control arm -- same imports, same RSS floor,
# no device state.
# ---------------------------------------------------------------------------
cat > /tmp/criu_child.py <<'PY'
import os, sys, time, torch
cuda = os.environ.get("CHILD_CUDA", "1") == "1"
rdy, go, out = sys.argv[1], sys.argv[2], sys.argv[3]
if cuda:
    N = 512 * 1024 * 1024 // 4
    a = torch.ones(N, device="cuda"); res = torch.zeros(N, device="cuda")
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        res.copy_(a * 2 + 1)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        res.copy_(a * 2 + 1)
    g.replay(); torch.cuda.synchronize()
    pre = res[0].item()
else:
    g = None
    pre = float(torch.ones(4).sum())  # touch torch, stay on the host
with open(out, "a") as f:
    f.write(f"pre={pre} cuda={cuda} pid={os.getpid()} torch={torch.__version__}\n")
open(rdy, "w").write(str(os.getpid()))
# The wait must survive a criu dump/restore, so poll a path rather than hold a
# pipe or a signal handler: a restored process wakes up inside this loop.
while not os.path.exists(go):
    time.sleep(0.2)
if cuda:
    g.replay(); torch.cuda.synchronize()
    post = res[0].item()
else:
    post = float(torch.ones(4).sum())
with open(out, "a") as f:
    f.write(f"post={post} (expect {pre})\n")
    f.write("VERDICT: " + ("IDENTICAL" if post == pre else f"DIFFERS {post} != {pre}") + "\n")
PY

start_child() { # $1=cuda(0|1) $2=tag  -> echoes pid
  local cuda=$1 tag=$2
  rm -f "/tmp/c_$tag.rdy" "/tmp/c_$tag.go" "/tmp/c_$tag.out"
  # setsid: criu refuses a tree whose session leader is outside it, and
  # --shell-job would otherwise be required to paper over the exec'ing shell.
  CHILD_CUDA=$cuda PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    setsid python3 /tmp/criu_child.py "/tmp/c_$tag.rdy" "/tmp/c_$tag.go" "/tmp/c_$tag.out" \
    </dev/null >"/tmp/c_$tag.log" 2>&1 &
  local i
  for i in $(seq 1 400); do [[ -s "/tmp/c_$tag.rdy" ]] && break; sleep 0.5; done
  cat "/tmp/c_$tag.rdy" 2>/dev/null
}

# criu dump, bounded, never fatal: the failure text is the finding.
try_dump() { # $1=pid $2=imgdir $3=extra...
  local pid=$1 img=$2; shift 2
  rm -rf "$img"; mkdir -p "$img"
  local s rc; s=$(now)
  timeout -k 5 "$TMO" "$CRIU" dump -t "$pid" -D "$img" -v4 --log-file dump.log \
      --tcp-close --file-locks --ext-unix-sk "$@" >"$img/dump.stdout" 2>&1
  rc=$?
  DUMP_SECS=$(el "$s" "$(now)")
  say "    criu dump rc=$rc in $DUMP_SECS  image=$(duh "$img")"
  if [[ $rc -ne 0 ]]; then
    say "    --- why it failed (criu log, error lines) ---"
    grep -E "Error|error|Warn.*unsupported|unsupported|can't|Can't" "$img/dump.log" 2>/dev/null \
      | grep -viE "Warn .*(cgroup|Set|no-fanout)" | tail -12 | sed 's/^/      /'
  fi
  return $rc
}

try_restore() { # $1=imgdir
  local img=$1 s rc; s=$(now)
  timeout -k 5 "$TMO" "$CRIU" restore -D "$img" -v4 --log-file restore.log -d \
      --tcp-close --file-locks --ext-unix-sk >"$img/restore.stdout" 2>&1
  rc=$?
  RESTORE_SECS=$(el "$s" "$(now)")
  say "    criu restore rc=$rc in $RESTORE_SECS"
  [[ $rc -ne 0 ]] && grep -E "Error|error" "$img/restore.log" 2>/dev/null | tail -10 | sed 's/^/      /'
  return $rc
}

ccdo() { # $1=action $2=pid [extra...]
  local action=$1 pid=$2; shift 2
  local s rc; s=$(now)
  timeout -k 5 90 "$CUDA_CKPT" --action "$action" --pid "$pid" "$@" >/tmp/ccout 2>&1
  rc=$?
  local d; d=$(el "$s" "$(now)")
  if [[ $rc -eq 0 ]]; then say "    cuda-checkpoint $action: ok in $d"
  elif [[ $rc -eq 124 || $rc -eq 137 ]]; then say "    cuda-checkpoint $action: HUNG (>90s)"
  else say "    cuda-checkpoint $action: rc=$rc in $d -- $(head -2 /tmp/ccout | tr '\n' ' ')"; fi
  return $rc
}

kill_tree() { pkill -9 -f criu_child.py 2>/dev/null; sleep 2; }

echo "=================================================================="
echo "CRIU x cuda-checkpoint matrix"
echo "criu    : $("$CRIU" --version 2>&1 | head -1)"
echo "cc      : $("$CUDA_CKPT" 2>&1 | grep -o 'Version [0-9.]*' | head -1)"
echo "torch   : $(python3 -c 'import torch;print(torch.__version__)' 2>/dev/null)"
echo "kernel  : $(uname -r)"
echo "caps    : $(grep CapEff /proc/self/status | awk '{print $2}')  seccomp=$(grep Seccomp: /proc/self/status | awk '{print $2}')"
echo "gpu     : $(gpu)"
echo "arms    : $ARMS"
echo "=================================================================="

for arm in $ARMS; do
  echo
  say "################ arm: $arm"
  IMG="$IMGROOT/$arm"
  case "$arm" in
    plugin|noplug)
      # Hide cuda-checkpoint from the plugin for `noplug` by pointing PATH at a
      # directory that does not contain it. The plugin looks the binary up by
      # name, so this is the whole mechanism -- no reinstall, no rebuild.
      SAVED_PATH="$PATH"
      if [[ "$arm" == noplug ]]; then
        mkdir -p /tmp/nocc
        export PATH=/tmp/nocc:/usr/sbin:/usr/bin:/bin
        say "  cuda-checkpoint hidden: command -v -> '$(command -v cuda-checkpoint || echo NONE)'"
      else
        say "  cuda-checkpoint visible: $(command -v cuda-checkpoint || echo NONE)"
      fi
      say "  plugin dir: $(ls /usr/lib/criu/ 2>/dev/null | tr '\n' ' ')"
      P=$(start_child 1 "$arm")
      if [[ -z "$P" ]]; then say "  child failed to start"; sed -n 1,20p "/tmp/c_$arm.log"; export PATH="$SAVED_PATH"; continue; fi
      say "  child pid=$P  rss=$(rss "$P")  gpu=$(gpu)"
      say "  /dev/nvidia* fds=$(ls -l /proc/$P/fd 2>/dev/null | grep -c nvidia)  nvidia maps=$(grep -c nvidia /proc/$P/maps)"
      say "  cuda state=$(PATH=$SAVED_PATH $CUDA_CKPT --get-state --pid "$P" 2>&1)"
      say "  NOT calling cuda-checkpoint by hand -- criu owns that decision now"
      if try_dump "$P" "$IMG"; then
        say "  process gone after dump? $(kill -0 "$P" 2>/dev/null && echo NO-still-alive || echo yes)  gpu=$(gpu)"
        # Did the plugin actually run? Its hooks log through criu's own log.
        say "  plugin trace in dump.log:"
        grep -aiE "cuda|plugin" "$IMG/dump.log" 2>/dev/null | tail -8 | sed 's/^/      /'
        if try_restore "$IMG"; then
          NP=$(cat "/tmp/c_$arm.rdy")
          say "  restored pid=$NP  state=$(PATH=$SAVED_PATH $CUDA_CKPT --get-state --pid "$NP" 2>&1)  gpu=$(gpu)  rss=$(rss "$NP")"
          say "  plugin trace in restore.log:"
          grep -aiE "cuda|plugin" "$IMG/restore.log" 2>/dev/null | tail -8 | sed 's/^/      /'
          # No manual `cc restore` either: if the plugin did its job the process
          # is already able to touch the device. Replaying the graph proves it.
          touch "/tmp/c_$arm.go"
          for _ in $(seq 1 60); do grep -q VERDICT "/tmp/c_$arm.out" 2>/dev/null && break; sleep 1; done
          say "  child output:"; sed 's/^/      /' "/tmp/c_$arm.out"
        fi
      fi
      export PATH="$SAVED_PATH"
      kill_tree
      ;;
    live)
      P=$(start_child 1 live)
      [[ -z "$P" ]] && { say "  child failed to start"; sed -n 1,20p /tmp/c_live.log; continue; }
      say "  child pid=$P  rss=$(rss "$P")  gpu=$(gpu)"
      say "  /dev/nvidia* fds=$(ls -l /proc/$P/fd 2>/dev/null | grep -c nvidia)  nvidia maps=$(grep -c nvidia /proc/$P/maps)"
      say "  cuda state=$($CUDA_CKPT --get-state --pid "$P" 2>&1)"
      try_dump "$P" "$IMG"
      say "  child alive after dump attempt? $(kill -0 "$P" 2>/dev/null && echo yes || echo no)"
      kill_tree
      ;;
    ckpt)
      P=$(start_child 1 ckpt)
      [[ -z "$P" ]] && { say "  child failed to start"; sed -n 1,20p /tmp/c_ckpt.log; continue; }
      say "  child pid=$P  rss=$(rss "$P")  gpu=$(gpu)"
      ccdo lock "$P" --timeout 30000 && ccdo checkpoint "$P"
      say "  after checkpoint: state=$($CUDA_CKPT --get-state --pid "$P" 2>&1) rss=$(rss "$P") gpu=$(gpu)"
      say "  /dev/nvidia* fds=$(ls -l /proc/$P/fd 2>/dev/null | grep -c nvidia)  restore-tid=$($CUDA_CKPT --get-restore-tid --pid "$P" 2>&1)"
      if try_dump "$P" "$IMG"; then
        say "  process gone after dump? $(kill -0 "$P" 2>/dev/null && echo NO-still-alive || echo yes)  gpu=$(gpu)"
        if try_restore "$IMG"; then
          NP=$(cat /tmp/c_ckpt.rdy)
          say "  restored pid=$NP (same pid? $([[ "$NP" == "$P" ]] && echo yes || echo no))  state=$($CUDA_CKPT --get-state --pid "$NP" 2>&1)"
          ccdo restore "$NP" && ccdo unlock "$NP"
          say "  after cuda restore: gpu=$(gpu) rss=$(rss "$NP")"
          touch /tmp/c_ckpt.go
          for _ in $(seq 1 60); do grep -q VERDICT /tmp/c_ckpt.out 2>/dev/null && break; sleep 1; done
          say "  child output:"; sed 's/^/      /' /tmp/c_ckpt.out
        fi
      fi
      kill_tree
      ;;
    noc)
      P=$(start_child 0 noc)
      [[ -z "$P" ]] && { say "  child failed to start"; sed -n 1,20p /tmp/c_noc.log; continue; }
      say "  child pid=$P  rss=$(rss "$P")  gpu=$(gpu)  (torch imported, CUDA untouched)"
      if try_dump "$P" "$IMG"; then
        if try_restore "$IMG"; then
          NP=$(cat /tmp/c_noc.rdy)
          touch /tmp/c_noc.go
          for _ in $(seq 1 60); do grep -q VERDICT /tmp/c_noc.out 2>/dev/null && break; sleep 1; done
          say "  child output:"; sed 's/^/      /' /tmp/c_noc.out
        fi
      fi
      kill_tree
      ;;
    *) say "  unknown arm"; ;;
  esac
done

echo
say "done. gpu=$(gpu)  images under $IMGROOT:"
du -sh "$IMGROOT"/* 2>/dev/null | sed 's/^/  /'
