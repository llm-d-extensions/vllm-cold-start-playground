#!/usr/bin/env bash
# Serialize a warmed vLLM engine with CRIU, restore it, and price the result
# against the cold boot it would replace.
#
# docs/sleep-mode.md#can-the-snapshot-be-serialized-and-reused-by-a-fresh-vllm
# priced this on paper and could not run it: criu is not in the vLLM image and
# the restricted pod has CapEff=0. manifests/pod-exec-criu.yaml removes both
# obstacles, and scripts/criu-matrix.sh establishes the mechanics on a small
# CUDA process first. This script is the real thing, on Qwen3-32B.
#
# Arms (--arm), all of which dump the whole vLLM process tree:
#
#   live    no cuda-checkpoint. The engine still holds /dev/nvidia* fds when
#           criu seizes it. The claim under test is that criu refuses, and the
#           exact refusal is the result -- it is what makes cuda-checkpoint a
#           prerequisite rather than an optimisation.
#   ckpt    cuda-checkpoint every CUDA-holding pid in the tree, then dump. The
#           full engine goes to host memory, so the image carries the weights,
#           the KV cache, the CUDA context and the graph pool.
#   sleep1  /sleep?level=1 (weights to a pinned host buffer, graphs kept) then
#           cuda-checkpoint the residual, then dump. Same order of image size,
#           different composition -- docs/sleep-mode.md measured 75.67 GiB RSS.
#   thin    /sleep?level=2 (weights discarded) then cuda-checkpoint, then dump.
#           The only variant whose arithmetic looked attractive: ~6.6 GiB. Note
#           the measured level-2 footgun -- /wake_up returns 200 and then
#           generates '!!!!!!' because nothing reloads the weights -- so this
#           arm is expected to restore *and be wrong*, which is the point.
#
# What gets measured, per arm:
#   boot            cold boot to health_200 (external poller, so it is directly
#                   comparable to the restore numbers below)
#   park            /sleep + cuda-checkpoint wall time, and GPU MiB after
#   dump            criu dump wall time, image bytes, largest image files
#   restore(cold)   criu restore after the image pages are evicted from the page
#                   cache with posix_fadvise(DONTNEED) -- a targeted eviction, so
#                   no other tenant on the node pays for a global drop_caches
#   restore(warm)   a second restore of the same image with the page cache hot
#   resume          cuda-checkpoint --action restore (+ /wake_up for sleep arms)
#   ready           restore start -> health_200, and -> a correct completion
#   correctness     the completion text compared byte-for-byte with the one the
#                   same engine produced before it was serialized
#
# Every criu and cuda-checkpoint call is wall-clock bounded. A hang is recorded
# and the tree is SIGKILLed; recovery is never left to the operator.
#
# Needs, inside the pod: criu on PATH (or $CRIU), /tmp/cuda-checkpoint (or $CUDA_CKPT),
# privileged container, and enough container memory to hold the whole device
# footprint on the host -- an 80 GiB GPU needs an ~80 GiB image.
#
# Run it under the subreaper, always -- see scripts/criu-reaper.py for why a
# zombie on pid 1 makes `criu restore` fail outright:
#
#   python3 criu-reaper.py bash criu-vllm-probe.sh --arm ckpt --tp 1
set -uo pipefail

ARM=ckpt
TP=1
MODEL=${MODEL:-Qwen/Qwen3-32B}
MAXLEN=${MAXLEN:-8192}
PORT=${PORT:-8300}
# NOT named CC: that is the C-compiler variable, and exporting it into the
# environment makes FlashInfer/Inductor JIT invoke this binary as a compiler.
# Observed 2026-09-04: `CC=/usr/local/bin/cuda-checkpoint` on the exec line gave
# "cuda-checkpoint: invalid option -- 'O'" and killed the 32B boot in
# profile_run, which looks nothing like the actual cause.
CUDA_CKPT=${CUDA_CKPT:-/tmp/cuda-checkpoint}
CRIU=${CRIU:-criu}
IMGROOT=${IMGROOT:-/criu/img}
BOOT_TIMEOUT=${BOOT_TIMEOUT:-900}
DUMP_TIMEOUT=${DUMP_TIMEOUT:-900}
RESTORE_TIMEOUT=${RESTORE_TIMEOUT:-900}
CCTIMEOUT=${CCTIMEOUT:-120}
EXTRA_VLLM=()
SKIP_WARM=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --arm) ARM="$2"; shift 2 ;;
    --tp) TP="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --max-model-len) MAXLEN="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --img-dir) IMGROOT="$2"; shift 2 ;;
    --skip-warm-restore) SKIP_WARM=1; shift ;;
    --) shift; EXTRA_VLLM=("$@"); break ;;
    -h|--help) sed -n '2,48p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
case "$ARM" in live|ckpt|sleep1|thin) ;; *) echo "--arm must be live|ckpt|sleep1|thin" >&2; exit 2 ;; esac

command -v "$CRIU" >/dev/null || { echo "no criu (set \$CRIU)" >&2; exit 2; }
[[ -x "$CUDA_CKPT" ]] || { echo "no cuda-checkpoint at $CUDA_CKPT" >&2; exit 2; }

IMG="$IMGROOT/vllm-tp$TP-$ARM"
LOG=/tmp/criu-vllm-tp$TP-$ARM.log
say()  { echo "[$(date -u +%H:%M:%S)] $*"; }
now()  { python3 -c 'import time;print(f"{time.monotonic():.3f}")'; }
el()   { python3 -c "print(f'{$2-$1:.2f}s')"; }
gpu()  { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '; }
gpusum(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1}END{print s" MiB"}'; }
rss()  { awk '/VmRSS/{printf "%.2f", $2/1048576}' "/proc/$1/status" 2>/dev/null || echo 0; }
bytes(){ du -sb "$1" 2>/dev/null | awk '{print $1}'; }
gib()  { python3 -c "print(f'{$1/1073741824:.2f} GiB')"; }

gen() { # deterministic completion; prints repr() of the text or ERR
  curl -s -m 180 "localhost:$PORT/v1/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"prompt\":\"The capital of France is\",\"max_tokens\":32,\"temperature\":0,\"seed\":1}" \
  | python3 -c 'import sys,json
try:
    d=json.load(sys.stdin)
    print(repr(d["choices"][0]["text"]) if "choices" in d else "ERR "+json.dumps(d)[:160])
except Exception as e:
    print("ERR "+str(e)[:80])'
}
health(){ curl -s -o /dev/null -w '%{http_code}' -m 3 "localhost:$PORT/health" 2>/dev/null || echo 000; }

# Wait for /health 200. Echoes the wall seconds from $1 (a monotonic stamp).
wait_health() { # $1=t0 $2=timeout
  local t0=$1 lim=$2 i
  for ((i=0; i<lim*10; i++)); do
    [[ "$(health)" == 200 ]] && { el "$t0" "$(now)"; return 0; }
    sleep 0.1
  done
  echo "TIMEOUT"; return 1
}

# Every pid in the tree, root first (vLLM's children are not all direct).
tree_pids() { # $1=root
  python3 - "$1" <<'PY'
import os, sys
root = int(sys.argv[1])
kids = {}
for p in os.listdir("/proc"):
    if not p.isdigit(): continue
    try:
        st = open(f"/proc/{p}/stat").read()
        ppid = int(st[st.rindex(")")+2:].split()[1])
    except Exception:
        continue
    kids.setdefault(ppid, []).append(int(p))
out, stack = [], [root]
while stack:
    p = stack.pop(0)
    out.append(p)
    stack.extend(sorted(kids.get(p, [])))
print(" ".join(str(p) for p in out))
PY
}

# Which of them hold the device? This is the whole question the `live` arm asks.
cuda_pids() { for p in $(tree_pids "$1"); do
    ls -l "/proc/$p/fd" 2>/dev/null | grep -q '/dev/nvidia' && echo "$p"
  done; }

ptitle(){ tr -d '\0' < "/proc/$1/comm" 2>/dev/null; }

ccdo() { # $1=action $2=pid [extra...]
  local action=$1 pid=$2; shift 2
  local s rc; s=$(now)
  timeout -k 5 "$CCTIMEOUT" "$CUDA_CKPT" --action "$action" --pid "$pid" "$@" >/tmp/ccout 2>&1
  rc=$?
  local d; d=$(el "$s" "$(now)")
  case $rc in
    0) say "    cc $action pid=$pid ($(ptitle "$pid")): ok in $d" ;;
    124|137) say "    cc $action pid=$pid ($(ptitle "$pid")): HUNG (>${CCTIMEOUT}s) -- wchan=$(cat /proc/$pid/wchan 2>/dev/null)" ;;
    *) say "    cc $action pid=$pid: rc=$rc in $d -- $(head -2 /tmp/ccout | tr '\n' ' ')" ;;
  esac
  return $rc
}

# Evict just this image's pages, so a restore reads from the device instead of
# the page cache. Targeted on purpose: a global drop_caches on a shared node
# would make every other tenant pay for our measurement.
evict() { python3 - "$1" <<'PY'
import os, sys
tot = 0
for dirpath, _, files in os.walk(sys.argv[1]):
    for f in files:
        p = os.path.join(dirpath, f)
        try:
            fd = os.open(p, os.O_RDONLY)
        except OSError:
            continue
        try:
            os.fsync(fd)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            tot += os.fstat(fd).st_size
        finally:
            os.close(fd)
print(f"evicted {tot/1073741824:.2f} GiB from the page cache")
PY
}

CRIU_FLAGS=(--tcp-close --file-locks --ext-unix-sk --manage-cgroups=ignore
            --force-irmap -v4)
# --link-remap is opt-OUT, because with it the image is single-use. It handles a
# deleted-but-still-open file by hard-linking it back into the filesystem at dump
# time; restore then consumes that link. Observed 2026-09-04 on the TP=1 spawn arm
# (reports/criu-raw/vllm-tp1-live-spawn.txt): the first restore succeeded, the
# second died in 0.19s with
#   Error (criu/files-reg.c:2258): Can't link dev/shm/link_remap.644 ->
#                                  dev/shm/sem.NCBGyB: No such file or directory
# on a POSIX semaphore vLLM's multiprocessing left unlinked under /dev/shm. That
# makes the image depend on state OUTSIDE itself, so a fresh replica -- the whole
# point -- would fail identically. Without the flag criu writes such files into
# the image as ghost files instead (they are 32 bytes; the default --ghost-limit
# is 1 MiB), which is what a portable image needs. Set LINK_REMAP=1 to restore the
# old behaviour and reproduce the failure.
[[ "${LINK_REMAP:-0}" == 1 ]] && CRIU_FLAGS+=(--link-remap)

do_dump() { # $1=root pid
  local root=$1 s rc
  rm -rf "$IMG"; mkdir -p "$IMG"
  s=$(now)
  timeout -k 10 "$DUMP_TIMEOUT" "$CRIU" dump -t "$root" -D "$IMG" \
      --log-file dump.log "${CRIU_FLAGS[@]}" >"$IMG/dump.stdout" 2>&1
  rc=$?
  DUMP_SECS=$(el "$s" "$(now)")
  DUMP_BYTES=$(bytes "$IMG")
  say "  criu dump rc=$rc in $DUMP_SECS  image=$(gib "${DUMP_BYTES:-0}")"
  if [[ $rc -eq 0 ]]; then
    say "  largest image files:"
    du -b "$IMG"/* 2>/dev/null | sort -rn | head -5 | \
      awk '{printf "      %8.2f GiB  %s\n", $1/1073741824, $2}'
  else
    say "  --- criu refused. error lines: ---"
    grep -aE "Error \(|error \(|unsupported|Can.t|can.t dump" "$IMG/dump.log" 2>/dev/null \
      | tail -14 | sed 's/^/      /'
  fi
  return $rc
}

do_restore() { # $1=label
  local label=$1 s rc
  # criu validates open regular files by size (--file-validation filesize is the
  # default). The restored engine appends to $LOG, so the second restore of the
  # same image would be refused with "bad size" unless the file is put back the
  # way the image remembers it.
  [[ -n "${LOG_SZ:-}" ]] && truncate -s "$LOG_SZ" "$LOG"
  s=$(now)
  timeout -k 10 "$RESTORE_TIMEOUT" "$CRIU" restore -D "$IMG" \
      --log-file "restore-$label.log" "${CRIU_FLAGS[@]}" -d >"$IMG/restore-$label.stdout" 2>&1
  rc=$?
  RESTORE_SECS=$(el "$s" "$(now)")
  RESTORE_T0=$s
  say "  criu restore ($label) rc=$rc in $RESTORE_SECS"
  [[ $rc -ne 0 ]] && grep -aE "Error \(|error \(" "$IMG/restore-$label.log" 2>/dev/null | tail -12 | sed 's/^/      /'
  return $rc
}

kill_vllm() {
  pkill -9 -f "vllm serve --model $MODEL" 2>/dev/null
  pkill -9 -f "VLLM::" 2>/dev/null
  sleep 5
  for _ in $(seq 1 30); do
    [[ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1}END{print s}')" -lt 1024 ]] && break
    sleep 2
  done
}

echo "======================================================================"
echo "criu-vllm-probe  arm=$ARM tp=$TP model=$MODEL max-model-len=$MAXLEN"
echo "criu $("$CRIU" --version 2>&1 | head -1 | tr -d '\n')   cc $("$CUDA_CKPT" 2>&1 | grep -o 'Version [0-9.]*' | head -1)"
echo "image dir: $IMG   log: $LOG"
echo "gpu at start: $(gpu)"
echo "container mem limit: $(awk '{printf "%.0f GiB\n", $1/1073741824}' /sys/fs/cgroup/memory.max 2>/dev/null || echo unknown)"
echo "======================================================================"

kill_vllm

# ---------------------------------------------------------------------------
# 1. boot
# ---------------------------------------------------------------------------
say "step 1: cold boot (this is the number the restore has to beat)"
export VLLM_SERVER_DEV_MODE=1
SLEEP_FLAG=()
[[ "$ARM" == sleep1 || "$ARM" == thin ]] && SLEEP_FLAG=(--enable-sleep-mode)
T_BOOT=$(now)
setsid vllm serve --model "$MODEL" --max-model-len "$MAXLEN" \
  --tensor-parallel-size "$TP" --port "$PORT" \
  "${SLEEP_FLAG[@]}" "${EXTRA_VLLM[@]+${EXTRA_VLLM[@]}}" </dev/null >"$LOG" 2>&1 &
ROOT=$!
BOOT_SECS=$(wait_health "$T_BOOT" "$BOOT_TIMEOUT")
if [[ "$BOOT_SECS" == TIMEOUT ]]; then
  say "  boot TIMEOUT after ${BOOT_TIMEOUT}s -- last log lines:"; tail -20 "$LOG"; kill_vllm; exit 1
fi
say "  health_200 at $BOOT_SECS   gpu=$(gpu)"
BASE=$(gen); say "  baseline completion: $BASE"
say "  tree: $(for p in $(tree_pids "$ROOT"); do printf '%s(%s) ' "$p" "$(ptitle "$p")"; done)"
CUDAP=$(cuda_pids "$ROOT" | tr '\n' ' ')
say "  pids holding /dev/nvidia*: ${CUDAP:-none}"
RSS_BEFORE=$(for p in $(tree_pids "$ROOT"); do printf '%s\n' "$(rss "$p")"; done | awk '{s+=$1}END{printf "%.2f", s}')
say "  tree RSS before park: ${RSS_BEFORE} GiB"

# ---------------------------------------------------------------------------
# 2. park
# ---------------------------------------------------------------------------
PARK_SECS=0
case "$ARM" in
  live) say "step 2: no park -- dumping a live CUDA process on purpose" ;;
  ckpt)
    say "step 2: cuda-checkpoint every CUDA-holding pid"
    s=$(now)
    for p in $CUDAP; do ccdo lock "$p" --timeout 60000 && ccdo checkpoint "$p"; done
    PARK_SECS=$(el "$s" "$(now)") ;;
  sleep1|thin)
    LVL=1; [[ "$ARM" == thin ]] && LVL=2
    say "step 2: /sleep?level=$LVL then cuda-checkpoint"
    s=$(now)
    curl -s -o /dev/null -X POST "localhost:$PORT/sleep?level=$LVL" -m 900
    say "    after sleep level $LVL: gpu=$(gpu)"
    for p in $CUDAP; do ccdo lock "$p" --timeout 60000 && ccdo checkpoint "$p"; done
    PARK_SECS=$(el "$s" "$(now)") ;;
esac
[[ "$ARM" != live ]] && say "  park took $PARK_SECS  gpu now: $(gpu)  (sum $(gpusum))"
RSS_PARKED=$(for p in $(tree_pids "$ROOT"); do printf '%s\n' "$(rss "$p")"; done | awk '{s+=$1}END{printf "%.2f", s}')
say "  tree RSS parked: ${RSS_PARKED} GiB   (this is the floor on image size)"
for p in $(tree_pids "$ROOT"); do
  say "    pid=$p ($(ptitle "$p")) rss=$(rss "$p") GiB nvidia_fds=$(ls -l /proc/$p/fd 2>/dev/null | grep -c nvidia) cc_state=$($CUDA_CKPT --get-state --pid "$p" 2>&1 | head -1)"
done

# ---------------------------------------------------------------------------
# 3. dump
# ---------------------------------------------------------------------------
say "step 3: criu dump the tree rooted at $ROOT"
if ! do_dump "$ROOT"; then
  say "  dump failed -- that is this arm's result. cleaning up."
  say "  tree still alive? $(kill -0 "$ROOT" 2>/dev/null && echo yes || echo no)   gpu=$(gpu)"
  # If criu left the tree stopped, the SIGKILL below still reclaims the GPU.
  kill_vllm
  say "  gpu after cleanup: $(gpu)"
  echo
  echo "=== SUMMARY arm=$ARM tp=$TP ==================================="
  echo "boot to health_200 : $BOOT_SECS"
  echo "park               : ${PARK_SECS}"
  echo "criu dump          : FAILED in $DUMP_SECS (see $IMG/dump.log)"
  exit 3
fi
LOG_SZ=$(stat -c %s "$LOG" 2>/dev/null || echo 0)
# criu killed the tree from the root down; the orphaned grandchildren are only
# reaped because this script runs under scripts/criu-reaper.py. Without it they
# sit as zombies on pid 1 (`sleep infinity`) and hold the pids criu must reuse.
wait "$ROOT" 2>/dev/null
say "  tree gone? $(kill -0 "$ROOT" 2>/dev/null && echo NO && echo "  (criu left it alive)" || echo yes)   gpu=$(gpu)"
DUMP_MB_S=$(python3 -c "print(f'{${DUMP_BYTES}/1048576/$(python3 -c "print(float('${DUMP_SECS%s}'))"):.0f} MB/s')" 2>/dev/null || echo n/a)
say "  dump throughput: $DUMP_MB_S"

# ---------------------------------------------------------------------------
# 4. restore, page cache cold
# ---------------------------------------------------------------------------
say "step 4: restore with the image evicted from the page cache"
say "  $(evict "$IMG")"
if ! do_restore cold; then say "  cold restore failed"; kill_vllm; exit 4; fi
COLD_RESTORE=$RESTORE_SECS
NEWROOT=$(pgrep -f "vllm serve --model $MODEL" | head -1)
say "  restored root pid=${NEWROOT:-none} (was $ROOT)   gpu=$(gpu)"
CUDAP2=$(for p in $(tree_pids "${NEWROOT:-$ROOT}"); do echo "$p"; done | tr '\n' ' ')

RESUME_SECS=0
if [[ "$ARM" != live ]]; then
  say "step 5: cuda-checkpoint --action restore"
  s=$(now)
  for p in $CUDAP; do
    kill -0 "$p" 2>/dev/null || continue
    [[ "$($CUDA_CKPT --get-state --pid "$p" 2>&1)" == *checkpointed* ]] || continue
    ccdo restore "$p" && ccdo unlock "$p"
  done
  RESUME_SECS=$(el "$s" "$(now)")
  say "  cuda restore took $RESUME_SECS  gpu=$(gpu)"
fi

if [[ "$ARM" == sleep1 || "$ARM" == thin ]]; then
  say "step 5b: /wake_up"
  s=$(now); curl -s -o /dev/null -X POST "localhost:$PORT/wake_up" -m 900
  say "  wake_up $(el "$s" "$(now)")  gpu=$(gpu)"
fi

say "step 6: readiness and correctness after restore"
READY_SECS=$(wait_health "$RESTORE_T0" 300)
say "  restore start -> health_200: $READY_SECS"
AFTER=$(gen)
GEN_SECS=$(el "$RESTORE_T0" "$(now)")
say "  restore start -> completion: $GEN_SECS"
say "  completion after restore: $AFTER"
if [[ "$AFTER" == "$BASE" ]]; then VERDICT="IDENTICAL"; else VERDICT="DIFFERS"; fi
say "  correctness: $VERDICT"

# ---------------------------------------------------------------------------
# 5. restore again, page cache warm
# ---------------------------------------------------------------------------
WARM_RESTORE=n/a
if [[ "$SKIP_WARM" == 0 ]]; then
  say "step 7: second restore of the same image, page cache warm"
  kill_vllm
  if do_restore warm; then
    WARM_RESTORE=$RESTORE_SECS
    for p in $CUDAP; do
      kill -0 "$p" 2>/dev/null || continue
      [[ "$($CUDA_CKPT --get-state --pid "$p" 2>&1)" == *checkpointed* ]] || continue
      ccdo restore "$p" && ccdo unlock "$p"
    done
    [[ "$ARM" == sleep1 || "$ARM" == thin ]] && curl -s -o /dev/null -X POST "localhost:$PORT/wake_up" -m 900
    W_READY=$(wait_health "$RESTORE_T0" 300)
    W_OUT=$(gen)
    say "  warm: restore=$WARM_RESTORE ready=$W_READY correct=$([[ "$W_OUT" == "$BASE" ]] && echo IDENTICAL || echo DIFFERS)"
  fi
fi

kill_vllm

echo
echo "=== SUMMARY arm=$ARM tp=$TP ============================================"
printf 'boot to health_200        : %s\n' "$BOOT_SECS"
printf 'baseline completion       : %s\n' "$BASE"
printf 'park (sleep+cuda-ckpt)    : %s   gpu after: %s\n' "${PARK_SECS}" "$(gpusum)"
printf 'tree RSS running / parked : %s / %s GiB\n' "$RSS_BEFORE" "$RSS_PARKED"
printf 'criu dump                 : %s for %s (%s)\n' "$DUMP_SECS" "$(gib "$DUMP_BYTES")" "$DUMP_MB_S"
printf 'criu restore (cache cold) : %s\n' "$COLD_RESTORE"
printf 'criu restore (cache warm) : %s\n' "$WARM_RESTORE"
printf 'cuda-checkpoint restore   : %s\n' "$RESUME_SECS"
printf 'restore -> health_200     : %s\n' "$READY_SECS"
printf 'restore -> completion     : %s\n' "$GEN_SECS"
printf 'correctness               : %s\n' "$VERDICT"
printf 'after-restore completion  : %s\n' "$AFTER"
echo "======================================================================"
