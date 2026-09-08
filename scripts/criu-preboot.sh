#!/usr/bin/env bash
# The CRIU arm that is actually about *cold start*: restore an interpreter that
# has already paid the imports, then let it serve.
#
# Everything else in this directory that touches cuda-checkpoint parks a warmed
# engine and brings it back -- warm standby. This script asks a different
# question. `python imports` is the largest phase in the boot at both TP=1 and
# TP=2 (docs/pending-experiments.md: 24.7s / 46.9% of a TP=2 boot *after* the
# pycache lever), and an interpreter that has imported torch and vLLM but never
# touched the GPU holds **no /dev/nvidia fds at all** -- so criu can serialize it
# with no cuda-checkpoint, no driver coupling, and a small image.
#
# Three steps:
#
#   --probe   report, per import stage, how long it took and whether it opened a
#             /dev/nvidia* fd. This decides how far the preload can go: the last
#             stage before the driver appears is the checkpoint point.
#   --build   run the preload up to --stage, criu dump it, keep the image.
#             This is off the critical path -- it happens once per (image, model,
#             flag set), like a container build.
#   --measure restore the image and hand it `vllm serve` argv; time restore ->
#             health_200 -> first completion. Compare with --arm cold, which is
#             the same script booting vLLM the ordinary way so the poller, the
#             env and the node are identical.
#
# What makes this different from the forkserver lever (coldstart/cs_forkserver.py)
# is that the preload does not have to happen in *this* pod's lifetime: the image
# is a file. A replica that has never run vLLM before can read it.
#
# Run every step under the subreaper (scripts/criu-reaper.py): a restored vLLM
# forks EngineCore, and when the tree is killed those grandchildren must be
# reaped or the pids criu needs are still held by zombies.
#
#   python3 criu-reaper.py bash criu-preboot.sh --probe
#   bash criu-preboot.sh --build --stage vllm
#   bash criu-preboot.sh --measure --arm restore --tp 1
#   bash criu-preboot.sh --measure --arm cold    --tp 1
set -uo pipefail

MODE=""
ARM=restore
STAGE=${STAGE:-vllm}
TP=1
MODEL=${MODEL:-Qwen/Qwen3-32B}
MAXLEN=${MAXLEN:-8192}
PORT=${PORT:-8400}
CRIU=${CRIU:-criu}
IMGROOT=${IMGROOT:-/criu/img}
DUMP_TIMEOUT=${DUMP_TIMEOUT:-300}
RESTORE_TIMEOUT=${RESTORE_TIMEOUT:-300}
BOOT_TIMEOUT=${BOOT_TIMEOUT:-900}
EVICT=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --probe) MODE=probe; shift ;;
    --build) MODE=build; shift ;;
    --measure) MODE=measure; shift ;;
    --arm) ARM="$2"; shift 2 ;;
    --stage) STAGE="$2"; shift 2 ;;
    --tp) TP="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --max-model-len) MAXLEN="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --no-evict) EVICT=0; shift ;;
    -h|--help) sed -n '2,36p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$MODE" ]] || { echo "give one of --probe / --build / --measure" >&2; exit 2; }

IMG="$IMGROOT/preboot-$STAGE"
RDY=/tmp/preboot.ready
GO=/tmp/preboot.go
ARGS=/tmp/preboot.argv
OUT=/tmp/preboot.out
LOG=/tmp/preboot.log

say() { echo "[$(date -u +%H:%M:%S)] $*"; }
now() { python3 -c 'import time;print(f"{time.monotonic():.3f}")'; }
el()  { python3 -c "print(f'{$2-$1:.2f}s')"; }
gpu() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '; }
gib() { python3 -c "print(f'{$1/1073741824:.2f} GiB')"; }
bytes(){ du -sb "$1" 2>/dev/null | awk '{print $1}'; }
health(){ curl -s -o /dev/null -w '%{http_code}' -m 3 "localhost:$PORT/health" 2>/dev/null || echo 000; }
gen() { curl -s -m 180 "localhost:$PORT/v1/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"prompt\":\"The capital of France is\",\"max_tokens\":32,\"temperature\":0,\"seed\":1}" \
  | python3 -c 'import sys,json
try:
    d=json.load(sys.stdin)
    print(repr(d["choices"][0]["text"]) if "choices" in d else "ERR "+json.dumps(d)[:160])
except Exception as e:
    print("ERR "+str(e)[:80])'; }
wait_health() { local t0=$1 lim=$2 i
  for ((i=0; i<lim*10; i++)); do
    [[ "$(health)" == 200 ]] && { el "$t0" "$(now)"; return 0; }
    sleep 0.1
  done; echo TIMEOUT; return 1; }
kill_all() { pkill -9 -f preboot_child.py 2>/dev/null; pkill -9 -f "vllm serve --model $MODEL" 2>/dev/null
  pkill -9 -f "VLLM::" 2>/dev/null; sleep 4
  for _ in $(seq 1 30); do
    [[ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1}END{print s}')" -lt 1024 ]] && break
    sleep 2
  done; }
evict() { python3 - "$1" <<'PY'
import os, sys
tot = 0
for dirpath, _, files in os.walk(sys.argv[1]):
    for f in files:
        p = os.path.join(dirpath, f)
        try: fd = os.open(p, os.O_RDONLY)
        except OSError: continue
        try:
            os.fsync(fd); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            tot += os.fstat(fd).st_size
        finally: os.close(fd)
print(f"evicted {tot/1073741824:.2f} GiB")
PY
}

# ---------------------------------------------------------------------------
# The preload child. Imports in named stages, reports fds per stage, then waits
# on a file so it survives dump/restore, then becomes `vllm serve`.
#
# It must never initialise CUDA: a process with a /dev/nvidia* fd cannot be
# dumped (scripts/criu-matrix.sh arm `live`), and a CUDA-initialised parent also
# forces vLLM to spawn its EngineCore instead of forking it
# (system_utils.py _maybe_force_spawn), which would throw away the very imports
# this image exists to carry.
# ---------------------------------------------------------------------------
cat > /tmp/preboot_child.py <<'PY'
import json, os, sys, time

STAGES = ("torch", "vllm", "server")


def nvfds():
    n = 0
    d = "/proc/self/fd"
    for f in os.listdir(d):
        try:
            if "/dev/nvidia" in os.readlink(os.path.join(d, f)):
                n += 1
        except OSError:
            pass
    return n


def rss_gib():
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1048576.0
    return 0.0


_log = None


def note(msg):
    if _log is not None:
        _log.write(msg + "\n")
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


def run():
    global _log
    stage_want = os.environ.get("PREBOOT_STAGE", "vllm")
    probe_only = os.environ.get("PREBOOT_PROBE", "0") == "1"
    rdy, go, argsf, outf = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    _log = open(outf, "a", buffering=1)

    note(f"start pid={os.getpid()} nvidia_fds={nvfds()} rss={rss_gib():.2f}GiB")
    t0 = time.monotonic()
    for stage in STAGES:
        s = time.monotonic()
        if stage == "torch":
            import torch  # noqa: F401
        elif stage == "vllm":
            import vllm  # noqa: F401
            from vllm import envs  # noqa: F401
        elif stage == "server":
            # The module graph `vllm serve` itself pulls in. An earlier version
            # of this comment claimed "minus anything that would touch the
            # device"; that is false, and the falseness is the whole point of
            # this stage. Measured 2026-09-04, importing these four in order:
            #   vllm.entrypoints.openai.api_server      +3.50s  nvidia_fds 0->11
            #   vllm.entrypoints.cli.main               +0.00s  nvidia_fds 11
            #   vllm.engine.arg_utils.AsyncEngineArgs   +0.00s  nvidia_fds 11
            #   uvicorn, fastapi                        +0.00s  nvidia_fds 11
            # The first import is the entire cost and the entire device contact
            # (platform resolution reaches CUDA); the other three are already
            # in via it. So the 3.5s worth prefetching and the fd set that
            # forces criu to need cuda_plugin.so are the same import. There is
            # no cheaper subset -- see the report section on --stage server.
            from vllm.entrypoints.openai import api_server  # noqa: F401
            from vllm.entrypoints.cli import main as _cli  # noqa: F401
            from vllm.engine.arg_utils import AsyncEngineArgs  # noqa: F401
            import uvicorn, fastapi  # noqa: F401
        note(f"stage={stage} took={time.monotonic()-s:.2f}s "
             f"cumulative={time.monotonic()-t0:.2f}s nvidia_fds={nvfds()} "
             f"rss={rss_gib():.2f}GiB threads={len(os.listdir('/proc/self/task'))}")
        if stage == stage_want:
            break

    note(f"preload done cumulative={time.monotonic()-t0:.2f}s "
         f"nvidia_fds={nvfds()} rss={rss_gib():.2f}GiB")
    if probe_only:
        note("PROBE-ONLY: exiting without waiting")
        raise SystemExit(0)

    open(rdy, "w").write(str(os.getpid()))
    # Poll a path rather than block on a pipe or a signal: a criu-restored process
    # has to wake up inside this loop with no help from the outside.
    while not os.path.exists(go):
        time.sleep(0.02)
    note(f"resumed after restore at {time.time():.3f} nvidia_fds={nvfds()}")

    argv = json.load(open(argsf))
    sys.argv = ["vllm"] + argv
    note("exec vllm serve in-process: " + " ".join(sys.argv))
    from vllm.entrypoints.cli.main import main
    main()


# This guard is load-bearing, not boilerplate, and the whole TP>1 arm depends on
# it. Because this file becomes `vllm serve`'s __main__, any vLLM code path that
# uses the `spawn` start method makes Python re-import __main__ in the new child
# to rebuild the pickled target -- and without a guard that re-runs this preload
# from the top instead of becoming EngineCore.
#
# Observed 2026-09-04 at --arm restore --tp 2: a criu-restored process has CUDA
# initialised, so system_utils.py:157 force-overrides fork -> spawn ("Reasons:
# CUDA is initialized"), the spawned EngineCore printed this script's own
# "stage=torch ... preload done cumulative=2.56s" lines, and the tree wedged in
# do_poll with 37 threads until the 900s poller gave up. GPU never left 4 MiB.
# Under a spawn re-import __name__ is "__mp_main__", so the guard skips the
# preload and multiprocessing's bootstrap proceeds normally.
#
# TP=1 hid the bug: UniProcExecutor forks EngineCore instead, and fork does not
# re-import __main__.
if __name__ == "__main__":
    run()
PY

case "$MODE" in

probe)
  say "import-stage probe (no dump): where does the driver first appear?"
  rm -f "$OUT"
  PREBOOT_PROBE=1 PREBOOT_STAGE=server python3 /tmp/preboot_child.py "$RDY" "$GO" "$ARGS" "$OUT"
  echo; say "verdict: the checkpoint point is the last stage with nvidia_fds=0"
  ;;

build)
  say "building an import-warm image at stage=$STAGE"
  kill_all
  rm -f "$RDY" "$GO" "$OUT"
  T=$(now)
  PREBOOT_STAGE="$STAGE" setsid python3 /tmp/preboot_child.py "$RDY" "$GO" "$ARGS" "$OUT" \
      </dev/null >"$LOG" 2>&1 &
  CHILD=$!
  for _ in $(seq 1 600); do [[ -s "$RDY" ]] && break; kill -0 $CHILD 2>/dev/null || break; sleep 0.5; done
  P=$(cat "$RDY" 2>/dev/null)
  [[ -z "$P" ]] && { say "preload failed:"; tail -20 "$LOG"; exit 1; }
  PRELOAD_SECS=$(el "$T" "$(now)")
  say "  preload ready pid=$P in $PRELOAD_SECS (this cost is what the image removes)"
  sed 's/^/    /' "$OUT"
  NVFD=$(ls -l "/proc/$P/fd" 2>/dev/null | grep -c '/dev/nvidia')
  say "  /dev/nvidia* fds: $NVFD  (must be 0 for a dump with no cuda-checkpoint)"
  rm -rf "$IMG"; mkdir -p "$IMG"
  s=$(now)
  timeout -k 10 "$DUMP_TIMEOUT" "$CRIU" dump -t "$P" -D "$IMG" --log-file dump.log \
      --tcp-close --file-locks --ext-unix-sk --manage-cgroups=ignore --link-remap \
      --force-irmap -v4 >"$IMG/dump.stdout" 2>&1
  rc=$?
  DS=$(el "$s" "$(now)")
  B=$(bytes "$IMG")
  say "  criu dump rc=$rc in $DS  image=$(gib "${B:-0}")"
  if [[ $rc -ne 0 ]]; then
    grep -aE "Error \(|error \(|unsupported" "$IMG/dump.log" | tail -12 | sed 's/^/      /'
    kill_all; exit 3
  fi
  # criu validates open regular files by size; record what the image expects so
  # --measure can restore that state instead of being refused with "bad size".
  { stat -c "%s %n" "$LOG"; stat -c "%s %n" "$OUT"; } > "$IMG/filesizes.txt" 2>/dev/null
  wait "$CHILD" 2>/dev/null
  du -b "$IMG"/* | sort -rn | head -4 | awk '{printf "      %8.2f GiB  %s\n", $1/1073741824, $2}'
  echo
  echo "=== BUILD SUMMARY stage=$STAGE ============================"
  printf 'preload (imports)   : %s\n' "$PRELOAD_SECS"
  printf 'criu dump           : %s\n' "$DS"
  printf 'image size          : %s\n' "$(gib "$B")"
  printf 'image dir           : %s\n' "$IMG"
  echo "==========================================================="
  ;;

measure)
  kill_all
  python3 -c "import json,sys; json.dump(['serve','--model','$MODEL','--max-model-len','$MAXLEN','--tensor-parallel-size','$TP','--port','$PORT'], open('$ARGS','w'))"
  say "arm=$ARM tp=$TP  gpu=$(gpu)"
  if [[ "$ARM" == cold ]]; then
    T=$(now)
    setsid vllm serve --model "$MODEL" --max-model-len "$MAXLEN" \
      --tensor-parallel-size "$TP" --port "$PORT" </dev/null >"$LOG" 2>&1 &
    R=$(wait_health "$T" "$BOOT_TIMEOUT")
    say "  cold boot -> health_200: $R"
    G=$(gen); GS=$(el "$T" "$(now)")
    say "  cold boot -> completion: $GS   out: $G"
    echo
    echo "=== MEASURE arm=cold tp=$TP =============================="
    printf 'exec -> health_200   : %s\n' "$R"
    printf 'exec -> completion   : %s\n' "$GS"
    printf 'completion           : %s\n' "$G"
    echo "=========================================================="
    kill_all
    exit 0
  fi

  [[ -d "$IMG" ]] || { echo "no image at $IMG -- run --build first" >&2; exit 2; }
  rm -f "$GO"
  # Put every file the image has open back to its dump-time size. Deleting $OUT
  # here instead would make criu fail to reopen it at all.
  if [[ -f "$IMG/filesizes.txt" ]]; then
    while read -r sz path; do [[ -n "$path" ]] && { : >>"$path"; truncate -s "$sz" "$path"; }; done < "$IMG/filesizes.txt"
  fi
  [[ "$EVICT" == 1 ]] && say "  $(evict "$IMG")"
  T=$(now)
  timeout -k 10 "$RESTORE_TIMEOUT" "$CRIU" restore -D "$IMG" --log-file restore.log \
      --tcp-close --file-locks --ext-unix-sk --manage-cgroups=ignore --link-remap \
      --force-irmap -v4 -d >"$IMG/restore.stdout" 2>&1
  rc=$?
  RS=$(el "$T" "$(now)")
  say "  criu restore rc=$rc in $RS"
  if [[ $rc -ne 0 ]]; then grep -aE "Error \(|error \(" "$IMG/restore.log" | tail -12 | sed 's/^/      /'; exit 4; fi
  # Release the restored interpreter into `vllm serve`. The clock keeps running:
  # everything from here is what a replica using this image would pay.
  touch "$GO"
  R=$(wait_health "$T" "$BOOT_TIMEOUT")
  say "  restore start -> health_200: $R"
  G=$(gen); GS=$(el "$T" "$(now)")
  say "  restore start -> completion: $GS   out: $G"
  say "  child log:"; sed 's/^/    /' "$OUT" 2>/dev/null | tail -6
  echo
  echo "=== MEASURE arm=restore stage=$STAGE tp=$TP ==============="
  printf 'criu restore         : %s\n' "$RS"
  printf 'restore -> health_200: %s\n' "$R"
  printf 'restore -> completion: %s\n' "$GS"
  printf 'completion           : %s\n' "$G"
  printf 'image                : %s (%s)\n' "$IMG" "$(gib "$(bytes "$IMG")")"
  echo "=========================================================="
  kill_all
  ;;
esac
