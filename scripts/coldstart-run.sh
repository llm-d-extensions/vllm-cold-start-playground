#!/usr/bin/env bash
# In-pod cold-start experiment driver.
#
# Runs `vllm serve` with the cold-start probe injected, polls the API until it
# is genuinely ready to serve, then stops vLLM -- once per --repeat. Everything
# lands under <out>/<run-id>/ as a self-contained run directory that
# analysis/coldstart_report.py can render.
#
# The measured window is the vLLM *process*: t0 is derived by the probe from
# /proc/self/stat inside vLLM itself, so nothing this script does before exec
# (copying the probe, writing metadata, clearing caches) is counted. Pod,
# container and image-pull time are out of scope by construction.
#
#   coldstart-run.sh [options] -- --model <id> [vllm serve args...]
#
# A run-id is refused if it already holds a trace: trace files are pid-keyed, so
# reusing one merges two runs into a single directory and the report renders the
# union without complaining. --reuse-run-id discards the old trace instead.
set -uo pipefail

RUN_ID="${CS_RUN_ID:-}"
OUT="${CS_OUT_DIR:-/var/log/coldstart}"
REPEAT=1
REUSE_RUN_ID=0
PORT="${CS_READY_PORT:-8000}"
PROBE_SRC="${CS_PROBE_SRC:-/opt/coldstart-src}"
PROBE_DIR="${CS_PROBE_DIR:-/tmp/coldstart}"
TIMEOUT="${CS_READY_TIMEOUT:-1800}"
FIRST_TOKEN=1
USE_PROBE=1
COLD_COMPILE=0
COLD_HF=0
# Clear only $VLLM_CACHE_ROOT/modelinfos -- the model-registry inspection cache.
# Separate from --cold-compile on purpose: --cold-compile wipes all of
# VLLM_CACHE_ROOT, which clears the compile artifacts *and* this, so a run that
# meant to measure compilation was also silently measuring a 15s registry
# subprocess. Varying them independently is the only way to price either.
COLD_REGISTRY=0
GRACE=20
# Seconds to wait for the driver to report no process holding a GPU between
# repeats. Polled, not slept: see the drain loop in run_once.
GPU_DRAIN=60
# A drained H100 reads 0 MiB here; allow a little slack for anything else
# sharing the device.
DRAIN_MIB=1024
ALLOW_CONCURRENT=0
VLLM_BIN="${CS_VLLM_BIN:-vllm}"

die() { echo "coldstart-run: $*" >&2; exit 2; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --repeat) REPEAT="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --probe-src) PROBE_SRC="$2"; shift 2 ;;
    --probe-dir) PROBE_DIR="$2"; shift 2 ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    --grace) GRACE="$2"; shift 2 ;;
    --gpu-drain) GPU_DRAIN="$2"; shift 2 ;;
    --drain-mib) DRAIN_MIB="$2"; shift 2 ;;
    --allow-concurrent) ALLOW_CONCURRENT=1; shift ;;
    --first-token) FIRST_TOKEN=1; shift ;;
    --no-first-token) FIRST_TOKEN=0; shift ;;
    --no-probe) USE_PROBE=0; shift ;;
    --cold-compile) COLD_COMPILE=1; shift ;;
    --cold-registry) COLD_REGISTRY=1; shift ;;
    --cold-hf) COLD_HF=1; shift ;;
    --reuse-run-id) REUSE_RUN_ID=1; shift ;;
    --) shift; break ;;
    -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
    *) die "unknown option: $1 (vLLM args go after --)" ;;
  esac
done
VLLM_ARGS=("$@")
[[ ${#VLLM_ARGS[@]} -gt 0 ]] || die "no vllm args given; use: coldstart-run.sh -- --model <id>"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)}"

# ---------------------------------------------------------------------------
# probe staging: the ConfigMap mount is read-only, so __pycache__ cannot be
# written there and every one of vLLM's processes would recompile the probe
# from source. Copy once, precompile once.
# ---------------------------------------------------------------------------
if [[ "$USE_PROBE" == 1 ]]; then
  [[ -f "$PROBE_SRC/sitecustomize.py" ]] \
    || die "no probe at $PROBE_SRC (expected sitecustomize.py); pass --probe-src or --no-probe"
  rm -rf "$PROBE_DIR"
  mkdir -p "$PROBE_DIR" || die "cannot create $PROBE_DIR"
  cp "$PROBE_SRC"/*.py "$PROBE_DIR/"
  python3 -m compileall -q "$PROBE_DIR" >/dev/null 2>&1 || true
fi

# Every cache dir must exist and be writable before vLLM looks at it: under
# OpenShift's restricted SCC the process runs as an arbitrary uid whose $HOME
# does not exist, and a missing TORCHINDUCTOR_CACHE_DIR silently degrades to "no
# cache", which would quietly make every run look cold.
ensure_dirs() {
  local d
  for d in "${HOME:-}" "${HF_HOME:-}" "${HF_HUB_CACHE:-}" "${XDG_CACHE_HOME:-}" \
           "${VLLM_CACHE_ROOT:-}" "${TORCHINDUCTOR_CACHE_DIR:-}" \
           "${TRITON_CACHE_DIR:-}" "${FLASHINFER_CACHE_DIR:-}" \
           "${FLASHINFER_WORKSPACE_BASE:-}" "$OUT"; do
    [[ -z "$d" ]] && continue
    mkdir -p "$d" 2>/dev/null || echo "coldstart-run: cannot create $d" >&2
    [[ -w "$d" ]] || echo "coldstart-run: $d is not writable -- expect cache misses" >&2
  done
}
ensure_dirs

# Refuse to run alongside another copy of this script.
#
# Stopping the *client* that launched a run (Ctrl-C, a killed `kubectl exec`, a
# task runner cancelling the local process) does NOT stop this script inside the
# pod: it keeps cycling repeats, launching vLLM and taking ~63 GiB each time. A
# second run started afterwards then competes for the same GPUs and fails with
# "Free memory on device cuda:0 (12.94/79.18 GiB) ... less than desired GPU
# memory utilization" -- or, worse, survives with a different memory budget and
# records a measurement that is quietly not comparable. That happened; the
# symptom looked like a teardown race and was not one.
#
# So check for a live sibling and stop, naming the pids, rather than producing
# numbers that silently share a device.
preflight_exclusive() {
  local self="$$" others=""
  local p ppid
  # `$(pgrep ...)` forks a subshell that has not exec'd yet, so it still carries
  # THIS script's cmdline and pgrep matches it -- a sibling pid, a few above our
  # own, that never existed as a second harness. Skipping "$$" and "$PPID" does
  # not catch it: the fork is neither, and under `kubectl exec` the parent lives
  # outside the pod's PID namespace so $PPID is 0 and that arm of the test is
  # dead code. Whether pgrep's /proc walk happens to see the fork before it execs
  # is a race, which is why this refused three boots in a row and none before it.
  #
  # So do not trust the pid list: re-validate every candidate. A pid whose
  # /proc entry has already gone was the fork, and a pid whose parent is us is
  # our own child. Only what survives both is another harness.
  # Fail loudly, not open. Without procps `pgrep` is absent, every scan comes
  # back empty, and the guard silently allows exactly the shared-GPU run it
  # exists to prevent -- an image change could switch it off and nothing would
  # say so. This is a warning rather than an exit because refusing to boot over
  # a missing utility would be worse than booting unguarded and saying so.
  if ! command -v pgrep >/dev/null 2>&1; then
    echo "coldstart-run: WARNING pgrep not found (no procps); cannot check for a" >&2
    echo "    second harness in this container. If one is running, these numbers" >&2
    echo "    share a GPU and are not comparable." >&2
    return 0
  fi
  for p in $(pgrep -f "coldstart-run[.]sh" 2>/dev/null); do
    [[ "$p" == "$self" ]] && continue
    [[ -r "/proc/$p/stat" ]] || continue          # vanished: the fork, not a peer
    ppid="$(awk '{print $4}' "/proc/$p/stat" 2>/dev/null)"
    [[ "$ppid" == "$self" ]] && continue          # our own child
    others="$others $p"
  done
  [[ -z "${others// /}" ]] && return 0
  echo "coldstart-run: another coldstart-run.sh is already running in this" >&2
  echo "    container (pid(s):$others). Two harnesses on one GPU produce" >&2
  echo "    measurements that share a device without saying so." >&2
  echo "    Inspect with: pgrep -af 'coldstart-run[.]sh'" >&2
  echo "    Then either wait for it, or: pkill -9 -f 'coldstart-run[.]sh'" >&2
  echo "    Pass --allow-concurrent to override (it will not be comparable)." >&2
  exit 3
}
[[ "$ALLOW_CONCURRENT" == 1 ]] || preflight_exclusive

cache_dirs() {
  # Compile caches only: these are what a warm re-run gets for free.
  for d in "${VLLM_CACHE_ROOT:-}" "${TORCHINDUCTOR_CACHE_DIR:-}" \
           "${TRITON_CACHE_DIR:-}" "${FLASHINFER_CACHE_DIR:-}" \
           "${FLASHINFER_WORKSPACE_BASE:-}"; do
    [[ -n "$d" ]] && echo "$d"
  done
}

run_once() {
  local i="$1" id run_dir
  if [[ "$REPEAT" == 1 ]]; then id="$RUN_ID"; else id="$RUN_ID-r$i"; fi
  run_dir="$OUT/$id"
  # Reusing a run-id silently corrupts the *old* run and the new one together.
  # meta.json, vllm.log and ready.log are overwritten (fixed names), but trace
  # files are pid-keyed -- events.<role>.<pid>.jsonl -- so the previous run's
  # events survive beside the new ones in one directory. The report then takes
  # t0 = min(process.exec) over the union and anchors to whichever run started
  # first: a real merge of two runs measured 4h39m apart rendered as a single
  # 16,843-second trace, and nothing in the output said so. Refuse instead of
  # clearing, because clearing would trade a corruption bug for a data-loss one
  # -- the operator picks which run they are willing to lose.
  if [[ -d "$run_dir/trace" ]] && compgen -G "$run_dir/trace/*.jsonl" >/dev/null; then
    if [[ "$REUSE_RUN_ID" != 1 ]]; then
      die "$run_dir already holds a trace; pick a different --run-id, or pass
    --reuse-run-id to discard the existing one. Merging two runs into one
    directory produces a report that is wrong without saying so."
    fi
    echo "coldstart-run: --reuse-run-id: discarding existing trace in $run_dir" >&2
    rm -f "$run_dir"/trace/*.jsonl
  fi
  mkdir -p "$run_dir/trace" || die "cannot create $run_dir/trace"

  if [[ "$COLD_COMPILE" == 1 ]]; then
    while read -r d; do
      [[ -n "$d" && "$d" != "/" ]] && rm -rf "${d:?}"/* 2>/dev/null
    done < <(cache_dirs)
  fi
  # After --cold-compile, so `--cold-compile --cold-registry` is not order
  # dependent, and so that clearing modelinfos alone is a no-op on the rest.
  if [[ "$COLD_REGISTRY" == 1 ]]; then
    mi="${VLLM_CACHE_ROOT:-$HOME/.cache/vllm}/modelinfos"
    [[ "$mi" != "/" && "$mi" == */modelinfos ]] && rm -rf "${mi:?}" 2>/dev/null
  fi
  if [[ "$COLD_HF" == 1 && -n "${HF_HOME:-}" ]]; then
    rm -rf "${HF_HOME:?}/hub" 2>/dev/null
  fi

  # ---- run metadata (written before launch; never on the measured path) ----
  {
    echo "{"
    printf '  "run_id": "%s",\n' "$id"
    printf '  "iteration": %s,\n' "$i"
    printf '  "started_utc": "%s",\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '  "hostname": "%s",\n' "$(hostname)"
    printf '  "probe": %s,\n' "$([[ "$USE_PROBE" == 1 ]] && echo true || echo false)"
    printf '  "cold_compile": %s,\n' "$([[ "$COLD_COMPILE" == 1 ]] && echo true || echo false)"
    printf '  "cold_hf": %s,\n' "$([[ "$COLD_HF" == 1 ]] && echo true || echo false)"
    printf '  "cold_registry": %s,\n' "$([[ "$COLD_REGISTRY" == 1 || "$COLD_COMPILE" == 1 ]] && echo true || echo false)"
    printf '  "vllm_version": "%s",\n' "$(python3 -c 'import vllm;print(vllm.__version__)' 2>/dev/null)"
    printf '  "torch_version": "%s",\n' "$(python3 -c 'import torch;print(torch.__version__)' 2>/dev/null)"
    printf '  "gpus": "%s",\n' "$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | paste -sd'; ' -)"
    printf '  "kernel": "%s",\n' "$(uname -sr)"
    printf '  "cmd": "%s"\n' "$VLLM_BIN serve ${VLLM_ARGS[*]//\"/\\\"}"
    echo "}"
  } > "$run_dir/meta.json"

  echo "== run $i/$REPEAT -> $run_dir"

  export CS_TRACE_DIR="$run_dir/trace"
  export CS_RUN_ID="$id"
  # t0 for the poller's live output only; the report derives the real t0 from
  # vLLM's own /proc exec time. CS_DISABLE keeps this helper untraced.
  local t0
  t0="$(CS_DISABLE=1 python3 -c 'import time; print(time.time())')"

  if [[ "$USE_PROBE" == 1 ]]; then
    PYTHONPATH="$PROBE_DIR${PYTHONPATH:+:$PYTHONPATH}" \
      "$VLLM_BIN" serve "${VLLM_ARGS[@]}" --port "$PORT" \
      > "$run_dir/vllm.log" 2>&1 &
  else
    "$VLLM_BIN" serve "${VLLM_ARGS[@]}" --port "$PORT" \
      > "$run_dir/vllm.log" 2>&1 &
  fi
  local vllm_pid=$!

  local ready_args=(--port "$PORT" --timeout "$TIMEOUT"
                    --trace-dir "$run_dir/trace" --run-id "$id" --t0 "$t0")
  [[ "$FIRST_TOKEN" == 1 ]] && ready_args+=(--first-token)

  local rc=0
  CS_DISABLE=1 python3 "${PROBE_SRC}/cs_ready.py" "${ready_args[@]}" \
    2>&1 | tee "$run_dir/ready.log" || rc=$?

  # vLLM may have died instead of becoming ready; say so loudly.
  if ! kill -0 "$vllm_pid" 2>/dev/null; then
    echo "!! vLLM exited before/while becoming ready; see $run_dir/vllm.log" >&2
    tail -20 "$run_dir/vllm.log" >&2
    rc=1
  fi

  # SIGTERM, not SIGKILL: the probe flushes each process's summary from a
  # signal handler, and vLLM gets to release the GPU before the next run.
  kill -TERM "$vllm_pid" 2>/dev/null
  local waited=0
  while kill -0 "$vllm_pid" 2>/dev/null && [[ "$waited" -lt "$GRACE" ]]; do
    sleep 1; waited=$((waited + 1))
  done
  kill -KILL "$vllm_pid" 2>/dev/null
  wait "$vllm_pid" 2>/dev/null

  # Free the GPU fully before the next iteration.
  #
  # Waiting on the launcher pid is not enough. At TP>1 the workers are separate
  # processes that setproctitle-rename themselves to "VLLM::Worker_TPn", so they
  # are neither the launcher nor matched by a name-based kill, and they can still
  # hold tens of GiB after the launcher is gone. A fixed `sleep 3` then hands the
  # next iteration a GPU that is not actually free, and vLLM refuses to start
  # with "Free memory on device cuda:0 ... is less than desired GPU memory
  # utilization" -- or, worse, starts with a different memory budget and yields a
  # measurement that is quietly not comparable. So poll the driver for the real
  # answer instead of guessing, and SIGKILL whatever is still holding on.
  # Gate the next repeat on the GPU actually being free, rather than on a fixed
  # sleep. Waiting on the launcher pid alone is not enough at TP>1: the workers
  # are separate processes that setproctitle-rename themselves to
  # "VLLM::Worker_TPn", so they are neither the launcher nor matched by a
  # name-based kill, and they can outlive it while still holding tens of GiB.
  #
  # Poll memory.used rather than the process list, because that is the quantity
  # vLLM's own startup check reads (it refuses to start when free memory is below
  # gpu_memory_utilization). A pid can leave the driver's compute-apps list
  # before its memory is reclaimed, so the process list can read empty a moment
  # early. Kill stragglers by pid as a best effort; believe the memory number.
  local gwaited=0 used=""
  while [[ "$gwaited" -lt "$GPU_DRAIN" ]]; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
             | tr -d ' ' | sort -rn | head -1)
    [[ -n "$used" ]] || break            # no nvidia-smi: nothing to gate on
    [[ "$used" -le "$DRAIN_MIB" ]] && break
    if [[ "$gwaited" -ge 10 ]]; then
      local h
      for h in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' '); do
        kill -KILL "$h" 2>/dev/null
      done
    fi
    sleep 1; gwaited=$((gwaited + 1))
  done
  if [[ -n "$used" && "$used" -gt "$DRAIN_MIB" ]]; then
    echo "!! GPU still holds ${used} MiB after ${GPU_DRAIN}s; the next run would" >&2
    echo "   start with a different memory budget. Failing this iteration rather" >&2
    echo "   than recording an incomparable one." >&2
    rc=1
  else
    [[ "$gwaited" -gt 3 ]] && echo "   (GPU drained to ${used:-?} MiB in ${gwaited}s)"
  fi
  sleep 2
  return $rc
}

status=0
for i in $(seq 1 "$REPEAT"); do
  run_once "$i" || status=1
done
echo "== done; runs under $OUT (fetch with kubectl cp, render with analysis/coldstart_report.py)"
exit $status
