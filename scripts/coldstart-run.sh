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
GRACE=20
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
    --first-token) FIRST_TOKEN=1; shift ;;
    --no-first-token) FIRST_TOKEN=0; shift ;;
    --no-probe) USE_PROBE=0; shift ;;
    --cold-compile) COLD_COMPILE=1; shift ;;
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
  sleep 3
  return $rc
}

status=0
for i in $(seq 1 "$REPEAT"); do
  run_once "$i" || status=1
done
echo "== done; runs under $OUT (fetch with kubectl cp, render with analysis/coldstart_report.py)"
exit $status
