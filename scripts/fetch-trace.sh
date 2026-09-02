#!/usr/bin/env bash
# Pull run directories out of an experiment pod and render the report locally.
#
#   scripts/fetch-trace.sh -n my-ns --run-id 20260901-120000
#   scripts/fetch-trace.sh -n my-ns --all
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NS=""
POD="vllm-coldstart"
RUN_ID=""
ALL=0
REMOTE_DIR="/var/log/coldstart"
OUT="$REPO/runs"
NO_REPORT=0

kube() { kubectl ${NS:+-n "$NS"} "$@"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--namespace) NS="$2"; shift 2 ;;
    --pod) POD="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --all) ALL=1; shift ;;
    --remote-dir) REMOTE_DIR="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --no-report) NO_REPORT=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$OUT"
list_runs() {
  if [[ "$ALL" == 1 ]]; then
    kube exec "$POD" -c vllm -- bash -lc "ls -1 $REMOTE_DIR"
  elif [[ -n "$RUN_ID" ]]; then
    # one run, or several iterations of it (<run-id>-r1, -r2, ...)
    kube exec "$POD" -c vllm -- bash -lc \
      "ls -1d $REMOTE_DIR/$RUN_ID $REMOTE_DIR/$RUN_ID-r* 2>/dev/null | xargs -n1 basename"
  else
    echo "give --run-id <id> or --all" >&2
    return 2
  fi
}

# read into an array without mapfile: macOS still ships bash 3.2
runs=()
while IFS= read -r line; do
  [[ -n "$line" ]] && runs+=("$line")
done < <(list_runs)

[[ ${#runs[@]} -gt 0 ]] || { echo "no runs found in $POD:$REMOTE_DIR" >&2; exit 1; }

for r in "${runs[@]}"; do
  r="${r%$'\r'}"
  [[ -z "$r" ]] && continue
  echo "== fetching $r"
  # `exec -- tar cf -` rather than `kubectl cp`. kubectl cp truncated a run
  # mid-stream ("Dropping out copy after 0 retries / unexpected EOF") and left a
  # short events file behind, which the report then happily rendered -- a
  # silently wrong measurement, the one failure mode this harness must not have.
  # Streaming a tarball through stdout fails loudly instead: tar exits non-zero
  # on a short archive, and staging it before extraction means a bad fetch never
  # overwrites a good local copy.
  # Stage inside $OUT rather than $TMPDIR: the destination is known writable,
  # which $TMPDIR is not on every host -- mktemp -t failed outright under a
  # sandbox that allows the repo but not /var/folders.
  tarball="$OUT/.fetch-$r.tgz"
  # Transfers really do truncate: kubectl cp lost a run mid-stream ("Dropping
  # out copy after 0 retries"), and so did a plain `exec -- tar cf -`, which is
  # why every attempt is verified and retried rather than trusted. A short
  # events file that still parses is the one failure this harness must not have
  # -- the report would render it and the number would simply be wrong.
  #   * -z: ~7x less to stream, so far less to lose.
  #   * 2>/dev/null in-pod: tar's warnings must never reach the byte stream.
  #   * tar tzf: reads the whole archive, so truncation is caught *before*
  #     anything lands in runs/, and gzip's own CRC catches corruption.
  ok=0
  for attempt in 1 2 3; do
    if kube exec "$POD" -c vllm -- \
         tar czf - -C "$REMOTE_DIR" "$r" 2>/dev/null > "$tarball" \
       && tar tzf "$tarball" >/dev/null 2>&1; then
      ok=1; break
    fi
    echo "   fetch attempt $attempt for $r came back short; retrying" >&2
  done
  if [[ "$ok" != 1 ]]; then
    rm -f "$tarball"
    echo "could not fetch $r intact after 3 attempts; it is still in the pod at $REMOTE_DIR/$r" >&2
    exit 1
  fi
  # Only now is the old copy expendable. Deleting it *before* the fetch cost me
  # a baseline trace: the rm succeeded, the transfer came back short, and the
  # run existed nowhere on this machine any more. Verified bytes in hand first,
  # destructive step second.
  rm -rf "$OUT/$r"
  tar xzf "$tarball" -C "$OUT"
  rm -f "$tarball"
  if [[ "$NO_REPORT" == 0 && -d "$OUT/$r/trace" ]]; then
    python3 "$REPO/analysis/coldstart_report.py" "$OUT/$r/trace" -o "$OUT/$r" \
      --csv "$OUT/phases.csv" | tail -2
  fi
done

echo "== local runs under $OUT"
