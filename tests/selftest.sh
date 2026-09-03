#!/usr/bin/env bash
# End-to-end self-test of the cold-start tooling, with no GPU and no vLLM.
#
# Runs tests/mock_vllm.py inside a Linux container with the probe injected
# exactly the way the Kubernetes manifests inject it (PYTHONPATH +
# sitecustomize), polls readiness with cs_ready.py, then renders the report on
# the host. Use it after changing the probe or the report to check the whole
# pipeline still agrees.
#
#   tests/selftest.sh [--cpus 2] [--weight-mb 512] [--workers 1]
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-python:3.12-slim}"
# Run natively: under QEMU emulation every import costs ~10x and the numbers
# stop resembling anything a cluster would produce.
case "$(uname -m)" in
  arm64|aarch64) HOST_PLATFORM=linux/arm64 ;;
  x86_64|amd64)  HOST_PLATFORM=linux/amd64 ;;
  *)             HOST_PLATFORM="" ;;
esac
PLATFORM="${PLATFORM:-$HOST_PLATFORM}"
CPUS="${CPUS:-2}"
WEIGHT_MB="${WEIGHT_MB:-512}"
WORKERS="${WORKERS:-1}"
RUN_ID="${RUN_ID:-selftest-$(date +%H%M%S)}"
OUT="$REPO/.selftest/$RUN_ID"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cpus) CPUS="$2"; shift 2 ;;
    --weight-mb) WEIGHT_MB="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --image) IMAGE="$2"; shift 2 ;;
    --platform) PLATFORM="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# The exclusivity guard first: it needs only /proc and a second later it is
# done, and a broken guard refuses real boots outright, so there is no reason to
# find out about it after the slow part.
echo "== exclusivity guard"
docker run --rm ${PLATFORM:+--platform "$PLATFORM"} -v "$REPO:/work" "$IMAGE" \
  bash /work/tests/guard_test.sh | sed 's/^/   /'

mkdir -p "$OUT/trace"
echo "== running mock vLLM in $IMAGE (cpus=$CPUS weights=${WEIGHT_MB}MB workers=$WORKERS)"

docker run --rm --cpus "$CPUS" ${PLATFORM:+--platform "$PLATFORM"} \
  -v "$REPO:/work" \
  -e CS_RUN_ID="$RUN_ID" \
  -e CS_TRACE_DIR=/work/.selftest/"$RUN_ID"/trace \
  -e CS_PROBE_DIR=/tmp/coldstart \
  "$IMAGE" bash -c '
set -e
# Copy the probe to a writable dir and precompile: in the cluster the probe is a
# read-only ConfigMap mount, so every process would otherwise recompile the
# modules from source. This mirrors what scripts/vllm-coldstart-run.sh does.
cp -r /work/coldstart /tmp/coldstart
python -m compileall -q /tmp/coldstart

# t0 for the readiness poller only; the report derives the real t0 from each
# process from its own /proc exec time. CS_DISABLE keeps this helper from being traced
# (it would otherwise be the earliest process and would *become* t0).
export CS_T0=$(CS_DISABLE=1 python -c "import time; print(time.time())")
export PYTHONPATH=/tmp/coldstart

python /work/tests/mock_vllm.py --port 8000 --weight-mb '"$WEIGHT_MB"' \
  --workers '"$WORKERS"' --serve-seconds 15 &
MOCK=$!

# Readiness poller runs without the probe on its path, like a sidecar would.
CS_DISABLE=1 python /work/coldstart/cs_ready.py \
  --port 8000 --first-token --interval-ms 20 --timeout 300 \
  --trace-dir "$CS_TRACE_DIR" --t0 "$CS_T0"
RC=$?
# SIGTERM, not SIGKILL: the probe flushes its per-process summaries from a
# signal handler, exactly as it must when Kubernetes tears the pod down.
kill -TERM $MOCK 2>/dev/null || true
wait $MOCK 2>/dev/null || true
exit $RC
'

echo
echo "== report"
python3 "$REPO/analysis/coldstart_report.py" "$OUT/trace" -o "$OUT" --min-ms 20
