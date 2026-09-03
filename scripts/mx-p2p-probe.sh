#!/usr/bin/env bash
# Measure ModelExpress's RDMA tier: a second engine pulling weights GPU-to-GPU
# from an engine that is already serving, instead of reading them from storage.
#
# WHY THIS IS THE ONLY FAIR TEST OF modelexpress
#
# `--load-format modelexpress` is a chain, not a loader
# (modelexpress/load_strategy/__init__.py): RDMA P2P via NIXL, then
# runai-model-streamer if MX_MODEL_URI is set, then GPUDirect Storage, then
# vLLM's DefaultModelLoader. On a lone cold pod the first tier has no peer to
# pull from and the third is unavailable without GDS, so the chain degrades to
# runai or to vLLM's *slowest* loader. Benchmarking only that would be measuring
# the fallback and calling it the product.
#
# HOW IT RUNS WITHOUT A MODELEXPRESS SERVER
#
# The decentralized metadata backend (MX_METADATA_BACKEND=k8s-service) skips the
# central coordinator: the source runs its own WorkerGrpcServer and the target
# dials it by Service DNS. When MX_K8S_SERVICE_PATTERN carries no port the client
# appends MX_WORKER_GRPC_PORT+rank itself -- so pointing the pattern at
# `localhost` makes the target dial the source's own gRPC server inside this same
# pod, and the whole two-tier topology fits on one 2-GPU node with no Service,
# no CRD and no server container.
#
# WHAT THE NUMBER MEANS
#
# GPU0 holds the source (loaded from disk, however slowly -- it is not timed).
# GPU1 is the measured cold start, and its weight-load phase is an NVLink/PCIe
# device-to-device transfer. That is the llm-d-relevant number: not "how fast can
# one pod read a checkpoint" but "what does the second replica pay".
#
#   CS_NS=my-ns scripts/mx-p2p-probe.sh
#   CS_NS=my-ns scripts/mx-p2p-probe.sh --run-id mxp2p-c1
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NS="${CS_NS:-}"; POD="${CS_POD:-vllm-coldstart}"
MODEL="${CS_MODEL:-Qwen/Qwen3-32B}"; MML="${CS_MML:-8192}"
LP="${CS_LP:-/cache/lp}"
SNAP="${CS_SNAP:-/cache/hf/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137}"
RUN_ID="mxp2p-$(date -u +%H%M%S)"
SRC_PORT=8100
KEEP=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    -n|--namespace) NS="$2"; shift 2 ;;
    --pod) POD="$2"; shift 2 ;;
    --keep-source) KEEP=1; shift ;;
    -h|--help) sed -n '2,33p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
kube() { kubectl ${NS:+-n "$NS"} "$@"; }

# Shared by source and target. MX_WORKER_HOST is what the source advertises as
# its NIXL endpoint, so it has to be an address the target can actually reach --
# inside one pod that is loopback.
MXENV=(MX_METADATA_BACKEND=k8s-service
       MX_K8S_SERVICE_PATTERN=localhost
       MX_WORKER_HOST=127.0.0.1
       "MX_MODEL_URI=$SNAP"
       PYTHONPATH="$LP"
       PYTHONPYCACHEPREFIX=/cache/pycache)

cleanup() {
  [[ "$KEEP" == 1 ]] && { echo "== leaving the source engine up (--keep-source)"; return; }
  echo "== stopping the source engine"
  kube exec "$POD" -c vllm -- bash -lc "pkill -f 'port $SRC_PORT' || true; sleep 3; pkill -9 -f 'port $SRC_PORT' || true" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "== 1/4 starting the SOURCE engine on GPU0 (not measured)"
kube exec "$POD" -c vllm -- bash -lc "
  rm -f /tmp/mx-source.log
  env CUDA_VISIBLE_DEVICES=0 ${MXENV[*]} \
    nohup vllm serve --model $MODEL --max-model-len $MML \
      --load-format modelexpress --port $SRC_PORT \
      >/tmp/mx-source.log 2>&1 &
  echo launched"

echo "== 2/4 waiting for the source to serve"
for i in $(seq 1 120); do
  code="$(kube exec "$POD" -c vllm -- bash -lc \
    "curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$SRC_PORT/health || true" 2>/dev/null)"
  [[ "$code" == 200 ]] && { echo "   source ready after ${i}0s-ish"; break; }
  if [[ "$i" == 120 ]]; then
    echo "!! source never became ready; last 40 lines:" >&2
    kube exec "$POD" -c vllm -- tail -40 /tmp/mx-source.log >&2
    exit 1
  fi
  sleep 5
done

echo "== 2b/4 what the source actually published"
kube exec "$POD" -c vllm -- bash -lc "
  grep -iE 'Eligible loaders|Trying strategy|strategy .* (failed|complete)|WorkerGrpcServer|publish|NIXL|nixl' /tmp/mx-source.log | tail -25 || true"

echo "== 3/4 measured TARGET cold start on GPU1, same pod, RDMA tier available"
ENVFLAGS=(--env CUDA_VISIBLE_DEVICES=1 --env VLLM_WORKER_MULTIPROC_METHOD=fork)
for kv in "${MXENV[@]}"; do ENVFLAGS+=(--env "$kv"); done
"$REPO/scripts/run-experiment.sh" ${NS:+-n "$NS"} --pod "$POD" --no-apply \
  --run-id "$RUN_ID" --repeat 1 --allow-concurrent "${ENVFLAGS[@]}" \
  -- --model "$MODEL" --max-model-len "$MML" --load-format modelexpress
rc=$?

echo "== 4/4 which strategy the target used"
grep -iE "Eligible loaders|Trying strategy|strategy .*(failed|complete)|rdma|nixl|GetTensorManifest" \
  "$REPO/runs/$RUN_ID/vllm.log" 2>/dev/null | tail -30 || echo "(no vllm.log)"
exit $rc
