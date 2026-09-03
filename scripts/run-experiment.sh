#!/usr/bin/env bash
# Host-side driver: build the probe ConfigMap, deploy the experiment pod, run N
# measured cold starts inside it, pull the traces back and render the report.
#
#   scripts/run-experiment.sh -n my-ns --model Qwen/Qwen2.5-1.5B-Instruct
#   scripts/run-experiment.sh -n my-ns --repeat 3 --cold-compile -- \
#       --model meta-llama/Llama-3.1-8B-Instruct --tensor-parallel-size 2
#
# --env K=V overrides one variable for this run only, so an A/B matrix does not
# need a manifest edit (and cannot leave one behind):
#
#   scripts/run-experiment.sh -n my-ns --run-id spawn      --env VLLM_WORKER_MULTIPROC_METHOD=spawn ...
#   scripts/run-experiment.sh -n my-ns --run-id forkserver --env CS_FORKSERVER=1 ...
#
# Anything after `--` goes to `vllm serve`. Re-running against an existing pod
# is the normal case: --keep (default) leaves it up so the next run reuses the
# warm page cache and the same GPU.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NS=""
POD="vllm-coldstart"
MANIFEST="$REPO/manifests/pod-exec.yaml"
RUN_ID="$(date -u +%Y%m%d-%H%M%S)"
REPEAT=1
MODEL=""
EXTRA=()
RUN_FLAGS=()
ENVS=()
DELETE_POD=0
APPLY=1
LOCAL_OUT="$REPO/runs"
TIMEOUT=1800

kube() { kubectl ${NS:+-n "$NS"} "$@"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--namespace) NS="$2"; shift 2 ;;
    --pod) POD="$2"; shift 2 ;;
    --manifest) MANIFEST="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --repeat) REPEAT="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    --out) LOCAL_OUT="$2"; shift 2 ;;
    --cold-compile) RUN_FLAGS+=(--cold-compile); shift ;;
    --cold-registry) RUN_FLAGS+=(--cold-registry); shift ;;
    --cold-hf) RUN_FLAGS+=(--cold-hf); shift ;;
    --no-probe) RUN_FLAGS+=(--no-probe); shift ;;
    --no-first-token) RUN_FLAGS+=(--no-first-token); shift ;;
    --env)
      [[ "$2" == *=* ]] || { echo "--env wants K=V, got: $2" >&2; exit 2; }
      ENVS+=("$2"); shift 2 ;;
    --no-apply) APPLY=0; shift ;;
    --delete) DELETE_POD=1; shift ;;
    --) shift; EXTRA=("$@"); break ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$MODEL" || ${#EXTRA[@]} -gt 0 ]] \
  || { echo "give --model <id> (or pass vllm args after --)" >&2; exit 2; }
[[ -n "$MODEL" ]] && EXTRA=(--model "$MODEL" "${EXTRA[@]+${EXTRA[@]}}")

echo "== 1/5 probe ConfigMap"
"$REPO/scripts/build-probe-configmap.sh" ${NS:+-n "$NS"} >/dev/null
kube apply -f "$REPO/manifests/generated/probe-configmap.yaml"

if [[ "$APPLY" == 1 ]]; then
  echo "== 2/5 pod"
  kube apply -f "$REPO/manifests/cache-pvc.yaml"
  kube apply -f "$MANIFEST"
else
  echo "== 2/5 pod (skipped, --no-apply)"
fi

echo "== 3/5 waiting for $POD to be running"
kube wait --for=condition=Ready "pod/$POD" --timeout=20m

# Wait for the kubelet to actually put the new probe in the pod's mount.
# `kubectl apply` above only reached the API server; the ConfigMap volume
# refreshes on the kubelet's own sync period, so an exec in the next few seconds
# runs the PREVIOUS probe. That fails loudly if the change added a flag, and
# silently if it only added spans -- producing a trace that is missing exactly
# what the run was meant to measure. Polling the stamp is the only way to know.
STAMP="$(cat "$REPO/manifests/generated/probe-stamp.txt")"
echo "== settle waiting for probe ${STAMP:0:12} to reach the pod"
for i in $(seq 1 90); do
  # `|| true`: on the first poll after adding the stamp key the file does not
  # exist yet, and under `set -e` a failed command substitution in an assignment
  # takes the whole script down before the loop can retry.
  POD_STAMP="$(kube exec "$POD" -c vllm -- cat /opt/coldstart-src/PROBE_STAMP 2>/dev/null | tr -d '[:space:]' || true)"
  [[ "$POD_STAMP" == "$STAMP" ]] && { echo "   probe current after ${i}s"; break; }
  if [[ "$i" == 90 ]]; then
    echo "!! pod still has probe '${POD_STAMP:0:12}' after 90s, wanted ${STAMP:0:12}" >&2
    echo "   refusing to measure with a probe that is not the one in this tree." >&2
    exit 1
  fi
  sleep 1
done

echo "== 4/5 running $REPEAT cold start(s) as run-id $RUN_ID"
# The probe ConfigMap ships coldstart-run.sh alongside the modules, so the pod
# does not need this repo mounted.
# `env` rather than a shell prefix: kubectl exec takes no --env, and going
# through `bash -c` would put the values through another round of word
# splitting. Overrides land only on this process tree, so the pod's own env --
# and the manifest that documents it -- stays the record of the default arm.
[[ ${#ENVS[@]} -gt 0 ]] && echo "   env: ${ENVS[*]}"
kube exec "$POD" -c vllm -- env "${ENVS[@]+${ENVS[@]}}" \
  bash /opt/coldstart-src/coldstart-run.sh \
  --run-id "$RUN_ID" --repeat "$REPEAT" --timeout "$TIMEOUT" \
  "${RUN_FLAGS[@]+${RUN_FLAGS[@]}}" -- "${EXTRA[@]}"

echo "== 5/5 fetching traces"
"$REPO/scripts/fetch-trace.sh" ${NS:+-n "$NS"} --pod "$POD" \
  --run-id "$RUN_ID" --out "$LOCAL_OUT"

if [[ "$DELETE_POD" == 1 ]]; then
  kube delete -f "$MANIFEST" --wait=false || true
fi
