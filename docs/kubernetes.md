# Running in Kubernetes

## Two modes

**Mode A — `manifests/pod-exec.yaml` (default).** The container starts as
`sleep infinity`, and each measured run is a `kubectl exec` of
`coldstart-run.sh`. This is the right default because:

* pod scheduling, image pull and container creation are out of scope, so there
  is no reason to pay them once per repetition;
* repeating inside one live container is the only way to vary the compile cache
  (or any other single variable) while node, GPU, page cache and kernel hold
  still.

**Mode B — `manifests/pod-serve.yaml`.** vLLM is the container entrypoint,
wrapped by `coldstart-run.sh`, with `readinessProbe` and `startupProbe` gating
traffic the way a real deployment would. Use it for questions about the
container's own lifecycle — "does our readiness probe let traffic in before vLLM
can answer it?" — not for repeated measurement.

`t0` is vLLM's own exec time in both modes, so their numbers are directly
comparable. In Mode B the pod's own startup is visible only as the gap between
the pod start time and `t0`, which the report does not count.

Both probes are `httpGet /health`, never `tcpSocket`. vLLM binds the port well
before the engine can answer, and the report quantifies exactly how many seconds
of false-ready a `tcpSocket` probe would have granted. `failureThreshold: 900`
with `periodSeconds: 2` allows a 30-minute honest cold start rather than
CrashLoopBackOff on a large model.

## Probe delivery: a ConfigMap

`scripts/build-probe-configmap.sh` renders `coldstart/*.py` plus
`scripts/coldstart-run.sh` into `manifests/generated/probe-configmap.yaml` and
fails if the result exceeds 900 kB (etcd's object limit is ~1 MiB). The pod
mounts it read-only at `/opt/coldstart-src` with `defaultMode: 0555`.

`coldstart-run.sh` then copies the probe to a writable directory
(`/tmp/coldstart`) and runs `compileall` on it. This is not cosmetic: a read-only
mount cannot hold `__pycache__`, so without the copy *every* Python process in
the tree recompiles the probe from source, which lands squarely inside the
measured window.

Regenerate and re-apply after any probe change:

```bash
make configmap
make deploy NS=my-ns
```

## Volumes

| mount | kind | why |
|---|---|---|
| `/cache` | PVC `coldstart-cache` | HF weights and every compile cache. Cold start is mostly a story about what is *not* cached, so the cache must be an explicitly controlled object |
| `/var/log/coldstart` | `emptyDir` | traces (a few hundred kB per run), pulled out with `kubectl cp` |
| `/dev/shm` | `emptyDir` `medium: Memory`, 16 GiB | vLLM's multiproc executor moves tensors through shared memory; the 64 MiB default is a classic silent stall in TP>1 startup |
| `/opt/coldstart-src` | ConfigMap, read-only | the probe |

Set `storageClassName` on the PVC. The storage tier behind the weights is one of
the variables under measurement, and leaving it to the cluster default means you
do not know what you measured. Size for the models under test: a 70B fp16
checkpoint is ~140 GiB, and compile caches add a few hundred MiB per
(model, TP, GPU, vLLM version) tuple.

## Cache directories must all be explicit

Every cache path is set as an env var in the manifests, and `coldstart-run.sh`
creates and writability-checks each one before launch:

```
HOME  HF_HOME  HF_HUB_CACHE  XDG_CACHE_HOME  VLLM_CACHE_ROOT
TORCHINDUCTOR_CACHE_DIR  TRITON_CACHE_DIR
FLASHINFER_CACHE_DIR  FLASHINFER_WORKSPACE_BASE
```

(FlashInfer has used both of its names across releases, so both are set to the
same path and the report accepts either.)

This is a correctness requirement, not tidiness. Under OpenShift's `restricted`
SCC the container runs as an arbitrary uid whose `$HOME` does not exist and whose
`/` is not writable. An unset `TORCHINDUCTOR_CACHE_DIR` then silently degrades to
"no cache" — and every run looks cold, including the ones that were supposed to
be warm. If you see `cannot create` or `is not writable` warnings in the run
output, fix them before believing any number.

## Keeping the baseline honest

`VLLM_NO_USAGE_STATS=1` and `DO_NOT_TRACK=1` are set: usage telemetry is an
outbound HTTPS POST on the startup path, and unless the experiment is
specifically about it, it is noise you do not control. `HF_HUB_OFFLINE=1` is
present but commented out — turn it on once weights are cached to remove hub
round trips from the critical path, and record that you did (it is a documented
experiment variable, not a default).

CPU requests equal CPU limits so the measurement is not at the mercy of
neighbours. Pin an image digest for anything you intend to compare across days;
the vLLM version is one of the strongest determinants of start time.

## Credentials

```bash
kubectl -n my-ns create secret generic hf-token --from-literal=token="$HF_TOKEN"
```

The `HF_TOKEN` env var is wired with `optional: true`, so the pod starts without
the secret. The probe redacts credential-looking env values into
`<redacted:N chars>` before they reach a trace, which makes traces safe to attach
to an issue — verify it for yourself on your first run anyway.

## Typical session

```bash
# once
make deploy NS=my-ns

# measure, fetch, report in one step
make run NS=my-ns MODEL=Qwen/Qwen2.5-1.5B-Instruct REPEAT=3

# re-measure in the already-running pod (no re-deploy, warm page cache)
make exec  NS=my-ns MODEL=Qwen/Qwen2.5-1.5B-Instruct RUN_ID=warm
make fetch NS=my-ns RUN_ID=warm

# extra vllm serve args
make run NS=my-ns MODEL=meta-llama/Llama-3.1-8B-Instruct \
         VLLM_ARGS="--tensor-parallel-size 2 --max-model-len 8192"
```

`scripts/run-experiment.sh` takes the same flags directly if you want more
control (`--cold-compile`, `--cold-hf`, `--no-probe`, `--no-first-token`,
`--no-apply`, `--delete`, and any `vllm serve` args after `--`).

## Troubleshooting

**`no probe at /opt/coldstart-src`** — the ConfigMap is not mounted or not
applied. `make configmap && make deploy`.

**vLLM exits instead of becoming ready** — the driver detects this, prints the
tail of `vllm.log` and exits non-zero. The full log is in the run directory.

**The report shows no vLLM spans, only interpreter boot and imports** — the
probe installed but `cs_vllm`'s patches did not apply. Check `probe.coverage` in
the trace: after a vLLM upgrade, renamed symbols show up as `not_applied`.

**`Pending` pod** — usually no GPU on the node matching the request, or the PVC's
storage class cannot bind. `kubectl describe pod` says which.

**Numbers look 5–10x too slow** — check you are not running an emulated
architecture, and check the throttling finding in the report.

**A `kubectl exec` run dies when your terminal disconnects** — the exec's process
group goes with it. For long runs use Mode B, or `nohup` the driver inside the
pod and fetch afterwards.
