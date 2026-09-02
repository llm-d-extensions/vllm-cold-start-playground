# The measurement model

Everything here exists to make one number defensible: **time from vLLM process
exec to API-server-ready.** This document says exactly what that number
contains, what it excludes, and where it can still mislead you.

## t0: the vLLM process exec

`t0` is not "when the probe started" and not "when the shell launched vLLM". It
is derived inside each Python process from `/proc/self/stat` field 22
(`starttime`, in clock ticks since boot) combined with `/proc/uptime`:

```
age  = uptime - starttime_ticks / CLK_TCK
t0   = time.time() - age
```

`/proc/uptime` has 10 ms resolution, which is far better than the integer-second
`btime` in `/proc/stat`. This matters because it puts the boundary *before the
first byte of Python bytecode runs*: interpreter startup, `site` processing and
the probe's own installation all land inside the measured window rather than
being invisible prologue.

The run's `t0` is the earliest exec time across all traced processes — normally
the API server process, since engine and worker processes are its descendants.

Two consequences worth internalising:

* Anything the harness does *before* exec (staging the probe, writing metadata,
  clearing caches, capturing a helper timestamp) is not counted, by construction.
  This is why `scripts/coldstart-run.sh` can do real work before launching vLLM
  without contaminating the result.
* A helper Python process that is itself traced would become the earliest exec
  and *would become t0*. That is why every helper the driver runs
  (`cs_ready.py`, the `CS_T0` timestamp capture) runs with `CS_DISABLE=1`.

## What is out of scope

Pod scheduling, image pull, container creation, and CNI setup are excluded.
They are a real part of end-to-end replica time and the charter tracks them, but
they belong to a different layer with different fixes; folding them in adds
variance measured in tens of seconds and hides everything happening inside the
process.

The consequence for experiment design is that **the pod should already be
running** when you measure. `manifests/pod-exec.yaml` (Mode A) does exactly
that: the container sleeps, and each measured run is a `kubectl exec`. Repeating
inside one live container is also the only way to vary the compile cache while
holding node, GPU, page cache and kernel still.

## The end of the window: five readiness edges

"Ready" is not one instant. Four edges are probed from outside by
`coldstart/cs_ready.py`; `api_startup_complete` comes from inside the server —
the probe wraps `uvicorn.Server.startup`. Each is its own trace event:

| edge | meaning | why it matters |
|---|---|---|
| `port_open` | TCP connect succeeds | the socket is bound; the engine may still be loading weights. A `tcpSocket` readinessProbe fires here — often seconds early |
| `api_startup_complete` | uvicorn's startup handlers have finished | the app is accepting requests; measured in-process, so it needs no polling |
| `health_200` | `GET /health` returns 200 | **the headline number.** What an `httpGet` readinessProbe uses |
| `models_200` | `GET /v1/models` returns 200 with the model listed | the model is registered, not just the app |
| `first_token` | a real 1-token completion returns | includes lazy kernel JIT and sampler warmup that `/health` does not |

The gap between `port_open` and `health_200` is engine bring-up, and the report
raises a finding when it is large — that gap is exactly how much too early a
TCP-based readiness probe would put a replica into the endpoint list.

The gap between `health_200` and `first_token` is what the first real user
request pays. It is not part of the headline number, but a synthetic warmup
request before joining the endpoint list is what removes it from user-visible
latency.

The poller runs at a 20 ms interval by default, so each edge carries up to
~20 ms of quantisation. Do not read significance into differences below ~50 ms.

## Phase attribution

Spans overlap by nature: the API server sits inside `api.run_server` while a
worker is inside `weights.load_model`, and a lazy `import` can happen inside
either. Summing span durations double-counts; taking a plain union loses the
attribution.

So the timeline is **swept**. Every elementary interval between two event
boundaries is credited to exactly one phase: the innermost span active during
that interval, with ties broken by phase priority (leaf-ish work such as
`torch.compile` outranks container spans such as `engine orchestration`). The
result is non-overlapping by construction and reconciles against wall clock:

```
sum(phases) + unaccounted == total
```

`unaccounted` is time inside the window that no span covered. It is the honest
measure of instrumentation blind spots. A few percent is normal; tens of percent
means a phase is missing a patch and the breakdown should not be trusted until
that is fixed.

Phases, roughly chronological:

```
interpreter boot            python bytecode has not run yet; site + probe install
python imports              module execution time (per-module, deduped)
config & tokenizer resolve  engine config, HF config, tokenizer
network: hub metadata       DNS/TCP/TLS/HTTP to the hub, telemetry
network: weight download    the transfers themselves
ipc handshake               ZMQ/message-queue setup, waiting for engine startup
device & collectives init   CUDA context, NCCL/gloo init
weight load                 safetensors open/read/convert
kv cache alloc              KV cache sizing and allocation
torch.compile               Dynamo/Inductor, including cache hits and misses
cudagraph capture           graph capture
warmup / profile run        dummy runs, memory profiling
api server startup          uvicorn, app build, route registration
engine orchestration        the containing spans that are not any of the above
other                       everything else
```

Note that phase time is **wall-clock on the critical path**, not summed CPU
across processes. Two workers each spending 800 ms in `torch.compile`
concurrently contribute ~800 ms, not 1.6 s. The report shows both: the phase
table is wall clock, while the subsystem section reports `n`, `sum` and `wall`
side by side so you can see how much parallelism you are getting.

## Probe overhead

The probe costs something, and the report says how much rather than hiding it:

```
probe cost   : 102ms worst / 37ms median of 4 processes to install the probe
```

This is measured per process from the first line of `sitecustomize` to the end
of installation. It is **never subtracted from any phase** — the numbers you see
are the numbers the instrumented process actually experienced. Keep it small:
the main reason network patching is deferred through the import hook is that
importing `http.client` eagerly pulls in `email` (~50 modules) and would move
real work onto the critical path before vLLM asked for it.

Per-event overhead is bounded by thresholds (`CS_MIN_DUR_MS`,
`CS_IMPORT_MIN_MS`, `CS_IO_MIN_MS`) and by a per-process event cap
(`CS_MAX_EVENTS`). Dropped events are counted and reported.

To quantify the probe's total effect on time-to-ready, run the same
configuration with `--no-probe` and compare the poller's edges. That control run
is cheap and worth doing once per environment.

## Where the number can still mislead you

* **Page cache.** A "cold start" whose weights are already in the host page
  cache is a warm re-run. The report detects this (weight bytes opened but no
  disk reads billed to the process) and labels the weight-load figure a floor,
  not a cold-node number.
* **Compile caches.** A start with a populated Inductor/Triton cache is a
  different experiment from one without. `--cold-compile` clears them; the run's
  `meta.json` and the report both record which you did.
* **CPU limit.** Imports, safetensors handling and Inductor compilation are all
  CPU-bound. cgroup throttling is sampled during startup and surfaced as a
  finding; a 2-core limit can double time-to-ready on the same GPU.
* **Emulation.** Never measure under QEMU. An amd64 image on an arm64 host
  inflated every import by roughly 10x during development; `tests/selftest.sh`
  detects the host architecture to avoid it.
* **Clock.** Cross-process correlation uses `CLOCK_REALTIME`, which is required
  because separate processes must share an origin. Within a process, durations
  come from `time.monotonic()`. An NTP step during a run would show up as
  disagreement between the two.

## Event schema

Traces are JSONL, one file per process, in Chrome Trace Event shape with
seconds-based `ts` (converted to Perfetto microseconds at report time):

| `ph` | meaning |
|---|---|
| `X` | a complete span: `name`, `cat`, `ts`, `dur`, plus `args` |
| `i` | an instant: readiness edges, `process.exec`, errors |
| `C` | a counter sample: RSS, CPU, throttling, GPU memory |
| `M` | metadata: `run.context`, `probe.installed`, `process.title`, summaries |

`cat` drives phase classification (see `CAT_PHASE` in
`analysis/coldstart_report.py`). Every process emits a `process.exec` (or
`process.fork`) header with its derived exec time, role, and cmdline, plus an
`interpreter.startup` span; exactly one of each per process. The report will
tell you if it sees more, because duplicates mean the tracer is measuring
itself.
