# The probe: how instrumentation gets injected

No vLLM fork. No image rebuild. No wrapper around the entrypoint. The probe
attaches itself to every Python process in the vLLM tree using two mechanisms
that Python gives us for free.

## 1. `sitecustomize.py` on `PYTHONPATH`

Python's `site` module imports a module named `sitecustomize` during interpreter
startup, before any user code runs. Putting the probe directory on `PYTHONPATH`
therefore instruments the process:

```bash
PYTHONPATH=/opt/coldstart CS_TRACE_DIR=/var/log/coldstart/<run> vllm serve ...
```

This covers the whole process tree, not just the first process:

* **`spawn`-ed children** (EngineCore, `WorkerProc`) inherit `PYTHONPATH` from
  the environment, so `site` runs the probe in them too.
* **forked children** are re-anchored via `os.register_at_fork`: the tracer
  reopens its own trace file, resets its per-process state and re-derives its
  role.

If the image already ships a `sitecustomize.py`, ours shadows it by being first
on `sys.path` — so after installing, the probe locates and executes the original
and records that it did (`probe.chained_sitecustomize`).

Everything in installation is best-effort: each step is wrapped, and failures are
recorded as trace events rather than raised. **A probe failure must never stop
vLLM from starting.** `probe.installed` lists what installed and what failed.

## 2. A `sys.meta_path` finder for deferred patching

vLLM and torch internals cannot be patched from `sitecustomize`: none of them are
importable yet, and importing them there would both distort the measurement and
disturb vLLM's own import order. So `cs_patch` installs one `meta_path` finder
that wraps each module's loader and fires registered callbacks the instant a
module finishes executing:

```python
cs_patch.after_import("uvicorn.server", _apply)   # runs when it lands
cs_patch.patch("vllm.v1.worker.gpu_worker", "Worker.load_model",
               name="weights.load_model", cat="weights")
```

`after_import` fires immediately for modules already imported, so registration
order does not matter. Patch results are version-tolerant: a symbol that moved
between vLLM releases is recorded as `missing` rather than raising, and
`cs_patch.coverage()` reports `applied` / `not_applied` / `never_imported` into
the trace as `probe.coverage`.

**Check `probe.coverage` after a vLLM upgrade.** A phase that quietly went to
zero usually means a renamed symbol, not a performance win.

## Module map

| module | what it measures |
|---|---|
| `cs_trace.py` | the tracer: JSONL sink, `t0` derivation from `/proc`, spans/instants/counters/meta, fork and signal safety |
| `cs_patch.py` | the deferred-patch machinery and coverage reporting |
| `cs_imports.py` | per-module import cost — cumulative, self time, origin (stdlib / site-packages / `.so`) |
| `cs_net.py` | DNS, TCP connect, TLS handshake, HTTP requests, HF hub metadata and downloads |
| `cs_io.py` | which files were opened and how big, plus `stat`/`exists`/`listdir` chatter counts |
| `cs_torch.py` | CUDA lazy init, collectives init, `torch.compile`/Inductor (cache hit vs miss), Triton JIT, CUDA graph capture |
| `cs_vllm.py` | ~120 patch points across vLLM's startup path: CLI, API server, engine config, tokenizer, EngineCore, executors, workers, model loader, KV cache, kernel warmups, and a hand-written wrapper that splits CUDA graph capture by mode and size |
| `cs_sampler.py` | background timeseries: CPU, RSS, threads, faults, `/proc/self/io`, netns bytes, cgroup throttling, page cache; NVML per-GPU from one elected process |
| `cs_env.py` | run context: cgroup limits, filesystem behind each cache, cache sizes, GPU/CPU facts, redacted env snapshot |
| `cs_ready.py` | the external readiness poller — the only piece that runs *without* the probe |

`cs_env` takes the `os.environ` snapshot synchronously (vLLM mutates its own
environment during startup, so a late snapshot would not be the environment the
process started with) and does everything slow — mountinfo parsing, cache
directory scans — on a daemon thread, off the critical path. Cache scans run in
one elected process only, chosen with an `O_EXCL` lock file.

## Rules the probe follows

These are the invariants that make the numbers trustworthy. Preserve them when
editing:

* **The probe must not measure itself.** The tracer captures a pristine
  `builtins.open` before `cs_io` wraps it, holds a thread-local reentrancy guard
  while writing, and `cs_io` skips paths under the trace directory. Without this,
  opening the trace file emits an event that re-enters the writer.
* **Flushing must survive `SIGTERM`.** `atexit` does not run on signals, and
  every process in this experiment dies by signal (Kubernetes teardown, the
  harness stopping vLLM). The tracer installs handlers for SIGTERM/SIGINT/SIGHUP
  that flush summaries, chain any prior handler, then re-raise through the
  default disposition. Use `at_finish(cb)` rather than `atexit.register` for
  anything that must reach the trace.
* **Nothing expensive at import time.** Network patching is deferred through the
  import hook precisely because importing `http.client` eagerly pulls in `email`
  (~50 modules).
* **Secrets never reach the trace.** Env values whose key matches
  `TOKEN|SECRET|PASSWORD|PASSWD|KEY|CRED|AUTH` are replaced with
  `<redacted:N chars>`. `HF_TOKEN` in particular must stay redacted.
* **Stdlib only.** The probe runs inside an unmodified vLLM image and must add
  no dependencies.

## Environment knobs

All optional; all read once at tracer construction.

| variable | default | effect |
|---|---|---|
| `CS_DISABLE` | unset | `1` turns the probe off entirely. Used for every helper process so it cannot become `t0` |
| `CS_TRACE_DIR` | `/var/log/coldstart` | output directory for JSONL event files |
| `CS_RUN_ID` | `run` | label recorded in every event file |
| `CS_ROLE` | auto | override the process role label (otherwise derived from argv, then from `setproctitle`) |
| `CS_PROBES` | `all` | comma list from `imports,net,io,torch,vllm,sampler,env`. `all,-net` disables one |
| `CS_MIN_DUR_MS` | `1` | drop spans shorter than this |
| `CS_IMPORT_MIN_MS` | `15` | drop import spans shorter than this |
| `CS_IMPORT_TOP_N` | `40` | modules kept in the per-process import summary |
| `CS_IO_MIN_MS` | `5` | drop file-open spans shorter than this |
| `CS_SAMPLE_MS` | `100` | per-process resource sampling interval |
| `CS_GPU_SAMPLE_MS` | `250` | NVML sampling interval |
| `CS_GPU_SAMPLE` | on | `0` disables NVML sampling |
| `CS_MAX_EVENTS` | `200000` | per-process event cap; drops are counted |
| `CS_MODEL` | unset | model path/id, used for filesystem detection and the first-token request |
| `CS_CUDAGRAPH_MIN_DUR` | `0` | drop CUDA-graph per-descriptor spans shorter than this many seconds. `0` keeps all ~300 of them, which is what makes the per-mode counts exact |
| `CS_DEBUG` | unset | print a traceback if installation raises |

The readiness poller has its own: `CS_READY_HOST`, `CS_READY_PORT`,
`CS_READY_INTERVAL_MS`, `CS_READY_TIMEOUT`, `CS_READY_FIRST_TOKEN`, `CS_T0`.

## Instrumenting something new

Add a row to `PATCHES` in `cs_vllm.py`:

```python
("vllm.v1.worker.gpu.model_runner", "GPUModelRunner.capture_model",
 "cudagraph.capture_model", "cudagraph", None)
#  module                            qualname          span name      cat   kind

# `kind` is None for a plain function, or "acm" for something returning an
# async context manager; async functions and async generators are detected.
```

The `cat` decides which phase the span is credited to — see `CAT_PHASE` in
`analysis/coldstart_report.py`, and add a mapping there if you introduce a new
category. If your new span is a leaf inside an existing container span, check its
phase priority in `PHASES` so the sweep credits the right one.

### Two traps that cost us 10.9s of blind spot

**Patch the namespace that resolves the name, not the one that defines it.**
`from x import f` copies `f` into the importer's module dict, so patching `x.f`
afterwards changes nothing at the call site. `kernel_warmup` is defined in
`vllm/model_executor/warmup/kernel_warmup.py` but called from `gpu_worker.py`,
which imported it by name — so the target is
`vllm.v1.worker.gpu_worker:kernel_warmup`. The same applies to the ~12
sub-warmups: they are bound into `vllm.model_executor.warmup.kernel_warmup`,
which is where they must be patched. A name imported *inside* the function body
(`trigger_inductor_lazy_init`, `minimax_m3_msa_warmup`) is resolved at call time,
so there the defining module is right. Listing both is harmless — only one can
fire.

**Check which of several coexisting implementations this build actually runs.**
vLLM 0.28 ships two model runners side by side: the legacy
`vllm/v1/worker/gpu_model_runner.py` and the v2 package
`vllm/v1/worker/gpu/model_runner.py`, selected by `Worker.use_v2_model_runner`.
Six patches aimed at the v1 module sat unresolved in `cs_patch._pending` for the
life of every run, leaving 81.7% of the 13.3s warmup phase with no span under it.
Nothing failed; the module was simply never imported. The same split is why
`cudagraph_num_of_warmups` does nothing on this build — only the v1 runner reads
it.

### Verify coverage, then verify there is no dark time

A patch that silently never applies is the failure mode to design against, so
the report answers it two ways and you should read both:

* `cs_patch` emits one `probe.patch_declared` meta at install and one
  `probe.patch` instant per target as it resolves. The report reconstructs
  coverage from those, unions across processes, and prints **two** lists: targets
  whose module was imported but whose attribute was missing (a version
  difference), and targets **whose module was never imported** (usually a probe
  defect). These events do not depend on the process exiting in a way the probe
  observes — which matters, because `EngineCore` does not, and its
  `at_finish` summary never reaches the trace.
* The `DARK SPAN` finding is the backstop that needs no foreknowledge: any span
  over 1s and 2% of total whose interior is less than a quarter covered by
  descendants gets named. That catches the next renamed module without anyone
  having to notice the rename.

For a new span inside the warmup phase, also check the `WARMUP / CUDA GRAPH
CAPTURE` section: it rolls up every `warmup.*` and `cudagraph.*` span by name and
splits capture by mode and by warmup-vs-captured forward, and its top line
(`compile_or_warm_up_model: N total`) is what the children must add up to.

### Then confirm the plumbing end to end

```bash
make selftest
```

`tests/mock_vllm.py` reproduces vLLM's process topology (API server + spawned
workers + resource tracker) and phase structure without a GPU, so the self-test
exercises the tracer, the fork/spawn paths, the poller, all readiness edges and
the full report. It will not tell you whether a *real* vLLM symbol still exists —
for that, run against the image and read the two coverage lines at the foot of
the report.
