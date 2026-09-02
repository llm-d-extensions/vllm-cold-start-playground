# llm-d cold start: vLLM startup measurement

Tooling to answer one question precisely: **where does the time go between the
moment a vLLM process is exec'd and the moment its API server can actually
serve a token?**

This is the measurement half of [SIG Fast Start](CHARTER.md) Goal #1 — a
canonical time-to-ready metric broken down by phase. By default it is
instrumentation and analysis only: it runs against unmodified upstream images and
changes nothing about how vLLM starts. Two probes *do* change startup, both
off unless explicitly switched on, and both exist to price a proposed upstream fix
before anyone writes the patch — see
[what is patched](#what-is-patched-and-what-should-be-upstreamed).

## Scope

```
pod scheduled   image pulled   container created │ vLLM exec ──────────► /health 200
◄──────────── out of scope ──────────────────────┤◄──── what we measure ────►
```

`t0` is the vLLM process's own exec time, read from `/proc/self/stat`, so
interpreter startup and `site` processing fall *inside* the window. Pod
scheduling, image pull and container creation are deliberately excluded — they
are a separate problem with separate owners, and mixing them in makes every
number noisier.

The end of the window is `/health` returning 200. Three more edges are recorded
around it (`port_open`, `models_200`, `first_token`) because they routinely
differ by seconds and because which one you pick changes what "ready" means.

## Results so far: Qwen3-32B on one H100

`Qwen/Qwen3-32B` (61.02 GiB, 17 shards), TP=1 on one H100 80GB, weights on GPFS,
`--max-model-len 8192`. Each step adds one change to the row above it. Median of 3,
with the five arms interleaved within each cycle and a warm-up cycle discarded, so
torch.compile misses and cold bytecode land on every arm equally instead of on
whichever one happened to run first.

| # | change added | median ready | the 3 runs | Δ | phase that moved |
|---|---|---|---|---|---|
| 1 | **baseline** — vLLM defaults: `spawn`, no writable `__pycache__`, `--load-format auto` | [**90.65s**](reports/32b-step1-baseline-spawn.txt) | 85.08 / 90.65 / 91.22 | — | imports 37.6s, weight load 32.2s |
| 2 | `PYTHONPYCACHEPREFIX=/cache/pycache` | [**68.34s**](reports/32b-step2-pycache.txt) | 63.41 / 68.34 / 68.98 | −22.31s | imports 37.6 → 16.6s |
| 3 | `VLLM_WORKER_MULTIPROC_METHOD=fork` | [**63.37s**](reports/32b-step3-fork.txt) | 58.42 / 63.37 / 63.82 | −4.97s | imports 16.6 → 10.5s |
| 4 | `--load-format fastsafetensors` | [**47.64s**](reports/32b-step4-fastsafetensors.txt) | 47.51 / 47.64 / 47.65 | −15.73s | weight load 32.4 → 17.6s |
| 5 | `cs_fst` patch: `nogds=True`, `max_threads=8`, `bbuf_size_kb=32768` | [**43.30s**](reports/32b-step5-cs-fst-patch.txt) | 42.85 / 43.30 / 44.38 | −4.34s | weight load 17.6 → 13.7s |

**90.65s → 43.30s, −47.35s (−52%).** Only step 5 needs patched code; steps 2-4 are
configuration. Compile time is 0.21-0.23s in every arm, so nothing here is a
recompile artifact.

Each median links to that run's full report in [`reports/`](reports/) — phase
breakdown, per-process span tree and subsystem detail. The report also shows the
process topology, which is the direct evidence for step 3: at the baseline
`EngineCore` is an `exec` of `multiprocessing.spawn` with its own
`resource_tracker`, and from step 3 on it is a `fork` that re-imports nothing. The
full traces stay local — `runs/` is 186 MiB for 92 runs, 98% of it trace events, so
it is gitignored.

What the phase breakdown says about each step:

* **Bytecode compilation is half the import wall.** vLLM's import graph is 6781
  files, and on a read-only image layer with nowhere to write `__pycache__` every
  interpreter recompiles all of them — 9.73s per interpreter, 56% of its import
  time. This is the largest single win in the table and it is a stock CPython
  environment variable.
* **The imports are then paid twice.** `vllm serve` forces `spawn`
  (`entrypoints/serve/utils/api_utils.py:166`), so `EngineCore` is a fresh
  interpreter that re-imports torch and vllm from scratch; `fork` makes it
  inherit them. Note this step's
  end-to-end Δ (−4.97s) is *inside* that arm's own 5.4s spread — the reliable
  evidence is the import phase, which drops 6.1s with a much tighter spread. At
  TP=1 there is exactly one child, so this is the smallest this step can be; at
  TP>1 one preload amortises across N workers.
* **Weight loading was the ceiling, and storage was never the problem.** GPFS
  delivers 9.30 GiB/s with O_DIRECT at 16 threads and the H100 takes 51.55 GiB/s
  over pinned host memory, but the default loader moved 61.02 GiB at ~2.0 GiB/s.
  `--load-format fastsafetensors` takes that to 3.79 GiB/s and step 5 to
  5.00 GiB/s. The bottleneck was read concurrency, and the flag is obscure enough
  that most deployments never set it.
* **Then fastsafetensors asks for hardware that is not installed** — the `nogds`
  row below.

Caveats, so the table is read for what it is:

* **Weights are warm in GPFS's pagepool.** A genuinely cold read of this model
  costs ~62s against ~30s warm; the discarded warm-up cycle showed a 63.4s weight
  load. The *deltas* are comparable across arms, but the absolute floor is
  optimistic for a first-ever pull on a node.
* Constants across all arms, set by `manifests/pod-exec.yaml` and not part of any
  step: caches on a PVC, `OMP_NUM_THREADS=16` against a 16-core cgroup limit, and
  `VLLM_ENABLE_STARTUP_PLAN=1`. The last one helps the baseline, so the
  reductions above are if anything understated.
* One node, one GPU, TP=1, and readiness is `/health` 200. Reproduce with
  `runs/lad2-*`.

## What is patched, and what should be upstreamed

Three of the four wins are **configuration** — no patched vLLM, no patched image.
One is a monkey-patch, and it is the one with a real upstream fix behind it.

| change | how it is applied here | what should happen upstream |
|---|---|---|
| `PYTHONPYCACHEPREFIX` | env var, stock CPython | Nothing in vLLM. The image should ship its own `.pyc`: `python -m compileall` at build time puts them in a read-only layer and no operator has to know this knob exists. |
| `VLLM_WORKER_MULTIPROC_METHOD=fork` | env var, already a supported value | `fork` works but is on borrowed time — it forks a multi-threaded, torch-loaded parent, vLLM silently reverts to `spawn` if CUDA is already initialised, and Python 3.14 moves the Linux default to `forkserver`. vLLM already contains forkserver support (`api_server.py:111-117`) but **rejects the value**: `envs.py:930` declares the choices as `["spawn", "fork"]` (and the annotation at `envs.py:67` agrees), so `get_mp_context()` (`utils/system_utils.py:168`) raises `ValueError`. Making it reachable is those two lines; making it *pay* also needs `forkserver.ensure_running()` moved to the top of `cli/main.py:main()`, because where it sits now only ~1.7s of its ~15s preload overlaps anything. |
| `--load-format fastsafetensors` | stock CLI flag | Nothing to patch. Worth documenting that the win is this large, since the flag is easy to miss. |
| `nogds=True`, `max_threads=8`, `bbuf_size_kb=32768` | **monkey-patch** — [`coldstart/cs_fst.py`](coldstart/cs_fst.py) wraps `fastsafetensors.parallel_loader.ParallelLoader.__init__`; opt-in via `CS_FST=1` | `weight_utils.py:1057` computes `nogds = pg.size() > 1`, and the comment above it shows why: at TP>1 `cuFileDriverOpen()` would create CUDA contexts on every visible GPU. Availability of GDS is never checked, so at TP=1 vLLM *always* asks for it. There *is* a fallback (`weight_utils.py:1083`), but it needs a `RuntimeError` with `"gds"` in the message — and fastsafetensors degrades internally rather than raising, so the fallback never fires: the `"GDS not enabled"` warning appears in none of our runs. The failed probe is then billed silently to every fresh `EngineCore` — 1.69s of one-time setup plus 0.91s of steady-state throughput. It should key on whether `libcufile` and `nvidia_fs` exist, not on world size. `max_threads` and `bbuf_size_kb` are not reachable from vLLM at all: not plumbed through, and absent from fastsafetensors' own `LoaderConfig`. |

Two hazards deliberately *not* fixed here, both prerequisites for the forkserver
work rather than wins of their own:

* three `flashinfer` modules call `torch.cuda.get_device_capability()` at **import**
  time, which initialises CUDA in the parent — precisely what makes `fork` unsafe;
* `vllm/third_party/flash_linear_attention` (`tilelang`) has the same hazard, on a
  path 32B does not hit.

The other opt-in probe, [`coldstart/cs_forkserver.py`](coldstart/cs_forkserver.py)
(`CS_FORKSERVER=1`), measures the forkserver ceiling without patching vLLM by
starting the preload at t≈0 and swapping `get_mp_context()`. It is not in the table
above: at TP=1 it lands at parity with `fork`, which is the expected result.

## Quickstart

Validate the whole pipeline locally, with no GPU and no vLLM installed — a mock
that reproduces vLLM's process topology and phase structure runs in a container,
and the real probe, poller and report run against it:

```bash
make selftest
```

Measure a real cold start in a cluster:

```bash
make run NS=my-ns MODEL=Qwen/Qwen2.5-1.5B-Instruct
```

That builds the probe ConfigMap, applies the PVC and experiment pod, runs the
measurement inside it, pulls the traces back to `runs/<run-id>/` and renders the
report. Compare two runs:

```bash
make compare BASE=runs/warm-cache NEW=runs/cold-cache
```

`make help` lists every target.

## What you get

Each run directory holds:

| file | what it is |
|---|---|
| `report.txt` | the human answer: total, phase breakdown, per-process span tree, subsystem detail, findings |
| `summary.json` | the same numbers, machine-readable |
| `trace.json` | Chrome Trace Event file — open in [ui.perfetto.dev](https://ui.perfetto.dev) or `chrome://tracing` |
| `phases.csv` | one row per run, for regression tracking across many runs |
| `trace/*.jsonl` | raw events, one file per process |
| `meta.json`, `vllm.log`, `ready.log` | provenance and raw logs |

The phase breakdown is non-overlapping and reconciles against wall clock:
`sum(phases) + unaccounted == total`. The `unaccounted` row is the honest
measure of the probe's blind spots — treat a large one as a bug in the
instrumentation, not a rounding artifact.

## Documentation

* **[docs/measurement.md](docs/measurement.md)** — the measurement model: how
  `t0` is derived, readiness edges, phase attribution, probe overhead, event
  schema. Read this before trusting a number.
* **[docs/instrumentation.md](docs/instrumentation.md)** — how the probe is
  injected, what each module measures, every `CS_*` knob, and how to instrument
  a new vLLM symbol.
* **[docs/kubernetes.md](docs/kubernetes.md)** — the manifests, Mode A vs
  Mode B, cache volumes, OpenShift/SCC notes, troubleshooting.
* **[docs/experiments.md](docs/experiments.md)** — the experiment matrix and how
  to run a defensible comparison.
* **[docs/foundry.md](docs/foundry.md)** — ingested notes on
  [Foundry](https://github.com/foundry-org/foundry), which persists CUDA graphs to
  disk: what it eliminates in our phase model, its operating constraints, and how
  to evaluate it here.

## Layout

```
coldstart/          the probe: sitecustomize.py + cs_*.py (stdlib only)
analysis/           coldstart_report.py — trace dir -> report/CSV/Perfetto
manifests/          experiment pod, cache PVC, probe ConfigMap source
reports/            committed report.txt for each run cited in this README
scripts/            in-pod driver, ConfigMap builder, host-side run/fetch
tests/              mock vLLM + end-to-end self-test
```

The probe is stdlib-only and imports nothing at module scope that vLLM would not
already import, so it runs inside any vLLM image without adding dependencies.
