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
with the arms of a matrix interleaved within each cycle and a warm-up cycle
discarded, so torch.compile misses and cold bytecode land on every arm equally
instead of on whichever one happened to run first. The steps come from one
five-arm matrix (`runs/lad2-*`); a second five-arm matrix (`runs/w6-*`) priced the
CUDA graph capture levers discussed below, and its baseline arm re-measured the
step-5 config and reproduced it to within 0.16s.

| # | change added | median ready | the 3 runs | Δ | phase that moved |
|---|---|---|---|---|---|
| 1 | **baseline** — vLLM defaults: `spawn`, no writable `__pycache__`, `--load-format auto` | [**90.65s**](reports/32b-step1-baseline-spawn.txt) | 85.08 / 90.65 / 91.22 | — | imports 37.6s, weight load 32.2s |
| 2 | `PYTHONPYCACHEPREFIX=/cache/pycache` | [**68.34s**](reports/32b-step2-pycache.txt) | 63.41 / 68.34 / 68.98 | −22.31s | imports 37.6 → 16.6s |
| 3 | `VLLM_WORKER_MULTIPROC_METHOD=fork` | [**63.37s**](reports/32b-step3-fork.txt) | 58.42 / 63.37 / 63.82 | −4.97s | imports 16.6 → 10.5s |
| 4 | `--load-format fastsafetensors` | [**47.64s**](reports/32b-step4-fastsafetensors.txt) | 47.51 / 47.64 / 47.65 | −15.73s | weight load 32.4 → 17.6s |
| 5 | `cs_fst` patch: `nogds=True`, `max_threads=8`, `bbuf_size_kb=32768` | [**43.30s**](reports/32b-step5-cs-fst-patch.txt) | 42.85 / 43.30 / 44.38 | −4.34s | weight load 17.6 → 13.7s |

**90.65s → 43.30s, −47.35s (−52%).** Only step 5 needs patched code; steps 2-4 are
configuration. Compile time is 0.21-0.25s in every arm, so nothing here is a
recompile artifact.

The next phase down, `cudagraph capture` at 11.4s, is deliberately **not** a step:
every configuration lever that shrinks it degrades steady-state inference. See
[CUDA graph capture](#cuda-graph-capture-114s-that-configuration-cannot-honestly-remove)
for the measurements, and
[the solution space](#the-solution-space-and-what-each-one-costs) for every option
explored against all three remaining phases — adopted, rejected, and closed —
sorted by what each one spends.

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
  TP=1 there is exactly one child, so this is the smallest this step can be. At
  TP>1 the step does not exist: CUDA is already initialised in the launching
  process, so vLLM overrides `fork` back to `spawn` and the lever measures
  nothing ([measured](#what-survives-at-tp2)).
* **Weight loading was the ceiling, and storage was never the problem.** GPFS
  delivers 9.30 GiB/s with O_DIRECT at 16 threads and the H100 takes 51.55 GiB/s
  over pinned host memory, but the default loader moved 61.02 GiB at ~2.0 GiB/s.
  `--load-format fastsafetensors` takes that to 3.79 GiB/s and step 5 to
  5.00 GiB/s. The bottleneck was read concurrency, and the flag is obscure enough
  that most deployments never set it.
* **Then fastsafetensors asks for hardware that is not installed** — the `nogds`
  row below.
* **What is left is CUDA graph capture — 11.4s that this harness can measure and
  should not "fix".** vLLM captures one graph per batch size per mode: 51 sizes ×
  2 modes = 102 graphs, **11.4s**, 26% of the step-5 total, in a phase this
  harness used to report as a single opaque `warmup` span. Every knob that
  shrinks it takes the time back out of steady-state inference, so it stays out
  of the table; the measurements are [below](#cuda-graph-capture-114s-that-configuration-cannot-honestly-remove).

Caveats, so the table is read for what it is:

* **Weights are warm in GPFS's pagepool.** A genuinely cold read of this model
  costs ~62s against ~30s warm; the discarded warm-up cycle showed a 63.4s weight
  load. The *deltas* are comparable across arms, but the absolute floor is
  optimistic for a first-ever pull on a node.
* Constants across all arms, set by `manifests/pod-exec.yaml` and not part of any
  step: caches on a PVC, `OMP_NUM_THREADS=16` against a 16-core cgroup limit, and
  `VLLM_ENABLE_STARTUP_PLAN=1` — which persists the memory-profiling result under
  `$VLLM_CACHE_ROOT` so a repeat boot with the same fingerprint skips profiling,
  worth ~0.6s. It helps the baseline, so the reductions above are if anything
  understated. All three are documented in
  [docs/kubernetes.md](docs/kubernetes.md#constants-not-steps).
* **These five reports were re-rendered after a phase-attribution fix**, so their
  per-phase numbers differ from earlier copies while `TOTAL` and `unaccounted`
  are byte-identical. `cuda.synchronize` nests *inside* CUDA graph capture and
  used to win the attribution sweep on depth, billing ~1.7-2.0s of capture cost
  to `device & collectives init`; syncs are now transparent to the sweep and that
  time sits in `cudagraph capture`. Consequence: `device & collectives init`
  reads ~2ms in these runs, not 1.7s. The residual real cost of
  `worker.init_device` (727ms) is fully overlapped by two helper subprocesses
  importing torch, which outrank it cross-process.
  These traces also predate the v2-runner probes, so their `warmup / profile run`
  vs `cudagraph capture` split is not comparable to `runs/w6-*` — the ladder's
  `TOTAL` column is.
* **The capture matrix independently reproduced step 5.** Its `base` arm *is* the
  step-5 config, and landed at 43.46s median against the 43.30s measured in the
  earlier matrix on a different day — 0.16s apart, which is the best evidence
  here that the ladder's absolute numbers are reproducible across days.
* One node, one GPU, TP=1, and readiness is `/health` 200. Reproduce the steps
  with `runs/lad2-*` and the capture measurements with `runs/w6-*`. The same seven
  arms at TP=2 are [below](#what-survives-at-tp2) — two of the four levers do not
  survive.

### What survives at TP=2

The ladder above is TP=1. Re-running the same seven arms on 2×H100 (NVLink, same
node, same model, median of 3, `scripts/tp2-ladder.sh`) answers which levers were
measuring something structural and which were measuring a single-worker artifact.

| # | change added | TP=2 median | the 3 runs | TP=2 Δ | TP=1 Δ | survives? |
|---|---|---|---|---|---|---|
| 1 | **baseline** — `spawn`, no writable `__pycache__`, `--load-format auto` | **94.7s** | 140 / 88.3 / 94.7 | — | — | — |
| 2 | `PYTHONPYCACHEPREFIX=/cache/pycache` | **58.9s** | 55.7 / 58.9 / 59.7 | **−35.8s** | −22.31s | **yes, and it grows** |
| 3 | `VLLM_WORKER_MULTIPROC_METHOD=fork` | 59.6s | 60.0 / 56.2 / 59.6 | +0.7s | −4.97s | **no — vetoed** |
| 3b | `forkserver` (probe) | 57.2s | 56.7 / 61.1 / 57.2 | noise | parity | **no — vetoed** |
| 4 | `--load-format fastsafetensors` | **52.7s** | 54.9 / 52.7 / 51.0 | **−6.2s** | −15.73s | yes, at ~40% |
| 5a | `cs_fst`: `nogds=True` alone | 52.9s | 52.9 / 55.2 / 50.8 | +0.2s | — | **no — structural** |
| 5b | `cs_fst`: + `max_threads=8`, `bbuf_size_kb=32768` | 53.0s | 53.2 / 53.0 / 50.5 | +0.1s | −4.34s (5a+5b) | in-phase only |

**94.7s → 52.7s, −42.0s (−44%)** against TP=1's −52%, and the whole of it is two
levers instead of four. Steps 3 and 3b are off the adopted chain, and because they
are, steps 4-5b were measured with `VLLM_WORKER_MULTIPROC_METHOD=fork` set but
overridden back to `spawn` by vLLM — so the 52.7s is `pycache + fastsafetensors`,
not a four-lever stack.

Why each one moved:

* **`PYTHONPYCACHEPREFIX` scales with worker count, as predicted.** Imports go
  54.4s → 24.6s, **−29.8s**, against −21.0s at TP=1, and in that baseline run
  imports are **61.7%** of the 88.3s boot. The reason is a process count: at TP=1
  the baseline has two interpreters importing torch from scratch (`api_server` and
  the spawned `EngineCore`), at TP=2 it has **four** — `EngineCore` plus one
  `WorkerProc` per rank — each recompiling vLLM's 6781-file import graph, and the
  lever removes that cost from every one of them. This is the one lever that gets
  *better* with parallelism.
* **`fork` and `forkserver` both die on the same veto, and it is not a tuning
  matter.** In **21 of 21** TP=2 runs — every repeat of every arm — CUDA is
  already initialised in the process that launches the engine: `cuda.lazy_init`
  fires **3.4-6.0s before** `engine.core_proc_manager`, inside
  `config.create_engine_config`. In the TP=1 runs traced with the same probe it
  never fires there at all. The two
  consequences are visible directly:
  `_maybe_force_spawn()` logs `Overriding VLLM_WORKER_MULTIPROC_METHOD to 'spawn'
  ... Reasons: CUDA is initialized` (`utils/system_utils.py:157`) in every `fork`
  repeat, and the forkserver probe records
  `forkserver.summary {"served": 0, "declined": ["CUDA is initialized"]}` — it
  armed, preloaded `vllm.v1.engine.core` in a CUDA-free single-threaded process,
  and then handed that context to nobody. Its 57.2s median is step 2's number: the
  ranges overlap (56.7-61.1 against 55.7-59.7) and no child was ever forked from
  it. **The prediction that "at TP>1 one preload serves N workers, which is where
  it should pay" is therefore not confirmed and cannot be tested until the veto is
  removed** — the preload is dead weight, not amortisation.

  The veto has one cause, and it is an accident of import order.
  `scripts/tp-cuda-init-probe.py` wraps `torch.cuda._lazy_init` — the single
  funnel through which a CUDA context is created — and then builds nothing but
  the engine config. At TP=1 it never fires. At TP=2 it fires once, on this chain
  ([transcript](reports/tp-cuda-init-probe-20260903-134530.txt)):

  ```
  EngineArgs.create_engine_config              arg_utils.py:2501
   VllmConfig.__post_init__                    config/vllm.py:1611
    _set_compile_ranges                        config/vllm.py:2078
     PassConfig.flashinfer_max_size            config/compilation.py:200
      default_fi_allreduce_fusion_max_size_mb  config/compilation.py:206
       import ...fusion.allreduce_rms_fusion   -- for one constant table
        import flashinfer.jit                  allreduce_rms_fusion.py:90
         FLASHINFER_WORKSPACE_DIR = _get_workspace_dir_name()   jit/env.py:161
          CompilationContext()                 jit/env.py:149
           torch.cuda.get_device_capability()  compilation_context.py:105
            torch.cuda._lazy_init()   <-- CUDA context, in the API server
  ```

  `flashinfer_max_size` returns early unless `world_size` is one of
  `[2, 4, 8, 16]` (`config/compilation.py:195-196`), and **that guard is the
  entire difference between TP=1 and TP=2.** Past it, vLLM imports
  `allreduce_rms_fusion` to read one constant (`FI_ALLREDUCE_FUSION_MAX_SIZE_MB`),
  that module imports `flashinfer` at module scope, and flashinfer's `jit/env.py`
  calls `torch.cuda.get_device_capability()` at import time merely to *name a
  cache directory*. Nothing on that path needs a CUDA context: the frontend
  acquires one (0.43s of it) several seconds before it decides how to start the
  engine, and thereby loses both `fork` and `forkserver` for the rest of the
  boot. Two plausible culprits
  were eliminated on the way — `has_flashinfer()` uses `find_spec` specifically to
  avoid this, and vLLM's own `get_device_capability()` resolves to the NVML
  implementation (`platforms/cuda.py:738`), which never touches the runtime. So it
  is flashinfer's import-time hazard
  ([already noted here](#what-is-patched-and-what-should-be-upstreamed)), reached
  through a vLLM import performed for a constant, behind a TP>1-only guard.
* **`fastsafetensors` survives, at 40% of its TP=1 win, because half the work was
  already parallel.** Weight load is 32.2s at TP=1 baseline but **14.3s** at TP=2:
  each rank reads its own shard concurrently, so the default loader's read
  concurrency problem is already half-solved by tensor parallelism. The flag takes
  14.3s → **8.81s**, −5.5s, against −14.8s at TP=1. The lever is real, the headroom
  is smaller.
* **`nogds` is dead at TP>1 by construction, and the measurement confirms it
  exactly.** `weight_utils.py:1057` computes `nogds = pg.size() > 1`, so at TP=2
  vLLM already passes `nogds=True` and the patch's headline lever has nothing left
  to fix: 8.81s → 8.67s in the weight phase, −0.14s, inside noise. The two tuning
  knobs still do something — 8.67s → **8.04s**, −0.63s — but that is under the
  arm's own 0.7s spread and it does not survive to the total (52.7 → 53.0). At
  TP>1, the `cs_fst` patch is worth keeping only for `max_threads`/`bbuf_size_kb`,
  and only if the phase number is what you are optimising.

Caveats specific to these runs:

* The baseline arm's first repeat is a 140s outlier — 11.0s of cold Inductor
  compile plus a cold weight read. The median is taken over three, and the other
  arms' r1 values show no such effect.
* `device & collectives init` is **472ms** at TP=2, so NCCL/communicator setup is
  not a tensor-parallel tax worth optimising: 0.9% of the boot.
* **The two ladders were not run at the same context length — checked, and it
  does not change the conclusions.** The seven arms above passed no
  `--max-model-len`, so they took this model's default 40960, while the published
  TP=1 ladder used 8192. Every TP=2-vs-TP=2 delta in the table is unaffected (all
  seven arms share one command), but the cross-TP phase comparisons needed
  settling, so both endpoints were re-run at TP=2 with `--max-model-len 8192`
  (`runs/tp2m8k-*`, median of 3). At the matched context length, against the TP=1
  runs traced with the *same* probe revision (`runs/cc-base-c*`, also 8192):

  | phase, median of 3 | TP=1 @ 8192 | TP=2 @ 8192 | TP=2 Δ |
  | --- | --- | --- | --- |
  | `cudagraph capture` | 11.1s (10.9 / 11.4 / 11.1) | **8.74s** (8.44 / 8.82 / 8.74) | −2.4s, −21% |
  | `weight load` | 15.8s | **7.71s** | −8.1s, −51% |

  So `cudagraph capture` really is cheaper at TP=2, and the earlier `~8.7s vs
  11.4s` reading survives the check almost exactly: capture at TP=2 is 8.74s at
  8192 against ~8.7s at 40960, i.e. **capture barely depends on context length at
  all**, which is why the mismatch turned out not to matter here. It shrinks by
  −21%, not the −50% that "each rank captures half a model" would predict, and
  that gap is the point: the per-size launch overhead across 51 sizes × 2 modes
  does not divide by rank count even though the per-rank forward does. `weight
  load` halves cleanly, because each rank reads its own shard.
* The `−44% vs −52%` headline is a within-TP ratio on each side, so each is
  internally valid, but the two totals are not the same workload — and at 8192
  the TP=2 endpoint lands at **51.8s** (51.7 / 54.6 / 51.8) against 52.7s at
  40960, so the headline is not sensitive to it either.
* Do not compare `TOTAL` or `python imports` between the two right-hand columns
  above: `cc-base` is a TP=1 arm with `fastsafetensors` but *without* the import
  levers (its imports are 22.5s where the full TP=1 stack reaches 10.2s), so it
  is a valid anchor for capture and weight load — phases the import levers do not
  touch — and not for the boot total.
* Re-anchoring in the other direction — TP=1 at 40960 — is not possible on this
  node, which is worth stating on its own. Qwen3-32B on **one** H100 leaves
  7.92 GiB for KV where the full 40960 context needs 10.0 GiB, so the engine
  refuses to start and reports an estimated maximum of 32432. Full-context 32B at
  TP=1 is a capacity limit, not a cold-start one, and it is the reason the
  like-for-like comparison had to be made at the smaller context.
* Two runs in this set were discarded rather than reported: an orphaned in-pod
  `coldstart-run.sh` left over from a cancelled job was competing for the same
  GPUs, which shows up as `Free memory on device cuda:0 ... less than desired GPU
  memory utilization` or as absurd readiness times (331s, 966s). Cancelling the
  local client does not stop the pod-side script, so `coldstart-run.sh` now
  refuses to start when a sibling is live (`--allow-concurrent` overrides) and
  gates each repeat on the GPU actually being free rather than on a fixed sleep.
* Reproduce with `scripts/tp2-ladder.sh -n <ns>` (renders in
  `runs/tp2-*/report.txt`), and the matched-context pair with
  `scripts/tp2-ladder.sh -n <ns> --prefix tp2m8k --max-model-len 8192 --arm s1-baseline --arm s5b-fsttuned`.

### CUDA graph capture: 11.4s that configuration cannot honestly remove

After step 5, `cudagraph capture` is the second-largest phase — **11.4s of 43.4s,
26%** — and three stock CLI levers shrink it. All three were measured in a full
interleaved median-of-3 matrix. **None is adopted, and none should be**: each one
buys startup time by making steady-state inference worse, and the amount worse is
something this harness cannot measure at all.

| arm | median ready | the 3 runs | Δ vs `base` | why it is not a step |
|---|---|---|---|---|
| `base` — step-5 config, 51 sizes, `FULL_AND_PIECEWISE` | 43.46s | 43.42 / 43.46 / 44.07 | — | — |
| 10 capture sizes ([report](reports/32b-cudagraph-sizes-experiment.txt)) | 34.32s | 34.01 / 34.32 / 34.90 | −9.14s | fewer graphs means batches pad up to the next captured size — see below |
| `cudagraph_mode=FULL_DECODE_ONLY`, 51 sizes | 35.95s | 35.46 / 35.95 / 36.46 | −7.52s | drops the PIECEWISE pass outright, so the cost lands on mixed prefill+decode batches under load |
| `--kernel-config` with flashinfer autotune, cutedsl and JIT warmups off | 43.09s | 42.27 / 43.09 / 43.35 | −0.37s | the only one with no steady-state cost, but 0.37s against a ~0.9s spread does not earn a row |
| 10 sizes **and** the kernel warmups off | 34.14s | 33.86 / 34.14 / 35.31 | −9.32s | 0.18s better than the size list alone, well inside both arms' spread |

**Why the size list is not a free lever.**
`_compute_bs_to_padded_graph_size` (`v1/cudagraph_dispatcher.py:72`) pads a batch
**up** to the next captured size. With the default 51-size ladder the padding is
at most ~8 tokens. With 10 sizes, a batch of 129 runs as a batch of 256 — roughly
twice the decode work for the same output, on every batch that lands in a gap, for
the entire life of the server. Trading a one-time 9s against that is a bad trade
for any deployment that serves more than a few minutes of traffic.

**And this harness is structurally blind to it.** `first_token − health_200` is
43ms in all five arms, because batch size 1 is captured in every one of them. A
single batch-1 request cannot observe padding waste. So the `first_token` parity
across these arms is *not* evidence that the cost is small — it is evidence that
we did not measure it. Pricing it needs a throughput benchmark under a realistic
batch-size distribution, which is out of scope here.

**What the measurement is good for** is sizing the prize and locating it
precisely, which is what makes it an upstream argument rather than an operator
workaround:

* capture cost is **linear in the number of sizes** (~124ms per PIECEWISE size),
  so the 51-size default is a deliberate 11.4s and nobody is told;
* it is **not** `torch.compile` — that is 0.24s on an AOT cache hit;
* it **is** the PIECEWISE forward: 124ms per size against 16ms for FULL — but
  **not** at identical graph counts, which is what this README said until the
  captures were counted without a duration floor. There are **3366** torch
  `CUDAGraph`s, not 102: 51 sizes × (65 piecewise pieces + 1 FULL), because
  `splitting_ops` cuts the graph at each of the 64 attention layers. PIECEWISE
  pays torch's per-graph prologue 65 times per size, which is 46ms of the 133ms;
  net of it the gap is **4.6x**, not 7.7x;
* and **a third of the phase is that prologue** — `graph.__enter__`/`__exit__`
  total **3.84s**, of which the 5ms-floored spans in the committed traces show
  only 1.73s. Each of the 3366 does a `torch.cuda.synchronize()` and an
  `empty_cache()`. Doing one of each *after* the loop instead takes the phase from
  11.9s to **7.7s** and gives the memory back — steady-state device memory is
  base's to 0.00 GiB, measured against NVML. The same intervention measured end to
  end is **−4.0s of time-to-ready**, negative in every paired pass
  ([detail](docs/concurrent-cudagraph-capture.md#4b-the-prerequisite-is-the-result-40s)).
  It is the largest startup win in this repo that needs no redesign, and it turned
  up while pricing the prerequisites of something else.

So the 11.4s should be attacked by making capture cheaper or reusable — hoisting
that prologue, a faster piecewise path,
[capture parallelism](docs/concurrent-cudagraph-capture.md), capturing **on
demand** so only the shapes a workload actually reaches are ever built
([docs/on-demand-cudagraph.md](docs/on-demand-cudagraph.md)), or persisting graphs
across boots ([Foundry](docs/foundry.md)) — not by asking operators to shorten the
list.

## The solution space, and what each one costs

At 43.30s, three phases are 85% of what is left: **weight load 13.3s**,
**CUDA graph capture 11.4s**, **Python imports 10.2s**. Everything this repo has
explored attacks one of those three, and the useful way to sort the options is not
by how many seconds they save but by **what they spend to get them**. Startup time
is cheap to buy with someone else's budget — steady-state throughput, host RAM, a
resident process, an operator's attention — and most of the levers that look
attractive are doing exactly that.

| what it spends | options | verdict |
|---|---|---|
| **nothing** — a stock env var or flag | `PYTHONPYCACHEPREFIX`, `fork`, `--load-format fastsafetensors` | **adopted**: −43.01s of the −47.35s |
| a monkey-patch, with a real upstream fix behind it | `cs_fst` (`nogds`, `max_threads`, `bbuf_size_kb`) | **adopted**, opt-in: −4.34s |
| **steady-state inference** | shorter `cudagraph_capture_sizes`, `FULL_DECODE_ONLY` | **rejected** — see below |
| engineering inside vLLM | hoisting the capture prologue (**−4.0s**), on-demand capture, a cheaper PIECEWISE forward, concurrent capture, the six warmup findings | **the real answer**, and not ours to ship |
| host RAM and a resident process | sleep mode, `+ cuda-checkpoint`, `+ --device-map` | **works today** — but it is warm standby, not cold start |
| an external dependency and a driver-welded artefact | Foundry graph persistence | worth evaluating |
| nothing that exists | serializing the graph pool for the *next* boot | **closed at the CUDA level** |

### Adopted: the 52% is all in imports and weight load

| solution | Δ | what it costs |
|---|---|---|
| `PYTHONPYCACHEPREFIX=/cache/pycache` | **−22.31s** | Nothing at runtime. The real fix is upstream: `compileall` at image build time, so no operator needs to know the knob exists. |
| `VLLM_WORKER_MULTIPROC_METHOD=fork` | **−4.97s** | Forking a multi-threaded, torch-loaded parent. vLLM silently reverts to `spawn` if CUDA is already initialised, and Python 3.14 moves the Linux default to `forkserver`. This is borrowed time, not a durable win — and at TP=2 the time is already gone: CUDA *is* initialised before the engine launches, so the revert fires on every run and the lever is worth 0s ([measured](#what-survives-at-tp2)). |
| `--load-format fastsafetensors` | **−15.73s** | Nothing. Storage was never the bottleneck — read concurrency was, and the flag is obscure enough that most deployments never set it. |
| `cs_fst`: `nogds=True`, `max_threads=8`, `bbuf_size_kb=32768` | **−4.34s** | A monkey-patch. At TP=1 vLLM always asks for GPUDirect Storage without checking whether it exists, and the two tuning knobs are not plumbed through vLLM at all. |
| `forkserver` (probe only) | **parity** | Nothing, and it buys nothing at TP=1 — one child means nothing to amortise. At TP=2 it buys nothing either, for a different and more interesting reason: the same "CUDA is initialized" veto that kills `fork` makes the probe decline, so the preload runs and serves **zero** children ([measured](#what-survives-at-tp2)). Whether one preload can amortise across N workers is still untested — removing the veto is the prerequisite. |

### Rejected: buying startup with steady-state inference

Three stock levers shrink the 11.4s capture phase. All three were measured in a
full interleaved median-of-3 matrix, and the two that work are the two that make
serving worse
([detail](#cuda-graph-capture-114s-that-configuration-cannot-honestly-remove)).

| lever | Δ | what it spends |
|---|---|---|
| 10 capture sizes instead of 51 | −9.14s | Every batch that lands in a gap pads **up** to the next captured size, for the life of the server: a batch of 129 runs as 256, roughly twice the decode work for the same output. |
| `cudagraph_mode=FULL_DECODE_ONLY` | −7.52s | Drops the PIECEWISE pass outright, so the cost re-appears on mixed prefill+decode batches under load. |
| kernel warmups off (flashinfer autotune, cutedsl, JIT) | −0.37s | The only one with no steady-state cost — and 0.37s against a ~0.9s spread does not earn a row. |

The honest part of this result is that **this harness cannot price what those two
arms cost.** `first_token − health_200` is 43ms in every arm, because batch size 1
is captured in all of them, and a single batch-1 request cannot observe padding
waste. The parity is evidence that we did not measure it, not evidence that it is
small.

### The real answer: make capture cheaper or reusable

None of these is a configuration change, and none has an operator-visible
trade-off. They are the upstream asks
([detail](#warmup-and-cuda-graph-capture-five-findings-with-no-lever)).

| change | worth | status |
|---|---|---|
| Capture **on demand**, so only the shapes a workload reaches are built | **11.4s** off readiness, 3.3s of it deleted rather than deferred | Needs a vLLM change, and has to fix a latent silent-corruption bug first ([design](docs/on-demand-cudagraph.md)). The first request touching an uncaptured shape pays for it — which is the trade, and it is a much better one than padding every batch forever. |
| **Hoist torch's per-capture prologue out of the capture loop** | **−4.0s of the 11.4s, measured end to end**, and it costs no memory | The only lever here already quantified against a paired control, and the cheapest: `torch.cuda.graph.__enter__` runs a `synchronize()` and an `empty_cache()` before **each of 3366** captures. Doing one of each after the loop instead takes the phase 11.9s → 7.7s with steady-state GPU memory unchanged. 21 interleaved runs across three matrices; the phase delta is negative in all twelve paired passes, and the end-to-end median is −4.00s ([matrix](docs/concurrent-cudagraph-capture.md#4b-the-prerequisite-is-the-result-40s)). |
| Make the PIECEWISE forward cheaper | **133ms per size against 17ms for FULL**, of which 46ms is the prologue above paid 65 times | A 4.6x residual gap, not the 7.7x this table used to claim: PIECEWISE builds 65 graphs per size against FULL's one, so the graph counts were never identical. What is left after the prologue is dispatch overhead. |
| Overlap capture across sizes (**[concurrent capture](docs/concurrent-cudagraph-capture.md)**) | **~0.5s**, and only after the row above | No longer unquantified. Concurrent capture *works* — bit-exact at 2, 4 and 8 threads — but it needs `capture_error_mode="thread_local"` and one mempool per thread, tops out at **1.10-1.21x on two threads** (8 threads is 0.60x, i.e. slower than serial), costs **+1.08 GiB of steady-state GPU memory per thread**, and needs six vLLM changes including one whose failure mode is silently wrong graphs. Feasible, priced, and last in line. |
| Honour `cudagraph_num_of_warmups` on the v2 runner | 2.19s | The knob is read only on the **v1** runner; v2 runs one eager forward per (mode, size) unconditionally. Either honour it or delete it. |
| Gate the tilelang JIT hook on `sys.modules` | 1.00s | Importing tilelang and TVM to install an observability hook that has nothing to watch if tilelang was never imported. No off switch exists. |
| Check the architecture string before importing | 417ms | `kimi_k3_triton_warmup` imports `deepseek_v4` and `compressed_tensors` to reach an `isinstance` check that returns `None` for Qwen3. |
| Say which profile run is skipped | 0s | `VLLM_ENABLE_STARTUP_PLAN=1` logs `Memory profiling will be skipped` and then runs `profile_run()` anyway (2.31s). A documentation bug, but it costs debugging time. |

### Works today, but it is warm standby

This is the one branch that removes the whole 11.4s *and* the 13.3s weight load —
by never paying either again. It is not a cold-start answer: it keeps the process
alive and spends host RAM instead ([measured](docs/sleep-mode.md)).

| solution | round trip | what it costs |
|---|---|---|
| `--enable-sleep-mode` (level 1) | **1.45s** to wake | Frees 69.70 GiB with the captured graphs still replayable, verified byte-identical. Keeps the process and **61.68 GiB** of pinned host RAM, and stops **4.17 GiB** short of releasing the card. The first park costs 26.03s, because that pinned buffer has to be allocated once. |
| `+ cuda-checkpoint` on the residual | **5.09s** | Takes the card to **0 MiB** with the graphs intact. Costs a **75.67 GiB** host image — so on a 128 GiB container, exactly one parked replica — plus a binary that is not in the vLLM image. And Kubernetes still will not reclaim the device: `nvidia.com/gpu: "1"` owns it for the pod's lifetime. |
| `+ --action restore --device-map` | **4.13s**, onto a *different* GPU | A warmed Qwen3-32B migrated between two H100s while another tenant held 76 GiB of the original card, output byte-identical, with `/wake_up`'s 61.68 GiB of **new** allocations following the remap. The price is structural: the remap target must be visible to the process, so the pod must request every GPU it might move between and do its own assignment — which discards the device-plugin isolation model. At TP>1 this is moot until the base case is fixed: a TP=2 checkpoint never completes at all, for reasons that are now understood — see [docs/sleep-mode.md](docs/sleep-mode.md). |
| Level 2 + refill weights on wake | ≈**16s** | The variant that trades RAM for seconds the right way: **6.61 GiB** parked instead of 75.67 GiB — a dozen parked replicas instead of one — for ~12s more unpark, still 2.7x better than a 43.3s boot. Three of its four parts already exist; the missing one is a wake path that runs the weight loader after a level-2 sleep. Today a level-2 wake returns HTTP 200 in 0.248s and the model emits `'!!!!!!!!!!'`. |

### Evaluated and closed

| option | outcome |
|---|---|
| **Foundry** — persist CUDA graphs to disk across boots | The only approach that makes a *warm node* capture nothing at all, and the one thing in this list that attacks the 11.4s without a vLLM change or a live process. Its cost is an external dependency plus an archive welded to one driver version, GPU model and config ([notes](docs/foundry.md)). Numbers are upstream-reported and unverified here. |
| **Serialize the `cuda-checkpoint` snapshot for the next boot** | **Impossible, not merely unimplemented.** Of 95 `cuGraph*` symbols libcuda exports, none produces or consumes bytes, while `cuModuleLoadData` does — CUDA has a serialized format for kernels and deliberately none for graphs. `cuda-checkpoint` has no on-disk format either: its interface is pid-keyed with no output path, and what `--action checkpoint` produces is a GPU-free *process* for a dumper to write out, never a file. Even granting a dumper, the level-1 image is 75.67 GiB and takes ~90s to read at this filesystem's 859 MB/s — longer than a cold boot takes to run ([priced](docs/sleep-mode.md#can-the-snapshot-be-serialized-and-reused-by-a-fresh-vllm)). |

### If you only take one thing from this

The 11.4s of CUDA graph capture is the phase where the temptation to trade is
strongest, and every *available* lever spends steady-state inference to buy it.
The options that do not — capturing on demand, a cheaper piecewise path, persisted
graphs — all require code that does not exist yet. That gap is the argument for
upstream work, and quantifying it precisely is what this harness is for.

## What is patched, and what should be upstreamed

Three of the four wins are **configuration** — no patched vLLM, no patched image.
One is a monkey-patch, and it is the one with a real upstream fix behind it.

| change | how it is applied here | what should happen upstream |
|---|---|---|
| `PYTHONPYCACHEPREFIX` | env var, stock CPython | Nothing in vLLM. The image should ship its own `.pyc`: `python -m compileall` at build time puts them in a read-only layer and no operator has to know this knob exists. |
| `VLLM_WORKER_MULTIPROC_METHOD=fork` | env var, already a supported value | `fork` works but is on borrowed time — it forks a multi-threaded, torch-loaded parent, vLLM silently reverts to `spawn` if CUDA is already initialised, and Python 3.14 moves the Linux default to `forkserver`. vLLM already contains forkserver support (`api_server.py:111-117`) but **rejects the value**: `envs.py:930` declares the choices as `["spawn", "fork"]` (and the annotation at `envs.py:67` agrees), so `get_mp_context()` (`utils/system_utils.py:168`) raises `ValueError`. Making it reachable is those two lines; making it *pay* also needs `forkserver.ensure_running()` moved to the top of `cli/main.py:main()`, because where it sits now only ~1.7s of its ~15s preload overlaps anything. |
| `--load-format fastsafetensors` | stock CLI flag | Nothing to patch. Worth documenting that the win is this large, since the flag is easy to miss. |
| **nothing** — CUDA graph capture is *not* configured away here | measured only, see [above](#cuda-graph-capture-114s-that-configuration-cannot-honestly-remove) | Capturing 51 sizes × 2 modes costs **11.4s** at TP=1 on a 32B model, 26% of time-to-ready, and it is unavoidable without degrading inference. The cost is concentrated in the PIECEWISE forward — 124ms per size against 16ms for FULL at identical graph counts, while torch-level capture is 1.66s of the 11.4s. Four things would move it without an operator trade-off: make the piecewise capture forward cheaper (it is 7.7x FULL for no obvious reason), overlap capture across sizes instead of running 102 forwards serially, capture on demand so only the shapes a workload reaches are built ([docs/on-demand-cudagraph.md](docs/on-demand-cudagraph.md) — 11.4s off readiness, 3.3s of it deleted outright, but it needs a vLLM change and fixes a latent silent-corruption bug first), or persist graphs across boots so a warm node captures nothing at all. The last is what [Foundry](docs/foundry.md) does — and note that snapshotting the graph pool with `cuda-checkpoint` is *not* a shortcut to it: CUDA exposes no graph serialization API at all, and captured kernels embed device addresses reaching into the weights and KV cache, so nothing smaller than the full 68.8 GiB address space is restorable ([why](docs/foundry.md#why-cuda-checkpoint-is-not-a-next-boot-shortcut)). Within a *live* process it is a different story: vLLM's VMM-based sleep mode frees 69.70 GiB and `cuda-checkpoint` then takes the residual to **0 MiB**, all the way back in **5.09s** with the graphs intact ([docs/sleep-mode.md](docs/sleep-mode.md)) — but that keeps the process and ~64 GiB of host RAM, so it is warm standby, not cold start. Serializing that snapshot for the *next* boot is not a `cuda-checkpoint` capability and cannot be made into one: its whole interface is pid-keyed with no output path, and what `--action checkpoint` produces is a GPU-free process (0 `/dev/nvidia*` fds, 0 MiB held) for a dumper to write out, never a file ([priced](docs/sleep-mode.md#can-the-snapshot-be-serialized-and-reused-by-a-fresh-vllm)). What the live snapshot *can* do is move: `--action restore --device-map` brings a parked engine back on a **different GPU** of the same node in **2.67s**, and the 61.68 GiB of new allocations at `/wake_up` follow the remap onto the new card ([measured](docs/sleep-mode.md#restoring-onto-a-different-gpu)) — so a pod can defragment its own devices without re-reading weights or re-capturing graphs, at the cost of having to see all of them. Shortening `cudagraph_capture_sizes` is *not* the fix — it moves the cost to steady state. |
| `nogds=True`, `max_threads=8`, `bbuf_size_kb=32768` | **monkey-patch** — [`coldstart/cs_fst.py`](coldstart/cs_fst.py) wraps `fastsafetensors.parallel_loader.ParallelLoader.__init__`; opt-in via `CS_FST=1` | `weight_utils.py:1057` computes `nogds = pg.size() > 1`, and the comment above it shows why: at TP>1 `cuFileDriverOpen()` would create CUDA contexts on every visible GPU. Availability of GDS is never checked, so at TP=1 vLLM *always* asks for it. There *is* a fallback (`weight_utils.py:1083`), but it needs a `RuntimeError` with `"gds"` in the message — and fastsafetensors degrades internally rather than raising, so the fallback never fires: the `"GDS not enabled"` warning appears in none of our runs. The failed probe is then billed silently to every fresh `EngineCore` — 1.69s of one-time setup plus 0.91s of steady-state throughput. It should key on whether `libcufile` and `nvidia_fs` exist, not on world size. Because it keys on world size, this half of the patch is worth exactly **−0.14s at TP=2** — vLLM already passes `nogds=True` there, so the bug it fixes only exists at TP=1 ([measured](#what-survives-at-tp2)). `max_threads` and `bbuf_size_kb` are not reachable from vLLM at all: not plumbed through, and absent from fastsafetensors' own `LoaderConfig`; they are the only part of the patch that still pays at TP>1, and only inside the weight phase (−0.63s). |

Two hazards deliberately *not* fixed here, both prerequisites for the forkserver
work rather than wins of their own:

* three `flashinfer` modules call `torch.cuda.get_device_capability()` at **import**
  time, which initialises CUDA in the parent — precisely what makes `fork` unsafe.
  At TP=1 nothing on the startup path imports them, so this stays latent. **At
  TP>1 it is not latent: it is what costs the boot both `fork` and `forkserver`**
  ([chain](#what-survives-at-tp2)). `config/compilation.py:206` imports
  `vllm.compilation.passes.fusion.allreduce_rms_fusion` — during *config
  creation*, in the frontend process, behind a `world_size in [2, 4, 8, 16]`
  guard — solely to read the constant `FI_ALLREDUCE_FUSION_MAX_SIZE_MB`; that
  module imports `flashinfer` at module scope; and `flashinfer/jit/env.py:161`
  builds a `CompilationContext()` at import time to name a cache directory, which
  calls `torch.cuda.get_device_capability()`. Two independent fixes, either
  sufficient: move the constant table into a module that does not import
  flashinfer, or have flashinfer name its workspace from NVML rather than the CUDA
  runtime. Both are strictly better than the status quo, in which a process that
  does no GPU work acquires a CUDA context and silently forfeits its choice of
  start method;
* `vllm/third_party/flash_linear_attention` (`tilelang`) has the same hazard, on a
  path 32B does not hit.

The other opt-in probe, [`coldstart/cs_forkserver.py`](coldstart/cs_forkserver.py)
(`CS_FORKSERVER=1`), measures the forkserver ceiling without patching vLLM by
starting the preload at t≈0 and swapping `get_mp_context()`. It is not in the table
above: at TP=1 it lands at parity with `fork`, which is the expected result, and at
TP=2 it declines outright (`served: 0`) for the reason just given — so the ceiling
it exists to measure is still unmeasured, and the flashinfer import is what stands
in the way.

### Warmup and CUDA graph capture: six findings with no lever

`warmup / profile run` was the #2 phase and a single opaque span. Instrumented
(`runs/w6-*`, and see [docs/instrumentation.md](docs/instrumentation.md)),
`compile_or_warm_up_model` is **14.0s**, 80% of it CUDA graph capture — which is
[not configurable away](#cuda-graph-capture-114s-that-configuration-cannot-honestly-remove).
The six findings below are the rest: each is an upstream bug, a missing knob, or
missing accounting, and the first five are the 3.6-3.9s of `warmup / profile run`
that survives every arm of that matrix.

1. **The eager warmup forward before every capture is unconditional on the v2
   runner.** `cudagraph_num_of_warmups` (default 0 at
   `config/compilation.py:643`, forced to 1 at `config/vllm.py:1539`) is read
   only at `v1/worker/gpu_model_runner.py:7067` — the **v1** runner. This build
   selects v2 (`Worker.use_v2_model_runner`), and `gpu/cudagraph_utils.py` runs
   one eager forward per (mode, size) with no test against the knob: 102 forwards,
   **2.19s** of the 11.2s capture phase, unreachable. Either honour the knob on
   v2 or delete it.
2. **Activating the JIT monitor imports tilelang and TVM — 1.00s to install an
   observability hook.** `activate_jit_monitor` → `_setup_tilelang_jit_hook` →
   `import tilelang` → `tvm`, including `tvm.relax.backend.adreno`, a mobile-GPU
   backend. `jit_monitor_mode` is `Literal["warn", "error"]`, so there is no off
   switch; it costs 1.00s cold and still 436ms with warm bytecode. If tilelang is
   not already in `sys.modules`, no tilelang kernel can JIT during inference and
   the hook has nothing to watch — gating `_setup_tilelang_jit_hook` on
   `sys.modules.get("tilelang")` makes it free on every model that does not use
   tilelang, which is most of them.
3. **`kimi_k3_triton_warmup`'s architecture check sits behind its own 417ms
   import.** `_get_kda_layer` runs
   `from vllm.models.kimi_k3.nvidia.kda import KimiK3DeltaAttention` — pulling in
   `vllm.models.deepseek_v4` and `compressed_tensors` — purely to reach an
   `isinstance` check that returns `None` for Qwen3. The gate is
   `enable_jit_warmup`, not the model architecture. Testing the config's
   architecture string before importing costs nothing.
4. **The startup plan does not skip the profile run it says it skips.**
   `determine_available_memory` logs `Memory profiling will be skipped` and then
   calls `self.model_runner.profile_run()` anyway — **2.31s** here — because it
   "still need[s] a profile run which compiles the model for
   max_num_batched_tokens". What `VLLM_ENABLE_STARTUP_PLAN=1` actually skips is
   the `memory_profiling` context and `profile_cudagraph_memory()`, not the
   forward. The log line should say which.
5. **PIECEWISE capture costs 7.7x FULL — and it is not for the same number of
   graphs, which took a floorless count to see.** `cudagraph.capture_begin` in the
   committed traces is floored at 5ms, so it counts *captures over 5ms*, and this
   README read that as 102 graphs. A `CS_CGDETAIL=1` run spans every one:
   **3366 `CUDAGraph`s**, tagged by the line that built them —
   3315 at `compilation/cuda_graph.py:313` and 51 at
   `v1/worker/gpu/cudagraph_utils.py:365`, i.e. 51 sizes × (65 + 1), one FULL
   graph and **65 piecewise pieces** per size, because `splitting_ops` cuts at
   each of the 64 attention layers. Three corrections follow. Torch's per-capture
   prologue is **3.84s** of the phase (34%), not 1.66s. The per-mode cost is not
   "identical to within 1%" — that was the floor talking; the true totals are
   0.83s FULL against 2.37s PIECEWISE. And 46ms of the 133ms PIECEWISE forward is
   that prologue entered 65 times, so the residual dispatch gap is **4.6x**. The
   forward is still nearly flat in batch size (98ms at 1 token, 174ms at 512),
   which is why trimming the size list pays close to linearly.
6. **On the v2 runner the CUDA graph pool is unbudgeted, so its memory comes out
   of headroom rather than out of the KV cache.** `gpu_worker.py:532` asks
   `profile_cudagraph_memory()` how much to reserve for graphs before sizing the
   KV cache; the v1 runner answers by capturing sample graphs
   (`v1/worker/gpu_model_runner.py:6780`), and the v2 implementation is
   `return 0` (`v1/worker/gpu/model_runner.py:843`, "It is TBD whether we keep
   this API or not"). vLLM then logs what the pool cost *after* capturing it —
   **1.84 GiB** at 32B — having subtracted nothing for it beforehand, so it is
   spent out of whatever `gpu_memory_utilization` left over. Anything that adds
   pools adds to it just as invisibly: rotating 2 and 4 mempools raises
   steady-state device memory by **+1.08 and +3.21 GiB**, confirmed against NVML
   independently of vLLM's own accounting
   ([matrix](docs/concurrent-cudagraph-capture.md#4-what-it-costs-on-the-real-model)).
   Same v1/v2 shape as finding 1: on v2 the memory accounting is missing, not just
   the knob.

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
  Mode B, cache volumes, the env vars pinned across every arm, OpenShift/SCC
  notes, troubleshooting.
* **[docs/experiments.md](docs/experiments.md)** — the experiment matrix and how
  to run a defensible comparison.
* **[docs/on-demand-cudagraph.md](docs/on-demand-cudagraph.md)** — the design
  exploration for taking the 11.4s `cudagraph capture` phase off the readiness
  path by capturing lazily on first use instead of eagerly at startup: what vLLM
  already supports, the silent-corruption bug that blocks it, and what it is worth.
* **[docs/concurrent-cudagraph-capture.md](docs/concurrent-cudagraph-capture.md)** —
  can the 102 captures run at the same time? Measured on bare torch: yes,
  bit-exact, but only via `capture_error_mode="thread_local"` and one mempool per
  thread, and the ceiling is **1.10-1.21x on two threads** — eight threads is
  *slower* than serial, and it is neither the GIL nor the caching allocator. On the
  real model each extra mempool costs **+1.08 GiB** of steady-state GPU memory,
  linear in threads and unbudgeted on the v2 runner. The document also carries what
  pricing the *prerequisites* turned up, which is worth more than the concurrency:
  a floorless census showing **3366 CUDA graphs, not 102**, and the per-capture
  `synchronize()` + `empty_cache()` in `torch.cuda.graph.__enter__` that together
  cost **4.0s of time-to-ready** and can be hoisted out of the loop.
* **[docs/sleep-mode.md](docs/sleep-mode.md)** — the one lever that already
  works today: `--enable-sleep-mode` frees 69.70 GiB and wakes in **1.45s** with
  the captured CUDA graphs still replayable (verified byte-identical output), and
  layering `cuda-checkpoint` on the ~3-4 GiB residual takes the card to **0 MiB**
  and back in **5.09s**. Not a cold-start fix — hibernation for a warm replica —
  plus what it costs, why the remaining blocker is Kubernetes rather than CUDA,
  and a silent level-2 footgun. Also: the snapshot can be restored onto a
  **different GPU** on the same node — a warmed Qwen3-32B engine migrated between
  two H100s in **4.13s** while another tenant held 76 GiB of the original card,
  with the 61.68 GiB of *new* allocations at wake following the remap and output
  byte-identical — subject to three measured `--device-map` rules, one of which
  (the target must be visible to the process) forces the pod to see every GPU it
  might move between. And it prices the follow-on question of whether the
  snapshot can be *serialized* and reused by a fresh vLLM: it cannot with
  `cuda-checkpoint`, which has no on-disk format at all, and the CRIU route is
  only arithmetically interesting in its thin level-2 form (**6.61 GiB** image,
  ~25s vs 43.3s) against a level-1 image of **75.67 GiB** that takes longer to
  read than a cold boot takes to run. That same thin shape is the more useful
  variant of the *live* composition too: parking at level 2 and refilling weights
  on wake would cost ~16s instead of 4.13s but hold 6.61 GiB of host RAM instead
  of 75.67 GiB — the difference between one parked replica and a dozen on a
  128 GiB container. It needs one path vLLM does not have yet.
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
