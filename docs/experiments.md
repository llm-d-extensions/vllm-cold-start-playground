# Running experiments

## The shape of a defensible run

1. **Deploy once, measure many.** `make deploy` puts the pod up; every
   measurement after that is a `kubectl exec`. Node, GPU, kernel, image and page
   cache stay fixed, so a difference between two runs is attributable to the one
   thing you changed.
2. **Change one variable at a time.** Each run's `meta.json` records the vLLM and
   torch versions, GPU model, kernel, the exact command line, and whether the
   compile or HF caches were cleared. The report prints the cache state it found
   at `t0`. Use them.
3. **Repeat.** `REPEAT=3` runs three iterations into `<run-id>-r1..r3`. The first
   iteration after a fresh pod is systematically different from the rest (page
   cache, GPU state), so quote a range, not a single number.
4. **Take a `--no-probe` control** once per environment, to quantify what the
   instrumentation itself costs.

```bash
make deploy NS=my-ns
make run NS=my-ns MODEL=Qwen/Qwen2.5-1.5B-Instruct REPEAT=3
```

## The matrix

Each row is a variable worth isolating, with the phases it should move. If a
change moves a phase it should not, that is the interesting result.

| variable | how | expect it to move |
|---|---|---|
| **compile cache cold vs warm** | `--cold-compile` on one run, not the other | `torch.compile`, `cudagraph capture` |
| **weights cold vs warm** | `--cold-hf`, or a fresh node | `network: weight download`, `weight load` |
| **page cache** | fresh node vs repeat run on the same pod | `weight load` (the report flags when reads came from page cache) |
| **hub offline** | `HF_HUB_OFFLINE=1` in the pod env | `network: hub metadata`, `config & tokenizer resolve` |
| **eager vs compiled** | `--enforce-eager` | `torch.compile` and `cudagraph capture` should go to ~0; `first_token` may get worse |
| **CUDA graph capture sizes** ⚠️ *diagnostic only* | `--compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,16,32,64,128,256,512]}'` | `cudagraph capture` — linear in list length (~124ms per PIECEWISE size at 32B, 11.4s → 2.5s cutting 51 sizes to 10). **Not a configuration to ship**: batches pad **up** to the next captured size, so this buys startup with steady-state throughput, and no edge this harness records can see the cost. Run it to size the phase, not to tune a deployment |
| **CUDA graph mode** ⚠️ *diagnostic only* | `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'` | `cudagraph capture` — drops the PIECEWISE pass outright, −7.5s at 32B. Same caveat: the cost lands on mixed prefill+decode batches under load, which this harness does not generate |
| **kernel warmups** | `--kernel-config '{"enable_flashinfer_autotune":false,"enable_cutedsl_warmup":false,"enable_jit_warmup":false}'` | `warmup / profile run` — only −0.4s at 32B, at the edge of the noise floor. Also removes the 169-417ms `kimi_k3_triton` import on models that are not Kimi |
| **tensor parallelism** | `--tensor-parallel-size 2,4,8` | `ipc handshake`, `device & collectives init`, `python imports` (once per worker) |
| **CPU budget** | change the pod's cpu limit/request | `python imports`, `weight load`, `torch.compile`; watch the throttling finding |
| **storage tier** | PVC `storageClassName` (NVMe / network / object-store cache) | `weight load` throughput |
| **weight loader** | `--load-format fastsafetensors` | `weight load` — −12.4s at 32B; check the progress-bar string, not the flag |
| **`/dev/shm` size** | shrink it below what TP needs | `ipc handshake` — this is a classic silent stall |
| **vLLM version** | pinned image digest | anything; version is one of the strongest determinants |
| **model size / dtype** | `--model`, `--dtype` | `weight load` roughly linearly; compile time much less so |
| **`--max-model-len`, KV cache** | serve args | `kv cache alloc`, `warmup / profile run` |
| **probe on/off** | `--no-probe` | nothing, ideally — that is the point of the control |
| **engine start method** | `--env VLLM_WORKER_MULTIPROC_METHOD=spawn\|fork`, or `--env CS_FORKSERVER=1` | `python imports` in EngineCore, `ipc handshake` |
| **bytecode cache** | `--env PYTHONPYCACHEPREFIX=/cache/pycache` | `python imports`, in *every* interpreter — 9.7s each on the stock image |
| **deferred imports** | patch a module-level import to fire on first use | `python imports` — but check the module count actually dropped |

Cold-compile and warm-compile back to back in one pod is the single most
informative pair to start with:

```bash
scripts/run-experiment.sh -n my-ns --run-id cold --cold-compile \
    -- --model meta-llama/Llama-3.1-8B-Instruct
scripts/run-experiment.sh -n my-ns --run-id warm --no-apply \
    -- --model meta-llama/Llama-3.1-8B-Instruct
make compare BASE=runs/cold NEW=runs/warm
```

Every row above is a flag or an env var on the same upstream image. Some
interesting comparisons are not: an arm that needs a patched engine, a preloaded
shared library, or a different image is a separate build and belongs in its own
run set, never as a variant of a baseline measured against a different image.
[foundry.md](foundry.md) is the worked example: CUDA graphs restored from disk,
including which phases it should move and what to check when it does not.

### The engine start method, arm by arm

`vllm serve` pays the engine's imports twice. `api_utils.py:164` forces `spawn`
whenever `VLLM_WORKER_MULTIPROC_METHOD` is unset, so `EngineCore` is a fresh
interpreter re-importing torch and vllm: 15.5s of a 55.3s run on
`20260901-191644-r2`. `fork` skips that but forks a multi-threaded,
torch-loaded parent. `forkserver` is the safe version of the same trick, and
`CS_FORKSERVER=1` makes it reachable — vLLM's own `envs.py:937` rejects the
value, so the feature at `api_server.py:108` cannot currently be switched on at
all (see `coldstart/cs_forkserver.py`).

```bash
# `fork` is listed twice on purpose: it is the baseline, and it drifts. See below.
for arm in fork spawn forkserver-late forkserver-early forkserver-upstream fork-again; do
  case $arm in
    fork|fork-again)  e=(--env VLLM_WORKER_MULTIPROC_METHOD=fork) ;;
    spawn)            e=(--env VLLM_WORKER_MULTIPROC_METHOD=spawn) ;;
    forkserver-late)  e=(--env CS_FORKSERVER=1 --env CS_FORKSERVER_MODE=late) ;;
    forkserver-early) e=(--env CS_FORKSERVER=1 --env CS_FORKSERVER_MODE=early) ;;
    # what api_server.py:110 preloads today -- the frontend class, not the
    # thing that gets forked. Sized in "Finding 3, sized" below.
    forkserver-upstream)
                      e=(--env CS_FORKSERVER=1 --env CS_FORKSERVER_MODE=early
                         --env CS_FORKSERVER_PRELOAD=vllm.v1.engine.async_llm) ;;
  esac
  scripts/run-experiment.sh -n my-ns --run-id "$arm" --repeat 3 --no-apply \
    "${e[@]}" -- --model Qwen/Qwen2.5-1.5B-Instruct
done
```

Compare **median of 3**, not best of 3, and compare arms measured *adjacently* —
same container, same sitting, one block after the other. Spread inside a single
block is ~0.7-2.7s; the same `fork` config drifted 35.60s -> 37.95s across 40
minutes of one session; and it measured 39.16s median the day before. So drift
(~2.4s within a session, ~3.5s across days) is larger than most effects worth
looking for, and "same sitting" is not a strong enough rule on its own. Re-run
your baseline arm *next to* the new one rather than reaching for the block you
measured half an hour ago; a stale baseline is how a real 6s win gets reported as
noise, or a nonexistent one as a result — see the warning under the results
table, where exactly that happened.

The antidote is not more repetitions, it is measuring inside one process tree.
`forkserver.launch_child` and the fork-to-`engine.core_proc_init` gap are
properties of a single run and do not drift; prefer them for anything smaller
than a couple of seconds.

Expect `forkserver-early` to land at **parity with `fork`, not ahead of it** — at
TP=1 there is one child, so one preload cannot beat one free fork. The ~15s is
won against `spawn`, which is the actual upstream default, and the margin grows
at TP>1 where one preload amortises across N workers and `fork` is least safe.
`forkserver-late` is the control that shows placement is the whole lever: it
reproduces upstream's `ensure_running()` position and should land at ≈`spawn`.

#### Measured: H100 80GB, Qwen2.5-1.5B-Instruct, vllm 0.28.0 / torch 2.13.0+cu130

Six blocks of 3 in one container, in the order shown. The `fork` baseline was
measured twice, first and last, and it moved — so each candidate is read against
the baseline block *next to it*, not against a single number.

| block | arm | runs (s) | median | vs adjacent `fork` |
|---|---|---|---|---|
| 00:13 | `spawn` | 50.91 / 49.97 / 52.71 | **50.89** | +15.3 |
| 00:16 | `fork` | 35.11 / 35.80 / 35.60 | **35.60** | — |
| 00:19 | `forkserver-late` | 50.42 / 51.72 / 49.19 | **50.42** | +14.8 |
| 00:23 | `forkserver-early` | 36.28 / 35.55 / 37.94 | **36.26** | **+0.66** |
| 00:53 | `forkserver-upstream` | 38.95 / 38.80 / 39.83 | **38.80** | +0.85 |
| 00:56 | `fork` again | 38.65 / 37.95 / 36.90 | **37.95** | — |

**`forkserver-early` reaches parity with `fork`** — +0.66s, inside a single
block's own spread — while `forkserver-late` lands on `spawn` at +14.8s. Placement
is the whole lever, as predicted, and the totals are corroborated by a span that
cannot drift: `forkserver.launch_child` is **14.9ms** early against **13,660ms**
late. Same mechanism, same preload list, ~900x from placement alone.

**The child's re-import cost**, fork to `engine.core_proc_init` — measured within
one process tree, so session drift cannot reach it. Median of 3, with the observed
range, because at 50ms the run-to-run spread is a real fraction of the number:

| arm | child created by | → `core_proc_init` | range (n=3) |
|---|---|---|---|
| `fork` | fork | 11ms | 10.8 - 10.8 |
| `forkserver-early` | fork | 55ms | 45.0 - 57.7 |
| `forkserver-upstream` | fork | 364ms | 353.5 - 383.8 |
| `spawn` | exec | 13,970ms | 13,821 - 14,661 |

The `forkserver-early` and `forkserver-upstream` ranges do not overlap, which is
what makes the ~309ms between them a result rather than a coincidence.

**The warning, and it cost me a result.** The `fork` baseline drifted +2.35s
(35.60 → 37.95) across 40 minutes of one session on a byte-identical config —
40 minutes that happened to contain three 600s timeouts from the failed
`gpu_worker` arm. Read across the session, `forkserver-upstream` looks 2.7s
slower than `forkserver-early`. Read against the baseline block beside it, it is
0.85s slower; read from the drift-immune span, 0.31s. **The 2.7s was almost
entirely drift**, and reporting it would have been a fabricated result with three
consistent repetitions behind it. Sub-second effects require adjacent blocks and
a within-tree span; nothing else here is trustworthy at that scale.

The drift is diagnosable, which is worth knowing before you blame the GPU: the
child's *import wall* grew with it, 6.8s in the 00:16 block to 7.5s in the 00:56
block, and `vllm.v1.worker.gpu_worker` alone went 4.90s -> 5.40s. Imports are
CPU-bound, so this was CPU contention on a 224-core node under a 16-core cgroup
limit — a neighbour, not thermal throttling and not anything about vLLM. Check the
child's import wall across blocks before trusting a total.

#### Finding 3, sized: the upstream preload list names the wrong module

`api_server.py:110` preloads `vllm.v1.engine.async_llm` — the *frontend* class.
What actually gets forked is `EngineCoreProc`. The wrong list is not simply
smaller: that forkserver loads *more* in total (6012 modules against 5874), and
its child still ends up importing 23 modules the `vllm.v1.engine.core` preload
had already covered. Among them, all worker-side:

```
vllm.v1.worker.utils            vllm.v1.worker.gpu.attn_utils
vllm.v1.worker.gpu.pcp_manager  vllm.v1.worker.gpu.mm.lora
vllm.lora.layers                vllm.lora.worker_manager
vllm.v1.sample.ops.topk_topp_sampler
```

Child import wall 7.06s → 7.71s; pre-init gap 55ms → 364ms. At TP=1 that is worth
a few hundred milliseconds, and the structural point is worth more than the
number: the preload should name what is forked. At TP>1 those 23 modules are
re-imported by *every* worker, so the same one-line fix scales with N.

#### The arm that fails closed, and why the ceiling turned out to be movable

As shipped, preloading the worker fails closed — loudly, which is the good
outcome:

```
CS_FORKSERVER_PRELOAD=vllm.v1.engine.core,vllm.v1.worker.gpu_worker
  -> RuntimeError: Cannot re-initialize CUDA in forked subprocess.
     (3/3 runs, forkserver.server_summary cuda_initialized=true)
```

An earlier version of this section concluded from that the ~7s the forked child
still pays is "a hard ceiling rather than a tuning knob", because every module in
`vllm.v1.worker.gpu_worker`'s graph initializes CUDA at import. The observation
reproduces; the conclusion was wrong.

Auditing 34 candidate modules, one per fresh interpreter, 15 are safe to preload
and 17 poison a fork — and **all 17 trace to the same frame**:
`flashinfer/compilation_context.py:105`, reached from `flashinfer/jit/env.py:161`,
a module-level `FLASHINFER_WORKSPACE_DIR = _get_workspace_dir_name()` that builds
a `CompilationContext` which loops `torch.cuda.get_device_capability(device)` to
form a cache key. One statement in a third-party package poisons vLLM's whole
worker import graph.

Enumerating the rest needs care, and both obvious methods are wrong. A column-0
grep *under*counts: it misses multi-line calls and module-level `if`/`try`
blocks. An `ast.walk` *over*counts: it descends into function bodies, which do
not run on import (that mistake reported 93 flashinfer sites, most of them
methods). Walking the AST while pruning exactly what is deferred — function and
lambda bodies, keeping class bodies, decorators and default arguments — gives:

| package | import-time `torch.cuda.*` sites |
|---|---|
| `flashinfer` | 3 — `gdn_kernels/gdn_decode_bf16_state.py:2701,2705`, `kda_kernels/__init__.py:32` |
| `vllm` | 2 — both in vendored `third_party/flash_linear_attention/ops/utils.py:152-153` |

**vLLM's own modules contain none.** Every blocker is third-party. The
`compilation_context` site above is a fourth, *indirect* one that this scan
deliberately does not count, because the call sits inside a constructor invoked
by a module-level statement.

Neutralising them — `FLASHINFER_CUDA_ARCH_LIST=9.0` for the first, deferring
`NUM_SMS` / `_USE_PACKED_FMA` to first use for the second, reading capability
from NVML for the third — by rewriting each module's source as it is imported, so
nothing on disk changes:

```
mod=vllm.v1.worker.gpu_worker  arch='9.0'
  import 17.46s  modules=7448  is_initialized=False  driver 3->3  bad_fork=False
  FORK: OK -- PRELOADABLE  ok
```

`driver 3->3` is the load-bearing part: `cuCtxGetCurrent` still returns
`CUDA_ERROR_NOT_INITIALIZED` after importing all 7448 modules, so the CUDA driver
was never touched — not merely torch's bookkeeping flag. The forked child then
allocates on the GPU successfully. The vendored `flash_linear_attention` pair
never needed patching: it is not on this import path.

So the ceiling is **three patch points in third-party code**, not a property of
CUDA — and vLLM already ships the fix pattern for them. `vllm/platforms/cuda.py`
states in its own docstring that it uses "pynvml. However, it should not
initialize cuda context", and `NvmlCudaPlatform.get_device_capability` reads
`nvmlDeviceGetCudaComputeCapability`. flashinfer reaches for
`torch.cuda.get_device_capability` for the identical query.

Two cautions before treating this as banked:

* This is an existence proof that the graph *can* be CUDA-free, not a merged
  patch. `FLASHINFER_CUDA_ARCH_LIST` was separately verified to produce a
  byte-identical workspace key (`90a`) three ways — `nvidia-smi
  --query-gpu=compute_cap` → 9.0, `_normalize_cuda_arch(9,0)` → `(9,"0a")`, and
  the run log's `flashinfer_autotune_cache/0.6.16.post3/90a/` — so that part is
  safe to set today; the other two are proposals.
* The blockers are **whack-a-mole**. Fixing the first moved the failure to
  `gdn:2701`; fixing that moved it to `kda:32`. Setting the env var alone looks
  like a fix and is not: it silently relocates the culprit. Any claim that a
  preload is CUDA-free must be re-verified *by forking*, on the exact image,
  after every version bump.

#### `torch.cuda.is_initialized()` is not a fork-safety test

`_maybe_force_spawn` (`vllm/utils/system_utils.py:148`) decides whether fork is
safe using `cuda_is_initialized()`, which is `torch.cuda._is_compiled() and
torch.cuda.is_initialized()` (`vllm/utils/platform_utils.py:14`). That predicate
can pass while fork is already broken.

`tilelang` is the counterexample, and the audit's only silent poison:

```
before: torch.cuda.is_initialized=False  driver={'rc': 3}
after : torch.cuda.is_initialized=False  driver={'rc': 0}
        _is_in_bad_fork(parent)=False
child : bad_fork=True  alloc=RuntimeError: Cannot re-initialize CUDA in forked subprocess
```

It calls `cuInit()` outside torch's bookkeeping, so `is_initialized()` reads
`False` before *and* after. torch's `atfork` handler keys on the driver rather
than that flag and marks the child bad anyway. Every parent-side predicate says
fork is safe; the child cannot allocate.

That is why this project's audit verdict is ground truth — fork a child and
allocate on the GPU — and never the predicate. It is also a genuine upstream
bug: with `tilelang` imported, vLLM would choose `fork` and hand back a child
that dies on first GPU use.

Two things to read out of the trace rather than the total:

* `forkserver.launch_child` — the span that contains the block-on-preload wait.
  Small means the preload was overlapped; large means it was not. Measured on
  the H100 node against a real vLLM preload: **14.9ms** (`early`) against
  **13,660ms** (`late`). Same mechanism, same preload list, ~900x difference
  from placement alone -- that span is the single number that says whether an
  arm actually overlapped anything.
* `forkserver.server_summary` in the forkserver process — `preload_missing` must
  be empty (`forkserver.main()` swallows `ImportError` on every entry, so a typo
  costs the whole win silently), and `cuda_initialized` must be false, because
  a CUDA-free parent is the entire safety argument for forkserver over fork.

If `forkserver.summary` shows `declined`, the probe deliberately stood down —
CUDA already initialised, Ray, `--numa-bind` or WSL — and that run measured
vLLM's own start method, not forkserver.

## Correction: the `ipc handshake` phase was an attribution bug

Every report rendered before this fix over-reported `ipc handshake`, and reported
`weight load`, `kv cache alloc` and `warmup / profile run` as **0.0s** on every
multi-process run. On `runs/fs-early-r2`:

```
ipc handshake               12.0s   33.9%  #########...................
weight load                 0ms      0.0%
kv cache alloc              0ms      0.0%
warmup / profile run        0ms      0.0%
```

There was no 12s of IPC. Of that 12.025s, **12.020s was a single span** —
`engine.wait_for_startup`, `cat="ipc"`, in the parent api_server — and all of it
was charged to the parent's pid while the child was spending 2.87s / 7.38s /
6.12s in exactly the three phases showing zero.

The cause was in `sweep_phases`: at each interval it picked a winning span across
all processes by **depth before priority**. Depth is only meaningful *within* one
process, where nesting is real; across processes it is noise. The parent sitting
in `wait_for_engine_startup` at depth 9 therefore outranked the child's actual
work at depth 3-4, for twelve seconds.

The fix has two parts:

* Priority decides across processes; depth only breaks ties within one.
* A process whose innermost active span means "blocked waiting on another
  process" counts as **idle** and is set aside — `engine.wait_for_startup`,
  `engine.launch_cores`, `engine.mp_client_init`, `engine.async_mp_client_init`,
  `engine.core_proc_manager`, `forkserver.launch_child`, `ipc.message_queue_wait`.
  A blocked process still wins intervals where nothing else is working, so the
  phases still partition the wall exactly. A span may also mark itself with
  `args={"blocking": true}`; the name list is what makes the fix apply to the 31
  traces already on disk.

After the fix `ipc handshake` is **14-39ms on all 28 runs that launched an
engine** (the three `gpu_worker` arms failed closed and have no ipc row at all),
and `fs-early-r2`
reads: python imports 21.1s / 59.3%, warmup 5.42s, engine orchestration 2.86s,
weight load 2.71s, kv cache 1.26s, hub metadata 802ms, api server 960ms, ipc
19ms, unaccounted 0ms. The genuine IPC spans (`ipc.message_queue_init`,
`ipc.stateless_pg_create`) never appear at TP=1 at all; the only real launch cost
is `engine.core_proc_manager` at 17ms. **There was no IPC cost to optimise — at
TP=1 there never was one.**

Two invariants were assert-checked against the old rule on all 31 runs: the
phases still partition the wall exactly, and `unaccounted` is unchanged to within
1e-9 everywhere — the fix moved time between phases and invented none.
`runs/phases.csv` was rebuilt from all 31 traces, keyed by directory name rather
than the trace's own `run_id`, which collides (six directories each record
`fork-r1..fork-r3`). The pre-fix file is kept verbatim at
`runs/phases.legacy-attribution.csv`; do not compare rows across the two.

The report now also prints, under the phase table, how much of the wall had more
than one process working at once — 48% on `fs-early-r2`. Over those intervals a
single-winner phase table is a choice, not a measurement, and the note says so.
The SUBSYSTEMS `ipc` row carries `(of which 16.7s is blocking on another process,
not handshake cost)`, so the raw span total no longer looks like it contradicts
the 19ms phase.

## The image ships no bytecode, and every interpreter pays for it

`vllm/vllm-openai` contains **zero** `.pyc` files — vllm 2438 `.py` / 0 `.pyc`,
flashinfer 1189 / 0, torch 2285 / 0 — and under OpenShift's arbitrary uid
`dist-packages` is read-only, so Python cannot create them at runtime either.
Every fresh interpreter re-parses and re-compiles the entire import graph.

Measured by importing `vllm.v1.worker.gpu_worker` in fresh interpreters, arms
alternating A,B,A,B,… so drift cannot favour one, and the cache warmed by a
throwaway pass *before* any timing:

| arm | median | the three runs |
|---|---|---|
| as shipped, no `.pyc` | **17.31s** | 17.45 / 17.31 / 17.13 |
| `PYTHONPYCACHEPREFIX` warm | **7.58s** | 7.58 / 7.41 / 8.79 |

**9.73s per interpreter — 56% of the import wall.** The cache is 6781 files /
138 MB and one warm-up pass (21.6s) builds it.

That cost is paid by the api_server, by the forkserver process, and — the
expensive one — by every spawned `EngineCore`, which is a fresh interpreter. Under
the upstream default (`spawn`) it is therefore paid about twice per cold start.

**Measured end to end**, 2x2 against `TOTAL TIME TO READY`, arms cycling
A,B,C,D,A,B,C,D,… so drift lands on all four equally, cache warmed by two
discarded runs before any timing, n=3 each:

| start method | `.pyc` | median | the three runs | `python imports` |
|---|---|---|---|---|
| `spawn` | off | 51.79s | 51.29 / 51.79 / 52.82 | 36.30s |
| `spawn` | **on** | **29.83s** | 29.61 / 29.83 / 31.25 | 16.20s |
| `fork` | off | 37.06s | 36.99 / 37.06 / 37.12 | 22.40s |
| `fork` | **on** | **23.98s** | 23.72 / 23.98 / 24.32 | 10.50s |

**−21.96s (−42%) under `spawn`, −13.07s (−35%) under `fork`.** The arm spreads are
≤1.6s against effects of 13s and 22s, so this needs no drift argument — but the
corroboration is what makes it a result rather than a total: essentially the whole
saving lands in the `python imports` phase (−20.10s of −21.96s under `spawn`,
−11.90s of −13.07s under `fork`), which is exactly the phase bytecode compilation
belongs to. Nothing else moved: `weight load` sat at 2.4-3.0s across all four arms.

Three readings worth separating:

* **The per-interpreter cost reproduces.** `fork` has one interpreter importing
  the full stack and saves 11.90s; `spawn` has two and saves 20.10s, or 10.05s
  each. Both bracket the 9.73s the isolated probe measured on a smaller graph.
  The `spawn`/`fork` ratio is **1.68×**, not 2× — under `spawn` the child's import
  partly overlaps the parent's, so the second interpreter's cost was never fully
  serial.
* **This one env var beats the `fork` hack.** `spawn` + `.pyc` (29.83s) is faster
  than `fork` with no `.pyc` (37.06s). A config change on stock upstream
  behaviour outperforms the start-method change we had been leaning on.
* **It does not replace the forkserver work, it stacks with it.** `fork` + `.pyc`
  is 23.98s against `spawn` + `.pyc` at 29.83s, so ~5.9s is still recoverable by
  not re-importing at all. **23.98s is the fastest cold start this harness has
  produced**, against a previous best of 35.60s.

There is also **no first-boot penalty**, which was the one cost I expected to have
to warn about. The warm-up run — `spawn`, `.pyc` on, against an empty cache —
came in at **42.73s**, still 9s faster than the `spawn` baseline: the api_server
compiles the graph, writes it, and the `EngineCore` that starts moments later
reads it back instead of recompiling. The cache pays for itself inside the run
that creates it.

The change is a config change, not a patch:

```yaml
- name: PYTHONPYCACHEPREFIX
  value: /cache/pycache
```

It lands on the same PVC as the torch.compile cache, so it is cross-pod for the
same reason and needs no privilege at all.

Two cautions. The cache is 8374 files / 157 MB for a full cold start (more than
the 6781 the isolated probe produced, which imported only `gpu_worker` and not the
API server stack), and it sits on network storage — reading it back is not free,
and that cost is already inside the numbers above. And a `.pyc` is keyed by the
source's path, mtime and size, so the cache survives pod restarts but is
correctly invalidated by an image change — which is the argument for putting it on
the volume rather than baking it per-pod.

These twelve runs measure the steady state a restarting pod sees, on one node with
a warm page cache. A fresh PVC on a cold node is the one case still unmeasured;
the warm-up run above suggests it is a gain there too, but that is an inference,
not a measurement.

Upstream the better fix is `python -m compileall` at image build: the `.pyc` then
ship read-only inside the image, cost nothing at runtime, and need no writable
volume. Worth raising against the vLLM image; it would benefit every deployment,
not just cold-start work.

This finding also explains an earlier confounded measurement, and the trap is
worth naming: pointing `-X pycache_prefix` at a fresh empty directory makes the
*first* interpreter pay full compilation (18.25s) while every later one inherits
a warm cache (~7.7s). Read as an A/B, that manufactures a 10s "effect" for
whatever variable happened to be in the first cell.

## Deferring an import saves nothing if the modules are shared

The `b12x` warmup chain looked like the obvious lazy-import candidate.
`vllm/v1/worker/gpu_worker.py:55` imports `kernel_warmup` at module level,
`kernel_warmup.py:15` imports `b12x_warmup` at module level, and that pulls in
five quantization providers. `-X importtime` attributed **4.54s** to the chain.
All five providers early-return on `is_device_capability_family(120)` before
touching the model, so on this H100 (`DeviceCapability(major=9, minor=0)`,
`family(120): False`) the whole chain warms exactly nothing — confirmed by
`grep -h "B12X\|Warmed up" runs/*/vllm.log` being empty across all 31 runs.

Deferring it — delete the module-level import, gate the call at
`kernel_warmup.py:161` on the same capability check, following the pattern
`kernel_warmup.py:101` already uses for `minimax_m3_msa_warmup` — measures:

| arm | median | the three runs | modules |
|---|---|---|---|
| baseline, b12x eager | **17.30s** | 17.30 / 17.52 / 17.23 | 7447 |
| b12x deferred | **18.12s** | 18.12 / 17.45 / 19.10 | 7446 |

**One module.** The saving is not small, it is absent — and the sign is negative,
which is drift and inter-run noise (±1s here), not a cost of laziness. The 4.54s
was `-X importtime` *cumulative* time: every submodule under the chain was
already being imported by something else in the graph, so the chain's exclusive
cost was one module's worth.

The finding is still real — vLLM runs a five-provider warmup that is a
guaranteed no-op on every non-SM120 GPU, which is A100, H100 and L40S — it just
is not a *startup-time* finding. The same applies to flashinfer autotune, which
logs `Loaded 0 configs` / `Saved 0 configs` and still costs something to reach.

The general lesson, and the reason this is written up as a null result rather
than dropped: **`-X importtime` cumulative time is an upper bound on what
deferring an import can save, often a wildly loose one.** Before writing a lazy
import, check that the module count actually drops. If it drops by one, there is
nothing to win, however large the attributed time.

## `fastsafetensors` is worth 12.4s at 32B, and every knob on it is not

`weight load` was 47% of a 32B cold start — the largest single phase once
bytecode caching had taken the imports down. `--load-format fastsafetensors`
(fastsafetensors 0.3.3, shipped in the upstream image) is a one-flag change and
it is the largest single win this harness has measured on a phase other than
imports.

Qwen/Qwen3-32B, 61.02 GiB in 17 shards on a GPFS PVC, `--max-model-len 8192`,
`VLLM_WORKER_MULTIPROC_METHOD=fork`, `PYTHONPYCACHEPREFIX=/cache/pycache`, arms
interleaved A,B,A,B,… after one discarded warm-up, n=3:

| arm | median total | the three runs | vLLM's `Loading weights took` |
|---|---|---|---|
| `--load-format auto` | 59.85s | 57.74 / 59.85 / 94.16 | 28.84s |
| `--load-format fastsafetensors` | **47.49s** | 46.58 / 47.49 / 48.58 | **16.19s** |

**−12.36s (−20.6%).** Effective throughput on the same files goes 2.12 → 3.77
GiB/s.

What makes this a result rather than a total is that two independent measurement
paths agree: the probe's phase attribution puts `weight load` at 30.30 → 17.60s
(**−12.70s**) and vLLM's own in-engine timer puts it at 28.84 → 16.19s
(−12.65s). Those instruments share no code. Nothing else moved — imports
−0.10s, warmup +0.00s, orchestration +0.00s, kv cache −0.02s, compile and
cudagraph ~0.

The arm is verified as actually engaged, not merely requested: `load_format=
fastsafetensors` in the engine config line, and a **different progress-bar
string** — `Loading fastsafetensors checkpoint shards` at 1.18 it/s against
`Loading safetensors checkpoint shards` at 1.69 s/it, a 2.0× shard rate. Check
the bar, not the flag: run-ids in this tree named `q32-fs-*` are the *forkserver*
arm and load plain safetensors.

Quote the range, not just the median. The baseline's 94.16s run was a **62.00s**
weight load against 26.52s and 28.84s for the other two, with compile time a
normal 0.22s throughout — the same code reading the same files varies 2.3× with
cache state. `fastsafetensors` was far tighter (16.07 / 16.19 / 16.69s), which is
itself part of the finding.

### Storage was never the bottleneck; read concurrency was

The tempting reading — 47% of the wall on a network filesystem, therefore
I/O-bound — is wrong, and measuring the device settles it. `O_DIRECT` reads of
the same 61.02 GiB, bypassing the page cache (the node has ~2 TiB RAM and will
happily cache the entire model, flattering any buffered benchmark):

| threads | 1 | 4 | 8 | 16 |
|---|---|---|---|---|
| GPFS, `O_DIRECT` | 1.15 | 4.12 | 7.79 | **9.30 GiB/s** |

9.30 GiB/s is 6.56s for the whole model. Host-to-device is not the limit either:
51.55 GiB/s from pinned memory, 16.55 pageable. So the baseline loader's 2.12
GiB/s is **its own read concurrency**, four ways off what the filesystem
delivers, and `fastsafetensors` recovers part of that by reading with more
threads.

Two consequences. Staging weights on node-local NVMe is a dead end here — the
local PVs write at 3.4 GB/s, *slower* than GPFS. And 16.2s still sits against a
6.6s floor, so ~9.5s remains inside the loader's pipeline, unreachable by any of
the knobs below.

### Both tunables are null, and one of them is a trap

`VLLM_FASTSAFETENSORS_QUEUE_SIZE` (`envs.py:123`, default 0) is the only
fastsafetensors parameter vLLM exposes. End to end at 32B, interleaved, n=3:

| arm | median total | weight load |
|---|---|---|
| `QUEUE_SIZE=0` | 48.03s (47.13 / 48.03 / 48.45) | 16.08s |
| `QUEUE_SIZE=2` | 47.51s (46.97 / 47.51 / 47.61) | 15.97s |

**−0.11s on the phase it targets.** `ParallelLoader`'s docstring promises deeper
pipelining at the cost of `(max_concurrent_producers + queue_size) * file_size`
of GPU memory; peak GPU stayed flat at 8.8G across queue depths 0–3, i.e. the
queue never fills. Leave it at the default.

`max_threads` and `bbuf_size_kb` are not reachable at all —
`fastsafetensors_weights_iterator` (`weight_utils.py:1033-1105`) constructs
`ParallelLoader` directly and passes neither, and neither appears in
fastsafetensors' own `LoaderConfig`, so `FASTSAFETENSORS_CONFIG` cannot set them
on vLLM's path either. Measured by driving `ParallelLoader` directly, they are
null anyway on the path vLLM takes: 14.05s against the default's 14.10s.

### The one real finding: `nogds` keys on the wrong variable

`weight_utils.py:1043` decides the transfer path with

```python
nogds = pg.size() > 1
```

so at TP=1 vLLM asks for GDS. This cluster has none — no `nvidia_fs` module, no
`/dev/nvidia-fs*`, no `libcufile`, no `cufile.json`,
`torch.cuda.gds_register_buffer` absent — because the GPU-operator ClusterPolicy
sets `gds: {enabled: false}`. fastsafetensors degrades internally rather than
raising, so the `nogds=True` fallback in that function never fires and the cost
is silent.

Fresh process per config, one load each, arms interleaved over 3 passes — which
is what a cold start actually does:

| config | cold start (load 1) | in-process repeat (load 2) | per-process setup |
|---|---|---|---|
| `nogds=False`, 16 thr, 16384 KiB — **what vLLM does** | **16.57s** | 14.13s | **2.44s** |
| `nogds=True`, 16 thr, 16384 KiB | **13.97s** | 13.22s | 0.75s |
| `nogds=True`, 8 thr, 32768 KiB | **12.77s** | 12.01s | 0.76s |

`nogds=True` is worth **−2.60s**, and the decomposition is the point: only 0.91s
is steady-state throughput (14.13 → 13.22s), while **1.69s is one-time
per-process setup** — probing for a feature that is not installed, paid afresh by
every `EngineCore`. This is not a knob to tune; it is a heuristic keying on world
size when it should key on whether `libcufile`/`nvidia_fs` exist.

With that flipped, `max_threads=8` / `bbuf_size_kb=32768` *does* pay — a further
−1.20s, having done nothing under `nogds=False`. Full combination **16.57 →
12.77s (−3.80s)**, projecting 32B total to ~43.7s. All three knobs need a patch;
none is reachable from the environment.

Upstream, in priority order: condition `nogds` on GDS availability rather than
`pg.size()`; expose `max_threads` and `bbuf_size_kb`; note that
`max_threads=16` is counterproductive under a CPU-limited cgroup (16 cores here
against 224 on the node).

### Two measurement traps this sweep walked into first

Both produced confident, wrong answers before being caught, and both apply to
every arm in this tree.

**Never measure a loader with in-process repeats.** The first load in a process is
2.44s slower than later loads on vLLM's path — same process, same files, cache
long since warm. Only load 1 corresponds to a cold start, and load-1 for the
default config (16.10 / 16.57 / 16.57s) matches vLLM's own in-engine timer across
six end-to-end runs (16.06–16.39s), which the warm figures never did. A sweep
built on in-process repeats understated every knob here by roughly half.

Worse, a sweep that iterates configs over the same files **warms the cache
monotonically down the matrix**, so improvements alias onto whichever config ran
last. That artifact produced a −2.50s "win" for `queue_size` and a −3.49s one for
`nogds` that end-to-end runs then refuted. Interleave configs across passes; do
not trust `posix_fadvise(POSIX_FADV_DONTNEED)` to fix it, because GPFS serves
reads from its own pagepool and `/proc/meminfo Cached` neither reflects nor
controls it — eviction reported 61.0G once and 0.0G on all eleven later calls.

**Any `VLLM_*` env arm pays a one-time recompile.** `compile_factors()`
(`envs.py:2216`) starts from *every* known vLLM env var and drops only those in
`ignored_factors` (`:2222`), so 233 vars are part of the torch.compile cache key
— `VLLM_FASTSAFETENSORS_QUEUE_SIZE` among them. Its first run cost **24.78s of
compilation** against 0.21-0.22s for every other run in the set, turning a null
knob into an apparent 72.07s catastrophe. `VLLM_WORKER_MULTIPROC_METHOD` *is* on
the drop list, which is why the fork/spawn arms never showed this. Discard the
first run after changing any env var, or measure a 25s regression that is not
there.

## Reading the report

`report.txt` is ordered so you can stop as soon as you have your answer.

**Header** — run id, model config, node facts, cache state at `t0`, process
count, and the probe's own cost. If the cache line says `fully cold` and you
expected warm, stop here; the run does not mean what you think.

**`TOTAL TIME TO READY`** — the headline, plus which signal defined "ready", plus
all five readiness edges as offsets from `t0`.

**`PHASE BREAKDOWN`** — non-overlapping wall clock per phase, with `unaccounted`
as the remainder. Followed by the single span that was credited the most time in
each phase, which is usually the actual answer to "what should I fix?".

**`PROCESSES`** — one row per Python process: when it was exec'd relative to
`t0`, its import cost, disk read, peak RSS, cmdline. Late-starting processes are
where serialisation hides.

**`SPAN TREE`** — what happened, in order, per process, filtered by `--min-ms`.
This is the view to read when a phase number surprises you.

**`PYTHON IMPORTS`**, **`SUBSYSTEMS`** — per-process import detail; then network,
weight loading, collectives, compilation, graph capture, KV cache, warmup, IPC,
and filesystem metadata chatter. Each subsystem shows `n`, `sum` and `wall` — the
gap between `sum` and `wall` is how much parallelism you got.

**`FINDINGS`** — the report's own reading of the trace: readiness-probe
semantics, CPU throttling, cold caches, page-cache-served weights, per-process
import cost, process serialisation, first-request cost. These are heuristics with
thresholds, not verdicts; each one names the evidence it used so you can check it.

For anything the text report flattens, open `trace.json` in
[ui.perfetto.dev](https://ui.perfetto.dev) — the counter tracks (CPU, RSS,
throttling, disk, network, GPU) sit under the spans and make "was this
disk-bound or CPU-bound?" a visual question.

## Tracking across many runs

Every render appends one row to `phases.csv` (and `scripts/fetch-trace.sh`
appends to a shared `runs/phases.csv`). Columns: run id, total, ready source,
model/dtype/tp/eager/compile level, vLLM and torch versions, CPU limit, model
filesystem, one column per phase, the readiness edges, disk read bytes, network
rx bytes, and throttled seconds. That is enough to plot time-to-ready by phase
against vLLM version or model size without re-reading any trace.

## Honesty checklist

Before quoting a number:

* Is `unaccounted` small? A large remainder means the breakdown is incomplete —
  check `probe.coverage` for patches that did not apply.
* Does the cache state in the header match the experiment you meant to run?
* Was the container throttled? Throttling makes every CPU-bound phase a
  measurement of your CPU limit rather than of vLLM.
* Did the weights come from the page cache? If so, `weight load` is a floor.
* Is this the first run in a fresh pod, or the third? Say which.
* Native architecture, not emulated?
* **Was the baseline measured next to this arm, or earlier in the session?**
  Anything under ~2.5s needs an adjacent baseline block; see the drift warning
  above. If you cannot re-run the baseline, quote the within-tree span instead of
  the total.
* **Is the phase table describing one process or several?** If the report's
  concurrency note says a large fraction of the wall had more than one process
  working, a single phase does not own that time; quote the per-process table.
  And no number from `runs/phases.legacy-attribution.csv` is comparable to a
  current one.
* **Did you claim a preload is fork-safe on the strength of a predicate?**
  `torch.cuda.is_initialized()` can read `False` while the fork is already
  broken. Fork a child and allocate on the GPU, or you have not tested it.
* **Did the source rewrite actually fire?** A patch-on-import experiment that
  silently matches nothing reads exactly like "the fix does not work". Assert the
  expected match count, and verify against the loaded module, not a log line.
* **Was the `.pyc` cache in the same state in both arms?** A cold
  `pycache_prefix` costs ~9.7s in whichever cell runs first. Warm it before
  timing anything.
* **Are the arms interleaved?** Run A,B,A,B,…, not a block of A then a block of
  B; within-session drift reached +2.35s over 40 minutes on byte-identical
  config.
* **Are you summing import spans?** They are nested and cumulative; summing them
  is meaningless. Use the interval union.
* **Did you measure a loader with in-process repeats?** The first load in a
  process is ~2.4s slower than the ones after it, and only the first corresponds
  to a cold start. One fresh process per config, or the knobs look half as large
  as they are.
* **Did the sweep iterate configs over the same files?** Then the cache warmed
  monotonically down the matrix and the win landed on whichever config ran last.
  Interleave across passes; `posix_fadvise(DONTNEED)` does not evict GPFS's
  pagepool, whatever `/proc/meminfo Cached` says.
* **Did you change a `VLLM_*` env var?** 233 of them are torch.compile cache
  factors, so the first run pays a full recompile — 24.8s at 32B. Discard it.
* **Is the pod already setting the variable you think you are testing?**
  `manifests/pod-exec.yaml` pins `VLLM_WORKER_MULTIPROC_METHOD=fork` (and
  `VLLM_ENABLE_STARTUP_PLAN=1`) at the container level, so an arm that passes
  no `--env` is *not* upstream-default — it inherits those. A baseline has to
  set the default back explicitly (`--env VLLM_WORKER_MULTIPROC_METHOD=spawn`),
  or the step you are trying to price reads as a no-op. Symptom: two arms whose
  `python imports` phase is identical to 0.1s. The full list of pinned
  variables is in [docs/kubernetes.md](kubernetes.md#constants-not-steps).
* **Did the run-id collide with an earlier run?** `coldstart-run.sh` now refuses
  this, but a directory containing two runs' pid-keyed trace files renders as one
  enormous run without complaint. If a total looks impossible, count the
  `process.exec` events: more than one root means two runs got merged.
