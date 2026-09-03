# Concurrent CUDA graph capture

`cudagraph capture` is **11.4s** of a 43.3s time-to-ready for Qwen3-32B on one
H100 — 26%, and the phase that
[no configuration removes honestly](../README.md#cuda-graph-capture-114s-that-configuration-cannot-honestly-remove).
The captures run one after another in a `for` loop, ~7.1s of the phase is pure CPU
work with the GPU almost idle, and the container has 16 cores and uses one. So:
**can the captures run concurrently?**

The answer, measured rather than reasoned:

* **CUDA and PyTorch permit it.** Two, four and eight threads captured 32 graphs at
  once and every replay was **bit-exact** against an eager reference. It needs
  `capture_error_mode="thread_local"` and **one mempool per capturing thread**.
* **It barely pays, and only at two threads.** 2 threads is **1.10-1.21x** across
  three independent invocations. 4 threads is 0.99-1.05x. 8 threads is **0.60x** —
  much *slower* than serial. The cores are consumed (cpu/wall up to 3.5) and not
  converted into progress, and it is neither the GIL nor the caching allocator:
  the serialization is below PyTorch.
* **It is not free.** One mempool per thread costs
  **[+1.08 GiB of GPU memory per extra pool](#4-what-it-costs-on-the-real-model)**
  at 32B, measured, exact, and linear in threads — and on the v2 runner that
  memory is **unbudgeted**, so the failure mode is a late OOM rather than a
  smaller KV cache.
* **What was in the way is worth 8x more than the concurrency is.** Concurrency
  requires that the `torch.cuda.synchronize()` and `empty_cache()` which
  `torch.cuda.graph.__enter__` runs before every capture not run inside a sibling's
  capture window. Pricing that
  prerequisite on its own turned out to be
  **[the largest single startup win in this repo that needs no vLLM redesign](#4b-the-prerequisite-is-the-result-40s)**:
  **−4.0s of the capture phase and −4.0s of time-to-ready**, with a spread under
  0.25s, from *moving* two calls out of a loop that runs them 3366 times — and
  measured to cost no memory at all.

**Verdict: feasible, worth ~0.5s, and last in line.** After the prologue fix there
is only ~4.25s of recording left to overlap, so 2-thread capture is worth ~0.4-0.7s
of a 7.7s phase — bought with ~1.08 GiB per thread and the six upstream changes in
§5. It is the only lever in this repo that charges *steady-state GPU memory* for a
startup win. Do §4b first; then re-price this against what remains.

Everything below is measured on the image this repo tracks (vLLM 0.28.0 vendor
build, torch 2.13.0+cu130, CUDA 13.0, driver 580.105.08, H100 80GB HBM3, TP=1),
in namespace `lionel-cold-start-cc`. Scripts:
[`scripts/cudagraph-concurrent-capture.py`](../scripts/cudagraph-concurrent-capture.py),
[`scripts/cudagraph-capture-legality.py`](../scripts/cudagraph-capture-legality.py),
[`scripts/cudagraph-pool-matrix.sh`](../scripts/cudagraph-pool-matrix.sh),
[`coldstart/cs_cgcapture.py`](../coldstart/cs_cgcapture.py). Raw probe output:
[`reports/cudagraph-concurrent-capture.txt`](../reports/cudagraph-concurrent-capture.txt).

## 1. Does CUDA allow it at all?

PyTorch says no, twice. `torch/cuda/graphs.py:419-421`:

> `# Not thread safe, but graphs already have the general (explicitly documented)`
> `# restriction that only one capture may be underway at a time in the process.`

and `libc10_cuda.so` carries the string `"Only one capture at a time is allowed
in a process."` — but that string sits among the `CudaMallocAsync` symbols, i.e.
it belongs to the *non-default* allocator backend. The native caching allocator
this build uses instead carries `num_active_captures_` and `"Could not find
memory pool id for capture."`, which are the artefacts of code that expects more
than one. That discrepancy is what made the experiment worth running.

CUDA itself is explicit: `cudaStreamBeginCapture` takes a
`cudaStreamCaptureMode`, and the whole point of `cudaStreamCaptureModeThreadLocal`
is to scope the capture's restrictions to the capturing thread. PyTorch exposes it
as `capture_begin(capture_error_mode=...)`, defaulting to `"global"`.

Measured, 32 graphs of ~400 nodes each, median of 5, every arm replayed and diffed
against an eager reference (the run saved in `reports/`):

| arm | median | vs serial | cpu/wall | overlap achieved | max abs diff |
|---|---|---|---|---|---|
| `serial` (what vLLM does) | 94.1 ms | 1.00x | 1.00 | — | 0 |
| `serial-globalmode` | 97.0 ms | 0.97x | 1.00 | — | 0 |
| `threads-2-ownpool` | **85.7 ms** | **1.10x** | 1.78 | 2.00 | **0** |
| `threads-4-ownpool` | 94.3 ms | 1.00x | 2.75 | 3.91 | **0** |
| `threads-8-ownpool` | 155.7 ms | **0.60x** | 2.89 | 7.57 | **0** |
| `threads-{2,4,8}-sharedpool` | — | — | — | — | `RuntimeError: beginAllocateToPool: already recording to mempool_id` |
| `threads-2-globalmode` | — | — | — | — | `AcceleratorError: CUDA error: operation not permitted when stream is capturing` |

Four results, in order of how much they matter:

**It works.** `maxdiff=0` on every graph in every threaded arm: concurrent capture
does not merely avoid crashing, it records the same kernels with the same
arguments. `overlap achieved` is busy-time over wall-time across the capturing
threads, so 2.00 of a possible 2.00 confirms the captures really were in flight
simultaneously rather than politely queueing.

**The ceiling is two threads, and it is low.** Three independent invocations of the
same benchmark give 2 threads **1.10x / 1.18x / 1.21x**, 4 threads
**0.99x / 1.00x / 1.05x**, 8 threads **0.60x / 0.64x**. Eight threads achieve 7.57x
overlap and land 65% *slower* than serial — the work is being done, in parallel,
and thrown away in contention. Two controls locate it: `cpu/wall` reaches 3.47, so
it is **not the GIL**; and the `-noalloc` arms (capture into a pre-warmed pool, no
fresh allocations) show the same shape — serial 84.7ms, 2 threads 72.0ms (1.18x),
4 threads 86.5ms, 8 threads 86.5ms — so it is **not the caching allocator** either.
What is left is the driver's own per-process capture serialization.

**A shared mempool is a hard stop.** `beginAllocateToPool` refuses a second
concurrent capture into the same pool, at every thread count. This is the finding
with consequences, because vLLM shares exactly one pool across all captures
*deliberately* (`v1/worker/gpu/cudagraph_utils.py:131`, and see §4).

**torch's default mode forbids it outright.** With
`capture_error_mode="global"` the second thread's capture aborts. Worse, the
failure is not contained: after it, the main thread's own
`torch.cuda.synchronize()` raises, and so does `empty_cache()` in the *next*
arm — a failed global-mode capture leaves the process with a capture still
underway and nothing can follow it. (Which is why that arm runs last in the
script.)

## 2. What is legal while another thread is capturing?

A threaded capture does not get to pick the ambient calls its process makes.
`scripts/cudagraph-capture-legality.py` holds a capture open on a barrier and probes
from another thread. Two columns per mode, because "did the call raise" is the
weaker question: **did the capture survive it** is what a design has to plan around.

| operation, issued from a non-capturing thread | `thread_local` | capture lives? | `global` | capture lives? |
|---|---|---|---|---|
| `torch.cuda.synchronize()` | **RAISES** | **no** | **RAISES** | **no** |
| `torch.cuda.empty_cache()` | ok | yes | **RAISES** | **no** |
| `torch.cuda.current_stream().synchronize()` | ok | yes | **RAISES** | **no** |
| fresh `cudaMalloc` (`torch.zeros(4 MiB)`) | ok | yes | ok | yes |
| eager matmul on the default stream | ok | yes | **RAISES** | **no** |
| `torch.cuda.mem_get_info()` | ok | yes | ok | yes |
| a second capture, own pool | ok | yes | **RAISES** | **no** |
| a second capture, same pool | **RAISES** (`beginAllocateToPool`) | **yes** | **RAISES** | **no** |

Each row gets its own fresh capture, which is a correction to how this probe
originally worked: it ran all eight checks against one capture, and check #1 kills
that capture, so checks 2-8 were being measured against a capture that was already
dead. Every verdict came out the same once isolated — so the conclusions below did
not move — but they are now sound rather than lucky, and the survival column is new.

`thread_local` is what makes concurrency possible: everything a well-behaved
sibling thread does is legal *and* leaves the capture intact. Note the shape of the
one refusal, which is the good kind: a second capture into the same pool raises
`beginAllocateToPool` and **the in-flight capture survives** — a recoverable "not
now", not a corrupted graph.

The one destructive row is the one that matters. **A device-wide
`torch.cuda.synchronize()` kills the in-flight capture in both modes** — the
capturing thread dies with `operation failed due to a previous error during
capture` — and under `global` so does an `empty_cache()`, a stream sync, an eager
matmul, and any second capture at all.

That single row disqualifies `torch.cuda.graph` — the context manager both of
vLLM's capture paths use — because `__enter__` calls `torch.cuda.synchronize()`
unconditionally (`torch/cuda/graphs.py:437`, comment: "Free as much memory as we
can for the graph"). **Any concurrent-capture implementation has to stop using
`torch.cuda.graph`, or change it.** That requirement is what §4b turns into a
measurement.

## 3. What there is to overlap

### 3a. First, how many captures are there?

The number this repo has been carrying is wrong, and it is worth fixing before
sizing any parallelism over it. `cudagraph.capture_begin` in the committed traces
is floored at 5ms (`coldstart/cs_torch.py`), so it counts *captures over 5ms* —
102 of them — and the README read that as 102 graphs. A run with `CS_CGDETAIL=1`
spans every `graph.__enter__`/`__exit__` with no floor and tags each with the line
that built it (`runs/cc-detail2`,
[`analysis/cudagraph_graph_census.py`](../analysis/cudagraph_graph_census.py)):

```
built by                                 graphs     enter   `- >5ms      exit   `- >5ms
compilation/cuda_graph.py:313              3315    2.369s    0.844s    0.419s    0.000s
v1/worker/gpu/cudagraph_utils.py:365         51    0.829s    0.829s    0.227s    0.061s
TOTAL                                      3366    3.198s    1.672s    0.646s    0.061s
```

**3366 `CUDAGraph`s, not 102** — and 3366 = 51 × (65 + 1). Each of the 51 sizes
builds **one** FULL graph and **65** piecewise ones: `splitting_ops` cuts the graph
at every attention op, so 64 attention layers give 65 pieces, and
`CUDAGraphWrapper` captures each piece separately. Two consequences:

* **Torch's per-capture prologue is 3.84s of the 11.3s phase (34%), not 1.66s.**
  Only **104** of the 3366 `__enter__` calls cross the 5ms floor; the floored view
  therefore shows 45% of the cost and hides 3262 calls, each of which still does a
  `synchronize()` and an `empty_cache()`.
* **The 7.7x PIECEWISE/FULL gap has an obvious cause after all.** Splitting the 102
  `capture_forward` spans by whether they contain piecewise captures: PIECEWISE
  **133.5ms** per size, FULL **17.3ms** — the same 7.7x the README reported — and
  **46.5ms of the piecewise 133.5ms is `graph.__enter__`, entered 65 times.** The
  FULL path's single `__enter__` sits *outside* `capture_forward` (0 of 3315
  piecewise enters land outside; 51 of 51 FULL enters do). Net of torch's
  per-graph fixed cost the gap is 4.6x rather than 7.7x, so about 40% of the
  piecewise penalty is simply "65 graphs instead of 1".

None of that changes the *unit* of parallelism — the 65 pieces of one forward are
the model's layers and are sequentially dependent, so the independent work is
still the 102 (mode, size) captures. It changes what the parallelism is competing
against, which §4b takes up.

### 3b. The decomposition

Median of `runs/w6-base-c{1,2,3}`, three runs in close agreement, from the
committed traces:

| component | n | c1 | c2 | c3 | what it is |
|---|---|---|---|---|---|
| `cudagraph.capture_forward` | 102 | 7.145s | 7.211s | 7.164s | CPU: recording the graphs. **The only overlappable work** — and a third of it is the prologue in §3a. |
| `cudagraph.warmup_forward` | 102 | 2.114s | 2.202s | 2.215s | CPU: launching one eager forward per (mode, size) |
| `cudagraph.capture_begin` | 102/101/101 | 1.696s | 1.650s | 1.637s | the >5ms part of §3a's 3.198s; **96% of it (1.633 / 1.586 / 1.575s) is a nested `torch.cuda.synchronize()`** |
| `cudagraph.capture_end` | 10/11/13 | 0.055s | 0.070s | 0.076s | likewise floored; the true total is 0.646s |
| `prepare_inputs` | 153 | ~0.17s | | | dummy batch construction |
| **`cudagraph.capture_model`** | 1 | **11.216s** | **11.376s** | **11.310s** | |

The `capture_begin` row is a *prerequisite* for concurrency rather than a target of
it: that nested device sync has to go before any second thread can exist (§2), so
its cost has to be priced on its own first. That is what §4b does — and it is where
the useful number turned out to be.

## 4. What it costs on the real model

Concurrency needs one mempool per capturing thread (§1). vLLM has one, on
purpose:

```python
# v1/worker/gpu/cudagraph_utils.py:131
self.pool = current_platform.get_global_graph_pool() if cudagraph_mode else None
```

and the capture loop is ordered around that sharing — descriptors sorted
`reverse=True` by token count, PIECEWISE before FULL, with the comment:

> `# Capture in order: PIECEWISE first, then FULL. PIECEWISE has larger`
> `# activations so FULL activations should fit in already allocated buffers in`
> `# the graph pool.`

N pools give up that reuse across pool boundaries. To price it without
confounding it with threading, [`coldstart/cs_cgcapture.py`](../coldstart/cs_cgcapture.py)
rotates N graph mempools round-robin across the captures **strictly serially**,
and vLLM's own log line reports the result.

Five arms, three interleaved passes each, 15 runs
(`runs/cc-{base,pool1,pool2,pool4,nosync}-c{1,2,3}`, rendered by
[`analysis/cudagraph_capture_matrix.py`](../analysis/cudagraph_capture_matrix.py)):

```
arm                 n   capture  spread   cap_fwd  warm_fwd   cap_beg    `-sync pool GiB  engaged
cc-base             3   11.097s  0.512s    7.019s    2.147s    1.679s    1.615s     1.84  upstream
cc-nosync           3   11.586s  0.369s    7.216s    2.108s    2.004s    0.000s     1.84  nosync x3366
cc-pool1            3   11.523s  0.882s    7.384s    2.187s    1.618s    1.551s     1.84  pools=1 (3366)
cc-pool2            3   12.240s  1.314s    7.932s    2.100s    1.711s    1.651s     2.93  pools=2 (1683/1683)
cc-pool4            3   11.601s  0.785s    7.407s    2.225s    1.633s    1.571s     5.06  pools=4 (842/842/841/841)
```

**The memory column is the result.** It is exact and it repeats: every one of the
three passes of every arm reported the same GiB to two decimals, from vLLM's own
driver-side accounting.

| graph mempools | pool size | delta | per extra pool |
|---|---|---|---|
| 1 (upstream) | 1.84 GiB | — | — |
| 2 | 2.93 GiB | +1.09 GiB | +1.09 GiB |
| 4 | 5.06 GiB | +3.22 GiB | +1.07 GiB |

Independently, from NVML rather than from vLLM — median device memory over the last
3s of each run, with the server up and serving
([`analysis/gpu_steady_memory.py`](../analysis/gpu_steady_memory.py)): 72.23 GiB at
1 pool, 73.32 at 2 (**+1.08**), 75.45 at 4 (**+3.21**), zero spread across the three
passes of every arm. Two accounting paths, agreeing to 0.01 GiB.

**~1.08 GiB per capturing thread, linear in threads.** `pool1` — the probe engaged
but still rotating a single pool — reproduces base's 1.84 GiB exactly, so the
rotation code itself costs nothing and the delta is the lost cross-pool reuse and
nothing else. The microbenchmark reproduces the same linearity independently at
toy scale: 32 graphs take 4 / 8 / 16 / 32 MiB over 1 / 2 / 4 / 8 pools.

**Who pays the 1.08 GiB is worse than "the KV cache".** Every arm here reports
`GPU KV cache size: 19,840 tokens` — identical — because this pod pins
`kv_cache_memory_bytes` through a persisted startup plan, so KV sizing cannot
react to the graph pool. But even without the plan it would not react, on this
runner: `gpu_worker.py:532` asks `profile_cudagraph_memory()` for an estimate to
subtract, and the **v2 runner's implementation returns 0**:

```python
# v1/worker/gpu/model_runner.py:843
def profile_cudagraph_memory(self) -> int:
    # NOTE(woosuk): It is TBD whether we keep this API or not.
    return 0
```

The v1 runner has a real implementation (`gpu_model_runner.py:6780`, which captures
sample graphs to measure). So on v2 the graph pool — the current 1.84 GiB *and*
anything a concurrency scheme adds — is spent out of whatever headroom
`gpu_memory_utilization` happened to leave, unbudgeted and unreported. The trade is
therefore not "less KV cache" but "less margin": to restore the margin an operator
must lower `gpu_memory_utilization`, which *then* costs KV cache. This is the same
v1/v2 gap shape as `cudagraph_num_of_warmups` (README finding 1).

**The timing columns are not readable, and say so.** The within-arm spread reaches
1.31s, larger than any between-arm difference in the table. That is the honest
answer to "do N pools cost time": not measurably, either way, at n=3.

**`nosync` is the surprising one.** It suppressed all 3366 device syncs — the
`` `-sync `` column is exactly 0.000s, attributed by span containment — and the
phase did **not** get shorter. `capture_begin` went *up* (1.679s → 2.004s). So the
1.6s is not the process idling in a barrier that concurrency could hide; the
barrier moves. The mechanism is next door in the same prologue: `graph.__enter__`
also calls `empty_cache()`, and `cudaFree` is itself device-synchronizing. Which
makes the whole prologue the thing to price, not the sync alone.

## 4b. The prerequisite is the result: −4.0s

Same harness, same pod, a fresh interleaved control, three passes
(`runs/cp-{base,noempty,nofree}-c{1,2,3}`):

```
arm                 n   capture  spread   cap_fwd  warm_fwd   cap_beg    `-sync pool GiB  engaged
cp-base             3   11.237s  0.186s    7.067s    2.179s    1.685s    1.628s     1.84  upstream
cp-noempty          3    8.839s  0.185s    4.875s    2.048s    1.648s    1.634s     2.87  noempty x3366
cp-nofree           3    7.200s  0.220s    3.953s    2.055s    0.000s    0.000s     2.87  nosync + noempty x3366
```

| arm | what it removes | capture phase | time-to-ready (median of 3) |
|---|---|---|---|
| `base` | — | 11.237s | 57.206s |
| `noempty` | the 3366 `empty_cache()` calls | **8.839s** (−2.40s) | 55.512s (−1.69s) |
| `nofree` | those **and** the 3366 `synchronize()` calls | **7.200s** (−4.04s) | **53.206s (−4.00s)** |

**This is a real 4 seconds, not a phase-accounting artefact.** It reaches
`health_200` end-to-end; the paired per-pass deltas are −2.88s, −4.71s, −3.54s
(negative in every pass); and the within-arm spreads are 0.19-0.22s, an order of
magnitude below the effect. Note where the time comes from: `capture_forward` falls
7.067s → 3.953s, i.e. most of the saving is *inside* the piecewise forward, exactly
where §3a found 65 prologues per size hiding under the 5ms floor.

Two honest caveats:

* `torch._C._host_emptyCache()` runs in the same prologue and is a C-extension
  attribute this probe leaves alone, so the measured 4.04s is a **lower bound** on
  what the prologue costs.
* The arms *delete* the frees, and deletion leaves memory behind: the pool grows
  1.84 → **2.87 GiB**, coincidentally about the price of one extra mempool. So
  `nofree` as measured trades ~1.03 GiB for 4.0s rather than being free.

Which raises the question an upstream patch actually turns on — not "delete the
frees" but **hoist** them: one `synchronize()` + `empty_cache()` after
`capture_model` instead of 3366 inside it. `CS_CGEMPTY=once` measures exactly that
(`runs/ch-{base,emptyonce}-c{1,2,3}`):

```
arm                 n   capture  spread   cap_fwd  warm_fwd   cap_beg    `-sync pool GiB  engaged
ch-base             3   11.534s  0.655s    7.265s    2.197s    1.673s    1.552s     1.84  upstream
ch-emptyonce        3    9.037s  0.285s    4.997s    2.153s    1.577s    1.563s     2.87  noempty x3366, hoisted free
```

and the same hoist applied to the device sync as well — the whole prologue moved
out of the loop rather than deleted (`runs/c2-{base,nofreeonce}-c{1,2,3}`):

```
c2-base             3   11.882s  0.203s    7.762s    2.193s    1.920s    1.580s     1.84  upstream
c2-nofreeonce       3    7.662s  1.218s    4.254s    2.219s    0.008s    0.000s     2.87  nosync + noempty x3366, hoisted free
```

| arm | what moves out of the loop | capture phase | steady device memory, NVML |
|---|---|---|---|
| `ch-emptyonce` | the 3366 `empty_cache()` calls | 9.037s (**−2.50s**) | 72.23 GiB (**+0.00**) |
| `c2-nofreeonce` | those **and** the 3366 `synchronize()` calls | 7.662s (**−4.18s**) | 72.23 GiB (**+0.00**) |

**Hoisting keeps both.** −4.18s of the capture phase, negative in every paired pass
(−3.36s, −4.74s, −4.18s), and steady-state device memory identical to base — the
1.03 GiB the deletion arms leave behind comes back when the free happens once after
the loop instead of never.

One trap is worth recording, because it very nearly produced the opposite
conclusion. **vLLM's own `took X GiB` reports 2.87 GiB for both hoist arms**, the
same as the deletion arms, which reads as "the hoist saves the time but not the
memory". It is wrong here, and structurally so: the figure is a delta measured
*inside* `capture_model` —

```python
start_free_gpu_memory = torch.accelerator.get_memory_info()[0]   # model_runner.py:871
...capture...
end_free_gpu_memory   = torch.accelerator.get_memory_info()[0]   # model_runner.py:901
```

— so a free issued after the loop returns cannot appear in it. NVML sampled to the
end of the run can, which is what
[`analysis/gpu_steady_memory.py`](../analysis/gpu_steady_memory.py) reads. The two
sources agree to 0.01 GiB wherever they overlap: for §4's pool arms NVML gives
+1.08 and +3.21 GiB against vLLM's +1.09 and +3.22.

Readiness for the hoist arms is negative in all six paired passes (−4.49s, −1.77s,
−5.36s for `ch`; −0.26s, −7.25s, −1.68s for `c2`) but the node was noisier during
them — base-arm readiness spread 3.6s against 1.8s in the matrix above, and the
smallest delta is a pass where imports alone drifted +2.25s. So the phase delta is
the figure to quote for the hoist, and the clean end-to-end number stays the −4.00s
from the deletion matrix, whose phase saving (−4.04s) matches the hoist's (−4.18s)
to within its own spread.

**What to propose upstream, then:** keep the frees, move them. One
`synchronize()` + `empty_cache()` after the capture loop in place of 3366 inside
`torch.cuda.graph.__enter__` — worth ~4s of a 43s startup, costing no memory, and a
precondition for §1 rather than a competitor to it. The call is there to "free as
much memory as we can for the graph" (torch's own comment), and the measurement says
one call after the loop achieves that purpose exactly as well as 3366 calls inside
it: same steady-state memory, four seconds cheaper.

Whether the right home for the fix is torch (make `torch.cuda.graph`'s prologue
optional) or vLLM (stop using `torch.cuda.graph` for bulk capture) is an upstream
question this measurement does not settle. Either way blocker #1 in §5 goes with
it.

## 5. What vLLM would have to change

Every item here is in the image this repo measures.

| # | blocker | where |
|---|---|---|
| 1 | `torch.cuda.graph.__enter__` device-syncs, which aborts a sibling capture (§2). Both vLLM capture paths use that context manager. | `torch/cuda/graphs.py:437`; `v1/worker/gpu/cudagraph_utils.py:365`, `compilation/cuda_graph.py:313` (the two lines §3a attributes all 3366 captures to) |
| 2 | One process-global graph pool, handed out by a cached accessor at three call sites and *also* published through a module global that a concurrent capture would race on. | `platforms/interface.py:1152`; `cudagraph_utils.py:131`, `compilation/cuda_graph.py:200`, `compilation/breakable_cudagraph.py:282`; `pynccl_allocator.py:43,63` (`_graph_pool_id`) |
| 3 | `capture_error_mode` is never set anywhere in the vLLM tree (grep: zero hits), so every capture runs in torch's `"global"` default — the mode §1 shows is fatal to concurrency. | both capture sites |
| 4 | The capture context is entered **once** around the whole loop; per-thread capture needs per-thread streams inside it, and at TP>1 lockstep entry/exit across ranks plus custom-all-reduce buffer registration. | `distributed/parallel_state.py:1451`, `:619-645` |
| 5 | Capture inputs are shared, persistent buffers (`input_buffers`, `block_tables`, `self.hidden_states`) and the forward context is a module global. Two concurrent captures would record reads of a buffer the other is writing — a **silent** wrong-graph failure, not a crash. | `v1/worker/gpu/model_runner.py:880-889`, `forward_context.py` |
| 6 | Sizes are captured largest-first *because* the pool is shared; with N pools the ordering guarantee has to be restated per pool. And the extra pool memory is invisible to memory planning, because v2's `profile_cudagraph_memory()` returns 0 (§4). | `cudagraph_utils.py:355-374`; `model_runner.py:843`, `gpu_worker.py:532` |

Item 5 is the one that makes this genuinely hard rather than merely fiddly, and
it is the reason this document does not include a threaded-vLLM arm: a
measurement whose failure mode is "the graphs are subtly wrong" is not a
measurement, it is a liability. Any real attempt must assert on generated
**output**, exactly as
[on-demand capture](on-demand-cudagraph.md#2-what-is-broken-the-capture-branch-returns-uncomputed-output)
must.

## 6. Where this sits against the other levers

| lever | startup Δ | what it spends | ready to try? |
|---|---|---|---|
| **Hoist the per-capture prologue** out of the loop (§4b) | **−4.0s measured, end to end** | **nothing** — the hoisted arm's steady-state memory is base's, to 0.00 GiB | **yes, and it is first** — and it is a prerequisite for everything below |
| Honour `cudagraph_num_of_warmups` on v2 | 2.05-2.20s | nothing | yes — a config read |
| Capture on demand ([design](on-demand-cudagraph.md)) | 11.4s off readiness, 3.3s deleted | one transient per shape, first touch | needs a latent bug fixed first |
| Make the PIECEWISE forward cheaper | up to ~6.3s, of which ~2.8s is the §4b prologue | nothing | the residual gap is 4.6x, not 7.7x (§3a) |
| **Concurrent capture, 2 threads** | **~0.4-0.7s** of what remains after §4b | **~1.08 GiB per thread**, unbudgeted on v2, a second core, and items 1-6 | last |

Concurrency is the only one of these that trades steady-state GPU memory for
startup seconds. That is the same shape of trade the README rejects for
shortening `cudagraph_capture_sizes` — with a much smaller prize, and one that
shrinks further once the lever above it lands. It is *feasible*; it is not next.

## Relationship to the other two answers

* [On-demand capture](on-demand-cudagraph.md) removes the loop rather than
  parallelising it, so the two do not compose: there is nothing left to overlap.
  If on-demand capture lands, concurrency is moot. §4b composes with it, though —
  a deferred capture still pays the prologue per shape.
* [Foundry](foundry.md) replays a saved pool on a warm node, so it also has
  nothing to overlap.

Concurrent capture only matters in the world where vLLM keeps capturing every
graph at startup. It is the smallest of the three answers to the largest phase
this repo cannot configure away — worth knowing the price of, worth not spending
first. The reason to have asked is what pricing its prerequisites turned up.
