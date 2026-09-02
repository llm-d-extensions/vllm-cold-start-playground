# On-demand CUDA graph capture

`cudagraph capture` is **11.4s** of a 43.3s time-to-ready for Qwen3-32B on one
H100 — 26%, and the largest phase after weight load. The README explains why no
*configuration* removes it: every knob that shrinks capture shrinks the set of
captured batch sizes, and batches then pad up to a larger graph forever
([why that is not a step](../README.md#cuda-graph-capture-114s-that-configuration-cannot-honestly-remove)).

This document asks a different question. Instead of capturing **fewer** graphs,
capture **the same** graphs **later** — lazily, on the serving path, the first
time each batch shape is actually seen.

The distinction matters because it is exactly the objection that killed the
configuration lever:

> On-demand capture reduces no graphs. The dispatcher's candidate set, the
> padding behaviour, and the steady-state replay path are **bit-for-bit what
> they are today**. The only thing that differs is a warm-up transient.

The conclusion up front: **the mechanism is already 80% present in vLLM, and the
remaining 20% includes one silent-corruption bug that must be fixed first.** It
is a genuine ~9-10s win on a realistic workload, but it is a vLLM code change,
not a knob, and it is not safe to switch on as-is.

Everything below is read out of the image this repo measures
(vLLM 0.28.0 vendor build, `/usr/local/lib/python3.12/dist-packages/vllm`,
**v2** model runner — see the v1/v2 trap in
[docs/instrumentation.md](instrumentation.md)) or measured from the `runs/w6-base-c{1,2,3}`
traces.

## 1. What already works

**The piecewise wrapper is already lazy.** `CUDAGraphWrapper.__call__`
(`compilation/cuda_graph.py:233`) creates a `CUDAGraphEntry` on first sight of a
batch descriptor and captures into it inline:

```python
entry = self.concrete_cudagraph_entries[batch_descriptor]
if entry.cudagraph is None:
    ...                       # capture
else:
    entry.cudagraph.replay()  # steady state
```

There is no startup-only guard on that branch. The 11.4s loop in
`compile_or_warm_up_model` is nothing but an *eager pre-warm* of this same code
path, driven over dummy batches.

**Upstream already anticipated this.** From `initialize_cudagraph_keys` in
`v1/cudagraph_dispatcher.py`:

> `# Note: we create all valid keys for cudagraph here but do not guarantee all`
> `# keys would be used. For example, if we allow lazy capturing in future PR,`
> `# some keys may never be triggered.`

**The capture-enabled gate is already open on v2.** `validate_cudagraph_capturing_enabled()`
(`compilation/monitor.py`) raises `RuntimeError("CUDA graph capturing detected at
an inappropriate time…")` when a module-level flag is off. The flag defaults to
`True`, and `set_cudagraph_capturing_enabled` is called from **exactly one
file** — `v1/worker/gpu_model_runner.py`, the *v1* runner (lines 6853, 6914,
6967, 7035). `grep -rn` across the v2 tree `v1/worker/gpu/` finds no caller. So
on the runner this build actually uses, capture outside startup is already legal;
nothing needs to be unlocked.

**Input and output addresses are stable by construction.** A captured graph bakes
in pointer values, so lazy capture would be unsafe if a real forward used
different buffers than `_dummy_run` does. It does not: the v2 runner passes
persistent `input_buffers`, `block_tables` and `intermediate_tensors` into
`CudaGraphManager.capture`, and `ModelCudaGraphManager.run_fullgraph` returns
slices of a persistent `self.hidden_states`. Real forwards write into those same
buffers. This is the one risk that turned out to be a non-issue.

## 2. What is broken: the capture branch returns uncomputed output

CUDA graph capture **records** kernels; it does not run them. Neither capture
site replays before returning:

* `compilation/cuda_graph.py` — `with torch.cuda.graph(...): output = self.runnable(...)`,
  then `entry.cudagraph = cudagraph; return output`.
* `compilation/breakable_cudagraph.py` — same shape, via
  `BreakableCUDAGraphCapture`.

The comment above the `return` explains why it returns the strong ref
("so that pytorch can correctly manage the memory during cuda graph capture") —
it does not claim the value is computed. Verified directly in the image:

```console
$ python3 -c '<capture x*2 with x=3, read out before and after replay>'
value returned by the capture branch (no replay): [0.0, 0.0, 0.0, 0.0]
value after an explicit replay():                 [6.0, 6.0, 6.0, 6.0]
```

At startup this is harmless — the output belongs to a dummy run and is thrown
away. On the serving path it is **silent corruption**: the first request at each
new batch size would return tokens sampled from an uninitialised buffer. No
exception, no warning, no log line. Wrong output that looks like output.

The fix is small — replay once after `capture_end`, before returning — but it
must land *before* laziness is enabled, and it needs a flag so the 102 startup
captures do not each pay an extra replay. **Any experiment that turns on lazy
capture without this produces garbage for the first request at every size**,
which is also why the arm must be validated on output content and not just on
`first_token` latency.

## 3. What is missing: FULL mode has no lazy path at all

The laziness in §1 is the **piecewise** wrapper only. FULL graphs are captured by
the manager itself (`v1/worker/gpu/cudagraph_utils.py:355-374`) into
`self.graphs[desc]`, and the runtime path has no capture branch:

```python
def run_fullgraph(self, desc):                      # line 412
    assert desc in self.graphs, f"No cudagraph for {desc}"
    self.graphs[desc].replay()
```

Worse, dispatch refuses to hand out any graph mode until the startup loop has
finished:

```python
def dispatch(self, ...):                            # line 382
    if self._graphs_captured and num_tokens > 0 and key in self._candidates:
        ...
        return desc
    return BatchExecutionDescriptor(cg_mode=CUDAGraphMode.NONE, ...)
```

`self._graphs_captured` is set `True` only on the last line of `capture()`.
So simply skipping `capture_model()` does **not** give on-demand capture — it
gives *eager forever*, i.e. the permanent throughput loss we already rejected,
just arrived at by a different route. This is the trap: the cheap version of this
experiment measures the wrong thing and looks like an 11.4s win.

Making it real needs three changes, in this order:

1. **Replay after capture** (§2), gated so startup does not pay it. Correctness
   prerequisite; nothing else is safe without it.
2. **Split the gate.** `_graphs_captured` currently means both "candidates are
   known" and "graphs exist". Separate them: let `dispatch` return a candidate
   whose graph is absent, and let `run_fullgraph` capture on miss instead of
   asserting.
3. **Enter the capture context per capture** — see §5.

## 4. What it is worth: 8.1s relocated, 3.3s eliminated

Median of `runs/w6-base-c{1,2,3}` (the step-5 config, 51 sizes × 2 modes =
102 graphs). Per-size figures are the median across runs of each run's
per-size mean.

| component | n | median | per size |
|---|---|---|---|
| `capture_forward` PIECEWISE | 51 | 6.325s | 124.0ms |
| `capture_begin` (FULL, manager) | 51 | 0.832s | 16.3ms |
| `capture_forward` FULL | 51 | 0.826s | 16.2ms |
| `capture_end` | 11 | 0.070s | — |
| **subtotal — real capture work** | | **8.053s** | |
| `warmup_forward` PIECEWISE | 51 | 1.270s | 24.9ms |
| `warmup_forward` FULL | 51 | 0.931s | 18.2ms |
| `prepare_inputs` | 153 | 0.168s | ~1ms |
| `capture_model` prologue + loop residue | | 0.886s | — |
| **subtotal — startup-only scaffolding** | | **3.258s** | |
| **`capture_model` total** | | **11.310s** | |

Two different fates:

* **3.26s is eliminated outright.** The eager warmup forward, the dummy input
  prep, and the `gc.collect()`/`empty_cache()`/`get_memory_info()` prologue exist
  only because startup capture has no real batch to work with. On the serving
  path the *real* forward that triggered the capture is the warmup, and the real
  batch is the input. This is dead work, not moved work.

  (The warmup forward is also the phase with the dead knob:
  `cudagraph_num_of_warmups` is read only by the v1 runner, so on v2 those 2.2s
  are unreachable by configuration. See the README's upstream table.)

* **8.05s is relocated**, not removed — spread over the first touch of each
  shape:

  | first batch at a new size | added latency |
  |---|---|
  | PIECEWISE (mixed prefill) | **~124ms** |
  | FULL (uniform decode) | **~33ms** |

  For scale, an eager forward at these sizes is ~25ms, so a capturing request is
  roughly 5x a normal one — once, per size, per mode.

**On a real workload the relocated cost is much smaller than 8.05s**, because
padding means only the sizes actually reached are ever captured. Startup captures
all 51 unconditionally; a decode-heavy deployment may only ever touch 10-15 of
them. At 10-15 sizes the transient is ~1.5-2.5s of extra latency total, and the
graph pool allocates a fraction of its 1.84 GiB. That is the case for doing this:
not "11.4s moved" but **11.4s off readiness, ~3.3s deleted, and only the graphs
you actually use ever built.**

## 5. The two risks that are not resolved

**Per-capture collective coordination (the hard one).** Capture must run inside
`graph_capture(device)` (`distributed/parallel_state.py:1451`), which enters
`get_tp_group().graph_capture(context)` and the PP equivalent; that in turn
enters `ca_comm.capture()` when custom all-reduce is active
(`parallel_state.py:619-645`). Today this context is entered **once**, around the
whole startup loop. On-demand capture enters it once **per capture, during
serving**. At TP=1 that is a stream swap and bookkeeping. At TP>1 every rank must
enter and leave in lockstep on the same shape, and custom all-reduce registers
graph buffers across ranks on exit. Lockstep is plausible — all ranks execute the
same batch shape each step — but "plausible" is not "verified", and a mismatch
here is a hang, not a wrong answer. **This is the blocker for anything beyond
TP=1.**

Also inside that path: `capture_begin` runs `gc.collect()` and
`empty_cache()` per graph — 16.3ms measured, and a full GC on the serving path.
That is already inside the 124ms/33ms above, but it is the kind of cost that
behaves worse under load than on an idle warmup.

**Memory moves from fail-fast to fail-under-load.** Deferring capture does *not*
shrink the KV cache: on v2 `profile_cudagraph_memory()` returns `0`
(`v1/worker/gpu/model_runner.py:843`, "NOTE(woosuk): It is TBD whether we keep
this API or not"), and capture already runs *after* the KV cache is sized and
allocated. So graphs come out of the same leftover headroom either way. What
changes is *when* a shortfall is discovered. Today, 1.84 GiB is claimed at boot
and an over-subscribed pod fails to start. With lazy capture it is claimed
incrementally while serving, so the same misconfiguration surfaces as an OOM
under traffic. Mitigation is to reserve the graph pool at startup without
capturing into it — which keeps the fail-fast property and still saves the 11.4s.

## 6. Verdict, and how to measure it here

On-demand capture is the right shape of answer to the 11.4s: it is the only
option examined in this repo that takes the whole phase off the readiness path
**without** giving anything back in steady state. It answers the objection to
shortening `cudagraph_capture_sizes` directly, because it changes no candidate
set and no padding decision.

It is not, however, a configuration change, and it is not free:

| | shorten `cudagraph_capture_sizes` | on-demand capture |
|---|---|---|
| startup saving | 9.1s measured | 11.4s |
| graphs captured | fewer | **same set, built on use** |
| padding / dispatch | **permanently changed** | identical |
| steady-state cost | **forever** | one transient per size |
| cost to adopt | a flag | vLLM code change + TP validation |
| correctness risk | none | **silent corruption until §2 is fixed** |

**A measurement in this harness is possible but should not be the naive one.**
Because the v2 capture gate is already open (§1), a `CS_*` probe could no-op
`capture_model()` with no vLLM patch, in the style of
[`coldstart/cs_fst.py`](../coldstart/cs_fst.py). But per §3 that arm measures
*eager forever*, not on-demand capture — it would report an ~11.4s win that does
not exist. A defensible arm needs the dispatch-gate split, and per §2 it must
assert on generated **output**, not just on `first_token`, since the failure mode
is silent.

Recommended order:

1. Report §2 upstream on its own. The capture branch returning uncomputed output
   is a latent bug independent of this idea — it is what makes the existing
   laziness in `CUDAGraphWrapper` unusable at runtime.
2. Fix §2 + §3 behind a flag, then measure at TP=1 with an output-correctness
   assertion, per [docs/experiments.md](experiments.md) (median of 3,
   interleaved, verify the arm engaged).
3. Only then look at TP>1 and §5.

## Relationship to Foundry

[Foundry](foundry.md) attacks the same 11.4s from the other end: persist the
captured graphs and reload them, so a warm node captures nothing. The two are
complements, not alternatives —

* Foundry wins on a **repeat** boot of a known configuration, and wins the whole
  phase including the 3.26s of scaffolding.
* On-demand capture wins on a **first** boot, an unknown shape mix, or any node
  where no snapshot exists, and needs no artefact store.
* Foundry's constraints (it pins `cudagraph_mode FULL_DECODE_ONLY`, and
  force-stomps `cudagraph_num_of_warmups` with `VLLM_USE_V2_MODEL_RUNNER=0`)
  do not apply here — on-demand capture works with the mode the deployment
  already wants.

If both existed, the ideal boot captures nothing at startup, loads what it has,
and lazily builds only the shapes the snapshot missed.
