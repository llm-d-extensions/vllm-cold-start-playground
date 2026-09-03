# Sleep mode: freeing the GPU without paying the 43.3s again

Every other document in this repo attacks *time-to-ready from a cold process*.
This one attacks a different, easier problem that turns out to be already
solved: **keep the vLLM process alive, hand the GPU back, and take it again in
~1.4s.**

The framing that makes it work: don't try to serialize the CUDA graphs. Free the
things that are *large* (weights, KV cache) and keep the things that are *small
but unreproducible* (the CUDA context, the loaded modules, the 1.84 GiB graph
pool, and — critically — the **virtual addresses** the captured graphs baked into
their kernel arguments). The CUDA VMM API is what makes that split possible:
`cuMemUnmap` + `cuMemRelease` return the *physical* pages to the driver while the
`cuMemAddressReserve`d range stays reserved, so a later `cuMemCreate` +
`cuMemMap` puts new physical memory behind the *same pointers* the graphs
already hold.

**vLLM ships this.** It is `--enable-sleep-mode`
(`vllm/device_allocator/cumem.py`, `vllm/device_allocator/sleep_mode_backend.py`),
and it needs no `cuda-checkpoint`, no CRIU, and no Foundry-style snapshot format.
This document is the measurement.

Sleep mode alone stops ~4 GiB short of releasing the card. Layering
`cuda-checkpoint` on top of it closes that gap — **measured to 0 MiB with the
graphs still replaying**, see [below](#composing-with-cuda-checkpoint-to-reach-0-mib).

And once the card is at 0 MiB, the snapshot can come back on a **different**
card: a warmed Qwen3-32B engine migrated from one H100 to another on the same
node in **4.13s** while a second tenant held 76 GiB of the original, and served
byte-identical output — see
[Restoring onto a different GPU](#restoring-onto-a-different-gpu).

## Measured, 2026-09-02

Qwen3-32B, one H100 80GB, `vllm serve --model Qwen/Qwen3-32B --load-format
fastsafetensors --max-model-len 8192 --enable-sleep-mode`, `VLLM_SERVER_DEV_MODE=1`.
Three cycles: level 1, level 1 again, level 2.

| | |
|---|---|
| GPU at ready | **76691 MiB** (weights 61.03 GiB + KV 7.92 GiB + graph pool 1.84 GiB) |
| GPU asleep (level 1) | **4269 MiB = 4.17 GiB**, `utilization.gpu` **0%** |
| freed | **69.70 GiB** — 61.68 GiB copied to pinned host RAM, 8.02 GiB discarded |
| **first** sleep | **26.03s** — dominated by first-time `cudaHostAlloc` of the 61.68 GiB pinned backup |
| **subsequent** sleeps | **1.26s** (52.6 GB/s D2H — pinned bandwidth; PyTorch's pinned-host cache is warm) |
| **wake_up** | **1.45s** and **1.46s** (45.7 GB/s H2D) |
| generation after wake | **byte-identical to the pre-sleep baseline, both cycles** |

The residual is not perfectly stable run to run: a second boot of the same arm
slept to **2.75 GiB / 2817 MiB** rather than 4.17 GiB. It is retained
PyTorch-allocator blocks and workspace state, so read it as "roughly 3-4 GiB",
not a constant.

The correctness check is the load-bearing one. The probe request was a 6-token
prefill plus 32 decode steps at `temperature 0` — so it replays a **PIECEWISE**
graph for the prefill and 32 **FULL** decode graphs, captured *before* the sleep,
against physical pages that were released and re-created *after* it. Identical
output means every baked device pointer in those graphs still resolves to the
right tensor. That is the whole hypothesis, verified end to end.

```
[21:14:01] GPU ready:    76691 MiB, 0 %
[21:14:02] OUT: ' Paris.  True or False?  Explain your answer. ...'
[21:14:28] GPU ASLEEP:    4269 MiB, 0 %      <- 69.70 GiB freed, process alive
[21:14:33] wake, then
[21:14:34] OUT: ' Paris.  True or False?  Explain your answer. ...'   IDENTICAL
```

## Why 4.17 GiB stays, and whether that is a problem

Only allocations made inside a `CuMemAllocator.use_memory_pool(tag=...)` context
are managed by sleep. There are exactly two such sites:

* `gpu_worker.py:452` — `tag="weights"`
* `gpu_worker.py:679` — `tag="kv_cache"` (one statement; the context closes at 680)

`capture_model()` is at `gpu_worker.py:732`, in a *different method*
(`compile_or_warm_up_model`) and outside every pool. So the **1.84 GiB graph pool
is a plain PyTorch allocation and survives sleep by design** — it has to, because
the graphs must stay replayable. The rest of the 4.17 GiB is the CUDA primary
context, the loaded cubins/modules, cuBLAS/FlashInfer workspaces, and persistent
input/output buffers. Upstream is explicit that it designs around this boundary:

```python
# gpu_worker.py:686
# Build KV-zero metadata outside the CuMem pool so the bookkeeping
# GPU tensors (seg_addrs, block-id buffers) use the standard PyTorch
# allocator and are not discarded during sleep/wake cycles.
```

Against the requirement "the processes may stay alive, but no GPU utilization":
**SM utilization is 0%** — a slept engine issues no work at all. What remains is
**4.17 GiB of held memory** — 5.3% of the card's 79.18 GiB usable. Whether that satisfies the
requirement depends on which resource is actually scarce:

* If the goal is **not to compute** — free the SMs for a co-scheduled tenant via
  MPS or time-slicing — sleep mode already delivers it, completely.
* If the goal is **to fit another 76 GiB model on the same card**, 4.17 GiB is in
  the way, and so is Kubernetes: `nvidia.com/gpu: "1"` assigns the device to the
  pod for the pod's lifetime. Freeing device memory does not return the device to
  the scheduler. That needs MIG, time-slicing, or a device plugin that can
  hot-detach — an orchestration change, not a CUDA one.
* If the goal is **absolute zero bytes**, only tearing down the CUDA context does
  it, and then the graph pool and the graphs go too. That is where
  `cuda-checkpoint` would earn its place — see below.

## Levels, and a level-2 footgun

`v1/engine/core.py:867-882` documents the semantics, and
`sleep_mode_backend.py:120-133` implements the mapping:

| level | `offload_tags` | weights | KV cache | prefix cache |
|---|---|---|---|---|
| 0 | — | untouched | untouched | kept |
| 1 | `("weights",)` | copied to pinned host RAM | **discarded** | cleared |
| 2 | `()` | **discarded** | **discarded** | cleared |

Discarding the KV cache is safe precisely because `clear_prefix_cache = level >= 1`
(`core.py:881`) — no scheduler state survives pointing at KV blocks whose contents
are gone.

Level 2 is a trap as exposed. It parks in 0.066s and "wakes" in 0.248s, both
HTTP 200 — and then generates `'!!!!!!!!!!...'`. That is the documented
consequence (level 2 means "reloaded from the model source on resume",
`sleep_mode_backend.py:56-58`) but `POST /wake_up` does not reload anything: it
re-maps fresh, uninitialised physical pages behind the weight pointers. The
caller is expected to push weights in itself (the RLHF weight-update path).
**The failure is silent** — same shape of silent failure as the VMM
address-ordering invariant, and worth an upstream guard.

### `reload_weights` exists — and crashes the engine rather than fixing it

vLLM already ships the piece that footgun seems to be missing:
`Worker.reload_weights()` → `GPUModelRunner.reload_weights()`
(`gpu_worker.py:470`, `gpu_model_runner.py:5643`) re-runs the model loader
against the original checkpoint path and calls `model.load_weights(...)`
straight into the existing, already-mapped parameter tensors — exactly
"reload from disk," with no route ever attached to it. Wiring one up
(`coldstart/cs_dev_routes.py`, `POST /reload_weights` → `collective_rpc`) and
calling it after a level-2 `/wake_up` on a warmed Qwen3-32B engine (TP=1,
graphs captured) does not fix the garbage output — it crashes EngineCore
outright on the next request:

```
NotImplementedError: _C_cache_ops::reshape_and_cache_flash: attempted to run
this operator with Meta tensors, but there was no fake impl or Meta kernel
registered. ...
vllm.v1.engine.exceptions.EngineDeadError: EngineCore encountered an issue.
```

`reload_weights()` only re-populates the `weights` tag; it does nothing about
whatever the level-2 `wake_up` left inconsistent elsewhere (a buffer ends up on
the `meta` device), and the CUDA graphs captured before the call make that
corruption fatal on the very next forward pass rather than merely wrong. So the
mechanism upstream ships for "reload from disk" is real and callable, but not
sufficient on its own — level 2 needs the rest of a reset path, not just a
weight refill. Full transcript:
`reports/demo-cold-start-tp1-20260903-105740.txt` (step 6).

For cold start, level 2 is the interesting one anyway *if* a **working**
weight-reload path existed: it would cost 13.3s of weight load on wake and
hold no host RAM, while still skipping imports, compile, capture and warmup —
~30s better than a cold boot. As shipped, use level 1.

## Cost, and what it is actually good for

Level 1 buys the 1.4s wake with **61.68 GiB of pinned host RAM per parked
replica**. `PIN_MEMORY` is `True` on this platform (`utils/torch_utils.py:74` →
`platform_utils.py:44`), which is what puts the transfers at ~50 GB/s rather than
~18 GB/s pageable — the difference between 1.4s and 3.9s. The container cgroup
caps memory at **128 GiB** (`manifests/pod-exec.yaml`), so **one parked 32B
replica per container**, not several. Pinned memory is also locked: it is not
reclaimable under pressure.

So this is **hibernation for a warm replica**, not a cold-start optimisation:

| | cold boot | level-1 wake |
|---|---|---|
| time | 43.3s | **1.45s** |
| GPU held while idle | 0 | 4.17 GiB |
| host RAM held while idle | 0 | 61.68 GiB pinned |
| process | none | alive |

It is the right tool for scale-to-zero-*compute* under bursty traffic, and the
wrong tool for packing more distinct models onto a card or for a genuinely cold
node. It composes with everything else in this repo rather than competing: the
43.3s still has to be paid once, per replica, per node.

## Composing with cuda-checkpoint to reach 0 MiB

The 4.17 GiB is the CUDA context, the loaded modules and the graph pool — the
state that has no other escape route, because the only way to release it is to
destroy the context, and destroying the context is what would take the graphs
with it. That is exactly `cuda-checkpoint`'s job description, and the residual is
a *far* better target for it than the whole 68.8 GiB: sleep mode has already moved
the bulk out through a path vLLM controls.

Two things had to be true for the two to compose, and neither was obvious:

1. **The graphs must survive a context destroy/recreate.** The executable object
   is driver-side — a DAG of kernel nodes holding `CUfunction` handles into loaded
   modules. Destroying the context ought to invalidate all of it.
2. **The unbacked VMM reservations must survive.** Proven above: `sleep()` calls
   `unmap_and_release` and leaves 69.70 GiB of *reserved-but-unmapped* virtual
   address space behind. Those ranges hold no bytes to copy — they are pure
   context-side bookkeeping — so there is no reason to assume a checkpoint tool
   would recreate them. If it does not, `wake_up`'s `cuMemMap` at the recorded
   pointer has nothing to map into.

**Both are true.** Isolated first, with a 512 MiB tensor, a captured graph, a
release-physical-keep-VA step, and an external checkpoint/restore in between:

```
GPU with graph + unbacked reservation:  2271 MiB
checkpoint  ->  0 MiB      (sustained; context destroyed, device released)
restore     ->  2271 MiB
a2@0x302000000  VA_MATCH=True      <- unbacked reservation came back
REPLAY out[0]=3.0 -> CORRECT       <- graph captured pre-teardown still valid
```

Then on real vLLM. `cuda-checkpoint` 580.105.08 (matching the driver) against the
**EngineCore** pid — the only process holding CUDA at TP=1:

| step | wall | GPU after |
|---|---|---|
| ready | — | 74627 MiB |
| `POST /sleep?level=1` | 22.20s first, ~1.3s warm | 2817 MiB (2.75 GiB) |
| `--action lock` | 0.03s | |
| `--action checkpoint` | **2.11s** | **0 MiB**, sustained over 5s |
| `--action restore` | **3.58s** | 2817 MiB |
| `--action unlock` | 0.03s | |
| `POST /wake_up` | 1.48s | 72139 MiB |
| generation | — | **byte-identical to baseline** |

**Unpark from a genuinely idle GPU to serving: 5.09s**, against 43.3s cold — and
the graphs replay correctly, verified the same way (PIECEWISE prefill + 32 FULL
decode graphs, exact string match).

Note what the timings say about where the cost is. The 2.75 GiB copy is worth
~0.08s at pinned bandwidth, but checkpoint takes 2.11s and restore 3.58s. Neither
is bandwidth-bound: they are **context teardown and rebuild**, including reloading
every module. So shrinking the residual further would buy almost nothing — 3.58s is
the floor for getting a CUDA context back, whatever it holds.

The APIServer survived a 5-second EngineCore freeze without declaring the engine
dead. A longer park may trip a heartbeat; untested.

### Is the composition actually better than checkpointing alone?

Honestly: **not on speed.** Checkpointing the un-slept process moves 68.8 GiB at
~1.38s D2H / ~1.36s H2D, so park and unpark land in the same few seconds either
way, and the host RAM held is ~the same (64.4 GiB composed — 61.68 GiB of pinned
sleep backup plus the 2.75 GiB checkpoint — against 68.8 GiB alone).

What the composition buys is **granularity and control**, which sleep mode has and
`cuda-checkpoint` does not:

* **Partial wake.** Restore the context (3.58s), then `wake_up(tags=["kv_cache"])`
  only — 8.02 GiB instead of 69.70 GiB — and bring weights up lazily or streamed.
  A whole-process checkpoint is all-or-nothing.
* **A choice about where the weights live.** With level 2 the composed park holds
  *no* host RAM for weights at all and reloads them from the page cache on wake
  (13.3s) — still ~30s better than cold, and it frees 61.68 GiB of pinned host RAM
  per parked replica, which is the binding constraint at 128 GiB per container.
  This needs the level-2 reload path that does not currently exist (see the footgun
  above).
* **Less state for the checkpoint to get right** — 2.75 GiB and no multi-GiB
  mappings, versus 68.8 GiB.

So the honest summary is that `cuda-checkpoint` alone is sufficient for warm
standby, sleep mode alone is sufficient if 4.17 GiB of held memory is acceptable,
and the composition is what you want if you need **both** zero GPU **and** control
over the 61.68 GiB.

### The remaining blocker is Kubernetes, not CUDA

`nvidia-smi` reads 0 MiB and the device is genuinely free — but
`nvidia.com/gpu: "1"` has assigned it to this pod for the pod's lifetime. Nothing
in the composition returns the device to the scheduler, so no other pod can use
the GPU it just released. Exploiting this needs MPS, time-slicing, MIG, or a device
plugin that can hot-detach. That is now the only unsolved part, and it is an
orchestration problem.

A weaker version of the goal *is* reachable today: handing the device to another
process **inside the same pod**, by restoring the snapshot onto a different card.
That is measured in [Restoring onto a different GPU](#restoring-onto-a-different-gpu)
below.

**TP>1 is not just untested — it deadlocks, and the reason is structural.**
Wiring `Worker.checkpoint_prepare` / `checkpoint_restore` (`gpu_worker.py:261`)
up as HTTP routes (`coldstart/cs_dev_routes.py`) and running the whole sequence
on a warmed Qwen3-32B TP=2 engine (2×H100, NVLink) reproduces the composition up
to a point and then hangs: `/sleep?level=1` and `/checkpoint_prepare` both
complete normally (11.36s and 0.06s), then `cuda-checkpoint --action checkpoint`
against the first rank's pid blocks — confirmed stuck for 29+ minutes, not slow.
Subsequent `--get-state` calls on the same pid block too, so the driver's
per-process checkpoint lock is held by the wedged call, not merely by the client.
Full transcript: `reports/demo-cold-start-tp2-20260903-110308.txt`.

Bisecting that with real vLLM boots costs ~90s a trial and moves a dozen
variables at once, so `scripts/cuda-checkpoint-ipc-matrix.py` reproduces it
without vLLM: two spawned children, one GPU each, each holding exactly *one*
kind of cross-rank state, then the same lock → checkpoint → restore → unlock
sequence. Measured on driver 580.105.08:

| cross-rank state held by the process | `checkpoint` | `restore` |
| --- | --- | --- |
| none (control) | ok, 0.37s / 0.41s → 0 MiB | ok, compute verified |
| `cuIpcOpenMemHandle` peer mapping live | ok, 0.62s / 0.67s → 0 MiB | **fails: `"invalid argument"`** |
| … peer mapping freed first | ok | ok, verified |
| **live NCCL communicator** | **hangs** | — |
| … `destroy_process_group()` first | ok, 0.46s / 0.52s | ok, verified |
| **torch symmetric memory live** | **hangs** | — |
| **CUDA graph with an all-reduce kept, then destroy** | `destroy_process_group()` **never returned in 30s** → **hangs** | — |
| … graph dropped first, then destroy | destroy 0.19s, checkpoint 0.47s → 0 MiB | ok, verified |

That pins the root cause to three facts that compose badly:

1. **`checkpoint_prepare` does not release what TP actually holds.**
   `CudaCommunicator.checkpoint_prepare` (`cuda_communicator.py:588`) calls
   exactly two things — `checkpoint_prepare_fi_ar_workspaces()` and
   `all2all_manager.checkpoint_prepare()` — under its own comment, *"Only
   FlashInfer all-reduce and FlashInfer all2all are supported for now"*. The
   stock all-reduce dispatch chain on this node logs as
   `Using ['CUSTOM', 'SYMM_MEM', 'PYNCCL'] all-reduce backends`, and
   `pynccl_comm` is built unconditionally whenever `world_size > 1`. None of
   those three is released. Hence 0.06s and 0 MiB freed: the call succeeds
   having done nothing that matters, which is why it reads as a working
   teardown right up to the hang.
2. **Live NCCL state, or torch symmetric memory, is enough to wedge the driver
   on its own** — rows 4 and 6 above, with nothing else shared. Releasing it
   first is sufficient to fix the checkpoint (rows 5 and 8). A live `cuIpc`
   mapping — mechanically what `CUSTOM` all-reduce uses — is subtler: it
   checkpoints *fine* and then fails to **restore**, which is consistent with
   `cuda-checkpoint`'s own README (`cuIpcGetMemHandle`-based checkpointing is a
   driver-610 feature; `cuMemExportToShareableHandle` memory, which is what
   symmetric memory allocates, is listed unsupported outright).
3. **Releasing the NCCL communicator requires destroying the CUDA graphs
   first**, and those graphs are the entire reason sleep level 1 exists.
   vLLM says so itself at `pynccl.py:148`: `ncclCommAbort` "can block until all
   CUDA graphs that captured NCCL ops on this comm are destroyed". Measured
   both ways in the last two rows: with a captured all-reduce still alive
   `destroy_process_group()` never returned in 30s; drop the graph first and it
   returns in **0.19s**, after which the full round trip to 0 MiB and back
   works, compute verified.

So on this stack **"park a TP>1 engine to 0 MiB" and "keep the captured CUDA
graphs" are mutually exclusive.** At TP=1 the warmup/capture phase those graphs
come out of is 11.3s of the 43.30s cold start (`warmup.compile_or_warm_up`, of
which 1.75s is instrumented `cudagraph.capture_begin` time) — precisely the cost
level 1 exists to preserve — so paying it back on every wake would cancel most of
what the park buys.

The earlier reading of this hang recorded in the transcript — a *mutual wait*
between the two ranks, each blocked on state the other could not supply because
only one rank was ever mid-checkpoint — **is wrong, and the matrix disproves
it.** `--parallel`, which puts both ranks mid-checkpoint simultaneously, hangs
identically; and the NCCL and symmetric-memory rows hang with only one child
ever touched. The `futex_wait_queue` in `/proc/<pid>/wchan` is a symptom of the
wedged driver call, not a rank-to-rank deadlock. Ordering is not the bug.

**The one backend with a real answer is not enough by itself.** flashinfer
implements exactly the operation this needs, under the name *Stable-VA
checkpointing* (`flashinfer/comm/allreduce.py:204`): unmap the physical backing
but keep the virtual address, so a CUDA graph that baked that address in still
replays. That is why `checkpoint_prepare` supports FlashInfer and nothing else.
`scripts/tp2-park-probe.sh --arm flashinfer` forces the engine onto it
(`VLLM_ALLREDUCE_USE_FLASHINFER=1`, `VLLM_ALLREDUCE_USE_SYMM_MEM=0`,
`--disable-custom-all-reduce`) and the detach demonstrably starts working: the
chain drops to `Using ['FLASHINFER', 'PYNCCL'] all-reduce backends` and
`/checkpoint_prepare` now frees **512 MiB** (4033 → 3521 MiB per GPU) where the
stock arm freed nothing. It still hangs — `PYNCCL` is still in the chain, is
still built unconditionally, is still captured in the graphs, and has no
stable-VA path. Transcript:
`reports/tp2-park-flashinfer-20260903-130336.txt`.

Which makes the upstream ask precise, and small:

* extend the stable-VA detach to `PyNcclCommunicator` — `custom_all_reduce.py:505`
  already has a `close()`, `symm_mem.py` has no teardown at all, and `pynccl.py`
  has a `destroy()` that cannot be called while graphs live;
* or have `checkpoint_prepare` fail loudly instead of returning 0.06s of success
  when the active backends are ones it cannot release. As shipped it reports
  success and the caller then hangs in the driver with the per-process
  checkpoint lock held, which is the worst available failure mode.

Recovery, for anyone who hits this: the wedge is process-level, not
unrecoverable driver state — `SIGKILL` the tree and both GPUs read 0 MiB again.
Kill by pid (`nvidia-smi --query-compute-apps=pid`), not by `pkill -f 'vllm
serve'`: the workers rename themselves to `VLLM::Worker_TP0` via setproctitle, so
that pattern misses them and leaves several GiB stranded.

## Restoring onto a different GPU

The Kubernetes blocker above is about giving a device back to the *scheduler*.
There is a weaker but immediately useful version of the same goal: giving it back
to *another process on the same node*. That needs the snapshot to come back on a
different card than it left, and `cuda-checkpoint` has a flag for exactly that:

```
--device-map oldUuid=newUuid,...
        Optionally remap devices during restore.
        Must contain all checkpointed devices.
```

It works. It is also considerably fussier than that one sentence suggests, and
every rule below was established by measurement rather than by reading the help
text — including one that made the flag look unsupported at first.

### Three rules, all load-bearing

Micro-probe first: a process holding 2 GiB and one captured CUDA graph on GPU0,
on a 2× H100 node
([`scripts/cuda-checkpoint-cross-gpu.sh`](../scripts/cuda-checkpoint-cross-gpu.sh)).

| case | map | rc | memory after | graph replay |
|---|---|---|---|---|
| all visible, identity (control) | `U0=U0,U1=U1` | 0 in 1.15s | `0, 2773  1, 4 MiB` | 3.0, all correct |
| **all visible, swap** | `U0=U1,U1=U0` | **0 in 1.04s** | **`0, 4  1, 2773 MiB`** | **3.0, all correct** |
| only GPU0 visible, remap to GPU1 | `U0=U1` | 1 `invalid argument` | unchanged | — |
| non-permutation | `U0=U1,U1=U1` | 1 `invalid argument` | — | — |

1. **The map must list every device *visible to the process*, not every device
   that holds a context.** This is the rule that hides the feature. With two
   GPUs visible and one context, every single-entry map is rejected — so a first
   pass through the plausible spellings (bare hex, `GPU-` prefix, uppercase,
   cross and identity alike) returns `invalid argument` for all of them, and the
   obvious conclusion is that cross-GPU restore is unsupported. The control that
   breaks the tie is an **identity** map: `U0=U0` also fails, which cannot mean
   "remapping is impossible" — it can only mean the flag is not parsing. Adding
   the second, entirely unrelated device makes it parse.
2. **The map must be a bijection.** Collapsing two devices onto one is rejected.
3. **The target must be visible to the process.** `CUDA_VISIBLE_DEVICES=0` plus a
   remap to GPU1 is rejected. This is the constraint that decides whether the
   whole technique is available, and it is why the premise *"assuming vLLM sees
   all GPUs"* is not a convenience — it is a requirement.

Two further properties, both worth having: a **failed** remap is
non-destructive (a plain `--action restore` afterwards still succeeds, so a
rejected map costs nothing), and the restored process is entirely **unaware** it
moved. It still reports `cuda:0`, the same virtual address `0x402000000`, and
even the *old* UUID from `get_device_properties()`. The remap happens below the
runtime; nothing in user code observes it.

### On real vLLM: Qwen3-32B changes cards in 4.13s

The composition that matters is sleep + checkpoint + remap + wake, with a real
tenant holding the original card in between
([`scripts/vllm-cross-gpu-restore.sh`](../scripts/vllm-cross-gpu-restore.sh)).
Qwen3-32B, TP=1, one engine warmed to ready with all 51×2 graphs captured:

| step | wall | GPU0 | GPU1 |
|---|---|---|---|
| ready | — | 74629 MiB | 4 MiB |
| `/sleep?level=1` | 22.807s | 2819 MiB | 4 MiB |
| `--action checkpoint` | ~2s | **0 MiB** | 0 MiB |
| another tenant claims GPU0 | — | 78351 MiB | 0 MiB |
| `restore --device-map` (swap) | **2.67s** | 78351 MiB | 2819 MiB |
| `/wake_up` | **1.46s** | 78351 MiB | **72141 MiB** |
| completion | — | byte-identical to baseline | |

The interesting line is `/wake_up`. A checkpoint/restore cycle plausibly only has
to relocate what it captured — but wake makes **61.68 GiB of entirely new
allocations** (weights re-mapped from the host copy, KV cache re-created), and
those are `cuMemCreate` calls issued *after* the remap, by a process that still
believes it is on `cuda:0`. They follow the remap. `nvidia-smi
--query-compute-apps` is unambiguous: EngineCore pid 497 at 71994 MiB on
`GPU-48596602` — the second card — while the other tenant holds 78342 MiB on
`GPU-069d6800`, the card the engine booted on.

So the remap is a property of the *process's* device binding, not of the
particular allocations that existed at checkpoint time. That is what makes this a
migration primitive rather than a curiosity.

Unpark-onto-a-different-card is **2.67 + 1.46 = 4.13s**, against 43.3s to boot a
new engine. The park itself is the slow half (22.8s of sleep), but the park is
off the critical path — it happens when the replica goes idle, not when traffic
arrives.

### What it buys, and the price on the invoice

It buys **defragmentation within the pod's own allocation**. A pod holding *n*
GPUs can park an idle engine, hand its physical card to whatever needs a
contiguous device, and bring the engine back on a different one — without
re-reading 62 GiB of weights or re-capturing a single graph.

The price is that rule 3 forces the pod to see every GPU it might move between,
which means requesting them all and doing its own assignment. That discards the
device-plugin isolation model: the scheduler's per-GPU accounting becomes
fiction, `nvidia.com/gpu` no longer describes what any single workload is using,
and nothing prevents two processes in the pod from targeting the same card. For a
single multi-GPU serving pod that already owns all its devices, that is not a
regression — it is describing what is already true. As a general mechanism it is
a real trade, and it should be written down as one rather than sold as free.

Still untested for a real remap: **TP>1**, which must additionally detach and
re-attach every device communicator across the remap via
`Worker.checkpoint_prepare` / `checkpoint_restore` (`gpu_worker.py:261`). At TP=1
there is no communicator, so the run above says nothing about it — and it is moot
until the base case is fixed: a same-GPU, identity-map TP=2 checkpoint never
completes at all, so a remap never comes into play. See
[the TP>1 finding above](#composing-with-cuda-checkpoint-to-reach-0-mib).

## The thin variant: park at level 2, refill weights on wake

The composition above holds 75.67 GiB of host RAM while parked, almost all of it
the pinned copy of the weights. There is an alternative shape that trades that
away: park with **level 2** — which discards the weights instead of copying them
— and on wake reload them from safetensors the way a cold start does, into the
address space that is still standing.

The mechanism this needs already exists, in three of its four parts:

* **The virtual addresses survive.** `CuMemAllocator.sleep()` calls
  `unmap_and_release`, which calls `cuMemUnmap` and `cuMemRelease` but never
  `cuMemAddressFree` (established by disassembling `cumem_allocator.abi3.so`).
  A slept engine keeps ~69.70 GiB of reserved-but-unmapped VA.
* **Re-mapping lands at the identical addresses.** This is not inference: a
  level-1 wake produces byte-identical output, which it could not if the captured
  graphs' baked-in pointers had moved. The micro-probe shows the same across a
  *remap* — same `0x402000000` before and after.
* **vLLM's weight loader already copies into existing parameter tensors** rather
  than allocating new ones, which is the right shape for refilling in place.
* **The wake-path call itself exists**: `Worker.reload_weights()` →
  `GPUModelRunner.reload_weights()` (`gpu_worker.py:470`,
  `gpu_model_runner.py:5643`) is exactly "run the loader after a level-2
  sleep" — it is just never wired to a route.

So the plumbing is not missing; it is broken. Today a level-2 wake is a footgun
— `/wake_up` returns HTTP 200 in 0.248s and the model then emits
`'!!!!!!!!!!...'`, because the VAs are re-mapped to *uninitialised* memory and
nothing refills them — and calling `reload_weights()` afterwards does not fix
that: measured on a warmed Qwen3-32B engine, it crashes EngineCore instead
(`NotImplementedError` on a `meta`-device tensor → `EngineDeadError`, see the
[finding above](#reload_weights-exists--and-crashes-the-engine-rather-than-fixing-it)).
`reload_weights()` refills the `weights` tag correctly but does nothing about
whatever else level 2's wake left inconsistent, and captured CUDA graphs turn
that into a hard crash on the next forward pass. The gap is not a missing code
path anymore — it is that the existing path only covers one of the things a
"reset the worker to just before the first CUDA context" wake needs to do.

The trade, in measured numbers:

| | host RAM while parked | unpark | correctness |
|---|---|---|---|
| level 1 + checkpoint | **75.67 GiB** | **4.13s** (2.67 restore + 1.46 wake) | verified byte-identical |
| level 2 + checkpoint + refill | **6.61 GiB** | ≈ **16s** (2.7 restore + 13.3 weight load) | **crashes EngineCore, measured** |
| cold start | 0 | 43.3s | — |

The 13.3s is this repo's measured `weight load` phase with
`--load-format fastsafetensors` and a warm page cache; the 2.7s is the measured
restore, and a level-2 image has strictly less to remap, so it is an upper bound.

That is roughly **69 GiB of host RAM for ~12s of extra unpark** — and on a 128 GiB
container the RAM is the binding constraint, not the seconds. At 75.67 GiB exactly
one engine can be parked. At 6.61 GiB a dozen can, with room left for the live
one. If the goal is a pool of warm-but-idle replicas rather than a single standby,
the thin variant is the only one that fits, and 16s versus 43.3s is still a 2.7×
improvement on the thing this repo is measuring.

Both variants also keep the property that matters most here: they never touch
steady state. Nothing is recompiled, no graph is dropped, no capture size is
reduced — the graphs that replay after the restore are the same graphs, at the
same addresses, that were captured before it.

## Can the snapshot be serialized and reused by a fresh vLLM?

No — not with `cuda-checkpoint`, and the reason is structural rather than a
missing flag. But the arithmetic of the *hypothetical* is interesting enough to
be worth writing down, because it points at the one variant that would pay.

### `cuda-checkpoint` has no serialization, by design

Its entire interface is keyed on a live pid:

```
--get-state --pid <pid>
--action lock | checkpoint | restore | unlock --pid <pid> [--timeout <ms>] [--device-map <uuids>]
--toggle --pid <pid>
--get-restore-tid --pid <pid>
```

No output path, no format, no `--save`. `strings` on the binary finds one
file-related symbol, `fwrite`, which is its own stdout. `--action restore`
requires a process already in the `checkpointed` state — i.e. the process that
was checkpointed. There is no way to hand it bytes.

What `--action checkpoint` actually does is move device state into **the
checkpointed process's own host address space** and release the GPU. Measured on
a 2.75 GiB CUDA process:

| | `/dev/nvidia*` fds | nvidia mappings | RSS | GPU |
|---|---|---|---|---|
| running | 29 | 117 (incl. 27 × `/dev/nvidiactl`) | 0.67 GiB | 2771 MiB |
| **checkpointed** | **0** | 86 — all ordinary `.so` file mappings | **3.30 GiB** | **0 MiB** |
| restored | 27 | 117 | 0.66 GiB | 2771 MiB |

So it does not snapshot a process; it *converts a GPU-holding process into an
ordinary one*. Afterwards nothing GPU-specific is left in the address space —
zero device fds, and the only remaining `nvidia`-matching mappings are
file-backed `libcublasLt.so`, `libnccl.so` and friends, which any dumper handles
as ordinary files. `--get-restore-tid` exists purely so an external dumper knows
which thread to resume CUDA on.

That is the whole design: `cuda-checkpoint` is a **plugin to** a process
checkpointer, and serializing the result is deliberately somebody else's job.
The only thing that does that job is CRIU, which is outside the scope this
exploration was given.

vLLM's own extension surface agrees. `SleepModeBackend` declares
`suspend()` / `resume()` and the predicates `preserves_communicators()`,
`preserves_compiled_artifacts()`, `preserves_graphs_with_communicators()`
(`sleep_mode_backend.py:78-100`) — every one of them about a **live** engine.
There is no serialize/deserialize pair and no hook at which a fresh boot could
ingest an image, even though the registry comment names "CUDA checkpoint, CRIU,
durable snapshot" as intended backends.

### Transplanting the graphs into a fresh process is closed at the CUDA level

The other reading of the question — let vLLM boot normally, then hand it the
captured graphs instead of re-capturing — fails one level lower. The driver in
this image exports **95** `cuGraph*` symbols and not one produces or consumes
bytes:

```
$ nm -D --defined-only /usr/lib64/libcuda.so.1 | grep -cE 'cuGraph'
95
$ ... | grep -iE 'serial|export|import|save|load|write|read|blob|binary|file'
cuGraphUpload          # uploads an instantiated CUgraphExec to a stream; not serialization
```

The contrast is the point: `cuModuleLoadData` / `cuLibraryLoadData` *do* ingest a
binary blob, so CUDA has a serialized format for **kernels** (cubin, PTX) and
deliberately none for **graphs**. A graph is a bag of baked pointers into one
address space, and NVIDIA has never offered to relocate it.

### If a dumper were available, what would the image weigh?

This is the part worth knowing, because the answer decides whether the idea is
even attractive. Measured RSS of the parked process tree at the moment CUDA state
is fully host-resident and the GPU reads 0 MiB:

| parked at | APIServer | EngineCore | **image** | dominant mapping |
|---|---|---|---|---|
| level 1 | 1.42 GiB | 74.25 GiB | **75.67 GiB** | 69.17 GiB `rw-s /dev/zero (deleted)` |
| level 2 | 1.41 GiB | 5.20 GiB | **6.61 GiB** | 3.25 GiB `[anon]`, 1.53 GiB `[heap]` |

The 69.17 GiB shared mapping is `CuMemAllocator`'s pinned host backup of the
weights. `cuda-checkpoint` does not touch it — the checkpoint only turned
2.62 GiB of *device* state into anonymous pages (`anon` 2.19 → 4.81 GiB). So a
level-1 image carries the full 61.68 GiB weight copy, and this filesystem does
859 MB/s:

```
4294967296 bytes (4.3 GB, 4.0 GiB) copied, 5.00268 s, 859 MB/s
```

75.67 GiB at 859 MB/s is **~90s just to read the image** — comfortably worse than
the 43.3s cold boot it was supposed to replace. Level 1 is a non-starter on disk
even before anything hard begins.

Level 2 is the interesting one. Discarding the weights instead of copying them
collapses the image to **6.61 GiB** — ~7.9s from cold disk, ~0 from page cache —
and the weights come back the way they always do, from the safetensors the page
cache already holds. Sketching it against measured numbers:

| step | cost | source |
|---|---|---|
| read 6.61 GiB image | ~7.9s cold, ~0s warm | 859 MB/s, measured |
| `cuda-checkpoint --action restore` | 3.58s | measured |
| reload weights into the restored VAs | 13.3s | measured, `fastsafetensors` |
| re-map KV, resume | ~0.25s | measured level-2 wake |
| **total** | **~25s** vs **43.3s** | |

Roughly an 18s saving, and it lands on exactly the phases that are hardest to
attack otherwise: the 11.4s of graph capture and ~15.5s of interpreter and
framework import, both of which come back inside the image already done.

Note one non-obvious upside. This is **not** "vLLM starts the same way and then
loads a snapshot" — a CRIU restore does not start vLLM at all, it resurrects the
identical address space. That makes the determinism problem that dominates
[foundry.md](foundry.md#why-cuda-checkpoint-is-not-a-next-boot-shortcut)
disappear: there is no need for a VMM cursor to make two independent boots agree
on virtual addresses, because there is only one boot. Every baked pointer in
every captured graph is valid because it is literally the same address space.

### Why it still is not a cold-start answer here

Four things stand between that table and a working system, in increasing order of
hardness:

1. **Nothing can write the bytes.** `cuda-checkpoint` cannot, as shown above. Not
   a gap to work around — there is no API to call.
2. **The cluster forbids it.** `criu` is not in the vLLM image, and the pod runs
   with `capabilities: drop: ["ALL"]` — `CapEff: 0000000000000000`, so no
   `CAP_SYS_ADMIN` and no `CAP_CHECKPOINT_RESTORE`. (`ptrace_scope=0`, which is
   why `cuda-checkpoint` itself works same-uid; that is a much lower bar.) None
   of this was testable on this node, and the numbers above are therefore a price
   list, not a result.
3. **vLLM cannot reload weights after level 2.** Measured footgun: `/wake_up`
   after a level-2 sleep returns HTTP 200 in 0.248s and then generates
   `'!!!!!!!!!!...'`. It re-maps uninitialised pages and expects the caller to
   push weights itself. The 13.3s row in the table above is a path that would
   have to be built.
4. **The image is welded to its environment.** It is a byte-identical process,
   so it is only valid for the same driver version, the same GPU model
   (`--device-map` remaps UUIDs, which covers a different card of the same model
   and nothing more), the same CUDA libraries, and the same vLLM flags. Change
   `--max-model-len` and the image is void. That is a fleet-wide cache-key
   problem of the same shape as
   [foundry.md](foundry.md#archive-shape-and-cache-key-implications) describes,
   with a stricter key.

The honest summary: serialization is not a `cuda-checkpoint` feature and cannot
be made into one. A CRIU-based thin (level-2) image is the only variant whose
arithmetic works, it would need a weight-reload path vLLM does not have, and it
buys ~18s at the cost of an artefact pinned to one driver, one GPU model and one
config. Meanwhile the live-process composition in the section above is real,
measured, and needs none of that — it is warm standby, and it is what this line
of work actually delivers.

## What this means for cuda-checkpoint

[docs/foundry.md](foundry.md#why-cuda-checkpoint-is-not-a-next-boot-shortcut) rejects `cuda-checkpoint`
for **next-boot** graph restore, and that reasoning stands — CUDA still exposes no
graph serialization API, and a fresh process cannot reproduce the address space.
But it also shows why `cuda-checkpoint` is unnecessary for the *live-process*
case: sleep mode already gives back 69.70 GiB — leaving 5.3% of the card held — with the graphs intact, for
free, today.

`cuda-checkpoint`'s value here is the last **4.17 GiB** — the context and the
graph pool that sleep mode cannot touch — and the section above shows that
measured, working, down to 0 MiB. vLLM has already built the seam for it:

* `SleepModeBackendFactory` is a plugin registry, and its own registration
  comment names the candidates: *"Third-party backends (CUDA checkpoint, CRIU,
  durable snapshot) register the same way through a `vllm.general_plugins` entry
  point, without changes to vLLM core."*
* The capability predicates a backend must answer are already declared:
  `preserves_communicators()` and `preserves_compiled_artifacts()`
  (`sleep_mode_backend.py:84-93`). `CuMemBackend` overrides the first to `True`
  with the reason — *"Communicator buffers (e.g. NCCL) live outside
  CuMemAllocator's pool"* — and leaves the base `False` default for the second.
* `Worker.checkpoint_prepare` / `checkpoint_restore` (`gpu_worker.py:261-265`)
  fan out to every device communicator
  (`parallel_state.py:2049-2060` → `base_device_communicator.py:223-227`,
  default no-op with a `warning_once` on the manager base).

That last point retires an item flagged unverified in `foundry.md`: whether NCCL
state survives a CUDA checkpoint at TP>1. Upstream's answer is that it does not
survive implicitly — a checkpointing backend must explicitly prepare and restore
each communicator, and the hooks to do it exist. Calling them is necessary but,
measured, nowhere near sufficient: the CUDA-side implementation behind those
hooks covers FlashInfer only, so with stock backends it releases nothing and the
checkpoint that follows never returns — see
[the TP>1 finding above](#composing-with-cuda-checkpoint-to-reach-0-mib).

## Reproducing

`/sleep`, `/wake_up` and `/is_sleeping` are dev-mode-only routes
(`entrypoints/serve/dev/sleep/api_router.py`, gated by `VLLM_SERVER_DEV_MODE=1`
at `api_server.py:225`). Note that `--enable-sleep-mode` changes the
torch.compile cache key: the run above recompiled from scratch (Dynamo bytecode
transform 11.94s) and reached ready in 85s rather than 43.3s. That is a
first-boot-of-a-new-fingerprint artefact and says nothing about sleep mode; the
second boot of the same arm would be warm.

Scripts:

* [`scripts/sleep-mode-probe.sh`](../scripts/sleep-mode-probe.sh) — launch,
  generate, sleep, wake, generate, compare; three cycles.
* [`scripts/cuda-checkpoint-graph-test.py`](../scripts/cuda-checkpoint-graph-test.py)
  — the isolated graph + unbacked-reservation survival test.
* [`scripts/sleep-plus-checkpoint-probe.sh`](../scripts/sleep-plus-checkpoint-probe.sh)
  — the full sleep + checkpoint + restore + wake composition on vLLM.
* [`scripts/cuda-checkpoint-cross-gpu.sh`](../scripts/cuda-checkpoint-cross-gpu.sh)
  — the `--device-map` rule matrix, plus the reallocation scenario (park, let
  another tenant take the card, come back on a different one). Needs 2 GPUs.
* [`scripts/vllm-cross-gpu-restore.sh`](../scripts/vllm-cross-gpu-restore.sh)
  — the same thing on a warmed Qwen3-32B, with a byte-for-byte output compare.
* [`scripts/vllm-parked-image-size.sh`](../scripts/vllm-parked-image-size.sh)
  — what a parked engine weighs in host RAM, per level. Run the two levels in
  **separate boots**: vLLM retains the pinned host backup across sleep/wake
  cycles, so a level-2 sleep following a level-1 cycle reports 75.69 GiB instead
  of 6.61 GiB.
* [`scripts/cuda-checkpoint-dumpability.sh`](../scripts/cuda-checkpoint-dumpability.sh)
  — what state a checkpointed process is actually left in (fds, mappings, RSS),
  and whether this container could host a dumper at all.
* [`scripts/cuda-checkpoint-ipc-matrix.py`](../scripts/cuda-checkpoint-ipc-matrix.py)
  — the vLLM-free bisection of the TP>1 hang: two children, one GPU each, one
  kind of cross-rank state at a time (`plain`, `nccl`, `nccl_teardown`,
  `nccl_graph`, `nccl_graph_free`, `ipc`, `ipc_teardown`, `symm`), then the same
  lock → checkpoint → restore → unlock. `--parallel` puts both ranks
  mid-checkpoint at once. Every call is wall-clock bounded, so a hang is
  recorded as a row rather than wedging the pod. This is what produced the
  matrix above. Needs 2 GPUs.
* [`scripts/tp2-park-probe.sh`](../scripts/tp2-park-probe.sh) — the same
  question on a real TP=2 engine, `--arm default` vs `--arm flashinfer`, with
  every `cuda-checkpoint` call bounded (the first attempt at this cost 29
  minutes of hang). Kills by pid on the way out, since the workers'
  setproctitle names defeat `pkill -f 'vllm serve'`.
* [`scripts/demo-cold-start.sh`](../scripts/demo-cold-start.sh) — the full live
  demo: boot, park, second tenant, migrate-or-restore-and-wake, at `--tp 1` and
  `--tp 2`. Everything is injected into the vanilla `vllm/vllm-openai` image via
  `coldstart/` monkeypatches (`CS_FORKSERVER=1`, `CS_FST=1`,
  `CS_DEV_ROUTES=1`), nothing is a custom image. It is also where the two
  findings above (`reload_weights` crashing EngineCore, and the TP=2
  `cuda-checkpoint` deadlock) were found —
  `reports/demo-cold-start-tp1-20260903-105740.txt` and
  `reports/demo-cold-start-tp2-20260903-110308.txt` are the full transcripts.
* [`coldstart/cs_dev_routes.py`](../coldstart/cs_dev_routes.py) — the
  `CS_DEV_ROUTES=1` monkeypatch `demo-cold-start.sh` depends on: adds
  `POST /checkpoint_prepare`, `/checkpoint_restore` and `/reload_weights` HTTP
  routes over `collective_rpc`, none of which vLLM exposes on its own.

The two cross-GPU scripts need a pod that can **see** both cards, which is
[`manifests/pod-exec-2gpu.yaml`](../manifests/pod-exec-2gpu.yaml):

```
kubectl -n <ns> apply -f manifests/pod-exec-2gpu.yaml
```

It is byte-identical to [`manifests/pod-exec.yaml`](../manifests/pod-exec.yaml)
apart from `nvidia.com/gpu` in `limits` and `requests`, and keeps the same pod
name, so `scripts/run-experiment.sh`, `make exec` and every probe script work
against it unchanged — applying one replaces the other. `CUDA_VISIBLE_DEVICES` is
left unset in both, so torch sees whatever the device plugin assigned.

A one-GPU pod cannot substitute: rule 3 above rejects a remap to a device the
process cannot see, and that is not a workaround-able limitation.

`cuda-checkpoint` is **not** in the vLLM image. The binary is a 5976-byte ELF at
`bin/x86_64_Linux/cuda-checkpoint` in
[NVIDIA/cuda-checkpoint](https://github.com/NVIDIA/cuda-checkpoint) and reports the
driver version it matches (`580.105.08` here); `ptrace_scope` is 0 in this
container, so same-uid ptrace needs no `CAP_SYS_PTRACE` despite
`capabilities: drop: ["ALL"]`.
