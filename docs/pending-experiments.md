# Pending experiments: closing the TP=2 gap

Everything in [README.md](../README.md), [docs/sleep-mode.md](sleep-mode.md) and
[docs/on-demand-cudagraph.md](on-demand-cudagraph.md) is a controlled,
median-of-3 (or better) measurement. **The boot half of the TP=2 gap is now
closed the same way**: #1 has been run — seven arms × 3 repeats, 21 runs
(`runs/tp2-*`, [`scripts/tp2-ladder.sh`](../scripts/tp2-ladder.sh)), plus a
matched-context pair at `--max-model-len 8192` (`runs/tp2m8k-*`) — and #2 is
half closed, with the capture phase total measured but its per-rank structure
still opaque. See [README — "What survives at TP=2"](../README.md#what-survives-at-tp2).
What is still single-sample is the **Part 2** material: the sleep-mode,
checkpoint/restore and artifact numbers rest on one uncontrolled boot from
[`scripts/demo-cold-start.sh --tp 2`](../scripts/demo-cold-start.sh)
([reports/demo-cold-start-tp2-20260903-110308.txt](../reports/demo-cold-start-tp2-20260903-110308.txt))
— which the ladder has at least corroborated: its baseline arm medians 94.7s
against that run's 95.79s. And one item was never a measurement gap at all:
parking a TP=2 engine to 0 MiB wedges the NVIDIA driver's checkpoint lock, and
[docs/sleep-mode.md](sleep-mode.md#composing-with-cuda-checkpoint-to-reach-0-mib)'s
isolating matrix ([`scripts/cuda-checkpoint-ipc-matrix.py`](../scripts/cuda-checkpoint-ipc-matrix.py)) now explains why:
live NCCL state or torch symmetric memory wedges it alone, and releasing NCCL
requires destroying the captured CUDA graphs first — the exact thing sleep
level 1 exists to keep. So of the seven: **#1 is done, #2 is half done, #4 is
root-caused rather than measured, and #3, #5, #6, #7 have not been run.** This
file exists so the rest of the gap stays written down instead of guessed at — see the [cold-start-slides.html](../cold-start-slides.html)
deck for what each one would complete.

## 1. TP=2 phase-level ladder — DONE (2026-09-03)

**Result**: `runs/tp2-*`, seven arms × 3 repeats. **94.7s → 52.7s, −42.0s
(−44%)**, against TP=1's −52%, and it is a two-lever stack rather than four:
`PYTHONPYCACHEPREFIX` survives and *grows* (−35.8s vs −22.31s; imports 54.4 →
24.6s, because the TP=2 baseline has four interpreters importing torch where
TP=1 has two), `fastsafetensors` survives at ~40% (−6.2s vs −15.73s; each rank
already reads its own shard), and **both process-model levers die on one veto**
— at TP=2 CUDA is already initialised in the launching process (21/21 runs,
`cuda.lazy_init` fires 3.4–6.0s before `engine.core_proc_manager`), so
`_maybe_force_spawn()` reverts `fork` to `spawn` and the forkserver probe
declines with `served: 0`. The `cs_fst` `nogds` lever is structurally dead:
`weight_utils.py:1057` computes `nogds = pg.size() > 1`. Phase breakdown of the
median arm is in `runs/tp2-s4-fst-r2/report.txt` (imports 24.7s / 46.9%, weight
load 8.84s, cudagraph capture 8.84s, unaccounted 9ms). Cross-TP phases were
re-measured at a matched 8192 (`runs/tp2m8k-*` vs `runs/cc-base-c*`): capture
11.1 → 8.74s (−21%), weight load 15.8 → 7.71s (−51%). Full write-up in
[README — "What survives at TP=2"](../README.md#what-survives-at-tp2).

**Note for anyone re-running it**: a TP=1 re-anchor at the model's default 40960
context is not possible on one H100 (7.92 GiB free for KV, 10.0 GiB needed,
estimated max 32432), which is why the cross-TP comparison is at 8192.

<details><summary>original plan</summary>

**What**: reproduce `runs/lad2-*` (the 5-step, median-of-3, interleaved matrix
behind the [README ladder table](../README.md#results-so-far-qwen3-32b-on-one-h100))
on [`manifests/pod-exec-2gpu.yaml`](../manifests/pod-exec-2gpu.yaml) with `--tensor-parallel-size 2`.

**Why it's missing**: no committed run in `runs/` passes `tensor-parallel-size
2` — confirmed by `grep -rl "tensor-parallel-size 2" runs/*/meta.json` (no
hits outside [`scripts/run-experiment.sh`](../scripts/run-experiment.sh), which only supports the flag). The
only TP=2 total-ready number that exists (95.79s) is a single boot under the
demo script's own env, not this harness's controlled arms — it is not safe to
read as "TP=2's version of 43.30s."

**What it unblocks**: a real TP=1-vs-TP=2 phase breakdown — in particular
whether `python imports` amortizes across the two `WorkerProc`s the way
[docs/experiments.md](experiments.md) predicts ("the margin grows at TP>1
where one preload amortises across N workers") rather than doubling.

</details>

## 2. TP=2 CUDA graph capture lever matrix — HALF DONE

**What**: reproduce `runs/w6-*` (the capture-size / `cudagraph_mode` /
kernel-warmup matrix behind [the CUDA graph capture
section](../README.md#cuda-graph-capture-114s-that-configuration-cannot-honestly-remove))
at TP=2. 51 sizes × 2 modes is captured **per rank**, so the 11.4s TP=1 figure
is a floor, not an estimate, for what two ranks pay concurrently or serially.

**Measured so far**: the capture *phase total* at TP=2 is **8.74s** at a matched
8192 context (`runs/tp2m8k-*`, median of 3) against **11.1s** for the same-probe
TP=1 anchor — **−21%, not −50%**. The per-rank forward halves but the per-size
launch overhead across 51 sizes × 2 modes does not divide by rank count, and that
undivided residue is what an on-demand design would have to attack.

**Still missing**: the lever matrix itself (`runs/w6-*` at TP=2), and — not
inferable from a phase total — whether the two ranks capture concurrently or
serialize against each other, plus where torch's 3366-graph prologue lands per
rank.

**What it unblocks**: whether capture across ranks overlaps (both `WorkerProc`s
capturing at once) or serializes, and whether the on-demand-capture proposal's
open TP>1 question — "per-capture collective coordination during serving...
unverified" ([docs/on-demand-cudagraph.md](on-demand-cudagraph.md)) — has an
answer.

## 3. TP=2 sleep-mode baseline, median of 3

**What**: reproduce the TP=1 table in [docs/sleep-mode.md](sleep-mode.md#measured-2026-09-02)
(GPU at ready, GPU asleep, freed, first/subsequent sleep, wake, byte-identical
check) at TP=2, three cycles, not one.

**Why it's missing**: the live TP=2 demo got exactly one number —
`/sleep?level=1` succeeded in **11.36s**, both GPUs to **4243 MiB** each — before
the checkpoint step (below) hung. One uncontrolled sample is not a baseline:
the TP=1 table shows first-sleep (26.03s) and subsequent-sleep (1.26s) differ
by 20x, so a single TP=2 number could be either.

## 4. Root cause found — now it's an upstream design trade, not a bug hunt

**Resolved, 2026-09-03.** The demo transcript's live read called this a
*rank-to-rank mutual deadlock*. That reading is wrong, and
[docs/sleep-mode.md](sleep-mode.md#composing-with-cuda-checkpoint-to-reach-0-mib)'s
isolating matrix ([`scripts/cuda-checkpoint-ipc-matrix.py`](../scripts/cuda-checkpoint-ipc-matrix.py), two spawned
children outside vLLM entirely) disproves it directly: `--parallel` — both
ranks mid-checkpoint at once — hangs identically to one-at-a-time, and a
*single* child holding a live NCCL communicator or torch symmetric memory
hangs completely alone. The `futex_wait_queue` in `wchan` is a symptom of the
driver's per-process checkpoint lock being wedged, not ranks waiting on each
other. `/checkpoint_prepare` also does not tear down NCCL as first reported:
[`CudaCommunicator.checkpoint_prepare`](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/distributed/device_communicators/cuda_communicator.py#L588-L589)
(`cuda_communicator.py:588`) releases
only FlashInfer's own all-reduce/all2all workspaces; the stock
`['CUSTOM','SYMM_MEM','PYNCCL']` chain has `pynccl_comm` built unconditionally
at `world_size>1` and none of it gets released — which is why the call
reports 0.06s of "success" having freed 0 MiB.

**The actual root cause**: live NCCL state or torch symmetric memory wedges
`cuda-checkpoint` on its own, and releasing the NCCL communicator requires
destroying the captured CUDA graphs first
([`pynccl.py:148`](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/distributed/device_communicators/pynccl.py#L148-L155)) — but those
graphs are the entire reason sleep level 1 exists. **Parking a TP>1 engine to
0 MiB and keeping its captured CUDA graphs are mutually exclusive on this
stack.** `--arm flashinfer`
([reports/tp2-park-flashinfer-20260903-130336.txt](../reports/tp2-park-flashinfer-20260903-130336.txt))
proves the shape of a fix exists: FlashInfer's own
[stable-VA detach](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.16.post3/flashinfer/comm/allreduce.py#L204)
frees 512 MiB where the stock arm frees 0 — but `PyNcclCommunicator` has no equivalent
detach, so `PYNCCL` staying in the chain still wedges it.

**What's actually still pending**: not a bug hunt, a design trade plus an
upstream ask — extend the stable-VA detach FlashInfer already has to
`PyNcclCommunicator` (its `destroy()` cannot be called while graphs live;
`symm_mem.py` has no teardown at all), or make `checkpoint_prepare` fail
loudly instead of reporting success when the active backends can't actually
be released. Neither has been attempted yet.

## 5. TP=2 demo steps 4-8, once the design trade in #4 resolves

**What**: the rest of [`scripts/demo-cold-start.sh --tp 2`](../scripts/demo-cold-start.sh) — second tenant B
boot, teardown, identity `--device-map` restore, `/checkpoint_restore`, final
wake + correctness check — mirroring the TP=1 demo's steps 4-8 in
[reports/demo-cold-start-tp1-20260903-105740.txt](../reports/demo-cold-start-tp1-20260903-105740.txt).

**Why it's missing**: all of it depends on step 3's checkpoint completing to 0
MiB, which #4 now shows requires either an upstream fix (stable-VA detach
extended to `PyNcclCommunicator`) or accepting destroy-and-recapture of the
CUDA graphs as the TP>1 park cost — a real measurement in its own right, not
yet run.

## 6. TP=2 cross-GPU restore

**What**: the `--device-map` migration measured at TP=1 in
[docs/sleep-mode.md](sleep-mode.md#restoring-onto-a-different-gpu) (4.13s,
one engine, one GPU pair) has no TP=2 analog — and TP=2 needs *two* GPUs per
rank moving in a coordinated remap, plus `Worker.checkpoint_prepare` /
`checkpoint_restore` rebuilding NCCL across the new device mapping. Moot until
#4's design trade resolves: a same-GPU, identity-map TP=2 checkpoint doesn't
reach 0 MiB today, so a remap never comes into play.

## 7. TP=2 artifact/cache trace

**What**: a traced TP=2 boot (`coldstart` probe attached, not just the demo
script's coarse timers) would give the HF-cache and compile-cache size figures
used on the artifact-sizes slide — cache footprint, shard distribution across
2 GPUs, whether `TRITON_CACHE_DIR`/`FLASHINFER_CACHE_DIR` grow with rank count.

**Why it's missing**: the demo script is a black-box timer around `curl`, not
a probed run; only `runs/lad2-*`/`runs/w6-*`-style runs carry the trace that
produces `hf_home=63.9 GiB/124 files`-style figures, and none of those exist
at TP=2 (see #1).

## What NOT to do meanwhile

Do not extrapolate a TP=2 phase breakdown by doubling the TP=1 numbers — this
is now measured rather than cautioned. `weight load` is **−51%** (each rank
reads its own shard), `cudagraph capture` only **−21%**, and `python imports`
goes the other way entirely: it stays the largest phase at TP=2 (24.7s, 46.9%)
*after* the pycache lever has already removed 35.8s, because two ranks mean two
more interpreters, not one amortized preload. Use `runs/tp2-*` and
`runs/tp2m8k-*` for the boot numbers.

What is still not safe to extrapolate is **Part 2**: every sleep-mode,
checkpoint/restore and artifact-size figure at TP=2 still rests on the demo's
single 95.79s boot, and #3, #5, #6 and #7 below are the experiments that would
fix that.
