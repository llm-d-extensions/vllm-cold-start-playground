# Foundry — ingested notes

External project ingested 2026-09-01. Source of record:
[`foundry-org/foundry`](https://github.com/foundry-org/foundry) (Apache-2.0,
created 2026-01-05, last push 2026-09-01, ~57 stars, primarily C++/CUDA + Python).
Paper: [arXiv:2604.06664](https://arxiv.org/abs/2604.06664), *Foundry:
Template-Based CUDA Graph Context Materialization for Fast LLM Serving Cold
Start* — Xueshen Liu, Yongji Wu, Yuncheng Yao, Danyang Zhuo, Ion Stoica,
Z. Morley Mao (submitted 2026-04-08).

This file is a reading of the upstream repo, not an evaluation. Nothing here has
been reproduced on our hardware. Numbers attributed to upstream are upstream's
claims; the repo's own `Performance` section is still a placeholder.

## Why it matters to this repo

Foundry attacks exactly two of the phases our report breaks out — `torch.compile`
and `cudagraph capture` — plus most of `warmup / profile run`. It is the closest
existing implementation of [CHARTER.md](../CHARTER.md) Goal #4 (caching and reuse
of inference-engine compilation artifacts, keyed by
`(model, hardware, config)`), and its roadmap reaches into Goal #5 territory
(elastic EP resize without recapture).

It is *not* a weight-distribution system. `weight load` and
`network: weight download` are untouched.

## What it does

A serving engine's CUDA graphs cannot simply be serialized: a captured graph
embeds raw device addresses in its kernel arguments, and the kernel code itself
was loaded lazily during warmup. Prior approaches either patch known kernels by
hand (fragile, per-kernel) or checkpoint/restore the whole process (inflexible
across parallelism changes).

Foundry instead makes the *execution context* reproducible, so captured pointers
resolve as-is with no rewriting. A driver-level `LD_PRELOAD` shim
(`libcuda_hook.so`) interposes three classes of CUDA driver call, each backed by
in-process state that SAVE serializes and LOAD rebuilds:

| Intercepted | Mechanism | Upstream |
|---|---|---|
| **Memory** (`cuMemAlloc_v2`, …) | every device allocation is funneled into one VMM region `[base_addr, base_addr + region_size)` with a **monotonic cursor**, giving each tensor a byte-deterministic offset; physical backing via `cuMemCreate` + `cuMemMap` | `csrc/hook.cpp` |
| **Modules / libraries** (`cuModuleLoad*`, `cuLibraryLoadData`) | fatbin bytes plus the `entry_name → CUfunction` table are recorded, so device code is captured regardless of what produced it (PyTorch, Inductor, cuBLAS NVJET, NVSHMEM, DeepEP, DeepGEMM) | `csrc/hook.cpp` |
| **Captured graphs** | graphs are grouped into topology equivalence classes; one per group is built as a **template** with a full `CUgraphExec`, the rest are *on-demand* graphs sharing that executor and carrying only per-node parameter overrides | `csrc/CUDAGraph.cpp`, `csrc/BinaryGraphIO.cpp` |

SAVE writes all three to a per-rank archive. LOAD pre-maps the same VMM range,
re-loads the modules from the packed fatbins, instantiates one `CUgraphExec` per
topology group, and applies node-param updates on demand at replay
(`cuGraphExecUpdate`). Graphs are stored both as JSON and as a binary
`.cugraph` (upstream: 84× faster parse than the JSON text path).

**The load-bearing invariant:** SAVE and LOAD must walk an *identical VMM cursor
trajectory*. If a LOAD-side allocation lands at a different offset, captured
kernels read unmapped or stale memory — and the failure is silent. Every
integration quirk below exists to enforce that.

## Reported numbers (upstream, unverified)

| Source | Claim |
|---|---|
| Paper abstract | cold-start latency reduced "by up to 99%"; Qwen3-235B-A22B init 10 min → **3.9 s**; evaluated on dense + MoE up to 235B |
| Repo README demo | Qwen3-30B-A3B-FP8, EP=2: **256 graphs rebuilt in ~1 s** (+ ~2 s sampler warmup + API server init) vs ~30 s of baseline warmup + capture |
| `RELEASE.md` 0.0.2 | SGLang: ~30 s of capture → **~0.4 s** restore; restored decode graphs match baseline throughput within run-to-run noise |

Note the SAVE side is *much slower*, not faster: the recipe troubleshooting table
documents `[TIMING] NCCL init: 122.992 s` on a Foundry SAVE run, attributed to
`LD_PRELOAD` module-load overhead. Upstream's position is that this does not
affect LOAD because all modules are preloaded from the archive. If we measure
this, SAVE and LOAD are two different configurations and must be reported
separately.

## Mapping to our phase model

Against the phases in `analysis/coldstart_report.py` (see
[docs/measurement.md](measurement.md)):

| our phase | effect on Foundry LOAD |
|---|---|
| `torch.compile` | → ~0. `do_not_compile=True` on LOAD; graphs replay through `CUDAGraphWrapper` instead |
| `cudagraph capture` | → ~0. Replaced by template instantiation + on-demand exec update |
| `warmup / profile run` | largely eliminated: `kernel_warmup` is a no-op, the KV `memory_profiling` forward is skipped (values read from `warmup_state.json`), the sampler-warmup full forward is replaced with zero hidden states. `_dummy_sampler_run` still runs |
| `kv cache alloc` | still happens; only the *sizing* comes from saved state |
| `weight load` | unchanged. Upstream tried overlapping background template builds with `load_weights` and found it **net-negative** (driver contention), so builds start after weight load and overlap the cheaper post-load init instead |
| `device & collectives init`, `python imports` | unchanged at best; plausibly *worse* on SAVE under `LD_PRELOAD` (see the 122 s NCCL figure). Untested on LOAD |
| `network: *`, `ipc handshake`, `interpreter boot` | untouched |

So a Foundry LOAD run should show `torch.compile` and `cudagraph capture`
collapsing while `weight load` holds steady — a clean, falsifiable prediction for
our harness.

## Operating contract (what an integrator must accept)

1. **`LD_PRELOAD` before any CUDA call.** `libcuda_hook.so` must be preloaded in
   every process that touches CUDA — plus `libnvshmem_host.so` for EP. The
   integration re-sets it at each worker spawn (`setup_ld_preload_env`), and the
   serve scripts export it from the shell as defense-in-depth.
2. **A forked engine.** Small but real: **~97 lines across 5 files** in `vllm/`,
   **~47 lines across 4 files** in `sglang/` — a shim module plus activation
   calls from the config object's `__post_init__`, `EngineCore.__init__`, and
   `Worker.__init__` (config is pickled across process boundaries, so
   `__post_init__` does not re-fire). Everything substantive lives in
   `foundry.integration.<engine>`. Forks are declared as `foundry-org/vllm`
   (branch `foundry`), `foundry-org/sglang`, `foundry-org/TensorRT-LLM`; the
   README describes them as forthcoming while `recipe/vllm/README.md` already
   instructs cloning them. **Verify they are public before planning around them.**
3. **Two-pass SAVE on vLLM.** vLLM's `_initialize_kv_caches` runs
   `memory_profiling` with a real forward whose activations are
   caching-allocator-driven and non-reproducible. Pass 1 runs it and writes
   `warmup_state.json`; pass 2 skips it and re-captures graphs at deterministic
   offsets; LOAD skips it too. Skipping pass 2 yields a LOAD that either fails
   the cursor check *or passes it and emits garbage* — both observed upstream.
   SGLang needs only one pass (no profile forward).
4. **A large reserved VA range.** Recipes use `base_addr = 0x600000000000`,
   `region_size = "256GB"`, `scratch_space_size = "1024MB"`; `base_addr` and
   `region_size` must be identical across SAVE and LOAD. Early
   non-deterministic allocations (CUDA context, cuBLAS handle, NCCL bring-up)
   land in scratch and are erased by `skip_to_scratch_boundary()`. The region is
   `stop`ped after capture so post-capture allocations use the normal caching
   allocator.
5. **Config pinning.** `cudagraph_mode FULL_DECODE_ONLY`;
   `cudagraph_num_of_warmups` force-stomped to 0 (vLLM hard-overrides it to 1);
   `VLLM_USE_V2_MODEL_RUNNER=0` (patches target the V1 `GPUModelRunner`; without
   the pin, SAVE silently writes an empty workspace); for EP,
   `NCCL_CUMEM_ENABLE=0` and `NCCL_NVLS_ENABLE=0` (NCCL fast paths need driver
   capability flags the VMM region does not carry), `--all2all-backend
   deepep_low_latency`, DeepEP forced to `use_fabric=True` (NVL's
   `cudaIpcGetMemHandle` handles are process-local and unpersistable), and
   `--max-num-batched-tokens ≤ 511` unless `NVSHMEM_QP_DEPTH` is raised.
6. **No `torch.cuda.empty_cache()` inside the capture loop.** The cursor is
   monotonic; freeing does not rewind it. Upstream observed `final_alloc_offset`
   inflating to ~194 GB instead of ~80 GB on Qwen3-30B-A3B EP2 from exactly this.

## Support matrix and the TP gap

| Engine | Single GPU | DP | TP | EP |
|---|:---:|:---:|:---:|:---:|
| SGLang (v0.5.13) | ✅ | ✅ | 🚧 | ✅ |
| vLLM | ✅ | ✅ | 🚧 | ✅ |
| TensorRT-LLM | 🚧 | 🚧 | 🚧 | 🚧 |

✅ = validated SAVE → LOAD → query upstream.

**Tensor parallelism is unsupported on both engines** — a deterministic NCCL
memory layout is still under construction. SGLang's EP path sidesteps it with
DP-attention (`--enable-dp-attention`) so no NCCL all-reduce is in the captured
path. For llm-d this is the single biggest gap: TP is the common topology for
large dense models, and it is the one Foundry cannot do today.

Requirements: CMake 4.0+, CUDA driver 12.0+, Boost 1.83.0+, PyTorch 2.9–2.11
(recipes install 2.11.0 + cu130). vLLM compat was v0.21 at 0.0.1; the current
recipe pins a nightly precompiled wheel.

## Archive shape and cache-key implications

```
foundry_archive_<model>/
  warmup_state.json                  # shared, rank 0: available_gpu_memory,
                                     # num_gpu_blocks, num_cpu_blocks,
                                     # final_alloc_offset
  rank_<N>/
    graph_*.json + graph_*.cugraph   # one pair per captured BatchDescriptor
    graph_manifest.json              # topology groups + template assignments
    fatbin_image_packed.img          # packed CUDA modules
    fatbin_entrypoint_packed.txt
    final_alloc_offset.json          # per-rank VMM watermark
```

Consequences for a SIG-level artifact cache:

- Archives are **per-rank**, and under EP different ranks legitimately have
  different `final_alloc_offset`s (different experts). The cache key is
  `(model, dtype, hardware, engine version, parallelism topology, capture sizes,
  rank)` — narrower than a `torch.compile` cache entry, so hit rates depend on
  fleet homogeneity.
- The paper claims a *single-GPU* offline capture can seed multi-GPU templates by
  patching only rank-dependent communication state. In the repo today that is
  **not shipped**: `ROADMAP.md` Stage 3 lists "Release NVSHMEM stub layer for
  single-GPU offline template capture" as unchecked. Capture happens on the
  target topology. If the stub layer lands, the cache-key story improves sharply.
- Archive size is not documented anywhere upstream. Unknown, and it matters for
  a multi-tier cache — measure it before designing around it.

## Roadmap items worth tracking

Stage 5–6 in `ROADMAP.md`, all unchecked, all adjacent to our charter:
PD-disaggregated serving with Foundry cold start; cross-node EP at scale
(multi-host NVLink + IB); end-to-end fast init with RDMA weight transfer;
NVIDIA Dynamo compatibility; and **instant-on elastic EP** — dynamic EP resize
by reusing templates and patching rank-dependent comm state, without full
recapture. That last one is the interesting one for SIG Autoscaling.

## How to evaluate it here

Our harness deliberately measures **unmodified upstream images**
([README](../README.md), [docs/measurement.md](measurement.md)). Foundry needs a
forked vLLM, a compiled `libcuda_hook.so`, and `LD_PRELOAD` in the container, so
it is a separate image and a separate row in the
[docs/experiments.md](experiments.md) matrix — not a flag on an existing run.

A defensible first comparison, three runs in one pod so node/GPU/kernel/page
cache stay fixed:

| run | what it establishes |
|---|---|
| baseline, warm compile cache | the number Foundry must beat |
| Foundry SAVE (pass 2) | the cost of producing an archive, including `LD_PRELOAD` module-load overhead |
| Foundry LOAD | the claim under test |

What to check in `report.txt`:

- `torch.compile` and `cudagraph capture` at ~0 on LOAD.
- `weight load` unchanged across all three — if it moved, something other than
  graph restoration changed.
- `device & collectives init` on SAVE vs baseline, to quantify the `LD_PRELOAD`
  penalty (upstream's 122 s NCCL figure needs independent confirmation).
- `unaccounted` staying small. Foundry's C++ hook does work our probe cannot see;
  a large `unaccounted` on LOAD means the phase breakdown is not capturing where
  restore time actually goes, and the probe needs a Foundry-aware span.
- `first_token` in addition to `/health`, since the whole point is that graphs
  are ready — a LOAD that reaches `/health` fast but stalls on first token would
  be the interesting negative result.

Open questions to answer before recommending it: archive size per model/rank;
whether LOAD is robust to a driver or GPU-model change under the same archive;
whether TP lands; whether the forks are maintained against vLLM main or drift.

## Upstream files worth reading directly

| Path | Why |
|---|---|
| `README.md` | mechanism, public Python API table, quick start |
| `docs/overview.md` | integration architecture shared by both engines, and the vLLM↔SGLang difference table |
| `docs/vllm/overview.md` | the four critical invariants; SAVE-pass-1 / pass-2 / LOAD lifecycle step by step |
| `docs/vllm/memory-lifecycle.md` | VMM region setup, allocation buckets A/B/C, `final_alloc_offset`, per-rank layout, high-risk seams |
| `docs/vllm/memory-consistency.md` | every non-deterministic path, the rule applied, and the exact failure mode when the rule breaks — the best single file for judging fragility |
| `docs/vllm/hooks.md`, `docs/vllm/direct-edits.md` | the monkey-patch set and the minimal fork contract; read these to port to another engine |
| `docs/vllm/moe-and-deepep.md` | DeepEP fabric mode and NVSHMEM init ordering |
| `recipe/vllm/README.md` | install, run sequence, env pins, and a genuinely useful troubleshooting table |
| `ROADMAP.md`, `RELEASE.md` | what is done vs claimed; 0.0.2 release notes carry the SGLang EP details |
| `python/foundry/integration/vllm/{config,runtime,hooks,graph_ops}.py` | TOML schema, VMM setup, the patches themselves |
