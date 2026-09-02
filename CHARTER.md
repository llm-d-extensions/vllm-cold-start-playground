# SIG Fast Start — Charter

> **Status:** Draft v0.3 — pending community review and maintainer approval
> **Sponsoring project:** [llm-d](https://github.com/llm-d/llm-d)
> **Proposed home repo:** [llm-d/llm-d-cold-start](https://github.com/llm-d/llm-d-cold-start)

## 1. Mission

Reduce the wall-clock time between the moment a new inference server replica
is requested and the moment it is ready to serve tokens at full quality of
service. SIG Fast Start coordinates the cross-cutting work — across model
weight distribution, compilation artifact reuse, inference engine startup,
and Kubernetes-level actuation — required to make scale-out responsiveness a
first-class property of llm-d deployments.

Today, bringing up an inference server for a large LLM commonly takes
**2–7+ minutes** end-to-end. This latency directly limits how aggressively
operators can scale to demand, how cheaply they can run spot/preemptible
capacity, how quickly they can recover from failures, and how elastic
adjacent workloads (such as RL rollouts and evaluations) can be. The SIG exists so that
"add another replica" stops being a multi-minute operation.

## 2. Goals

1. **Define and track a canonical "time-to-ready" metric** for llm-d
   inference replicas, broken down by phase (image pull, pod scheduling,
   process launch, weight load, compilation/warmup, health-check pass).
2. **Drive measurable reductions in time-to-ready** across the phases the
   SIG owns, with public targets per llm-d release.
3. **Provide multi-tiered cache storage for model weights** (object store →
   shared filesystem → node-local NVMe → DRAM → GPU VRAM → peer GPU VRAM)
   that integrates cleanly with both vLLM and SGLang.
4. **Provide caching and reuse for inference-engine compilation artifacts**
   (e.g., `torch.compile` graphs, CUDA graphs, Triton kernel caches,
   vLLM/SGLang engine state) so warmup is paid once per
   (model, hardware, config) tuple rather than once per replica.
5. **Accelerate the Kubernetes-side cold path** — pod actuation patterns,
   sleep/wake of warm replicas, launcher-based model swapping, and the
   CRDs/controllers needed to express warm-pool and seed-replica intents,
   in collaboration with SIG Installation and SIG Autoscaling. (Container
   image distribution itself is owned by SIG Installation; see §4.2.)
6. **Establish upstream collaboration** with vLLM and SGLang so that the
   hooks needed for fast start (weight ingestion, compilation cache export,
   sleep/wake, warm-pool semantics) live in the engines, not as forks.
7. **Publish reproducible benchmarks** (extending SIG Benchmarking
   infrastructure) covering representative model sizes, tensor-parallel
   topologies, and accelerator generations.

## 3. Non-Goals

- **Steady-state serving performance** (throughput, TTFT, ITL under load) —
  owned by SIG Router, SIG PD-Disaggregation, SIG KV-Disaggregation.
- **KV-cache transfer between active replicas** — owned by SIG
  KV-Disaggregation. Cold Start covers the first-token-ever path; KV
  disaggregation covers the warm path.
- **Autoscaling policy** (when to add/remove replicas) — owned by SIG
  Autoscaling. SIG Fast Start makes their decisions cheaper to act on but
  does not decide them.
- **Authoring a new inference engine.** All work integrates with existing
  engines (vLLM, SGLang, and others as adopted by llm-d).
- **Container image build, distribution, and pull acceleration** — owned
  by SIG Installation. Fast Start defines requirements (e.g., what
  time-to-ready budget the image-pull phase must fit into) but does not
  build or maintain the image-distribution path.
- **Model training, fine-tuning, or quantization pipelines** — except where
  they produce artifacts (weights, compiled graphs) that the SIG's caches
  must consume.

## 4. Scope

### 4.1 In scope

**Weight distribution and caching**
- Multi-tiered cache hierarchy: registry/object store, shared PVC,
  node-local NVMe, DRAM, GPU VRAM, peer GPU VRAM via RDMA/NIXL.
- Promotion/demotion policies across tiers based on access patterns and
  capacity pressure.
- Coordination of concurrent first-pulls so N replicas don't each download
  the same model from HuggingFace.
- Coordination with [ai-dynamo/modelexpress](https://github.com/ai-dynamo/modelexpress),
  which implements a multi-tier cache and seed-pod / GPUDirect-RDMA
  weight distribution pattern. modelexpress is governed by the
  ai-dynamo project; SIG Fast Start is a collaborator and consumer,
  not its owner. The SIG drives requirements and contributes
  upstream.

**Compilation and warmup artifact caching**
- Persistent caches for `torch.compile`, CUDA graphs, Triton autotuner
  results, and engine-specific compiled state, keyed by
  (model, dtype, parallelism, hardware, engine version).
- Mechanisms for engines to export and import these artifacts.

**Kubernetes actuation**
- [llm-d-incubation/llm-d-fast-model-actuation](https://github.com/llm-d-incubation/llm-d-fast-model-actuation)
  is adopted as the SIG's **reference implementation** for fast pod
  actuation, including the *dual-pods* and launcher-based
  model-swapping patterns. The SIG governs its roadmap.
- Sleep/wake of vLLM (and equivalent SGLang) replicas to keep capacity
  warm without holding GPUs.
- CRDs and controllers needed to express "warm pool", "model-swappable
  slot", and "seed replica" intents.

**Engine collaboration**
- Upstream features in vLLM and SGLang for: streaming weight ingestion,
  compilation-cache import/export, sleep/wake, weight hot-swap.
- Reference adapters that let llm-d consume these features uniformly.
- Initial engine coverage is **vLLM and SGLang only**. Additional
  engines (e.g., TGI, TensorRT-LLM, llama.cpp) are added on demand when
  an llm-d use case requires them, not preemptively.

**Measurement**
- Canonical time-to-ready benchmark suite, phase breakdown, regression
  tracking, and per-release public targets.
- The SIG's measurement substrate is the [`nop` harness][nop] in
  [llm-d/llm-d-benchmark](https://github.com/llm-d/llm-d-benchmark),
  which is purpose-built for model-load-time benchmarking. SIG Fast
  Start extends nop with the phase breakdown defined in Goal #1 and
  with cold-start-specific scenarios; the harness itself remains under
  SIG Benchmarking's ownership.

[nop]: https://github.com/llm-d/llm-d-benchmark/tree/main/workload/harnesses

### 4.2 Adjacent (collaborate, do not own)

- **SIG Installation** — Helm charts, operator integration, cluster
  prereqs, **and container image distribution / image-pull
  acceleration**. SIG Fast Start consumes their improvements but does
  not own this layer.
- **SIG Autoscaling** — consumer of the latency improvements.
- **SIG Benchmarking** — owns [llm-d-benchmark](https://github.com/llm-d/llm-d-benchmark),
  including the existing `nop` harness for model-load-time
  measurement. Fast Start contributes upstream: cold-start scenarios,
  phase breakdown, and regression tracking extensions. Fast Start does
  not fork or replace the harness.
- **SIG RL** — important downstream consumer of fast elasticity.
- **vLLM and SGLang upstream communities** — code lives there for engine
  changes; SIG Fast Start drives the requirements and reference impls.

## 5. Initial Leads

> *To be filled in before the charter is submitted for approval.*
> Per llm-d governance, SIGs have 2–3 leads responsible for direction,
> coordination, and decision-making. Leads should ideally span multiple
> sponsoring organizations and cover the three major workstreams (caching,
> engine integration, k8s actuation).

| Role | Name | Affiliation | Workstream focus |
| --- | --- | --- | --- |
| Lead | _TBD_ | _TBD_ | Multi-tier cache / weight distribution |
| Lead | _TBD_ | _TBD_ | Engine integration (vLLM / SGLang) |
| Lead | _TBD_ | _TBD_ | Kubernetes actuation |

## 6. Communication

- **Slack:** `#sig-fast-start` (proposed) on the llm-d Slack workspace.
- **Meetings:** weekly, on the shared llm-d project calendar. Recordings
  and notes archived in the SIG's Google Drive folder.
- **GitHub:** issues and PRs in `llm-d/llm-d-cold-start`, plus the
  `sig/fast-start` label across the broader llm-d org.
- **Decision-making:** lazy consensus among leads and active
  contributors, consistent with llm-d project governance. Cross-SIG
  decisions are escalated to project maintainers.

## 7. Roadmap (TO BE REVIEWED AND REFINED)

Targets are illustrative pending baseline measurement; final numbers will
be set after the v0 benchmark suite lands.

### Phase 0 — Foundation (charter approval + ~1 month)
- Approve charter; stand up Slack channel, meeting cadence, GitHub labels.
- Adopt `llm-d-cold-start` as the SIG's coordination repo.
- Inventory existing work: modelexpress, fast-model-actuation, vLLM
  sleep/wake, SGLang equivalents, torch.compile cache work.
- Draft and publish the SIG's **North Star design document** covering
  the target architecture across the three workstreams.
- Land a v0 time-to-ready benchmark by extending llm-d-benchmark's
  `nop` harness with end-to-end phase breakdown, run against one model
  in {8B, 70B} on one accelerator class. Contributed upstream to
  llm-d-benchmark; not maintained as a fork.

### Phase 1 — Baseline and quick wins (~1 quarter)
- Publish baseline time-to-ready numbers per phase.
- Integrate modelexpress (or equivalent) as the reference multi-tier
  weight cache behind a stable llm-d interface.
- Land vLLM sleep/wake + dual-pods pattern as a supported llm-d
  deployment mode.
- Define the compilation-artifact cache interface; prototype for vLLM.
- **Target:** ≥2× reduction in median time-to-ready for the v0 benchmark.

### Phase 2 — Engine parity and breadth (~2 quarters)
- SGLang parity for sleep/wake, weight ingestion, compilation cache.
- Peer-to-peer VRAM weight distribution (RDMA/NIXL) in production-grade
  config.
- DRAM and NVMe shard streaming; GPUDirect Storage path where supported.
- Expand benchmark coverage to additional model sizes, TP/PP topologies,
  and accelerator generations.

### Phase 3 — Elastic by default (~1 year)
- Time-to-ready low enough that autoscaling on per-minute traffic shifts
  is practical for ≥70B-class models.
- Warm-pool and seed-replica patterns standardized across engines.
- Compilation-cache sharing across clusters (artifact registry).
- **Stretch target:** sub-30s median time-to-ready for a 70B-class model
  on supported hardware, given a warm cache tier.

---
