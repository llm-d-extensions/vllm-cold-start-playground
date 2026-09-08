# vLLM cold start from a launcher process (TP=1 and TP=2)

**Question.** A launcher process fronts vLLM: it stays running and, on request,
creates a vLLM instance as a child process. How long does vLLM take to become
ready *from the moment the launcher issues process creation*, what makes that
number move, and — because the launcher is permanent and reusable — which warm
caches can we exploit?

Reference launcher: `llm-d-incubation/llm-d-fast-model-actuation`
(`inference_server/launcher`). The copy exercised here is in
`scripts/launcher/` (upstream launcher + a cold-import variant built to isolate
one effect; see *Method*).

- **Model:** Qwen/Qwen3-32B — 61.02 GiB, 17 safetensors shards, bf16.
- **Serve args:** `--max-model-len 8192` (+ `--tensor-parallel-size 2` for TP=2).
- **Hardware / stack:** H100 80GB HBM3, vLLM 0.28.0 / torch 2.13 / CUDA 13,
  16 CPU cores, 128 GiB RAM, 16 GiB `/dev/shm`.
- **Weights + compile caches:** on a GPFS-backed PVC (`/cache`), warm.
- **Namespace / pod:** `lionel-cold-start-cc`, pod `vllm-coldstart`
  (`manifests/pod-exec-2gpu.yaml`).

---

## TL;DR

| | Baseline (naive launcher) | Best measured | Reduction |
|---|---|---|---|
| **TP=1** | 89.2 s | **33.9 s** | −55 s (−62%) |
| **TP=2** | 96.0 s | **70.5 s** | −25 s (−27%) |

> **Read the "reduction" carefully.** The baseline above is a *naive launcher*
> (cold-import child, no warm bytecode) — the right control for isolating what
> the launcher's *own design* is worth, but **not** the right control for
> "should I use a launcher at all." A **fully tuned *standalone* `vllm serve`**
> (warm bytecode + fork + instanttensor + startup plan, `cold-start-slides.html`)
> reaches **36.9 s** (TP=1, median of 3). The launcher's best is 33.9 s, so the
> launcher's **net per-boot advantage over a tuned standalone is only ~3 s** at
> TP=1 and **≈0 at TP=2**. See *Launcher vs. tuned standalone* below — the
> launcher is an **operational** win (resident, reusable, owns lifecycle), not a
> cold-start-*speed* win.

- The launcher's own design is the biggest single lever at TP=1 *relative to a
  naive launcher*: by importing the vLLM server stack at module load and
  **forking** children, the child inherits the already-imported interpreter.
  That alone takes TP=1 from **89 s → 58 s** with no change to vLLM itself.
  (But a standalone process with a warm bytecode cache already imports cheaply,
  which is why this advantage largely evaporates against a *tuned* standalone.)
- Swapping the weight loader carries TP=1 the rest of the way. **fastsafetensors**
  gets it to 45 s; **instanttensor** — the fastest loader tested — cuts the
  weight phase from ~30 s to ~8 s and lands the best measured number, **33.9 s**.
- **At TP=2 the fork lever silently disappears:** vLLM detects CUDA is already
  initialized in the parent and *overrides* `VLLM_WORKER_MULTIPROC_METHOD` back
  to `spawn`. So at TP=2 the only launcher win is import inheritance for the API
  server (~12 s); the EngineCore and both workers still spawn and re-import.
- Because the launcher is permanent, its one-time warm-import boot cost
  (~15–17 s) is **paid once and amortized to zero** across all subsequent
  launches. Every warm cache below is shared across launches for free.

---

## Method

Measurement is done entirely **inside the pod** (`scripts/launcher/bench.py`,
`drive.py`) so no kube-apiserver round trips pollute timing.

- **t0** = the launcher's `PUT /v2/vllm/instances/{id}` returns. The launcher
  calls `multiprocessing.Process.start()` synchronously inside that handler, so
  the POST return is the process-creation instant to within a millisecond
  (measured POST overhead: 4–17 ms, reported per launch).
- **ready** = the instance's OpenAI server answers `GET /health` with 200.
- **total_s** = ready − t0. Phase durations (weight load, CUDA-graph capture,
  torch.compile) are read from vLLM's *own* log lines, not from diffing
  1-second log timestamps.
- Each arm boots one launcher and runs **2 sequential launches** on that same
  launcher (launch #0 = first launch on a fresh launcher; launch #1 = reuse).
  The instance is deleted between launches; its log is copied out first because
  the launcher unlinks logs on stop.

### The two launchers compared

- **warm-import** (`launcher.py`, upstream): imports
  `vllm.entrypoints.openai.api_server` etc. at module top. A forked child
  inherits torch+vllm already imported. This is the stock launcher.
- **cold-import** (`launcher_coldimport.py`): the same file with the four vLLM
  imports moved *inside* the kickoff function, so a forked child imports vLLM
  fresh. This is a synthetic "naive" baseline built only to quantify what the
  stock launcher's warm import is worth.

(One unrelated patch: `gputranslator.py` imports `kubernetes` lazily — the vLLM
image doesn't ship that package and only the mock-GPU path uses it. The
real-GPU NVML path is untouched.)

### The arms

| arm | launcher | `VLLM_WORKER_MULTIPROC_METHOD` | loader |
|---|---|---|---|
| A / A2 | cold-import | spawn | auto (safetensors) |
| B | warm-import | spawn | auto (safetensors) |
| C / C2 | warm-import | fork | auto (safetensors) |
| D / D2 | warm-import | fork | fastsafetensors |
| E / E2 | warm-import | fork | instanttensor |

(B2 is omitted: at TP=2 fork is forced to spawn, so C2 *is* the warm-import
spawn case.)

`instanttensor` is not in the base vLLM image; it was installed into the pod
with `pip install "vllm[instanttensor]"` (instanttensor 0.1.9). fastsafetensors
ships with vLLM.

---

## Results

Times in seconds. **launch#0** = cold launcher (first launch), **launch#1** =
warm reuse. `launcher_boot` is the one-time cost to bring the launcher to
healthy and is *not* part of any launch number.

### TP=1

| arm | launcher | EngineCore | launcher_boot | launch#0 | **launch#1 (warm)** | weight load | cudagraph |
|---|---|---|---|---|---|---|---|
| A | cold-import | spawn | 0.8 | 126.9 | **89.2** | 65→29 | 11 |
| B | warm-import | spawn | 15.3 | 72.3 | **72.1** | 30.5 | 10 |
| C | warm-import | fork | 17.1 | 58.7 | **58.1** | 29.5–30.1 | 12 |
| D | warm-import | fork + fastsafetensors | 16.0 | 47.8 | **45.4** | 19.1→18.8 | 11 |
| E | warm-import | fork + instanttensor | 14.0 | 34.6 | **33.9** | 9.4→8.1 | 10–11 |

**Where the TP=1 time goes, step by step (warm weights):**

- **A→B, −17 s (89→72):** warm-import launcher. The forked API server inherits
  torch+vllm from the launcher instead of importing them. Evidence: in B the
  launcher forks the API server at `13:16:54.98` and the banner prints
  instantly — but the **spawned** EngineCore doesn't reach "Initializing a V1
  LLM engine" until `13:17:13`, an **~18 s gap** spent re-importing torch+vllm
  in a fresh interpreter.
- **B→C, −14 s (72→58):** fork the EngineCore. Forking instead of spawning it
  removes that 18 s re-import gap — the forked EngineCore inherits the imported
  stack and begins engine init almost immediately. Confirmed: C's logs show the
  EngineCore (a forked pid) loading weights directly, with no "Overriding
  ... to spawn" line.
- **C→D, −13 s (58→45):** fastsafetensors loader. Weight load drops from ~30 s
  to ~19 s.
- **D→E, −11 s (45→34):** instanttensor loader. Weight load drops again, ~19 s
  → ~8 s (measured ~9 GB/s; log: "Loading safetensors using InstantTensor
  loader"). This is the fastest loader tested and gives the best TP=1 number.

Everything from A→C (89→58, −31 s) is bought purely by *how the launcher
creates the child* — no change to vLLM's own code or flags. E→ the loader
choice is orthogonal and stacks on top.

### TP=2

| arm | launcher | EngineCore | launcher_boot | launch#0 | **launch#1 (warm)** | weight load | cudagraph |
|---|---|---|---|---|---|---|---|
| A2 | cold-import | spawn | 1.0 | 155.5 | **96.0** | 15.2–15.7 | 9–10 |
| C2 | warm-import | fork→**spawn** | 15.4 | 82.0 | **88.3** | 15.4–15.7 | 9–10 |
| D2 | warm-import | fork→**spawn** + fastsafetensors | 16.4 | 78.9 | **81.6** | 9.6 | 9 |
| E2 | warm-import | fork→**spawn** + instanttensor | 14.4 | 70.4 | **70.5** | 8.5→8.7 | 8 |

**The TP=2 fork override (the key TP=2 finding).** Requesting fork at TP=2 does
*not* give a forked EngineCore. Every TP=2 run with fork requested logs:

> `system_utils.py:157] We must use the 'spawn' multiprocessing start method.
> Overriding VLLM_WORKER_MULTIPROC_METHOD to 'spawn'. ... Reasons: CUDA is
> initialized.`

By the time the multiproc executor is created at TP>1, CUDA is already
initialized in the parent, and vLLM refuses to fork from a CUDA-initialized
process (forking after CUDA init is unsafe). Consequences:

- C2's ~11–14 s win over A2 is **entirely** the warm-import launcher letting the
  *API server* skip its imports. The EngineCore and **both** TP workers are
  still spawned and each re-imports torch+vllm.
- fastsafetensors (D2) still helps: TP=2 weight load 15.4 s → 9.6 s, ~6 s off.
- instanttensor (E2) is the best TP=2 arm at **70.5 s**, ~10 s under D2. Note
  that TP=2 weight load is already small (~9–15 s, since the 61 GiB is sharded
  across two ranks reading in parallel), so most of E2's edge shows up as an
  earlier CUDA-graph capture start (capture begins at ~54 s vs ~61–63 s for D2)
  rather than as raw weight-byte time. The loader still pays off, just less
  dramatically than at TP=1 where it owns the whole 30 s single-rank read.
- TP=2 also carries a first-time compile penalty that the cache absorbs on
  reuse: A2 launch#0 = 155.5 s includes a fresh TP=2 torch.compile (dynamo
  13.4 s) whose artifact is cached to the PVC; launch#1 = 96.0 s.

**Why TP=2 weight load is *smaller* than TP=1** (15 s vs 30 s): the 61 GiB is
sharded across 2 GPUs, so each rank reads ~half. This is real wall-clock
because the two ranks read in parallel.

---

## Warm-cache inventory (what a permanent launcher gets to reuse)

Ordered by size of effect. (1)–(2) are launcher-process caches; (3)–(5) are
filesystem/OS caches the launcher benefits from but does not own.

| # | Warm cache | Mechanism | Worth | Scope |
|---|---|---|---|---|
| 1 | **Imported vLLM stack** | launcher imports at module load; child forks | ~17 s (TP=1 API+engine via fork); ~12 s (TP=2 API only) | launcher process; per launch |
| 2 | **Forked EngineCore** | `MULTIPROC_METHOD=fork` | ~14 s at **TP=1 only** (nil at TP=2 — forced to spawn) | launcher process; per launch |
| 3 | **torch.compile / inductor / triton** | `VLLM_CACHE_ROOT` etc. on PVC | first-config compile ~13 s, then ~0 | PVC; cross-pod |
| 4 | **Model-registry modelinfos** | `$VLLM_CACHE_ROOT/modelinfos` on PVC | ~15 s cold → ~0 warm (measured separately: `reports/registry-cache-cold-vs-warm-*.txt`) | PVC; cross-pod |
| 5 | **Weight page cache (GPFS)** | node kernel page cache | first cold read costs ~2× the weight phase | node-local; warms after 1st read |

**Cache (5), cold vs warm weights.** Only the very first launch on a cold node
reads weights from GPFS cold. TP=1 arm A: launch#0 weight load ≈ 65 s vs
launch#1 ≈ 29 s — a cold read roughly doubles the weight phase. Every launch
here after the first hit a warm page cache. A launcher that has served the same
model recently gets this for free; a fresh node does not.

**Launcher boot is amortized.** The warm-import launcher costs ~15–17 s to
reach healthy (it imports the whole server stack). That is paid **once** for
the life of the process. The cold-import launcher boots in ~1 s but then pays
far more on *every* launch. For a permanent, reusable launcher the warm-import
trade is unambiguously correct:

```
total wall for N launches (TP=1, warm):
  cold-import : 0.8 + N·89.2
  warm-import : 16   + N·58.1   (fork) / N·72.1 (spawn)
break-even vs cold-import spawn is < 1 launch; from launch 1 on, warm-import+fork wins.
```

---

## Launcher vs. tuned standalone (the "is the launcher even worth it?" check)

Every number above compares launcher arms *to each other*. That answers "what
does the launcher's design buy?" but not "does the launcher beat just running
`vllm serve` well?" So here is the honest side-by-side, same pod, same model,
same t0 definition (process-creation instant → `/health` 200), all the same
levers on (warm bytecode, fork EngineCore, instanttensor, startup plan, warm
PVC caches):

| | best TP=1 | best TP=2 |
|---|---|---|
| tuned standalone `vllm serve` (`cold-start-slides.html`) | **36.9 s** (median of 3: 35.9 / 36.9 / 37.9) | 53.5 s |
| this launcher | **33.9 s** (samples 34.6 / 33.9) | 70.5 s |
| **launcher net advantage** | **~3 s** | **none (launcher is slower here)** |

**TP=1 — where the ~3 s comes from and why it isn't more.** The launcher's one
structural advantage is import inheritance: the forked child skips the vLLM
module-load imports the standalone pays up front. In the arm-E log that
front-end (process start → "Initializing a V1 LLM engine") is **~3 s**, versus
the standalone's **10.2 s** "python imports" phase — so the launcher really does
remove ~7 s there. But the net is only ~3 s because (a) the standalone number is
a median while the launcher is a single warm reuse, and (b) in that sample the
launcher's post-import phases ran ~4 s slower (single-run variance + the CPU
throttling the slides flag), eating most of the front-end win. The deeper reason
the advantage is small at all: a standalone with a **warm bytecode cache already
imports cheaply**, so fork-inheritance only claws back the residual.

**TP=2 — the launcher is *behind*.** 70.5 s vs 53.5 s standalone. The launcher
arm here does not carry the TP=2-specific levers the standalone deck adopted
(e.g. sleep-mode / the deck's TP=2 tuning), and fork is forced to spawn anyway,
so the launcher has no structural lever left to offset. Do not read the launcher
as a TP=2 speedup.

**So what is the launcher for?** Not per-boot latency. It is an *operational*
component: a permanently resident process that owns vLLM instance lifecycle
(create / health / delete), amortizes its own ~16 s warm-import boot to zero, and
gives you a control-plane API in front of the server. Its cold-start numbers are
*competitive with* a hand-tuned standalone at TP=1 (~3 s better) and behind at
TP=2 — the reason to run it is the lifecycle management, not the clock.

*Caveat: the launcher figures are single warm reuses, not medians. A properly
interleaved launcher-vs-standalone median (cycle 0 discarded, arms interleaved,
same as the slide methodology) would pin the ~3 s down; it is not yet run.*

## Recommendations

1. **If you run a launcher, keep its warm-import + fork design.** It is the
   single largest win *within the launcher family* at TP=1 (−31 s, 89→58) and it
   is free — it's how the stock launcher already works, so do not "fix" the
   top-level imports. But note (see *Launcher vs. tuned standalone*) that this
   only claws the launcher back up to *parity-plus-~3 s* with a tuned standalone;
   choose the launcher for lifecycle management, not for the ~3 s.
2. **Swap the weight loader — `instanttensor` first, `fastsafetensors` as the
   in-image fallback.** instanttensor is the fastest tested (weight load ~30 s →
   ~8 s at TP=1): −11 s beyond fastsafetensors at TP=1 (best number 33.9 s) and
   ~10 s at TP=2 (best 70.5 s). It must be pip-installed (`vllm[instanttensor]`);
   if you cannot add it to the image, `--load-format fastsafetensors` ships with
   vLLM and still buys −13 s at TP=1 / −6 s at TP=2. Both are loader-only, no
   correctness change.
3. **At TP=2, do not expect fork to help the engine.** vLLM forces spawn once
   CUDA is initialized. The launcher's import inheritance still helps the API
   server (~12 s), but the EngineCore + workers re-import regardless. If TP>1
   engine startup must be cut, the lever is a *forkserver*/preloaded-worker
   scheme (see `CS_FORKSERVER` notes in `manifests/pod-exec-2gpu.yaml`), not
   `MULTIPROC_METHOD=fork`.
4. **Keep every cache on the PVC warm and explicit** (`VLLM_CACHE_ROOT`,
   `TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR`, `HF_HOME`, …). A cold
   torch.compile is ~13 s and a cold model registry is ~15 s; both are
   cross-pod on the PVC. Do **not** set `HF_HUB_OFFLINE` — it rewrites `--model`
   to a snapshot path and busts the compile cache (documented in the manifest).
5. **Pre-warm the node page cache** for the model you expect to serve if
   first-launch latency matters: the first cold weight read roughly doubles the
   weight phase.
6. **Remaining floor.** After the above (instanttensor), weight load is down to
   ~8 s and the largest remaining phase is CUDA-graph capture + warmup
   (~10–12 s), with the rest spread across engine init and profiling. Cutting
   weight load further means keeping weights resident (a snapshot/streaming
   scheme); cutting capture means fewer capture sizes (a latency/throughput
   trade). At TP=2, the engine + both workers still spawn-and-reimport (fork is
   forced off), so that ~12 s import gap is the next-biggest TP=2 target — see
   the forkserver note in rec. 3.

---

## Artifacts

- Scripts: `scripts/launcher/{launcher.py, launcher_coldimport.py,
  gputranslator.py, bench.py, drive.py}`
- Raw per-launch JSON: `scripts/launcher/results/tp{1,2}{C,D,E}.json`
  (also in pod at `/cache/launcher/results/`; tp2A pulled too)
- Preserved instance logs (in pod): `/cache/launcher/logs/*.log`
- Related prior reports: `reports/registry-cache-cold-vs-warm-*.txt`,
  `reports/loader-comparison-32b.txt`, `reports/32b-step*.txt`.

*Numbers are single-run per launch (2 launches per arm); treat ±2 s as noise.
The cross-launch and cross-arm gaps reported here are all well outside that.*
