# vLLM cold start from a launcher process (TP=1 and TP=2)

**Question.** A launcher process stays running and, on request, starts vLLM as
a child process. How long from "start the child" to vLLM being ready, what
moves that number, and — since the launcher is long-lived — which warm caches
can it reuse?

Setup: Qwen/Qwen3-32B (61 GiB, bf16), H100, vLLM 0.28.0 / torch 2.13 / CUDA 13,
16 CPU cores, weights on a warm GPFS cache. Launcher code from
`llm-d-incubation/llm-d-fast-model-actuation`, exercised via `scripts/launcher/`.
t0 = the launcher's process-creation call returns; ready = `/health` returns 200.

## Headline numbers

| | Naive baseline | Best measured | Improvement |
|---|---|---|---|
| **TP=1** | 89.2s | **33.9s** | −55s (−62%) |
| **TP=2** | 96.0s | **70.5s** | −25s (−27%) |

**Important caveat:** the "naive baseline" here is a launcher that imports
vLLM cold, which is a fair comparison for "what does the launcher's own design
buy you" — but not for "should I use a launcher at all." A fully tuned
*standalone* `vllm serve` (no launcher, just warm bytecode + fork + the fastest
loader) reaches 36.9s at TP=1. So the launcher's real advantage over a
well-tuned standalone is only about **3 seconds at TP=1, and roughly zero at
TP=2**. The launcher's value is operational (it stays running and manages
instance lifecycle), not raw cold-start speed. See "Launcher vs. tuned
standalone" below.

Main levers, in order of impact:
1. **Warm imports + forking the child** (the launcher's own design): the
   launcher imports vLLM once at startup; a forked child inherits that instead
   of re-importing. Takes TP=1 from 89s → 58s with zero changes to vLLM.
2. **Faster weight loader**: fastsafetensors gets TP=1 to 45s; instanttensor
   (fastest tested) gets weight loading from ~30s down to ~8s, landing at 33.9s.
3. **At TP=2, forking silently stops working.** Once CUDA is initialized (which
   happens before the multi-GPU workers start), vLLM forces the "spawn" method
   instead of fork, regardless of what was requested. So at TP=2 only the API
   server benefits from warm imports (~12s); the engine and both GPU workers
   still spawn fresh and re-import everything.
4. The launcher's own one-time warm-import startup cost (~15–17s) is paid once
   and amortized across every subsequent launch — free after the first.

## How this was measured

Each test ran two launches back-to-back on the same launcher (launch #0 = cold
launcher, launch #1 = warm reuse). Two launcher variants were compared:
- **warm-import** (the real launcher): imports vLLM at startup, so a forked
  child inherits it.
- **cold-import** (a synthetic test-only variant): same code, but imports moved
  so a forked child re-imports vLLM fresh — built only to measure what warm
  import is worth.

Five combinations ("arms") were tested, combining launcher type, fork-vs-spawn,
and weight loader (default safetensors / fastsafetensors / instanttensor —
the last needed a manual pip install, not in the base image).

## TP=1 results

| arm | setup | launch#1 (warm) | weight load |
|---|---|---|---|
| A | cold-import, spawn, default loader | 89.2s | 29-65s |
| B | warm-import, spawn, default loader | 72.1s | 30.5s |
| C | warm-import, fork, default loader | 58.1s | ~30s |
| D | warm-import, fork, fastsafetensors | 45.4s | ~19s |
| E | warm-import, fork, instanttensor | 33.9s | 8-9s |

Step by step: warm imports save ~17s (A→B) by skipping the re-import in the
API server process. Forking the engine itself (not just the API server) saves
another ~14s (B→C) — the forked engine inherits the loaded interpreter instead
of importing from scratch. Switching to fastsafetensors saves ~13s more on
weight loading (C→D), and instanttensor saves another ~11s (D→E). Everything
through step C (89s→58s) comes purely from how the launcher starts its child
process — no vLLM changes needed.

## TP=2 results

| arm | setup | launch#1 (warm) | weight load |
|---|---|---|---|
| A2 | cold-import, spawn, default loader | 96.0s | ~15s |
| C2 | warm-import, fork→forced-spawn, default loader | 88.3s | ~15s |
| D2 | warm-import, fork→forced-spawn, fastsafetensors | 81.6s | 9.6s |
| E2 | warm-import, fork→forced-spawn, instanttensor | 70.5s | 8.5-8.7s |

**Key TP=2 finding: requesting "fork" doesn't get you a forked engine.** Every
TP=2 run logs vLLM overriding the setting back to "spawn," because by the time
the multi-GPU workers start, CUDA is already initialized in the parent process,
and forking a CUDA-initialized process is unsafe. So at TP=2:
- The only import saving is for the API server (~12s); the engine and both GPU
  workers still spawn and re-import fully.
- fastsafetensors still saves ~6s of weight-load time; instanttensor is best
  overall (70.5s), mostly by starting CUDA-graph capture earlier rather than by
  further shrinking weight load (which is already small at TP=2 since each of
  the 2 GPUs only loads half the weights — ~15s vs ~30s at TP=1).
- The very first launch on a cold node also pays a one-time ~13s torch.compile
  cost that gets cached for later launches.

## What the launcher gets to reuse (because it stays running)

| Cache | What it saves | Owned by |
|---|---|---|
| Imported vLLM stack + forked engine | ~17s at TP=1 (fork works); ~12s at TP=2 (API server only, fork is forced off) | launcher process |
| torch.compile / triton compiled kernels | ~13s first time, then ~0 | shared disk cache, reusable across pods |
| Model registry lookup cache | ~15s cold → ~0 warm | shared disk cache, reusable across pods |
| OS page cache for model weight files | first cold read roughly doubles weight-load time | per-node, warms after first read |

The launcher's own ~15-17s warm-import startup cost is paid once per launcher
process and amortized to zero over many launches — a clear win for a
long-lived launcher, even though a short-lived one would never recoup it.

## Launcher vs. a tuned standalone `vllm serve` (is the launcher even worth it?)

Comparing the launcher's best result to a separately-tuned standalone `vllm
serve` (same model, same warm caches, same fork+instanttensor tricks applied
directly, no launcher in front):

| | best TP=1 | best TP=2 |
|---|---|---|
| tuned standalone `vllm serve` | 36.9s | 53.5s |
| this launcher | 33.9s | 70.5s |
| launcher's advantage | ~3s | none — launcher is slower |

At TP=1 the launcher's only structural edge is skipping vLLM's own import step
via fork — worth about 7s in isolation, but a standalone with a warm bytecode
cache already imports cheaply on its own, so the net gain shrinks to ~3s. At
TP=2 the launcher has no advantage at all: fork is forced off, and the
standalone benchmark used TP=2-specific tuning that this launcher setup
doesn't have.

**Bottom line: the launcher is not a cold-start speed tool.** Its value is
operational — it's a long-lived process that owns instance lifecycle (create /
health-check / delete) and amortizes its own warm-up cost across many launches.
Choose it for that reason, not to shave seconds off boot time.

*Caveat: the launcher numbers above are single warm-reuse runs, not medians
like the standalone comparison — a properly matched apples-to-apples test has
not been run yet, so treat the ~3s TP=1 gap as approximate.*

## Recommendations

1. Keep the launcher's warm-import + fork design — it's free (how the stock
   launcher already works) and the single biggest win within the launcher
   family at TP=1. But pick a launcher for lifecycle management, not for the
   ~3s speed edge over a tuned standalone.
2. Swap the weight loader: instanttensor first (fastest, needs
   `pip install vllm[instanttensor]`), fastsafetensors as the in-image fallback
   if you can't add packages. Both are drop-in, no correctness impact.
3. At TP=2, don't expect forking to help the engine — CUDA initialization
   forces spawn regardless. A forkserver/preloaded-worker scheme would be
   needed to close that gap; plain fork won't do it.
4. Keep compile and model-registry caches on shared, persistent storage across
   pods. A cold compile costs ~13s, a cold registry lookup ~15s. Don't set
   `HF_HUB_OFFLINE` — it changes the model path in a way that breaks the
   compile cache.
5. Pre-warm the node's page cache for the model if first-launch latency
   matters — a cold weight read roughly doubles weight-load time.
6. After applying all of the above, weight loading is down to ~8s and
   CUDA-graph capture (~10-12s) becomes the largest remaining phase. Further
   gains would require keeping weights resident in memory, or capturing fewer
   graph sizes (a latency/throughput tradeoff). At TP=2 the ~12s import gap
   from forced spawning is the next biggest target.

## Artifacts

- Scripts: `scripts/launcher/{launcher.py, launcher_coldimport.py, bench.py, drive.py}`
- Raw per-launch data: `scripts/launcher/results/tp{1,2}{C,D,E}.json`
- Related reports: `reports/registry-cache-cold-vs-warm-*.txt`,
  `reports/loader-comparison-32b.txt`, `reports/32b-step*.txt`.

*Numbers are single runs (2 launches per arm) — treat ±2s as noise. The gaps
reported here are all well outside that.*
