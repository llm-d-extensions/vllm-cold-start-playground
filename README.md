# llm-d cold start: vLLM startup measurement

Tooling to answer one question precisely: **where does the time go between the
moment a vLLM process is exec'd and the moment its API server can actually
serve a token?**

This is the measurement half of [SIG Fast Start](CHARTER.md) Goal #1 — a
canonical time-to-ready metric broken down by phase. It is instrumentation and
analysis only: it changes nothing about how vLLM starts, and it is meant to be
run against unmodified upstream images.

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
  Mode B, cache volumes, OpenShift/SCC notes, troubleshooting.
* **[docs/experiments.md](docs/experiments.md)** — the experiment matrix and how
  to run a defensible comparison.
* **[docs/foundry.md](docs/foundry.md)** — ingested notes on
  [Foundry](https://github.com/foundry-org/foundry), which persists CUDA graphs to
  disk: what it eliminates in our phase model, its operating constraints, and how
  to evaluate it here.

## Layout

```
coldstart/          the probe: sitecustomize.py + cs_*.py (stdlib only)
analysis/           coldstart_report.py — trace dir -> report/CSV/Perfetto
manifests/          experiment pod, cache PVC, probe ConfigMap source
scripts/            in-pod driver, ConfigMap builder, host-side run/fetch
tests/              mock vLLM + end-to-end self-test
```

The probe is stdlib-only and imports nothing at module scope that vLLM would not
already import, so it runs inside any vLLM image without adding dependencies.
