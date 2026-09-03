#!/usr/bin/env python3
"""Steady-state device memory per run, from NVML rather than from vLLM.

vLLM's "Graph capturing finished in N secs, took X GiB" is the obvious source for
what a graph-pool change costs, and analysis/cudagraph_capture_matrix.py reads it.
But it is measured *inside* `capture_model` --

    start_free_gpu_memory = torch.accelerator.get_memory_info()[0]   # model_runner.py:871
    ...capture...
    end_free_gpu_memory   = torch.accelerator.get_memory_info()[0]   # model_runner.py:901

-- so it cannot see anything that happens after the capture loop returns, which is
exactly what the CS_CGEMPTY=once arm does (one free after the loop instead of 3366
inside it). Any claim about that arm's memory needs a measurement taken later.

cs_sampler already writes one: NVML `mem_used` for the whole device, as `gpu0`
counter events, sampled to the end of the run. This reports the median of the last
`--window` seconds of those -- i.e. device memory with the server up and serving,
which is the number an operator actually pays.

It is device-wide and NVML-rounded, so it includes ~0.5 GiB of context and is
coarser than vLLM's delta. Read it for *differences between arms on the same node*,
not as an absolute.

  analysis/gpu_steady_memory.py runs/ch-*-c[0-9] runs/cp-*-c[0-9]
  analysis/gpu_steady_memory.py --window 5 runs/cc-pool*-c[0-9]
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
import statistics
import sys

GIB = 1 << 30


def one_run(run_dir: str, window: float) -> dict | None:
    samples = []
    for f in glob.glob(os.path.join(run_dir, "trace", "*.jsonl")):
        for line in open(f):
            if '"gpu0"' not in line:
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get("name") != "gpu0" or d.get("ph") != "C":
                continue
            used = d.get("args", {}).get("mem_used")
            ts = d.get("ts")
            if used is not None and ts is not None:
                samples.append((ts, used))
    if not samples:
        return None
    samples.sort()
    t_end = samples[-1][0]
    tail = [u for ts, u in samples if ts >= t_end - window]
    return {"run": os.path.basename(run_dir), "n_all": len(samples),
            "n_tail": len(tail), "span_s": t_end - samples[0][0],
            "peak_GiB": max(u for _, u in samples) / GIB,
            "steady_GiB": statistics.median(tail) / GIB}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--window", type=float, default=3.0,
                    help="seconds at the end of the run to median over")
    ap.add_argument("--raw", action="store_true")
    args = ap.parse_args()

    rows = [r for r in (one_run(d, args.window) for d in sorted(args.runs)) if r]
    if not rows:
        sys.exit("no gpu0 samples in: %s" % " ".join(args.runs))

    groups = collections.OrderedDict()
    for r in rows:
        key = r["run"] if args.raw else re.sub(r"-c\d+$", "", r["run"])
        groups.setdefault(key, []).append(r)

    hdr = "%-16s %3s %10s %8s %10s  %s" % ("arm", "n", "steady GiB", "spread",
                                           "peak GiB", "per-run steady")
    print(hdr)
    print("-" * len(hdr))
    base = None
    for name, rs in groups.items():
        st = [r["steady_GiB"] for r in rs]
        med = statistics.median(st)
        if base is None:
            base = med
        print("%-16s %3d %10.2f %8s %10.2f  %s"
              % (name, len(rs), med,
                 ("%.2f" % (max(st) - min(st))) if len(st) > 1 else "-",
                 statistics.median([r["peak_GiB"] for r in rs]),
                 " ".join("%.2f" % v for v in st)))
    print("\ndeltas vs the first arm listed (%.2f GiB):" % base)
    for name, rs in groups.items():
        med = statistics.median([r["steady_GiB"] for r in rs])
        print("  %-16s %+.2f GiB" % (name, med - base))
    print("\nsampled over %.0fs per run, median of the last %.0fs"
          % (statistics.median([r["span_s"] for r in rows]), args.window))


if __name__ == "__main__":
    main()
