#!/usr/bin/env python3
"""Summarise the CUDA graph capture phase across runs of the pool/sync matrix.

`analysis/coldstart_report.py` renders one run; this renders the *comparison*
that scripts/cudagraph-pool-matrix.sh produces, because the interesting columns
here are inside one phase and are not in the phase table:

  capture_model      the whole phase
  capture_forward    CPU recording -- the only part concurrency could overlap
  warmup_forward     the eager forward per (mode, size), dead work on v2
  capture_begin      torch's per-capture prologue
   `- device sync    the nested torch.cuda.synchronize(), which a concurrent
                     capture cannot survive (docs/concurrent-cudagraph-capture.md)
  graph pool GiB     from vLLM's own "Graph capturing finished" log line
  pools / nosync /
  noempty / once     cs_cgcapture events, i.e. proof the arm engaged

The `pool GiB` column cannot score a `CS_CGEMPTY=once` arm: vLLM measures it
*inside* `capture_model` (`v1/worker/gpu/model_runner.py:871,901`), so a free that
happens after the capture loop is invisible to it and the arm reads as if it kept
the memory. Use analysis/gpu_steady_memory.py, which reads NVML at the end of the
run, for anything about memory that outlives the phase.

Usage:
  analysis/cudagraph_capture_matrix.py runs/cc-*-c[0-9]     # groups by arm
  analysis/cudagraph_capture_matrix.py --raw runs/cc-base-c1

The glob deliberately requires the `-cN` pass suffix: a run without one is not
part of an interleaved matrix, and averaging it into an arm would compare it
against runs it never interleaved with.
"""

from __future__ import annotations

import argparse
import bisect
import collections
import glob
import json
import os
import re
import statistics
import sys

SPANS = ("cudagraph.capture_model", "cudagraph.capture_forward",
         "cudagraph.warmup_forward", "cudagraph.capture_begin",
         "cudagraph.capture_end")


def one_run(run_dir: str) -> dict | None:
    begins, syncs = [], []
    tot = collections.Counter()
    cnt = collections.Counter()
    pools = collections.Counter()
    nosync = noempty = once = 0
    for f in glob.glob(os.path.join(run_dir, "trace", "*.jsonl")):
        for line in open(f):
            if "cudagraph" not in line and "cuda.synchronize" not in line:
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            name = d.get("name", "")
            if name == "cgcapture.pool":
                pools[d.get("args", {}).get("pool")] += 1
                continue
            if name == "cgcapture.nosync":
                nosync += 1
                continue
            if name == "cgcapture.noempty":
                noempty += 1
                continue
            if name == "cgcapture.empty_once":
                once += 1
                continue
            ts, dur = d.get("ts"), d.get("dur")
            if ts is None or dur is None:
                continue
            if name in SPANS:
                tot[name] += dur
                cnt[name] += 1
            if name == "cudagraph.capture_begin":
                begins.append((ts, ts + dur))
            elif name == "cuda.synchronize":
                syncs.append((ts, dur))
    if not cnt:
        return None

    # The sync we care about is the one torch does inside graph.__enter__, so
    # attribute by containment rather than by name: cuda.synchronize is called
    # from several places during startup.
    begins.sort()
    starts = [b[0] for b in begins]
    nested = 0.0
    for ts, dur in syncs:
        i = bisect.bisect_right(starts, ts) - 1
        if i >= 0 and ts < begins[i][1]:
            nested += dur

    gib = None
    log = os.path.join(run_dir, "vllm.log")
    if os.path.exists(log):
        for line in open(log, errors="replace"):
            m = re.search(r"Graph capturing finished in \d+ secs, took ([\d.]+) GiB",
                          line)
            if m:
                gib = float(m.group(1))
    return {"run": os.path.basename(run_dir), "tot": tot, "cnt": cnt,
            "sync": nested, "gib": gib,
            "pools": dict(sorted(pools.items(), key=lambda kv: (kv[0] is None, kv[0]))),
            "nosync": nosync, "noempty": noempty, "once": once}


def arm_of(run: str) -> str:
    return re.sub(r"-c\d+$", "", run)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--raw", action="store_true", help="one line per run")
    args = ap.parse_args()

    rows = [r for r in (one_run(d) for d in sorted(args.runs)) if r]
    if not rows:
        sys.exit("no capture spans found in: %s" % " ".join(args.runs))

    hdr = ("%-14s %6s %9s %7s %9s %9s %9s %9s %8s  %s"
           % ("arm", "n", "capture", "spread", "cap_fwd", "warm_fwd", "cap_beg",
              "`-sync", "pool GiB", "engaged"))
    print(hdr)
    print("-" * len(hdr))

    groups = collections.OrderedDict()
    for r in rows:
        groups.setdefault(r["run"] if args.raw else arm_of(r["run"]), []).append(r)

    def med(rs, key, span=None):
        vals = [(x["tot"][span] if span else x[key]) for x in rs
                if (x["tot"][span] if span else x[key]) is not None]
        return statistics.median(vals) if vals else float("nan")

    for name, rs in groups.items():
        engaged = []
        pools = rs[0]["pools"]
        if pools:
            engaged.append("pools=%d (%s)"
                           % (len(pools), "/".join(str(v) for v in pools.values())))
        if rs[0]["nosync"]:
            engaged.append("nosync x%d" % rs[0]["nosync"])
        if rs[0]["noempty"]:
            engaged.append("noempty x%d" % rs[0]["noempty"])
        if rs[0]["once"]:
            # The hoisted free. Its presence is why the pool GiB column above is
            # not the right place to read this arm's memory (see the docstring).
            engaged.append("hoisted free")
        gibs = [r["gib"] for r in rs if r["gib"] is not None]
        # The spread within an arm is the noise floor: every timing column here
        # is only readable to the extent the arms differ by more than it.
        caps = [r["tot"]["cudagraph.capture_model"] for r in rs]
        spread = ("%.3fs" % (max(caps) - min(caps))) if len(caps) > 1 else "-"
        print("%-14s %6d %8.3fs %7s %8.3fs %8.3fs %8.3fs %8.3fs %8s  %s"
              % (name, len(rs),
                 med(rs, None, "cudagraph.capture_model"),
                 spread,
                 med(rs, None, "cudagraph.capture_forward"),
                 med(rs, None, "cudagraph.warmup_forward"),
                 med(rs, None, "cudagraph.capture_begin"),
                 med(rs, "sync"),
                 ("%.2f" % statistics.median(gibs)) if gibs else "-",
                 ", ".join(engaged) or "upstream"))

    print("\nper-run graph pool, as vLLM reports it:")
    for r in rows:
        print("  %-16s %s GiB   captures=%d"
              % (r["run"], r["gib"], r["cnt"]["cudagraph.capture_forward"]))


if __name__ == "__main__":
    main()
