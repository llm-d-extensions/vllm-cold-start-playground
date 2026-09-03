#!/usr/bin/env python3
"""Per-arm medians for the loader ladder: total to ready, and where it went.

  python3 analysis/loader_ladder.py runs/lf1-*


Same shape as analysis/registry_ladder.py -- cycle 0 dropped as warm-up, median
of the rest -- but the interesting column here is the `weight load` phase, since
that is the only phase the load format can move.
"""
import os, re, statistics, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coldstart_report import Run, sweep_phases

WATCH = ["weight load", "python imports", "warmup / profile run", "kv cache alloc",
         "cudagraph capture", "engine orchestration"]

def med(xs):
    return statistics.median(xs) if xs else float("nan")

rows = {}
for p in sys.argv[1:]:
    rid = os.path.basename(p.rstrip("/"))
    m = re.match(r"^[^-]+-(.+)-c(\d+)$", rid)
    if not m or int(m.group(2)) == 0:
        continue
    trace = os.path.join(p, "trace")
    run = Run(trace if os.path.isdir(trace) else p)
    t0 = run.t0(); t_ready, _ = run.t_ready()
    per, _ = sweep_phases(run, t0, t_ready)[:2]
    rows.setdefault(m.group(1), []).append((t_ready - t0, per))

hdr = f"{'arm':<12} {'n':>2} {'ready':>7} " + " ".join(f"{w[:11]:>11}" for w in WATCH)
print(hdr); print("-" * len(hdr))
base = None
for arm in sorted(rows):
    v = rows[arm]
    tot = med([x[0] for x in v])
    if base is None:
        base = tot
    cells = " ".join(f"{med([x[1].get(w, 0.0) for x in v]):>10.2f}s" for w in WATCH)
    print(f"{arm:<12} {len(v):>2} {tot:>6.2f}s {cells}   ({tot - base:+.2f}s)")
print()
for arm in sorted(rows):
    print(f"{arm:<12} totals per cycle: " +
          ", ".join(f"{x[0]:.2f}s" for x in rows[arm]))
