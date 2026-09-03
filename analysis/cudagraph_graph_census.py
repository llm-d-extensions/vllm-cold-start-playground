#!/usr/bin/env python3
"""Census of every `torch.cuda.graph` a startup builds, from a CS_CGDETAIL run.

`cudagraph.capture_begin`/`capture_end` in the normal traces are floored at 5ms
(`coldstart/cs_torch.py`), so they count *captures over 5ms*, not captures. That
floor is why the README says "exactly one torch CUDAGraph per (mode, size): 51
PIECEWISE, 51 FULL". A run with CS_CGDETAIL=1 spans every `__enter__`/`__exit__`
with no floor and tags each capture with the source line that constructed it,
which is what this renders:

  * how many CUDAGraphs each capture path really builds, and
  * how much of the phase is torch's per-capture prologue/epilogue -- both the
    part above the 5ms floor and the part underneath it.

Usage:
  analysis/cudagraph_graph_census.py runs/cc-detail2

The src attribution pairs each `cgcapture.pool` instant with the next
`cgcapture.enter` span on the same thread: the instant is emitted from
`graph.__init__` and the capture that follows on that thread is that object's.
"""

from __future__ import annotations

import collections
import glob
import json
import os
import sys

FLOOR = 0.005          # cs_torch's min_dur on cudagraph.capture_begin/_end


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    run = sys.argv[1]

    by_thread = collections.defaultdict(list)
    for f in glob.glob(os.path.join(run, "trace", "*.jsonl")):
        for line in open(f):
            if "cgcapture." not in line:
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get("name", "").startswith("cgcapture."):
                by_thread[(d.get("pid"), d.get("tid"))].append(d)
    if not by_thread:
        sys.exit("no cgcapture.* events in %s -- was it run with CS_CGDETAIL=1?" % run)

    tot = collections.defaultdict(collections.Counter)   # src -> {enter,exit}
    cnt = collections.Counter()
    over = collections.defaultdict(collections.Counter)  # src -> above the floor
    for events in by_thread.values():
        events.sort(key=lambda d: d["ts"])
        src = None
        for d in events:
            name = d["name"]
            if name == "cgcapture.pool":
                src = d.get("args", {}).get("src") or "?"
                cnt[src] += 1
            elif name in ("cgcapture.enter", "cgcapture.exit"):
                dur = d.get("dur")
                if dur is None:
                    continue
                key = name.split(".")[1]
                tot[src][key] += dur
                if dur > FLOOR:
                    over[src][key] += dur
                    over[src][key + "_n"] += 1

    hdr = "%-40s %6s %9s %9s %9s %9s" % ("built by", "graphs", "enter",
                                         "`- >5ms", "exit", "`- >5ms")
    print(hdr)
    print("-" * len(hdr))
    for src, n in cnt.most_common():
        print("%-40s %6d %8.3fs %8.3fs %8.3fs %8.3fs"
              % (src, n, tot[src]["enter"], over[src]["enter"],
                 tot[src]["exit"], over[src]["exit"]))
    te = sum(t["enter"] for t in tot.values())
    tx = sum(t["exit"] for t in tot.values())
    oe = sum(o["enter"] for o in over.values())
    ox = sum(o["exit"] for o in over.values())
    print("%-40s %6d %8.3fs %8.3fs %8.3fs %8.3fs"
          % ("TOTAL", sum(cnt.values()), te, oe, tx, ox))
    print("\ntorch per-capture overhead in this run: %.3fs enter + %.3fs exit "
          "= %.3fs" % (te, tx, te + tx))
    print("of which the 5ms-floored spans in a normal run show only %.3fs "
          "(%.0f%%)" % (oe + ox, 100.0 * (oe + ox) / (te + tx) if te + tx else 0))


if __name__ == "__main__":
    main()
