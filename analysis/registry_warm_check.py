#!/usr/bin/env python3
"""Does (cold total - registry phase) actually equal the warm total?

analysis/registry_ladder.py prints a "second boot" column computed by
subtracting the `model registry resolve` phase from a registry-cold total, on the
argument that the phase is serial, on the critical path, and simply absent on a
warm boot. This checks that argument against paired boots of the same arm
(scripts/tp1-registry-warm-check.sh): cold and warm, back to back, same cycle.

Reads as agreement when |predicted - measured| is inside the arm's own spread.
If it is not, the subtraction is hiding something -- overlap with a neighbouring
phase, or a cost the cold run pays outside the phase -- and the ladder's second
boot column should not be trusted.

  python3 analysis/registry_warm_check.py runs/regw-*
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coldstart_report import Run, sweep_phases, fmt_s  # noqa: E402

REGISTRY_PHASE = "model registry resolve"


def median(xs):
    ys = sorted(xs)
    n = len(ys)
    return None if not ys else (ys[n // 2] if n % 2 else (ys[n // 2 - 1] + ys[n // 2]) / 2.0)


def load(path):
    trace = os.path.join(path, "trace")
    run = Run(trace if os.path.isdir(trace) else path)
    t0 = run.t0()
    t_ready, _edge = run.t_ready()
    per_phase = sweep_phases(run, t0, t_ready)[0]
    return t_ready - t0, per_phase.get(REGISTRY_PHASE, 0.0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--prefix", default=None)
    args = ap.parse_args(argv)

    prefix = args.prefix or os.path.basename(args.runs[0].rstrip("/")).split("-")[0]
    pat = re.compile(r"^%s-(.+)-(cold|warm)-c(\d+)$" % re.escape(prefix))

    data = {}      # arm -> state -> [(total, registry)]
    for p in args.runs:
        if not os.path.isdir(p):
            continue
        m = pat.match(os.path.basename(p.rstrip("/")))
        if not m:
            continue
        arm, state = m.group(1), m.group(2)
        try:
            data.setdefault(arm, {}).setdefault(state, []).append(load(p))
        except Exception as exc:                            # noqa: BLE001
            print("  skipped %s (%s)" % (p, exc), file=sys.stderr)

    if not data:
        print("no paired runs found", file=sys.stderr)
        return 1

    print("=== is the ladder's 'second boot' subtraction sound? ===")
    print()
    hdr = ("%-14s %3s   %10s %10s   %10s   %10s   %9s"
           % ("arm", "n", "cold", "registry", "predicted", "warm", "error"))
    print(hdr)
    print("-" * len(hdr))
    verdicts = []
    for arm in sorted(data):
        cold = data[arm].get("cold", [])
        warm = data[arm].get("warm", [])
        if not cold or not warm:
            print("%-14s   -- missing %s arm" % (arm, "warm" if cold else "cold"))
            continue
        c_tot = median([t for t, _ in cold])
        c_reg = median([r for _, r in cold])
        w_tot = median([t for t, _ in warm])
        pred = c_tot - c_reg
        err = pred - w_tot
        spread = max(t for t, _ in warm) - min(t for t, _ in warm)
        ok = abs(err) <= max(spread, 1.0)
        verdicts.append((arm, ok, err, spread))
        print("%-14s %3d   %10s %10s   %10s   %10s   %+8.2fs"
              % (arm, min(len(cold), len(warm)), fmt_s(c_tot), fmt_s(c_reg),
                 fmt_s(pred), fmt_s(w_tot), err))
        # A warm boot that still shows registry time means --cold-registry leaked,
        # or the cache key changed between the pair -- either way the pair is void.
        w_reg = median([r for _, r in warm])
        if w_reg and w_reg > 0.5:
            print("   !! the warm run still spent %s in the registry phase; it was"
                  % fmt_s(w_reg))
            print("      not warm, so this pair proves nothing.")

    print()
    print("predicted = cold total - cold registry phase (what the ladder reports)")
    print("warm      = the same arm measured with the cache already populated")
    print("error     = predicted - warm; agreement means inside the warm spread")
    print()
    for arm, ok, err, spread in verdicts:
        print("%-14s %s  (error %+.2fs vs warm spread %.2fs)"
              % (arm, "AGREES" if ok else "DISAGREES", err, spread))
    if verdicts and all(ok for _, ok, _, _ in verdicts):
        print()
        print("VERDICT: the subtraction holds; the ladder's second-boot column is")
        print("         a measurement of the warm case, not just an inference.")
    elif verdicts:
        print()
        print("VERDICT: at least one arm disagrees. The registry phase is not the")
        print("         only difference between a cold and a warm boot -- find out")
        print("         what else moved before quoting the second-boot column.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
