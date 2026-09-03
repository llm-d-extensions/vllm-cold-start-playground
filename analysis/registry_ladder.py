#!/usr/bin/env python3
"""Summarise a registry-cold TP=1 ladder: what a real first boot costs.

Reads the run directories a ladder produced and prints one row per arm, so the
two numbers that were previously conflated can be read separately:

  first boot       median time to health_200 on a pod with a fresh
                   $VLLM_CACHE_ROOT -- what a genuinely new pod pays
  second boot      that minus the "model registry resolve" phase -- what every
                   boot after the first pays, and the figure the published
                   README ladder was actually reporting

The subtraction is sound because the registry phase is serial and on the
critical path: it happens inside config resolution, before the engine core is
spawned, with nothing else in flight to overlap it. A warm run skips it outright
(the phase measures 0.0002s), so removing it is not a model of a warm run -- it
is what the warm run of the same arm measures.

Cycle 0 of each arm is dropped: the pod's first boot after a probe change also
pays page-cache misses on the probe modules themselves, which is a cost of the
measurement, not of vLLM.

  python3 analysis/registry_ladder.py runs/reg1-*
  python3 analysis/registry_ladder.py --csv out.csv runs/reg1-*
"""
import argparse
import csv
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coldstart_report import (Run, registry_subprocess_s, sweep_phases,  # noqa: E402
                             fmt_s)

REGISTRY_PHASE = "model registry resolve"


def median(xs):
    if not xs:
        return None
    ys = sorted(xs)
    n = len(ys)
    return ys[n // 2] if n % 2 else (ys[n // 2 - 1] + ys[n // 2]) / 2.0


def arm_of(run_id, prefix):
    """`reg1-s3-fork-c2` -> (`s3-fork`, 2)."""
    m = re.match(r"^%s-(.+)-c(\d+)$" % re.escape(prefix), run_id)
    if not m:
        return None, None
    return m.group(1), int(m.group(2))


def load(path):
    # Accept either the run directory or its trace/ subdirectory. A ladder
    # produces runs/<id>/{meta.json,trace/,...} and the events live one level
    # down, so `runs/reg1-*` -- the obvious thing to type, and what the ladder
    # itself suggests -- must work.
    trace = os.path.join(path, "trace")
    run = Run(trace if os.path.isdir(trace) else path)
    t0 = run.t0()
    t_ready, edge = run.t_ready()
    total = t_ready - t0
    per_phase, _unacc = sweep_phases(run, t0, t_ready)[:2]
    reg_phase = per_phase.get(REGISTRY_PHASE, 0.0)
    sub = registry_subprocess_s(run)
    rc = getattr(run, "registry_cache", None) or {}
    return {
        "run_id": os.path.basename(path.rstrip("/")),   # the run dir, not trace/
        "total": total,
        "edge": edge,
        "registry_phase": reg_phase,
        # The subprocess span alone. Slightly smaller than the phase, which also
        # holds the cache hash/read and resolve_model_cls around it.
        "registry_subprocess": sub,
        "registry_cold": sub is not None and sub > 0.1,
        "modelinfos_at_t0": rc.get("n"),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run directories")
    ap.add_argument("--prefix", default=None,
                    help="run-id prefix (default: inferred from the first run)")
    ap.add_argument("--drop-cycle", type=int, default=0,
                    help="cycle to discard as warm-up (default 0; -1 keeps all)")
    ap.add_argument("--csv", help="also write the per-arm table here")
    ap.add_argument("--emit-reports", metavar="DIR",
                    help="copy each arm's median-cycle report.txt into DIR, named "
                         "<prefix>-<arm>.txt, so the README can link the run that "
                         "produced the median rather than a hand-picked one")
    args = ap.parse_args(argv)

    first = os.path.basename(args.runs[0].rstrip("/"))
    prefix = args.prefix or first.split("-")[0]

    rows, skipped = [], []
    for p in args.runs:
        if not os.path.isdir(p):
            continue
        rid = os.path.basename(p.rstrip("/"))
        arm, cycle = arm_of(rid, prefix)
        if arm is None:
            skipped.append("%s (does not match %s-<arm>-c<n>)" % (rid, prefix))
            continue
        if cycle == args.drop_cycle:
            continue
        try:
            r = load(p)
        except Exception as exc:                       # noqa: BLE001
            skipped.append("%s (%s: %s)" % (rid, type(exc).__name__, exc))
            continue
        r["arm"], r["cycle"] = arm, cycle
        rows.append(r)

    if not rows:
        print("no usable runs", file=sys.stderr)
        for s in skipped:
            print("  skipped %s" % s, file=sys.stderr)
        return 1

    # Preserve the order the arms were given in -- the ladder is cumulative, so
    # sorting alphabetically would be sorting by accident of naming.
    arms = []
    for r in rows:
        if r["arm"] not in arms:
            arms.append(r["arm"])
    arms.sort()   # s1..s5 sort correctly and are the intended reading order

    print("=== TP=1 ladder, model registry accounted ===")
    print("runs: %d over %d arm(s); cycle %s discarded as warm-up"
          % (len(rows), len(arms), args.drop_cycle))
    edges = sorted({r["edge"] for r in rows})
    print("readiness edge: %s" % ", ".join(edges))
    ncold = sum(1 for r in rows if r["registry_cold"])
    print("registry-cold: %d/%d measured runs" % (ncold, len(rows)))
    if ncold != len(rows):
        print("!! not every run was registry-cold -- the 'first boot' column is")
        print("   only a first boot for the ones that were. Check --cold-registry.")
    print()

    hdr = ("%-14s %3s   %10s   %10s   %10s   %10s"
           % ("arm", "n", "first boot", "registry", "second", "step"))
    print(hdr)
    print("-" * len(hdr))
    out, prev_first, prev_second = [], None, None
    for arm in arms:
        rs = [r for r in rows if r["arm"] == arm]
        first_boot = median([r["total"] for r in rs])
        reg = median([r["registry_phase"] for r in rs])
        second = median([r["total"] - r["registry_phase"] for r in rs])
        d = "" if prev_first is None else "%+.2fs" % (first_boot - prev_first)
        print("%-14s %3d   %10s   %10s   %10s   %10s"
              % (arm, len(rs), fmt_s(first_boot), fmt_s(reg), fmt_s(second), d))
        out.append({
            "arm": arm, "n": len(rs),
            "first_boot_s": round(first_boot, 3),
            "registry_s": round(reg, 3),
            "second_boot_s": round(second, 3),
            "step_first_boot_s": None if prev_first is None
            else round(first_boot - prev_first, 3),
            "step_second_boot_s": None if prev_second is None
            else round(second - prev_second, 3),
            "spread_s": round(max(r["total"] for r in rs)
                              - min(r["total"] for r in rs), 3),
        })
        prev_first, prev_second = first_boot, second

    print()
    print("first boot = fresh $VLLM_CACHE_ROOT (what a new pod pays)")
    print("second     = first boot minus the registry phase (what the published")
    print("             README ladder measured, and every boot after the first)")
    print("step       = change in first boot vs the arm above")

    if len(out) >= 2:
        a, z = out[0], out[-1]
        print()
        print("ladder total: %s -> %s first boot (%+.2fs, %.0f%%)"
              % (fmt_s(a["first_boot_s"]), fmt_s(z["first_boot_s"]),
                 z["first_boot_s"] - a["first_boot_s"],
                 100.0 * (z["first_boot_s"] - a["first_boot_s"]) / a["first_boot_s"]))
        print("              %s -> %s second boot (%+.2fs, %.0f%%)"
              % (fmt_s(a["second_boot_s"]), fmt_s(z["second_boot_s"]),
                 z["second_boot_s"] - a["second_boot_s"],
                 100.0 * (z["second_boot_s"] - a["second_boot_s"]) / a["second_boot_s"]))
        # The registry subprocess is itself a full `import vllm`, so anything that
        # speeds imports up should shrink it too. Worth stating either way: if it
        # does, the published step-2 win was understated on a real first boot.
        r0, rN = out[0]["registry_s"], out[-1]["registry_s"]
        print("registry phase: %s -> %s across the ladder (%+.2fs)"
              % (fmt_s(r0), fmt_s(rN), rN - r0))

    print()
    print("per-run detail")
    d = ("%-26s %-12s %10s %10s %8s %7s"
         % ("run", "arm", "total", "registry", "cold?", "mi@t0"))
    print(d)
    print("-" * len(d))
    for r in sorted(rows, key=lambda x: (x["arm"], x["cycle"])):
        print("%-26s %-12s %10s %10s %8s %7s"
              % (r["run_id"], r["arm"], fmt_s(r["total"]),
                 fmt_s(r["registry_phase"]),
                 "yes" if r["registry_cold"] else "no",
                 "-" if r["modelinfos_at_t0"] is None else r["modelinfos_at_t0"]))

    if skipped:
        print()
        print("skipped:")
        for s in skipped:
            print("  %s" % s)

    if args.emit_reports:
        import shutil
        os.makedirs(args.emit_reports, exist_ok=True)
        print()
        print("emitted reports")
        for arm in arms:
            rs = sorted((r for r in rows if r["arm"] == arm), key=lambda x: x["total"])
            # The middle run by total, so the linked report is the one behind the
            # median number in the table. With an even count there is no single
            # median run; take the lower middle and say so rather than linking a
            # run whose total is not the figure printed.
            pick = rs[(len(rs) - 1) // 2]
            src = os.path.join([p for p in args.runs
                                if os.path.basename(p.rstrip("/")) == pick["run_id"]][0],
                               "report.txt")
            if not os.path.isfile(src):
                print("  !! %s has no report.txt" % pick["run_id"])
                continue
            dst = os.path.join(args.emit_reports, "%s-%s.txt" % (prefix, arm))
            shutil.copyfile(src, dst)
            note = "" if len(rs) % 2 else "  (lower middle of %d)" % len(rs)
            print("  %-24s <- %s (%s)%s"
                  % (dst, pick["run_id"], fmt_s(pick["total"]), note))

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
            w.writeheader()
            w.writerows(out)
        print("\nwrote %s" % args.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
