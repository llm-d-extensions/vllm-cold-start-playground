#!/usr/bin/env python3
"""Turn a cold-start trace directory into an answer to "where did the time go?"

Reads the JSONL event files written by the in-process probe and the readiness
poller, then produces:

  * the headline number: vLLM process exec -> API server ready
  * a non-overlapping phase breakdown that sums to that number, plus the
    unaccounted remainder (the honest measure of instrumentation blind spots)
  * a per-process span tree (the readable "what happened, in order" view)
  * subsystem detail: imports, network, filesystem/weights, GPU, compilation
  * findings: cgroup throttling, cold caches, network on the critical path,
    serialisation between processes
  * a Perfetto / chrome://tracing file, a CSV row for regression tracking, and
    a JSON summary

Usage:
    coldstart_report.py RUNDIR [-o OUTDIR] [--min-ms 50]
    coldstart_report.py --compare RUNDIR_A RUNDIR_B [...]

Stdlib only: it runs inside the vLLM image as happily as on a laptop.

Phase attribution
-----------------
Spans overlap: the API server is inside ``api.run_server`` while a worker is
inside ``weights.load_model``, and a lazy ``import`` can happen inside either.
Summing span durations therefore double counts, and taking a plain union hides
which phase owned the time. So the timeline is swept: every elementary interval
between two event boundaries is credited to exactly one phase -- the innermost
span active in that interval, ties broken by phase specificity. The result is
non-overlapping by construction and reconciles against wall clock, so
``sum(phases) + unaccounted == total``.
"""

import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict

# ---------------------------------------------------------------------------
# phase model: (phase label, priority) -- higher priority wins ties when two
# equally deep spans are active. Leaf-ish work outranks container spans.
# ---------------------------------------------------------------------------
PHASES = [
    ("interpreter boot", 10),
    ("python imports", 60),
    ("config & tokenizer resolve", 50),
    ("network: hub metadata", 70),
    ("network: weight download", 75),
    ("ipc handshake", 40),
    ("device & collectives init", 55),
    ("weight load", 65),
    ("torch.compile", 80),
    ("cudagraph capture", 80),
    ("kv cache alloc", 58),
    ("warmup / profile run", 57),
    ("api server startup", 30),
    ("engine orchestration", 20),
    ("other", 15),
]
PHASE_ORDER = [p for p, _ in PHASES]
PRIORITY = dict(PHASES)

# canonical phase ordering for reports (roughly chronological)
REPORT_ORDER = [
    "interpreter boot", "python imports", "config & tokenizer resolve",
    "network: hub metadata", "network: weight download", "ipc handshake",
    "device & collectives init", "weight load", "kv cache alloc",
    "torch.compile", "cudagraph capture", "warmup / profile run",
    "api server startup", "engine orchestration", "other",
]

CAT_PHASE = {
    "import": "python imports",
    "config": "config & tokenizer resolve",
    "platform": "config & tokenizer resolve",
    "ipc": "ipc handshake",
    "device": "device & collectives init",
    "collective": "device & collectives init",
    "gpu": "device & collectives init",
    "weights": "weight load",
    "compile": "torch.compile",
    "cudagraph": "cudagraph capture",
    "kvcache": "kv cache alloc",
    "warmup": "warmup / profile run",
    "apiserver": "api server startup",
    "entrypoint": "engine orchestration",
    "engine": "engine orchestration",
    "executor": "engine orchestration",
    "telemetry": "network: hub metadata",
    "fileio": "weight load",
    "misc": "other",
    "probe": "other",
}

DOWNLOAD_HINTS = ("download", "http_get", "xet_get", "snapshot")


def classify(ev):
    name = ev.get("name", "")
    cat = ev.get("cat", "misc")
    if name == "interpreter.startup":
        return "interpreter boot"
    if cat == "network":
        if any(h in name for h in DOWNLOAD_HINTS):
            return "network: weight download"
        return "network: hub metadata"
    return CAT_PHASE.get(cat, "other")


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

class Run(object):
    def __init__(self, path):
        self.path = path
        norm = os.path.normpath(path)
        self.name = os.path.basename(norm)
        if self.name in ("trace", "traces", "."):
            # runs are laid out as <run-id>/trace/, so the leaf says nothing
            self.name = os.path.basename(os.path.dirname(norm)) or self.name
        self.spans = []        # X events
        self.instants = []     # i events
        self.counters = []     # C events
        self.metas = []        # M events
        self.procs = {}        # pid -> info
        self.bad_lines = 0
        self._load()
        self._index()
        rid = ((self.probe_installed or {}).get("run_id")
               or (self.context or {}).get("run_id"))
        if rid and rid != "run":
            self.name = rid

    def _load(self):
        files = sorted(glob.glob(os.path.join(self.path, "events.*.jsonl")))
        if not files:
            raise SystemExit("no events.*.jsonl found in %s" % self.path)
        self.files = files
        for f in files:
            with open(f, "r", errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ev = json.loads(line)
                    except Exception:
                        self.bad_lines += 1
                        continue
                    ph = ev.get("ph")
                    if ph == "X" and isinstance(ev.get("ts"), (int, float)):
                        self.spans.append(ev)
                    elif ph == "i":
                        self.instants.append(ev)
                    elif ph == "C":
                        self.counters.append(ev)
                    elif ph == "M":
                        self.metas.append(ev)
        self.spans.sort(key=lambda e: e["ts"])
        self.instants.sort(key=lambda e: e.get("ts") or 0)
        self.counters.sort(key=lambda e: e.get("ts") or 0)

    def _index(self):
        for m in self.metas:
            pid = m.get("pid")
            if m["name"] in ("process.exec", "process.fork"):
                a = m.get("args", {})
                self.procs[pid] = {
                    "pid": pid, "role": m.get("role"), "kind": m["name"],
                    "exec_ts": m.get("ts"), "cmdline": a.get("cmdline", ""),
                    "ppid": a.get("ppid"), "python": a.get("python"),
                    "title": None, "summary": None, "imports": None,
                }
            elif m["name"] == "process.title":
                self.procs.setdefault(pid, {"pid": pid}).setdefault("title", None)
                self.procs[pid]["title"] = m.get("args", {}).get("title")
            elif m["name"] == "process.summary":
                self.procs.setdefault(pid, {"pid": pid})["summary"] = m.get("args")
            elif m["name"] == "imports.summary":
                self.procs.setdefault(pid, {"pid": pid})["imports"] = m.get("args")
        self.context = self._meta("run.context")
        self.caches = self._meta("run.caches")
        self.config = self._meta("vllm.config")
        self.vllm_info = self._meta("vllm.info")
        self.torch_info = self._meta("torch.info")
        self.coverage = self._coverage()
        self.gpu_inventory = self._meta("gpu.inventory")
        self.ready_summary = self._meta("ready.summary")
        self.probe_installed = self._meta("probe.installed")

    def _coverage(self):
        """Probe coverage for the whole run, rebuilt from per-target events.

        ``probe.coverage`` is written from ``Tracer.at_finish``, which needs the
        process to die in a way the probe observes. EngineCore does not, so the
        only ``probe.coverage`` in a run came from the API server -- and the API
        server never imports the worker modules, so its ``never_imported`` list
        says nothing about the process that does the work. That is how a renamed
        model-runner module hid 10.9s of the warmup phase.

        So prefer the incremental events (``probe.patch_declared`` at install,
        one ``probe.patch`` per resolution), which survive any exit path, and
        fall back to the ``probe.coverage`` meta for traces recorded before
        those existed. A target counts as applied if it applied in *any*
        process: several are legitimately unreachable in the API server.
        """
        declared = set()
        for m in self.metas:
            if m["name"] == "probe.patch_declared":
                declared.update((m.get("args") or {}).get("targets") or [])
        status = {}
        for ev in self.instants:
            if ev.get("name") != "probe.patch":
                continue
            args = ev.get("args") or {}
            tgt = args.get("target")
            if tgt is None:
                continue
            status.setdefault(tgt, set()).add(args.get("status"))
        if not declared and not status:
            return self._legacy_coverage()
        applied = sorted(t for t, s in status.items() if "ok" in s)
        not_applied = sorted(t for t, s in status.items() if "ok" not in s)
        never = sorted(declared - set(status))
        return {"applied": len(applied), "not_applied": not_applied,
                "never_imported": never, "reported_by": None}

    def _legacy_coverage(self):
        """Coverage for traces recorded before the per-target events existed.

        Only a process that died in a way the probe observed wrote a
        ``probe.coverage`` summary, so these lists cover some subset of the
        run's interpreters. The two operators differ: a patch that failed to
        apply anywhere is a defect wherever it was seen, so ``not_applied`` is
        a union; but "never imported" is only true of a target no process
        imported, so that one is an *intersection* -- the helper subprocesses
        import a few hundred modules and would otherwise drown the list.

        Even the intersection overstates the blind spot, because the process
        that does the work is usually the one that did not report: this run's
        own ``fastsafetensors.parallel_loader`` patch demonstrably applied in
        EngineCore and still shows up below. ``reported_by`` is what the
        renderer uses to caveat that instead of presenting it as the truth.
        """
        legacy = [m for m in self.metas if m["name"] == "probe.coverage"]
        if not legacy:
            return None
        not_applied, never, applied = set(), None, 0
        for m in legacy:
            a = m.get("args") or {}
            not_applied.update(a.get("not_applied") or [])
            n = set(a.get("never_imported") or [])
            never = n if never is None else (never & n)
            applied = max(applied, a.get("applied") or 0)
        return {"applied": applied, "not_applied": sorted(not_applied),
                "never_imported": sorted(never or ()),
                "reported_by": len(legacy)}

    def _meta(self, name):
        for m in self.metas:
            if m["name"] == name:
                return m.get("args") or {}
        return None

    # -- timeline anchors -------------------------------------------------
    def t0(self):
        """vLLM process exec: the earliest exec'd (not forked) process."""
        cands = [p["exec_ts"] for p in self.procs.values()
                 if p.get("kind") == "process.exec" and p.get("exec_ts")]
        if not cands:
            cands = [s["ts"] for s in self.spans] or [0]
        return min(cands)

    def ready_edges(self):
        """name -> wall ts for each readiness edge we observed."""
        out = {}
        for ev in self.instants:
            n = ev.get("name", "")
            if n.startswith("ready.") and n != "ready.transition":
                out.setdefault(n[len("ready."):], ev["ts"])
            elif n == "api.startup_complete":
                out.setdefault("api_startup_complete", ev["ts"])
        return out

    def t_ready(self):
        """The moment we call it ready, and which signal we used."""
        edges = self.ready_edges()
        for key in ("health_200", "models_200", "api_startup_complete",
                    "first_token"):
            if key in edges:
                return edges[key], key
        if self.spans:
            last = max(s["ts"] + s.get("dur", 0) for s in self.spans)
            return last, "last event (no readiness signal found)"
        return self.t0(), "unknown"

    def label(self, pid):
        p = self.procs.get(pid, {})
        t = p.get("title")
        if t:
            return "%s (pid %s)" % (t, pid)
        role = p.get("role") or "?"
        return "%s (pid %s)" % (role, pid)


# ---------------------------------------------------------------------------
# interval maths
# ---------------------------------------------------------------------------

def union_len(intervals):
    if not intervals:
        return 0.0
    ivs = sorted(intervals)
    total = 0.0
    cs, ce = ivs[0]
    for s, e in ivs[1:]:
        if s > ce:
            total += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    return total + (ce - cs)


# ---------------------------------------------------------------------------
# spans that mean "this process is blocked on another process"
#
# These are not work. A parent sitting in wait_for_engine_startup while the
# EngineCore boots is idle, and crediting that wall to "ipc handshake" was the
# single largest lie this report told: on runs/fs-early-r2 it reported
#   ipc handshake  12.0s  33.9%
# and no weight load, no kv cache alloc and no warmup phase at all, while the
# child spent 2.87s / 7.38s / 6.12s in exactly those. The real IPC cost on that
# run is 17ms (CoreEngineProcManager.__init__); the 12.0s was the child's boot,
# wearing the parent's label.
#
# Names, not categories, because "ipc" legitimately covers both the waits and
# the genuine socket/shm setup, and we want to keep the latter. A trace may also
# mark a span itself via args={"blocking": true}, which is honoured below; the
# list is what makes the fix apply to traces already on disk.
BLOCKING_SPANS = frozenset((
    # v1/engine/utils.py: the parent polling the child's handshake socket, and
    # the contextmanager whose body is that poll.
    "engine.wait_for_startup",
    "engine.launch_cores",
    # v1/engine/core_client.py: MPClient.__init__ spends its whole span inside
    # launch_core_engines. Real client setup is the part with nothing else
    # running, and that still gets credited -- see the fallback below.
    "engine.mp_client_init",
    "engine.async_mp_client_init",
    # v1/engine/utils.py CoreEngineProcManager.__init__ -> Process.start(), and
    # under CS_FORKSERVER the block-on-preload inside popen_forkserver._launch.
    # In the forkserver-late arm that wait is 13.7s and the forkserver process
    # is importing throughout: those imports are the truthful owner of it.
    "engine.core_proc_manager",
    "forkserver.launch_child",
    # shm_broadcast MessageQueue.wait_until_ready: waiting for the peer to
    # attach. Appears only at TP>1.
    "ipc.message_queue_wait",
))


# Spans that are real work but say nothing about *which* phase that work belongs
# to. They nest inside whatever called them, they are deeper than their caller,
# and the sweep picks the deepest span -- so left alone they steal the interval
# and bill it to their own category. cuda.synchronize is the case that mattered:
# it fires inside CUDA graph capture (once per captured graph) and carried ~1.6s
# of capture cost into "device & collectives init", which is why that row's top
# sink was cuda.synchronize in every report.
#
# Demoted, not dropped: they still win an interval where they are the only
# active span (a bare synchronize between phases is device time and nothing
# else's), and their totals are unchanged in the SUBSYSTEMS section.
TRANSPARENT_SPANS = frozenset((
    "cuda.synchronize",
    "cuda.empty_cache",
    "cuda.mem_get_info",
    "cuda.reset_peak_memory_stats",
    "cuda.memory_stats",
    "gc.collect",
    "gc.freeze",
))


def is_transparent(ev):
    return ev.get("name", "") in TRANSPARENT_SPANS


def is_blocking(ev):
    args = ev.get("args")
    if isinstance(args, dict) and "blocking" in args:
        return bool(args["blocking"])
    return ev.get("name", "") in BLOCKING_SPANS


def sweep_phases(run, t0, t1):
    """Credit every elementary interval in [t0, t1] to exactly one phase.

    Within a process the winner is the innermost (deepest) active span, with
    phase specificity as a tie-break: the deepest span is the actual leaf work
    and the ones above it are containers.

    Across processes, depth is meaningless -- pid 4443's depth 3 is not "outside"
    pid 4357's depth 9 -- so the cross-process winner is chosen by phase
    specificity, with depth kept only as a deterministic stabiliser. Comparing
    depth first is what let a parent's depth-9 wait outrank a child's depth-3
    weight load.

    A process whose innermost active span is a BLOCKING_SPAN is *idle*: it is
    blocked on another process, so it is set aside entirely for that interval
    rather than merely demoted, because its enclosing container spans are not
    work either. Blocked processes still win intervals where nobody else is
    active -- that is genuine serialised handshake cost and it stays visible.

    Intervals where nothing is active are unaccounted. Non-overlapping by
    construction, so ``sum(per_phase) + unaccounted == t1 - t0``.
    """
    by_pid = defaultdict(list)
    for s in run.spans:
        ts = s["ts"]
        dur = s.get("dur") or 0.0
        end = ts + dur
        if end <= t0 or ts >= t1:
            continue
        by_pid[s.get("pid")].append((
            max(ts, t0), min(end, t1), s.get("depth") or 0, classify(s),
            s.get("name", ""), is_blocking(s), is_transparent(s)))

    bounds = {t0, t1}
    for spans in by_pid.values():
        for s, e, _, _, _, _, _ in spans:
            bounds.add(s)
            bounds.add(e)
    bounds = sorted(b for b in bounds if t0 <= b <= t1)

    # per-process sweep state
    state = {}
    for pid, spans in by_pid.items():
        state[pid] = {"spans": sorted(spans, key=lambda x: x[0]), "i": 0,
                      "active": []}

    per_phase = defaultdict(float)
    per_phase_proc = defaultdict(float)
    unaccounted = 0.0
    # wall where two or more processes were doing real work at once. Any
    # single-winner attribution is a choice over these intervals, not a
    # measurement, so the report says how much of the total they cover.
    concurrent = 0.0
    detail = defaultdict(float)     # (phase, span name) -> seconds credited

    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        width = b - a
        if width <= 0:
            continue
        working, blocked = None, None
        n_working = 0
        for pid, st in state.items():
            spans = st["spans"]
            while st["i"] < len(spans) and spans[st["i"]][0] <= a:
                st["active"].append(spans[st["i"]])
                st["i"] += 1
            st["active"] = [s for s in st["active"] if s[1] > a]
            if not st["active"]:
                continue
            # innermost active span: the best statement of what this process is
            # doing right now. Transparent spans (cuda.synchronize and friends)
            # are considered only when nothing else is active -- see
            # TRANSPARENT_SPANS.
            pool = [s for s in st["active"] if not s[6]] or st["active"]
            cand = max(pool, key=lambda s: (s[2], PRIORITY.get(s[3], 0)))
            # priority first, depth only to break ties -- see the docstring.
            key = (PRIORITY.get(cand[3], 0), cand[2], cand[3], cand[4], pid)
            if cand[5]:
                if blocked is None or key[:2] > blocked[:2]:
                    blocked = key
            else:
                n_working += 1
                if working is None or key[:2] > working[:2]:
                    working = key
        best = working if working is not None else blocked
        if n_working > 1:
            concurrent += width
        if best is None:
            unaccounted += width
        else:
            per_phase[best[2]] += width
            detail[(best[2], best[3])] += width
            per_phase_proc[(best[2], best[4])] += width
    return per_phase, unaccounted, detail, per_phase_proc, concurrent


# ---------------------------------------------------------------------------
# formatting helpers
# ---------------------------------------------------------------------------

def fmt_s(x):
    if x is None:
        return "-"
    if x >= 100:
        return "%.0fs" % x
    if x >= 10:
        return "%.1fs" % x
    if x >= 1:
        return "%.2fs" % x
    return "%.0fms" % (x * 1000)


def fmt_bytes(n):
    if not n:
        return "-"
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return "%.1f %s" % (n, unit)
        n /= 1024.0


def na(v):
    return "n/a" if v in (None, "") else v


def fmt_rate(mbps):
    """Never round a real rate down to a bare 0 MB/s."""
    if mbps <= 0:
        return "0 MB/s"
    if mbps < 1:
        return "%.2f MB/s" % mbps
    return "%.0f MB/s" % mbps


def bar(frac, width=28):
    n = int(round(frac * width))
    return "#" * n + "." * (width - n)


def table(rows, headers, out):
    widths = [len(h) for h in headers]
    for r in rows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(str(c)))
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    out.append(line)
    out.append("  ".join("-" * widths[i] for i in range(len(headers))))
    for r in rows:
        out.append("  ".join(str(c).ljust(widths[i]) if i else
                             str(c).ljust(widths[i])
                             for i, c in enumerate(r)))
    return out


# ---------------------------------------------------------------------------
# subsystem summaries
# ---------------------------------------------------------------------------

def counter_series(run, key, cat=None):
    """[(ts, value)] for a counter key, merged across processes."""
    out = defaultdict(list)
    for c in run.counters:
        args = c.get("args") or {}
        if key in args and isinstance(args[key], (int, float)):
            out[c.get("pid")].append((c["ts"], args[key]))
    return out


def counter_delta(run, key, t0=None, t1=None):
    """Sum over processes of (max-min) of a cumulative counter in a window."""
    total = 0.0
    for pid, series in counter_series(run, key).items():
        vals = [v for ts, v in series
                if (t0 is None or ts >= t0) and (t1 is None or ts <= t1)]
        if len(vals) >= 2:
            total += max(vals) - min(vals)
    return total


def spans_by_name(run, prefix=None, cat=None):
    out = []
    for s in run.spans:
        if cat and s.get("cat") != cat:
            continue
        if prefix and not s.get("name", "").startswith(prefix):
            continue
        out.append(s)
    return out


def aggregate(run, cat):
    """count / sum / max of spans in a category, plus their union wall time."""
    sel = [s for s in run.spans if s.get("cat") == cat]
    if not sel:
        return None
    return {
        "count": len(sel),
        "sum_s": sum(s.get("dur", 0) for s in sel),
        "max_s": max(s.get("dur", 0) for s in sel),
        "union_s": union_len([(s["ts"], s["ts"] + s.get("dur", 0))
                              for s in sel]),
    }


def top_spans(run, cat, n=10, key=None):
    sel = [s for s in run.spans if s.get("cat") == cat]
    sel.sort(key=lambda s: -(s.get("dur") or 0))
    return sel[:n]


def import_totals(run):
    """Per-process import self-time, from each process's imports.summary."""
    rows = []
    for pid, p in sorted(run.procs.items(), key=lambda kv: kv[0] or 0):
        imp = p.get("imports")
        if not imp:
            continue
        rows.append((pid, run.label(pid), imp.get("total_self_s"),
                     imp.get("modules"), imp.get("by_top_package") or {}))
    return rows


def proc_counts(run, key_prefix):
    """Merge the per-process ``counts`` aggregates from process.summary."""
    out = defaultdict(lambda: [0, 0.0])
    for p in run.procs.values():
        summ = p.get("summary") or {}
        for k, v in (summ.get("counts") or {}).items():
            if k.startswith(key_prefix):
                out[k][0] += v.get("n", 0)
                out[k][1] += v.get("total_s", 0.0)
    return out


def weight_window(run):
    """Wall window during which any process was loading weights."""
    sel = [s for s in run.spans
           if s.get("cat") == "weights" or s.get("name", "").startswith(
               "weights.")]
    if not sel:
        return None
    start = min(s["ts"] for s in sel)
    end = max(s["ts"] + s.get("dur", 0) for s in sel)
    return start, end


def warmup_breakdown(run):
    """The interior of ``compile_or_warm_up_model``, which is 26% of a tuned
    startup and used to render as one opaque span.

    Two things need to be readable without opening a trace. First, the steps:
    kernel_warmup's chain of gated sub-warmups, graph capture, then
    warmup_kernels and inductor's lazy init. Second, the shape of graph capture,
    which is the bulk of it: one eager warmup forward plus one captured forward
    per (mode, batch size), so the cost is a multiplier over
    ``cudagraph_capture_sizes`` and it splits by mode.
    """
    L = []
    top = [s for s in run.spans if s.get("name") == "warmup.compile_or_warm_up"]
    steps = [s for s in run.spans
             if s.get("cat") in ("warmup", "cudagraph")
             and (s.get("dur") or 0) > 0
             and s.get("name") not in ("warmup.compile_or_warm_up",)]
    if not top and not steps:
        return L

    if top:
        s = top[0]
        L.append("compile_or_warm_up_model: %s total" % fmt_s(s.get("dur") or 0))

    # steps directly under the top span, largest first
    if top:
        a = top[0]["ts"]
        b = a + (top[0].get("dur") or 0)
        pid = top[0].get("pid")
        inner = [s for s in steps if s.get("pid") == pid
                 and s["ts"] >= a and s["ts"] + (s.get("dur") or 0) <= b]
    else:
        inner = steps
    # roll up by span name: the per-descriptor capture spans repeat ~200 times
    roll = {}
    for s in inner:
        k = s.get("name")
        e = roll.setdefault(k, {"n": 0, "sum": 0.0})
        e["n"] += 1
        e["sum"] += s.get("dur") or 0.0
    rows = []
    for name, e in sorted(roll.items(), key=lambda kv: -kv[1]["sum"]):
        if e["sum"] < 0.001:
            continue
        rows.append([name, str(e["n"]), fmt_s(e["sum"])])
    if rows:
        L.append("")
        table(rows, ["step", "n", "wall"], L)

    # capture shape, by mode
    fwd = defaultdict(lambda: {"n": 0, "sum": 0.0})
    for s in run.spans:
        n = s.get("name") or ""
        if n not in ("cudagraph.warmup_forward", "cudagraph.capture_forward",
                     "cudagraph.prepare_inputs"):
            continue
        mode = (s.get("args") or {}).get("mode") or "?"
        key = (mode, n.split(".")[-1])
        fwd[key]["n"] += 1
        fwd[key]["sum"] += s.get("dur") or 0.0
    if fwd:
        modes = sorted({m for m, _ in fwd})
        rows = []
        for m in modes:
            row = [m]
            for kind in ("prepare_inputs", "warmup_forward", "capture_forward"):
                e = fwd.get((m, kind))
                row.append("%d / %s" % (e["n"], fmt_s(e["sum"])) if e else "-")
            rows.append(row)
        L.append("")
        table(rows, ["capture mode", "prep n / wall", "eager warmup n / wall",
                     "captured n / wall"], L)
        L.append("    one eager warmup forward per captured graph is "
                 "unconditional on the v2 model runner "
                 "(gpu/cudagraph_utils.py) -- cudagraph_num_of_warmups is read "
                 "only by the v1 runner, so it cannot turn this off.")
    return L


def dark_spans(run, total, min_dur=1.0, min_share=0.02, max_cov=0.5):
    """Long spans whose interior is mostly not covered by child spans.

    "Child" is by containment within the same process, not by parent id, so a
    span instrumented at any depth below counts. Returns
    ``(name, dur, covered_fraction, pid)`` for the worst offenders, largest dark
    time first.
    """
    if total <= 0:
        return []
    by_pid = defaultdict(list)
    for s in run.spans:
        by_pid[s.get("pid")].append(s)
    out = []
    for pid, spans in by_pid.items():
        spans.sort(key=lambda s: s["ts"])
        for s in spans:
            dur = s.get("dur") or 0.0
            if dur < min_dur or dur < min_share * total:
                continue
            # A blocking span is empty by definition -- it is one process
            # waiting on another. Reporting it as a blind spot would bury the
            # spans that are empty because nobody instrumented them.
            if is_blocking(s):
                continue
            a, b = s["ts"], s["ts"] + dur
            kids = [(x["ts"], x["ts"] + (x.get("dur") or 0.0)) for x in spans
                    if x is not s and (x.get("dur") or 0.0) > 0
                    and x["ts"] >= a and x["ts"] + (x.get("dur") or 0.0) <= b]
            covered, end = 0.0, a
            for ks, ke in sorted(kids):
                if ke <= end:
                    continue
                covered += ke - max(ks, end)
                end = ke
            frac = covered / dur if dur else 1.0
            if frac < max_cov:
                out.append((s.get("name", "?"), dur, frac, pid))
    # A dark parent and its equally dark child are the same blind spot reported
    # twice; keep the outermost by dropping any span contained in a reported one.
    out.sort(key=lambda x: -x[1])
    kept = []
    for item in out:
        if not any(item[1] <= k[1] and item[0] != k[0] and _contains(
                run, k, item) for k in kept):
            kept.append(item)
    kept.sort(key=lambda x: -(x[1] * (1.0 - x[2])))
    return kept[:3]


def _contains(run, outer, inner):
    """Is `inner`'s span nested inside `outer`'s, in the same process?"""
    if outer[3] != inner[3]:
        return False
    o = next((s for s in run.spans if s.get("name") == outer[0]
              and s.get("pid") == outer[3]), None)
    i = next((s for s in run.spans if s.get("name") == inner[0]
              and s.get("pid") == inner[3]), None)
    if o is None or i is None:
        return False
    return (i["ts"] >= o["ts"]
            and i["ts"] + (i.get("dur") or 0) <= o["ts"] + (o.get("dur") or 0))


def findings(run, t0, t_ready, per_phase, unaccounted):
    """Actionable observations, ordered by how much time they explain."""
    out = []
    total = t_ready - t0
    ctx = run.context or {}
    cg = ctx.get("cgroup") or {}
    cpu = ctx.get("cpu") or {}
    caches = (run.caches or {}).get("caches") or {}
    edges = run.ready_edges()

    # cgroup CPU throttling during startup
    thr = counter_delta(run, "cg_throttled_usec", t0, t_ready) / 1e6
    thr += counter_delta(run, "cg_throttled_time", t0, t_ready) / 1e9
    nthr = counter_delta(run, "cg_nr_throttled", t0, t_ready)
    if thr > 0.5 or nthr > 0:
        out.append((thr, "CPU THROTTLING: the container was throttled for "
                    "%s across %d periods during startup (cpu limit %s cores). "
                    "Raise the CPU limit/request: imports, safetensors "
                    "conversion and Inductor compilation are all CPU-bound."
                    % (fmt_s(thr), int(nthr),
                       cg.get("cpu_limit_cores") or "unset")))

    # imports paid per process
    imp_rows = import_totals(run)
    if len(imp_rows) > 1:
        tot = sum(r[2] or 0 for r in imp_rows)
        biggest = max(imp_rows, key=lambda r: r[2] or 0)
        out.append((tot, "IMPORTS x%d PROCESSES: %s of module execution summed "
                    "across %d Python processes (worst single process %s in %s). "
                    "Only the wall-clock portion on the critical path counts, "
                    "but this is the cost that a pre-warmed or forked engine "
                    "process would avoid."
                    % (len(imp_rows), fmt_s(tot), len(imp_rows),
                       fmt_s(biggest[2]), biggest[1])))

    # network on the critical path
    net = per_phase.get("network: hub metadata", 0) + \
        per_phase.get("network: weight download", 0)
    if net > 0.5:
        dl = per_phase.get("network: weight download", 0)
        out.append((net, "NETWORK ON CRITICAL PATH: %s attributed to HTTP "
                    "(%s of it weight download). If the model is already local, "
                    "set HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 to remove "
                    "the metadata round trips; also check "
                    "VLLM_NO_USAGE_STATS=1 to drop the telemetry POST."
                    % (fmt_s(net), fmt_s(dl))))

    # compile cache state
    comp = per_phase.get("torch.compile", 0)
    ind = caches.get("inductor_cache") or {}
    vc = caches.get("vllm_cache") or {}
    if comp > 2:
        cold = (ind.get("files", 0) == 0) and (vc.get("files", 0) == 0)
        out.append((comp, "COMPILATION: %s in torch.compile/Inductor%s. "
                    "Persisting VLLM_CACHE_ROOT + TORCHINDUCTOR_CACHE_DIR on a "
                    "shared volume, keyed by (model, dtype, TP, GPU, vLLM "
                    "version), is the highest-leverage fix here; -O0 or "
                    "--enforce-eager trades steady-state throughput for start "
                    "time." % (fmt_s(comp),
                               " with an empty compile cache at t0" if cold
                               else "")))

    # weight read throughput
    ww = weight_window(run)
    if ww:
        read = counter_delta(run, "read_bytes", ww[0], ww[1])
        dur = ww[1] - ww[0]
        io_summary = None
        for m in run.metas:
            if m["name"] == "io.summary":
                io_summary = m.get("args")
        wbytes = (io_summary or {}).get("weight_bytes") or 0
        fs = ((ctx.get("mounts_of_interest") or {}).get("model")
              or (ctx.get("mounts_of_interest") or {}).get("hf_hub") or {})
        nfiles = (io_summary or {}).get("weight_files", "?")
        fstype = (fs or {}).get("fstype", "unknown fs")
        if dur > 0 and read:
            out.append((dur, "WEIGHT LOAD: %s read from disk over %s => %.0f "
                        "MB/s (%s weight files, %s of weights on %s). Compare "
                        "against the device's sequential read ceiling; a low "
                        "number with idle CPU points at the storage tier, a low "
                        "number with one core pegged points at format "
                        "conversion."
                        % (fmt_bytes(read), fmt_s(dur), (read / dur) / 1e6,
                           nfiles, fmt_bytes(wbytes), fstype)))
        elif dur > 0 and wbytes:
            out.append((dur, "WEIGHT LOAD FROM PAGE CACHE: %s of weights (%s "
                        "files on %s) were opened in %s but the process billed "
                        "no disk reads, so the bytes came from the host page "
                        "cache -- this is a warm re-run and %s is a floor, not "
                        "the cold-node number. Drop caches or use a fresh node "
                        "to measure the real weight-load cost."
                        % (fmt_bytes(wbytes), nfiles, fstype, fmt_s(dur),
                           fmt_s(dur))))

    # metadata syscall storms (NFS/CSI)
    fsc = proc_counts(run, "fs.")
    stat_n = sum(v[0] for v in fsc.values())
    stat_s = sum(v[1] for v in fsc.values())
    if stat_s > 1.0:
        out.append((stat_s, "FILESYSTEM METADATA: %d stat/exists/listdir calls "
                    "costing %s. That is round-trip bound -- expected on NFS or "
                    "a CSI volume, near-free on local disk. A node-local cache "
                    "tier removes it." % (stat_n, fmt_s(stat_s))))

    # readiness semantics
    if "port_open" in edges and "health_200" in edges:
        gap = edges["health_200"] - edges["port_open"]
        if gap > 1:
            out.append((gap, "READINESS SEMANTICS: the TCP port accepted "
                        "connections %s before /health returned 200. A "
                        "TCP-socket readinessProbe would mark this pod ready "
                        "that much too early; use an httpGet on /health."
                        % fmt_s(gap)))
    if "first_token" in edges and "health_200" in edges:
        gap = edges["first_token"] - edges["health_200"]
        if gap > 0.5:
            out.append((gap, "FIRST REQUEST COST: the first real completion "
                        "took %s after /health went green (lazy kernel JIT, "
                        "sampler warmup). Traffic sent at the readiness edge "
                        "pays this; a synthetic warmup request before joining "
                        "the endpoint list hides it." % fmt_s(gap)))

    # serialisation between processes
    # multiprocessing's resource_tracker and the readiness poller are
    # bookkeeping, not engine bring-up: their start time says nothing about
    # serialisation of the work we care about.
    NOISE_ROLES = ("resource_tracker", "forkserver", "ready")
    execs = sorted((p["exec_ts"], run.label(p["pid"])) for p in run.procs.values()
                   if p.get("exec_ts") and p.get("kind") == "process.exec"
                   and p.get("role") not in NOISE_ROLES)
    if len(execs) > 1:
        last = execs[-1]
        out.append((last[0] - t0, "PROCESS SERIALISATION: the last process "
                    "(%s) only started at t+%s, so everything it must do "
                    "(imports, CUDA init, weight load) begins after that. "
                    "Overlapping engine/worker bring-up with API server "
                    "startup is pure win." % (last[1], fmt_s(last[0] - t0))))

    # A span with nothing under it. `unaccounted` only catches wall clock that
    # no span covers at all; it says nothing about a 13s span whose interior is
    # empty, which reads in the phase table as a measured phase and is not one.
    # This is the check that would have caught the renamed v2 model-runner
    # module without anyone noticing the rename.
    for name, dur, cov_frac, pid in dark_spans(run, total):
        out.append((dur * (1.0 - cov_frac),
                    "DARK SPAN: %s of %s (pid %s) has no probe underneath it "
                    "(%.0f%% of its interior is covered by child spans). The "
                    "phase table will show this as one opaque block. Add spans "
                    "for what it calls -- and check cs_patch coverage for "
                    "never_imported modules first, since a renamed upstream "
                    "module looks exactly like this."
                    % (fmt_s(dur * (1.0 - cov_frac)), name, pid,
                       100.0 * cov_frac)))

    # instrumentation blind spot
    if unaccounted > 0.15 * total and total > 0:
        cov = run.coverage or {}
        out.append((unaccounted, "UNACCOUNTED: %s (%.0f%% of total) is not "
                    "inside any instrumented span. %d probe targets did not "
                    "apply to this vLLM version -- see probe.coverage in the "
                    "trace; add the moved paths to cs_vllm.PATCHES."
                    % (fmt_s(unaccounted), 100.0 * unaccounted / total,
                       len(cov.get("not_applied") or []))))

    # page cache / major faults
    majflt = counter_delta(run, "majflt", t0, t_ready)
    if majflt > 10000:
        out.append((0.0, "PAGE CACHE: %d major faults during startup -- weight "
                    "pages were fetched from storage rather than the node page "
                    "cache. Expected on a truly cold node; if this repeats on a "
                    "warm node, something is evicting the cache."
                    % int(majflt)))

    out.sort(key=lambda x: -x[0])
    return [msg for _, msg in out]


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def render(run, min_ms=50.0, tree_depth=4):
    t0 = run.t0()
    t_ready, ready_src = run.t_ready()
    total = t_ready - t0
    per_phase, unaccounted, detail, per_phase_proc, concurrent = sweep_phases(
        run, t0, t_ready)

    L = []
    ctx = run.context or {}
    cfg = run.config or {}
    cg = ctx.get("cgroup") or {}
    cpu = ctx.get("cpu") or {}

    L.append("=" * 78)
    L.append("vLLM COLD START: process exec -> API server ready")
    L.append("=" * 78)
    L.append("run dir      : %s" % run.path)
    L.append("run id       : %s" % ((run.probe_installed or {}).get("run_id")
                                    or ctx.get("run_id") or run.name))
    L.append("model        : %s  (dtype=%s tp=%s pp=%s eager=%s)"
             % (cfg.get("model"), cfg.get("dtype"),
                cfg.get("tensor_parallel_size"),
                cfg.get("pipeline_parallel_size"), cfg.get("enforce_eager")))
    comp = cfg.get("compilation") or {}
    if comp:
        L.append("compilation  : level=%s mode=%s cudagraph=%s sizes=%s "
                 "cache_dir=%s" % (comp.get("level"), comp.get("mode"),
                                   comp.get("cudagraph_mode")
                                   or comp.get("use_cudagraph"),
                                   comp.get("n_cudagraph_sizes"),
                                   comp.get("cache_dir")))
    L.append("vllm / torch : %s / %s (cuda %s)"
             % (na((run.vllm_info or {}).get("version")),
                na((run.torch_info or {}).get("torch")),
                na((run.torch_info or {}).get("cuda_build"))))
    L.append("node         : %s cores (cgroup limit %s), %s"
             % (cpu.get("cpu_count"), cg.get("cpu_limit_cores") or "none",
                (ctx.get("gpu") or {}).get("nvidia_driver", "no nvidia driver")))
    gpus = (run.gpu_inventory or {}).get("devices") or []
    if gpus:
        L.append("gpus         : %s" % ", ".join(
            "%s[%s]" % (g.get("name"), g.get("index")) for g in gpus))
    fs_model = (ctx.get("mounts_of_interest") or {}).get("model") \
        or (ctx.get("mounts_of_interest") or {}).get("hf_hub")
    if fs_model:
        L.append("model fs     : %s on %s (%s)"
                 % (fs_model.get("fstype"), fs_model.get("mountpoint"),
                    fs_model.get("source")))
    caches = (run.caches or {}).get("caches") or {}
    if any(caches.values()):
        L.append("caches at t0 : " + ", ".join(
            "%s=%s/%s files" % (k, fmt_bytes((v or {}).get("bytes")),
                                (v or {}).get("files"))
            for k, v in caches.items() if v))
    elif caches:
        L.append("caches at t0 : none of %s exist (fully cold)"
                 % ", ".join(sorted(caches)))
    L.append("processes    : %d python processes traced (%d event files)"
             % (len(run.procs), len(run.files)))
    ovs = [m["args"]["overhead_s"] for m in run.metas
           if m["name"] == "probe.installed" and m.get("args", {}).get("overhead_s")]
    if ovs:
        L.append("probe cost   : %s worst / %s median of %d processes to install "
                 "the probe (never subtracted from any phase)"
                 % (fmt_s(max(ovs)), fmt_s(sorted(ovs)[len(ovs) // 2]), len(ovs)))
    L.append("")
    L.append("TOTAL TIME TO READY : %s   (t0 = first vLLM process exec, "
             "ready = %s)" % (fmt_s(total), ready_src))

    edges = run.ready_edges()
    if edges:
        L.append("")
        L.append("readiness edges (from t0):")
        for k in ("port_open", "api_startup_complete", "health_200",
                  "models_200", "first_token"):
            if k in edges:
                L.append("  %-22s t+%s" % (k, fmt_s(edges[k] - t0)))

    # ---- phase table ----
    L.append("")
    L.append("-" * 78)
    L.append("PHASE BREAKDOWN (non-overlapping; credited to the innermost "
             "active span)")
    L.append("-" * 78)
    if concurrent > 0.05 and total > 0:
        # Where two processes work at once, one of them had to be picked. Say
        # so, rather than letting the table imply the split was measured.
        L.append("  note: %s (%.0f%%) had >1 process working at once; over "
                 "those intervals the" % (fmt_s(concurrent),
                                          100.0 * concurrent / total))
        L.append("        phase shown is the more specific one, not the only "
                 "one. A process blocked")
        L.append("        on another (wait_for_engine_startup and friends) "
                 "counts as idle, not as ipc.")
    rows = []
    for phase in REPORT_ORDER:
        secs = per_phase.get(phase, 0.0)
        if secs <= 0:
            continue
        rows.append([phase, fmt_s(secs), "%5.1f%%" % (100.0 * secs / total
                                                      if total else 0),
                     bar(secs / total if total else 0)])
    rows.append(["unaccounted", fmt_s(unaccounted),
                 "%5.1f%%" % (100.0 * unaccounted / total if total else 0),
                 bar(unaccounted / total if total else 0)])
    rows.append(["TOTAL", fmt_s(total), "100.0%", ""])
    table(rows, ["phase", "wall", "share", ""], L)

    # ---- biggest single spans per phase ----
    L.append("")
    L.append("top time sinks (span credited the most wall time in each phase):")
    seen = set()
    for (phase, name), secs in sorted(detail.items(), key=lambda kv: -kv[1]):
        if secs < 0.05 or phase in seen and len([1 for p, _ in seen if p == phase]) > 2:
            continue
        seen.add((phase, name))
        L.append("  %-26s %-40s %s" % (phase, name[:40], fmt_s(secs)))
        if len(seen) >= 18:
            break

    # ---- per process ----
    L.append("")
    L.append("-" * 78)
    L.append("PROCESSES")
    L.append("-" * 78)
    rows = []
    for pid, p in sorted(run.procs.items(),
                         key=lambda kv: kv[1].get("exec_ts") or 0):
        if not p.get("exec_ts"):
            continue
        summ = p.get("summary") or {}
        io = summ.get("proc_io") or {}
        imp = p.get("imports") or {}
        rows.append([
            run.label(pid), p.get("kind", "").replace("process.", ""),
            "t+" + fmt_s(p["exec_ts"] - t0),
            fmt_s(imp.get("total_self_s")) if imp else "-",
            fmt_bytes(io.get("read_bytes")),
            fmt_bytes((summ.get("rusage") or {}).get("maxrss_kb", 0) * 1024
                      if summ.get("rusage") else None),
            (p.get("cmdline") or "")[:44],
        ])
    table(rows, ["process", "start", "exec@", "imports", "disk read",
                 "peak rss", "cmdline"], L)

    # ---- span tree of the critical processes ----
    L.append("")
    L.append("-" * 78)
    L.append("SPAN TREE  (>= %.0fms, depth <= %d)" % (min_ms, tree_depth))
    L.append("-" * 78)
    for pid in sorted(run.procs,
                      key=lambda p: run.procs[p].get("exec_ts") or 0):
        spans = [s for s in run.spans if s.get("pid") == pid
                 and (s.get("dur") or 0) * 1000 >= min_ms
                 and (s.get("depth") or 0) <= tree_depth
                 and s.get("cat") != "import"]
        if not spans:
            continue
        L.append("")
        L.append("%s  [exec t+%s]" % (run.label(pid),
                                      fmt_s((run.procs[pid].get("exec_ts") or t0)
                                            - t0)))
        for s in sorted(spans, key=lambda s: s["ts"]):
            d = s.get("depth") or 0
            L.append("  %st+%-8s %-46s %s"
                     % ("  " * d, fmt_s(s["ts"] - t0),
                        s.get("name", "")[:46], fmt_s(s.get("dur"))))

    # ---- imports ----
    imp_rows = import_totals(run)
    if imp_rows:
        L.append("")
        L.append("-" * 78)
        L.append("PYTHON IMPORTS")
        L.append("-" * 78)
        rows = []
        for pid, label, tot, nmod, pkgs in imp_rows:
            top = ", ".join("%s %s" % (k, fmt_s(v)) for k, v in
                            list(sorted(pkgs.items(), key=lambda kv: -kv[1]))[:5])
            rows.append([label, fmt_s(tot), nmod, top])
        table(rows, ["process", "module exec", "modules", "top packages"], L)

    # ---- subsystem detail ----
    L.append("")
    L.append("-" * 78)
    warm = warmup_breakdown(run)
    if warm:
        L.append("")
        L.append("-" * 78)
        L.append("WARMUP / CUDA GRAPH CAPTURE")
        L.append("-" * 78)
        L.extend(warm)

    L.append("SUBSYSTEMS")
    L.append("-" * 78)
    for cat, title in (("network", "network (dns/tcp/tls/http)"),
                       ("weights", "weight loading"),
                       ("gpu", "gpu / cuda"),
                       ("collective", "collectives (nccl etc)"),
                       ("compile", "compilation"),
                       ("cudagraph", "cuda graph capture"),
                       ("kvcache", "kv cache"),
                       ("warmup", "warmup"),
                       ("ipc", "inter-process handshakes")):
        agg = aggregate(run, cat)
        if not agg:
            continue
        L.append("%-26s n=%-5d sum=%-8s wall=%-8s max=%s"
                 % (title, agg["count"], fmt_s(agg["sum_s"]),
                    fmt_s(agg["union_s"]), fmt_s(agg["max_s"])))
        if cat == "ipc":
            # These are raw span durations, so most of this wall is the parent
            # sitting in wait_for_engine_startup while the child boots. The
            # phase table deliberately credits that to the child's real work,
            # which is why the two numbers differ by ~1000x -- see
            # BLOCKING_SPANS. Say so here rather than letting it read as a
            # contradiction.
            blk = [x for x in run.spans
                   if x.get("cat") == "ipc" and is_blocking(x)]
            if blk:
                L.append("    (of which %s is blocking on another process, "
                         "not handshake cost)"
                         % fmt_s(union_len([(x["ts"], x["ts"] + (x.get("dur")
                                                                or 0))
                                            for x in blk])))
        for s in top_spans(run, cat, 3):
            args = s.get("args") or {}
            extra = ""
            for k in ("url", "path", "file", "peer", "host", "repo_id",
                      "filename"):
                if k in args and args[k]:
                    extra = " %s=%s" % (k, str(args[k])[:60])
                    break
            L.append("    %-42s %-8s pid=%s%s"
                     % (s.get("name", "")[:42], fmt_s(s.get("dur")),
                        s.get("pid"), extra))

    fsc = proc_counts(run, "fs.")
    if fsc:
        tot_n = sum(v[0] for v in fsc.values())
        tot_s = sum(v[1] for v in fsc.values())
        L.append("%-26s n=%-5d sum=%s" % ("filesystem metadata calls", tot_n,
                                          fmt_s(tot_s)))
        for k, v in sorted(fsc.items(), key=lambda kv: -kv[1][1])[:4]:
            L.append("    %-42s %-8s n=%d" % (k, fmt_s(v[1]), v[0]))

    # ---- resource peaks ----
    L.append("")
    read_total = counter_delta(run, "read_bytes", t0, t_ready)
    rx = counter_delta(run, "rx_bytes", t0, t_ready)
    tx = counter_delta(run, "tx_bytes", t0, t_ready)
    L.append("disk read during startup : %s (%s avg over the window)"
             % (fmt_bytes(read_total),
                fmt_rate(read_total / total / 1e6 if total else 0)))
    L.append("netns traffic            : rx %s / tx %s"
             % (fmt_bytes(rx), fmt_bytes(tx)))
    gmem = counter_series(run, "mem_used")
    if gmem:
        for pid, series in gmem.items():
            if series:
                L.append("gpu memory used          : %s -> %s"
                         % (fmt_bytes(series[0][1]), fmt_bytes(series[-1][1])))
            break

    # ---- findings ----
    f = findings(run, t0, t_ready, per_phase, unaccounted)
    if f:
        L.append("")
        L.append("-" * 78)
        L.append("FINDINGS")
        L.append("-" * 78)
        for i, msg in enumerate(f, 1):
            L.append("%2d. %s" % (i, _wrap(msg, 74, "    ")))

    cov = run.coverage or {}
    if cov.get("not_applied"):
        L.append("")
        L.append("probe targets not applied to this vLLM version (%d): %s"
                 % (len(cov["not_applied"]),
                    ", ".join(cov["not_applied"][:8]) +
                    (" ..." if len(cov["not_applied"]) > 8 else "")))
    if cov.get("never_imported"):
        # Distinct from not_applied, and the more dangerous of the two: the
        # module was never imported, so the patch never even got the chance to
        # fail. A module that upstream renamed looks exactly like a module this
        # build does not use.
        L.append("")
        L.append("probe targets whose module was never imported (%d): %s"
                 % (len(cov["never_imported"]),
                    ", ".join(cov["never_imported"][:8]) +
                    (" ..." if len(cov["never_imported"]) > 8 else "")))
        if cov.get("reported_by"):
            L.append("    (trace predates per-target coverage events: list is "
                     "from the %d process(es) that reported, so a target may "
                     "have applied in one that did not)" % cov["reported_by"])
    if run.bad_lines:
        L.append("NOTE: %d unparseable trace lines (process killed mid-write?)"
                 % run.bad_lines)
    return "\n".join(L), {
        "total_s": total, "t0": t0, "t_ready": t_ready,
        "ready_source": ready_src, "phases": dict(per_phase),
        "unaccounted_s": unaccounted, "edges": {k: v - t0 for k, v
                                                in edges.items()},
        "findings": f, "config": cfg, "context": ctx,
        "processes": {str(p): run.label(p) for p in run.procs},
    }


def _wrap(text, width, indent):
    words = text.split()
    lines = []
    cur = ""
    for w in words:
        if cur and len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    return ("\n" + indent).join(lines)


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def chrome_trace(run):
    """Perfetto / chrome://tracing view: spans per process+thread, resource
    counters as tracks, readiness edges as global markers."""
    t0 = run.t0()
    ev = []
    for pid, p in run.procs.items():
        ev.append({"name": "process_name", "ph": "M", "pid": pid, "tid": 0,
                   "args": {"name": run.label(pid)}})
        ev.append({"name": "process_sort_index", "ph": "M", "pid": pid,
                   "tid": 0, "args": {"sort_index": int(
                       (p.get("exec_ts") or t0) - t0)}})
    for s in run.spans:
        ev.append({"name": s.get("name", "?"), "cat": s.get("cat", ""),
                   "ph": "X", "ts": (s["ts"] - t0) * 1e6,
                   "dur": max((s.get("dur") or 0) * 1e6, 1),
                   "pid": s.get("pid"), "tid": s.get("tid", 0),
                   "args": s.get("args") or {}})
    for i in run.instants:
        ev.append({"name": i.get("name", "?"), "cat": i.get("cat", ""),
                   "ph": "i", "ts": (i["ts"] - t0) * 1e6,
                   "pid": i.get("pid"), "tid": i.get("tid", 0),
                   "s": "g" if str(i.get("name", "")).startswith("ready.")
                   else "t", "args": i.get("args") or {}})
    # cumulative counters plus derived rates
    prev = {}
    for c in run.counters:
        args = c.get("args") or {}
        nums = {k: v for k, v in args.items()
                if isinstance(v, (int, float))}
        if not nums:
            continue
        pid = c.get("pid")
        ev.append({"name": c.get("name", "counter"), "ph": "C",
                   "ts": (c["ts"] - t0) * 1e6, "pid": pid, "args": nums})
        key = (pid, c.get("name"))
        p = prev.get(key)
        if p:
            dt = c["ts"] - p[0]
            if dt > 0:
                rates = {}
                for k, label, scale in (
                        ("read_bytes", "disk_read_MBps", 1e6),
                        ("rx_bytes", "net_rx_MBps", 1e6),
                        ("tx_bytes", "net_tx_MBps", 1e6),
                        ("write_bytes", "disk_write_MBps", 1e6)):
                    if k in nums and k in p[1]:
                        rates[label] = round(
                            (nums[k] - p[1][k]) / dt / scale, 2)
                cpu = 0.0
                for k in ("cpu_utime_s", "cpu_stime_s"):
                    if k in nums and k in p[1]:
                        cpu += nums[k] - p[1][k]
                if cpu:
                    rates["cpu_cores"] = round(cpu / dt, 2)
                if rates:
                    ev.append({"name": (c.get("name") or "") + "_rate",
                               "ph": "C", "ts": (c["ts"] - t0) * 1e6,
                               "pid": pid, "args": rates})
        prev[key] = (c["ts"], nums)
    return {"traceEvents": ev, "displayTimeUnit": "ms",
            "otherData": {"run": run.name, "t0_epoch": t0,
                          "note": "ts are microseconds since vLLM process exec"}}


CSV_FIELDS = ["run_id", "run_dir", "total_s", "ready_source", "model", "dtype",
              "tp", "enforce_eager", "compile_level", "vllm", "torch",
              "cpu_limit_cores", "model_fs", "unaccounted_s"] + \
    ["phase_" + p.replace(" ", "_").replace(":", "").replace("/", "")
     for p in REPORT_ORDER] + \
    ["edge_port_open", "edge_health_200", "edge_models_200", "edge_first_token",
     "disk_read_bytes", "net_rx_bytes", "throttled_s"]


def csv_row(run, summary):
    cfg = summary["config"] or {}
    ctx = summary["context"] or {}
    t0, t_ready = summary["t0"], summary["t_ready"]
    row = {
        "run_id": ((run.probe_installed or {}).get("run_id")
                   or (run.context or {}).get("run_id") or run.name),
        "run_dir": run.path,
        "total_s": round(summary["total_s"], 4),
        "ready_source": summary["ready_source"],
        "model": cfg.get("model"), "dtype": cfg.get("dtype"),
        "tp": cfg.get("tensor_parallel_size"),
        "enforce_eager": cfg.get("enforce_eager"),
        "compile_level": (cfg.get("compilation") or {}).get("level"),
        "vllm": (run.vllm_info or {}).get("version"),
        "torch": (run.torch_info or {}).get("torch"),
        "cpu_limit_cores": (ctx.get("cgroup") or {}).get("cpu_limit_cores"),
        "model_fs": ((ctx.get("mounts_of_interest") or {}).get("model")
                     or {}).get("fstype"),
        "unaccounted_s": round(summary["unaccounted_s"], 4),
        "disk_read_bytes": int(counter_delta(run, "read_bytes", t0, t_ready)),
        "net_rx_bytes": int(counter_delta(run, "rx_bytes", t0, t_ready)),
        "throttled_s": round(counter_delta(run, "cg_throttled_usec",
                                           t0, t_ready) / 1e6, 3),
    }
    for p in REPORT_ORDER:
        key = "phase_" + p.replace(" ", "_").replace(":", "").replace("/", "")
        row[key] = round(summary["phases"].get(p, 0.0), 4)
    for e in ("port_open", "health_200", "models_200", "first_token"):
        row["edge_" + e] = round(summary["edges"].get(e), 4) \
            if e in summary["edges"] else ""
    return row


def compare(runs, out):
    """Side-by-side phase table: run A is the baseline, others show deltas."""
    summaries = []
    for r in runs:
        _, s = render(r)
        summaries.append((r, s))
    base = summaries[0][1]
    out.append("=" * 78)
    out.append("COLD START COMPARISON (baseline = %s)" % summaries[0][0].name)
    out.append("=" * 78)
    headers = ["phase"] + [r.name for r, _ in summaries] + \
              ["delta vs base" for _ in summaries[1:]]
    rows = []
    for p in REPORT_ORDER + ["unaccounted", "TOTAL"]:
        vals = []
        for _, s in summaries:
            if p == "TOTAL":
                vals.append(s["total_s"])
            elif p == "unaccounted":
                vals.append(s["unaccounted_s"])
            else:
                vals.append(s["phases"].get(p, 0.0))
        if not any(v > 0.01 for v in vals):
            continue
        row = [p] + [fmt_s(v) for v in vals]
        for v in vals[1:]:
            d = v - vals[0]
            row.append(("%+.1fs" % d) if abs(d) >= 0.05 else "~")
        rows.append(row)
    table(rows, headers, out)
    out.append("")
    for r, s in summaries:
        cfg = s["config"] or {}
        out.append("%-20s total=%-8s model=%s tp=%s eager=%s compile=%s"
                   % (r.name, fmt_s(s["total_s"]), cfg.get("model"),
                      cfg.get("tensor_parallel_size"),
                      cfg.get("enforce_eager"),
                      (cfg.get("compilation") or {}).get("level")))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rundir", nargs="+", help="trace directory/directories")
    ap.add_argument("--compare", action="store_true",
                    help="compare the given run directories side by side")
    ap.add_argument("-o", "--out-dir", help="write report.md, trace.json, "
                                            "summary.json, phases.csv here")
    ap.add_argument("--csv", help="append the CSV row to this file "
                                  "(for regression tracking across runs)")
    ap.add_argument("--min-ms", type=float, default=50.0,
                    help="span tree threshold (default 50ms)")
    ap.add_argument("--tree-depth", type=int, default=4)
    ap.add_argument("--json", action="store_true",
                    help="print the machine-readable summary instead of text")
    args = ap.parse_args(argv)

    runs = [Run(d) for d in args.rundir]

    if args.compare or len(runs) > 1:
        out = compare(runs, [])
        print("\n".join(out))
        return 0

    run = runs[0]
    text, summary = render(run, args.min_ms, args.tree_depth)
    if args.json:
        print(json.dumps(summary, indent=2, default=str))
    else:
        print(text)

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        with open(os.path.join(args.out_dir, "report.txt"), "w") as fh:
            fh.write(text + "\n")
        with open(os.path.join(args.out_dir, "summary.json"), "w") as fh:
            json.dump(summary, fh, indent=2, default=str)
        with open(os.path.join(args.out_dir, "trace.json"), "w") as fh:
            json.dump(chrome_trace(run), fh)
        _write_csv(os.path.join(args.out_dir, "phases.csv"),
                   csv_row(run, summary))
        print("\nwrote %s/{report.txt,summary.json,trace.json,phases.csv}"
              % args.out_dir)
        print("open trace.json in https://ui.perfetto.dev or chrome://tracing")
    if args.csv:
        _write_csv(args.csv, csv_row(run, summary))
        print("appended row to %s" % args.csv)
    return 0


def _write_csv(path, row):
    exists = os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerow(row)


if __name__ == "__main__":
    sys.exit(main())
