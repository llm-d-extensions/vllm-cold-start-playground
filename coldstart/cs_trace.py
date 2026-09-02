"""Core tracing primitives for llm-d cold-start instrumentation.

Stdlib only, no third-party imports, no imports of anything heavy at module
import time: this module is loaded from ``sitecustomize`` before the
interpreter has finished starting up, in *every* Python process of the vLLM
process tree (API server, EngineCore, workers).

Event stream format: one JSON object per line, written to
``$CS_TRACE_DIR/events.<role>.<pid>.jsonl``.

    {"ph":"X","name":"weights.load","cat":"weights","ts":1756.., "dur":12.3,
     "pid":17,"tid":140..,"role":"engine_core","id":42,"parent":41,"depth":3,
     "args":{...}}

``ph`` follows the Chrome Trace Event convention so the analysis tool can emit
a Perfetto/chrome://tracing file with no translation:

    X = complete span (ts + dur)      i = instant
    C = counter sample                M = metadata

Timestamps are **seconds** (float) on CLOCK_REALTIME so events from different
processes on the same node share one timeline; ``mono`` carries the
CLOCK_MONOTONIC reading of the same instant for drift-free intra-process math.
"""

import atexit
import json
import os
import sys
import threading
import time

__all__ = [
    "tracer", "span", "instant", "counter", "meta", "count", "enabled",
    "emit_span",
    "process_exec_time", "read_proc_io", "read_proc_stat", "on_fork",
]

_UNSET = object()
_fork_callbacks = []
_finish_callbacks = []

# The pristine builtins.open, captured before cs_io wraps it. The tracer must
# never use the instrumented one: opening the trace file would emit an event,
# which would re-enter _file() before self._fh is assigned, duplicating the
# process header and leaking a second handle to the same file.
_raw_open = open


def on_fork(callback):
    """Run ``callback`` in a forked child, after the tracer has re-anchored."""
    _fork_callbacks.append(callback)


def env(name, default=None):
    return os.environ.get("CS_" + name, default)


def env_flag(name, default=False):
    v = env(name)
    if v is None:
        return default
    return v.strip().lower() not in ("", "0", "false", "no", "off")


def env_float(name, default):
    try:
        return float(env(name, ""))
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------
# /proc helpers (Linux). All return None off-Linux so the probe can be
# developed/tested on macOS without exploding.
# --------------------------------------------------------------------------

def read_proc_stat(pid="self"):
    """Return selected /proc/<pid>/stat fields as a dict."""
    try:
        with open("/proc/%s/stat" % pid, "rb") as fh:
            data = fh.read()
        # comm may contain spaces and parens; split after the last ')'
        tail = data[data.rindex(b")") + 2:].split()
        # tail[0] is field 3 (state), so 1-based field N is tail[N - 3]:
        # minflt=10, majflt=12, utime=14, stime=15, num_threads=20,
        # starttime=22, rss=24.  See proc(5).
        return {
            "minflt": int(tail[7]),
            "majflt": int(tail[9]),
            "utime": int(tail[11]) / os.sysconf("SC_CLK_TCK"),
            "stime": int(tail[12]) / os.sysconf("SC_CLK_TCK"),
            "num_threads": int(tail[17]),
            "starttime_ticks": int(tail[19]),
            "rss_pages": int(tail[21]),
        }
    except Exception:
        return None


def process_exec_time():
    """Wall-clock time at which *this* process was exec'd, and its age.

    Derived from /proc/self/stat starttime (ticks since boot) and /proc/uptime,
    which has 10ms resolution -- far better than /proc/stat's integer-second
    ``btime``. This is the t0 for "process start to ready": it predates the
    first byte of Python bytecode, so interpreter startup and ``site``
    processing land inside the measured window.
    """
    try:
        st = read_proc_stat()
        with open("/proc/uptime", "r") as fh:
            uptime = float(fh.read().split()[0])
        hz = os.sysconf("SC_CLK_TCK")
        age = uptime - (st["starttime_ticks"] / hz)
        return time.time() - age, age
    except Exception:
        return None, None


def read_proc_io(pid="self"):
    try:
        out = {}
        with open("/proc/%s/io" % pid, "r") as fh:
            for line in fh:
                k, _, v = line.partition(":")
                out[k.strip()] = int(v.strip())
        return out
    except Exception:
        return None


def read_net_dev():
    """Bytes in/out summed over all non-loopback interfaces of this netns."""
    try:
        rx = tx = 0
        with open("/proc/self/net/dev", "r") as fh:
            for line in fh.read().splitlines()[2:]:
                name, _, rest = line.partition(":")
                if name.strip() == "lo":
                    continue
                f = rest.split()
                rx += int(f[0])
                tx += int(f[8])
        return {"rx_bytes": rx, "tx_bytes": tx}
    except Exception:
        return None


def read_meminfo_cached():
    try:
        with open("/proc/meminfo", "r") as fh:
            for line in fh:
                if line.startswith("Cached:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    return None


# --------------------------------------------------------------------------
# Role detection: which process of the vLLM tree are we?
# --------------------------------------------------------------------------

def full_cmdline():
    """Full argv of this process. ``sys.argv`` is only ``['-c']`` for processes
    started by multiprocessing spawn, which is exactly the case we most need to
    tell apart, so prefer /proc/self/cmdline."""
    try:
        with open("/proc/self/cmdline", "rb") as fh:
            parts = fh.read().split(b"\0")
        return " ".join(p.decode("utf-8", "replace") for p in parts if p)
    except Exception:
        return " ".join(sys.argv or [])


def _program_words():
    """The parts of the command line that identify the *program*, with user
    flags and their values removed.

    Matching role needles against the whole command line is wrong: a run with
    ``--workers 1`` or a model called ``.../engine-7b`` would be labelled a
    worker or an engine. Only argv[0] and the ``-m`` / ``-c`` payload say what
    this process actually is.
    """
    words = [w for w in full_cmdline().split() if w]
    out = []
    i = 0
    while i < len(words):
        w = words[i]
        if i == 0:
            out.append(w)                      # interpreter or script path
        elif w in ("-m", "-c"):
            out.extend(words[i + 1:])          # module path / inline code
            break
        elif w.startswith("-"):
            i += 1
            if i < len(words) and not words[i].startswith("-"):
                i += 1                         # skip this flag's value
            continue
        else:
            out.append(w)                      # subcommand, e.g. "vllm serve"
        i += 1
    return " ".join(out).lower()


def _guess_role():
    hay = _program_words()
    for needle, role in (
        ("resource_tracker", "resource_tracker"),
        ("semaphore_tracker", "resource_tracker"),
        ("spawn_main", "spawned"),
        ("forkserver", "forkserver"),
        ("vllm serve", "api_server"),
        ("api_server", "api_server"),
        ("vllm.entrypoints", "api_server"),
        ("engine", "engine_core"),
        ("worker", "worker"),
    ):
        if needle in hay:
            return role
    # Name it after the script (``mock_vllm.py`` -> ``mock_vllm``); the file
    # name becomes part of the trace file name, so keep it filesystem-safe.
    for w in reversed(hay.split()):
        if w.endswith(".py"):
            base = os.path.basename(w)[:-3]
            break
    else:
        base = os.path.basename((sys.argv or [""])[0] or "")
        if base in ("", "-c"):
            base = "python"
    safe = "".join(c if (c.isalnum() or c in "-_") else "_"
                   for c in base).strip("-_")
    return safe or "python"


class Tracer(object):
    """Append-only JSONL event sink, safe across fork/spawn and threads."""

    def __init__(self):
        self.dir = env("TRACE_DIR", "/var/log/coldstart")
        self.run_id = env("RUN_ID", "run")
        self.disabled = env_flag("DISABLE", False)
        self.probes = set(
            p.strip() for p in env("PROBES", "all").split(",") if p.strip())
        self.min_dur = env_float("MIN_DUR_MS", 1.0) / 1000.0
        self.max_events = int(env_float("MAX_EVENTS", 200000))
        self.role = env("ROLE") or _guess_role()
        self._lock = threading.RLock()
        self._fh = None
        self._fh_pid = None
        self._seq = 0
        self._n = 0
        self._dropped = 0
        self._local = threading.local()
        self.counts = {}
        self.t0, self.age0 = process_exec_time()
        self.start_wall = time.time()
        self.start_mono = time.monotonic()
        self._closed = False
        self._summarised = False
        self._forked = False
        if not self.disabled:
            atexit.register(self._on_exit)
            self._install_signal_flush()
            try:
                os.register_at_fork(after_in_child=self._after_fork)
            except (AttributeError, TypeError):
                pass

    def _install_signal_flush(self):
        """Flush the per-process summary on SIGTERM/SIGINT as well as on exit.

        ``atexit`` never runs when a process dies of a signal, and every process
        in this experiment dies that way: Kubernetes SIGTERMs the container at
        teardown, and the harness SIGTERMs vLLM once the readiness edge is
        recorded. Without this the last process to be measured is exactly the
        one missing its ``process.summary`` (I/O, faults, rusage).

        Only the *default* disposition is replaced. If the application already
        installed a handler -- vLLM installs its own -- it is chained after the
        flush, so shutdown semantics are unchanged. A non-main thread cannot
        touch signal dispositions at all; that raises and is ignored.
        """
        try:
            import signal
        except Exception:
            return
        for signame in ("SIGTERM", "SIGINT", "SIGHUP"):
            sig = getattr(signal, signame, None)
            if sig is None:
                continue
            try:
                prev = signal.getsignal(sig)
            except Exception:
                continue

            def handler(signum, frame, sig=sig, prev=prev):
                try:
                    self.instant("process.signal", cat="process",
                                 signal=int(signum))
                    # close=False: a chained application handler may return
                    # and keep serving, so leave the file writable.
                    self._on_exit(close=False)
                except Exception:
                    pass
                if callable(prev):
                    return prev(signum, frame)
                if prev == signal.SIG_IGN:
                    return None
                # Default disposition: re-raise so the exit status still says
                # "killed by signal N" rather than a clean 0.
                try:
                    signal.signal(signum, signal.SIG_DFL)
                    os.kill(os.getpid(), signum)
                except Exception:
                    os._exit(128 + int(signum))

            try:
                signal.signal(sig, handler)
            except Exception:
                pass

    def _after_fork(self):
        """A forked child inherits our patches but not our threads or file
        handle. Re-anchor the timeline (its exec time *is* the fork time) so the
        child's spans are attributable, and let probes restart their samplers.

        This matters because vLLM may fork rather than spawn its EngineCore /
        worker processes, in which case ``sitecustomize`` never runs again."""
        self._forked = True
        self._fh = None
        self._fh_pid = None
        self._closed = False
        self._summarised = False
        self.counts = {}
        self._local = threading.local()
        self.role = env("ROLE") or _guess_role()
        for cb in list(_fork_callbacks):
            try:
                cb()
            except Exception:
                pass

    # -- probe gating ------------------------------------------------------
    def probe_enabled(self, name):
        if self.disabled:
            return False
        if "all" in self.probes:
            return ("-" + name) not in self.probes and ("no" + name) not in self.probes
        return name in self.probes

    # -- io ----------------------------------------------------------------
    def _file(self):
        pid = os.getpid()
        if self._fh is not None and self._fh_pid == pid:
            return self._fh
        # First write in this process (or after a fork): open our own file.
        try:
            os.makedirs(self.dir, exist_ok=True)
        except Exception:
            pass
        path = os.path.join(
            self.dir, "events.%s.%d.jsonl" % (self.role, pid))
        try:
            self._fh = _raw_open(path, "a", buffering=1)
        except Exception:
            try:
                self._fh = _raw_open(os.path.join(
                    "/tmp", os.path.basename(path)), "a", buffering=1)
            except Exception:
                self.disabled = True
                return None
        self._fh_pid = pid
        self._emit_process_header()
        return self._fh

    def _emit_process_header(self):
        t0, age = process_exec_time()
        st = read_proc_stat()
        args = {
            "cmdline": full_cmdline()[:2000],
            "argv": " ".join(sys.argv)[:500],
            "executable": sys.executable,
            "python": sys.version.split()[0],
            "ppid": os.getppid(),
            "cwd": os.getcwd(),
            "age_at_probe_s": age,
            "hostname": os.uname().nodename if hasattr(os, "uname") else "",
            "run_id": self.run_id,
        }
        if st:
            args["threads"] = st["num_threads"]
        # _write_locked, not _write: we are called from inside _file() with the
        # reentrancy guard held, and these two events must not be dropped.
        self._write_locked({
            "ph": "M",
            "name": "process.fork" if self._forked else "process.exec",
            "cat": "process",
            "ts": t0 if t0 else self.start_wall,
            "probe_ts": self.start_wall, "args": args,
        })
        if age is not None and not self._forked:
            # exec -> first bytecode we can observe: interpreter boot + site
            self._write_locked({
                "ph": "X", "name": "interpreter.startup", "cat": "process",
                "ts": t0, "dur": age, "depth": 0,
                "args": {"note": "exec() to sitecustomize import; includes "
                                 "dynamic linking, libpython init, site/pth"},
            })

    def _write(self, ev):
        if self.disabled or self._closed:
            return
        if getattr(self._local, "writing", False):
            # Reentrant: something on the write path (an instrumented open, a
            # /proc read, a probe callback) tried to emit its own event. Drop it
            # rather than recursing -- the tracer must not measure itself.
            self._dropped += 1
            return
        if self._n >= self.max_events:
            self._dropped += 1
            return
        self._local.writing = True
        try:
            self._write_locked(ev)
        finally:
            self._local.writing = False

    def _write_locked(self, ev):
        fh = self._fh if self._fh_pid == os.getpid() else self._file()
        if fh is None:
            return
        ev.setdefault("pid", os.getpid())
        ev.setdefault("tid", threading.get_ident())
        ev.setdefault("role", self.role)
        try:
            line = json.dumps(ev, default=_json_default, separators=(",", ":"))
        except Exception:
            return
        with self._lock:
            try:
                fh.write(line + "\n")
                self._n += 1
            except Exception:
                self.disabled = True

    # -- public API --------------------------------------------------------
    def next_id(self):
        with self._lock:
            self._seq += 1
            return self._seq

    def _stack(self):
        s = getattr(self._local, "stack", None)
        if s is None:
            s = self._local.stack = []
        return s

    def begin(self, name, cat="misc", **args):
        """Open a span. Returns an opaque token for :meth:`end`."""
        if self.disabled:
            return None
        sid = self.next_id()
        stack = self._stack()
        tok = (sid, name, cat, time.time(), time.monotonic(),
               stack[-1][0] if stack else 0, args)
        stack.append(tok)
        return tok

    def end(self, tok, min_dur=_UNSET, **extra):
        if tok is None or self.disabled:
            return
        sid, name, cat, ts, mono, parent, args = tok
        stack = self._stack()
        # Tolerate unbalanced spans (exceptions, generators): pop back to tok.
        if tok in stack:
            while stack and stack[-1] is not tok:
                stack.pop()
            if stack:
                stack.pop()
        dur = time.monotonic() - mono
        key = cat + ":" + name
        c = self.counts.setdefault(key, [0, 0.0])
        c[0] += 1
        c[1] += dur
        floor = self.min_dur if min_dur is _UNSET else min_dur
        if dur < floor:
            return
        if extra:
            args = dict(args)
            args.update(extra)
        self._write({
            "ph": "X", "name": name, "cat": cat, "ts": ts, "dur": dur,
            "id": sid, "parent": parent, "depth": len(stack),
            "args": args or None,
        })

    def current_span_id(self):
        stack = self._stack()
        return stack[-1][0] if stack else 0

    def emit_span(self, name, cat, ts, dur, parent=0, depth=0, args=None):
        """Write a pre-measured span (used by probes that do their own timing)."""
        if self.disabled:
            return
        self._write({"ph": "X", "name": name, "cat": cat, "ts": ts,
                     "dur": dur, "id": self.next_id(), "parent": parent,
                     "depth": depth, "args": args or None})

    def instant(self, name, cat="misc", **args):
        if self.disabled:
            return
        stack = self._stack()
        self._write({
            "ph": "i", "name": name, "cat": cat, "ts": time.time(),
            "parent": stack[-1][0] if stack else 0, "args": args or None,
        })

    def counter(self, name, values, cat="metric"):
        if self.disabled:
            return
        self._write({"ph": "C", "name": name, "cat": cat,
                     "ts": time.time(), "args": values})

    def meta(self, name, **args):
        self._write({"ph": "M", "name": name, "cat": "meta",
                     "ts": time.time(), "args": args})

    def count(self, key, n=1, seconds=0.0):
        """Cheap aggregate counter for events too hot to emit individually."""
        c = self.counts.setdefault(key, [0, 0.0])
        c[0] += n
        c[1] += seconds

    def error(self, where, exc):
        self.instant("probe.error", cat="probe", where=where,
                     err="%s: %s" % (type(exc).__name__, exc))

    # -- shutdown ----------------------------------------------------------
    def at_finish(self, cb):
        """Run ``cb`` when this process is summarised -- on clean exit *or* on
        SIGTERM. Probes register their ``*.summary`` writers here instead of
        with ``atexit``, which a signal death skips."""
        _finish_callbacks.append(cb)

    def _on_exit(self, close=True):
        if self.disabled or self._closed:
            return
        if self._summarised:
            self._finish(close)
            return
        for cb in list(_finish_callbacks):
            try:
                cb()
            except Exception:
                pass
        try:
            io = read_proc_io()
            st = read_proc_stat()
            net = read_net_dev()
            summary = {
                "counts": {k: {"n": v[0], "total_s": round(v[1], 6)}
                           for k, v in sorted(self.counts.items())},
                "dropped_events": self._dropped,
                "events": self._n,
            }
            if io:
                summary["proc_io"] = io
            if st:
                summary["proc_stat"] = st
            if net:
                summary["net_dev"] = net
            try:
                import resource
                ru = resource.getrusage(resource.RUSAGE_SELF)
                summary["rusage"] = {
                    "utime": ru.ru_utime, "stime": ru.ru_stime,
                    "maxrss_kb": ru.ru_maxrss, "majflt": ru.ru_majflt,
                    "minflt": ru.ru_minflt, "inblock": ru.ru_inblock,
                }
            except Exception:
                pass
            self._write({"ph": "M", "name": "process.summary", "cat": "process",
                         "ts": time.time(), "args": summary})
        except Exception:
            pass
        finally:
            self._summarised = True
            self._finish(close)

    def _finish(self, close):
        try:
            if self._fh:
                self._fh.flush()
                # fsync, not just flush: the last thing this process writes is
                # its summary, and the container is usually destroyed moments
                # later. A flush only reaches the kernel -- on a bind mount or
                # an overlay backed by a VM, the tail page can be lost when the
                # container goes away, which shows up as a truncated final JSONL
                # line. This happens after the readiness edge, so it costs the
                # measurement nothing.
                try:
                    os.fsync(self._fh.fileno())
                except Exception:
                    pass
                if close:
                    self._fh.close()
        except Exception:
            pass
        if close:
            self._closed = True


def _json_default(o):
    try:
        return str(o)[:500]
    except Exception:
        return "<unserializable>"


tracer = Tracer()


class _Span(object):
    """Context manager + decorator-friendly span."""

    __slots__ = ("tok",)

    def __init__(self, name, cat, args):
        self.tok = tracer.begin(name, cat, **args)

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        if et is not None:
            tracer.end(self.tok, min_dur=0.0,
                       error="%s: %s" % (getattr(et, "__name__", et), ev))
        else:
            tracer.end(self.tok)
        return False

    def set(self, **kw):
        if self.tok is not None:
            self.tok[6].update(kw)


def span(name, cat="misc", **args):
    return _Span(name, cat, args)


def instant(name, cat="misc", **args):
    tracer.instant(name, cat, **args)


def counter(name, values, cat="metric"):
    tracer.counter(name, values, cat)


def meta(name, **args):
    tracer.meta(name, **args)


def count(key, n=1, seconds=0.0):
    tracer.count(key, n, seconds)


def emit_span(name, cat, ts, dur, parent=0, depth=0, args=None):
    tracer.emit_span(name, cat, ts, dur, parent, depth, args)


def enabled(probe):
    return tracer.probe_enabled(probe)
