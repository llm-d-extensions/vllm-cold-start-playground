"""Python import cost attribution.

Equivalent in spirit to ``python -X importtime`` but structured: every module
execution becomes a span in the shared trace, so import cost is visible on the
same timeline as CUDA init and weight loading -- and, crucially, is attributed
per process (the API server, EngineCore and each worker all pay their own
import bill, and in vLLM v1 they pay it *serially* unless they overlap).

Reported per module:
  * cumulative time  -- module exec including its transitive imports
  * self time        -- excluding nested imports (the real per-module cost)
  * origin           -- stdlib / site-packages / frozen, and ``.so`` extensions
                        (native extension load = dynamic linker + relocations,
                        a large share of ``import torch``)
"""

import os
import sys
import sysconfig
import time

import cs_patch
import cs_trace as T

_MIN = T.env_float("IMPORT_MIN_MS", 15.0) / 1000.0
_TOP_N = int(T.env_float("IMPORT_TOP_N", 40))

_stack = []            # [name, t_mono, child_time, span_parent, tracer_ts]
_self = {}             # module -> self seconds
_cum = {}              # module -> cumulative seconds
_origin = {}           # module -> origin tag
_order = []            # first-exec order
_installed = False


def _stdlib_dirs():
    out = set()
    for key in ("stdlib", "platstdlib"):
        p = sysconfig.get_paths().get(key)
        if p:
            out.add(p)
    return tuple(out)


_STDLIB = _stdlib_dirs()


def _classify(name):
    mod = sys.modules.get(name)
    f = getattr(mod, "__file__", None) or ""
    if not f:
        if name in getattr(sys, "builtin_module_names", ()):
            return "builtin"
        return "frozen/namespace"
    if f.endswith((".so", ".pyd", ".dylib")):
        return "native"
    if "site-packages" in f or "dist-packages" in f:
        return "site-packages"
    if _STDLIB and f.startswith(_STDLIB):
        return "stdlib"
    return "other"


def _on_start(name):
    _stack.append([name, time.monotonic(), 0.0,
                   T.tracer.current_span_id(), time.time()])


def _on_end(name, exc):
    if not _stack:
        return
    # Unwind defensively in case a nested exec_module never reported.
    while _stack and _stack[-1][0] != name:
        _stack.pop()
    if not _stack:
        return
    entry = _stack.pop()
    total = time.monotonic() - entry[1]
    self_t = total - entry[2]
    if _stack:
        _stack[-1][2] += total
    _cum[name] = _cum.get(name, 0.0) + total
    _self[name] = _self.get(name, 0.0) + self_t
    if name not in _origin:
        _origin[name] = _classify(name)
        _order.append(name)
    if total >= _MIN:
        T.tracer.emit_span(
            "import " + name, "import", entry[4], total,
            parent=entry[3], depth=len(_stack),
            args={"self_s": round(self_t, 6), "origin": _origin.get(name),
                  "failed": bool(exc)})


def install():
    global _installed
    if _installed or not T.enabled("imports"):
        return
    _installed = True
    cs_patch.add_exec_observer(_on_start, _on_end)
    T.tracer.at_finish(summarize)


def summarize():
    if not _self:
        return
    total_self = sum(_self.values())
    by_pkg = {}
    by_origin = {}
    for name, s in _self.items():
        top = name.split(".")[0]
        by_pkg[top] = by_pkg.get(top, 0.0) + s
        o = _origin.get(name, "?")
        by_origin[o] = by_origin.get(o, 0.0) + s
    top_pkgs = sorted(by_pkg.items(), key=lambda kv: -kv[1])[:_TOP_N]
    top_mods = sorted(_self.items(), key=lambda kv: -kv[1])[:_TOP_N]
    T.tracer.meta(
        "imports.summary",
        modules=len(_self),
        total_self_s=round(total_self, 4),
        by_top_package={k: round(v, 4) for k, v in top_pkgs},
        by_origin={k: round(v, 4) for k, v in by_origin.items()},
        slowest_modules=[{"module": k, "self_s": round(v, 4),
                          "cum_s": round(_cum.get(k, 0.0), 4),
                          "origin": _origin.get(k)} for k, v in top_mods],
        first_import_order=_order[:200],
    )
