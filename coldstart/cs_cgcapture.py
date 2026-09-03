"""CUDA graph capture: price the two things concurrent capture would need.

`cudagraph capture` is 11.4s of a 43.3s time-to-ready at 32B, and "overlap
capture across sizes" is the one lever the README lists as unquantified.
[scripts/cudagraph-concurrent-capture.py](../scripts/cudagraph-concurrent-capture.py)
answers the primitive half on bare torch: concurrent capture *works* (bit-exact,
`capture_error_mode="thread_local"`), but it needs **one mempool per capturing
thread** and it needs **nobody to call `torch.cuda.synchronize()`**. Both of
those are properties of vLLM's capture loop, not of CUDA, and both are
measurable here without writing a threaded capture at all:

    CS_CGPOOLS=N     rotate N graph mempools round-robin across the 102
                     captures, still strictly serial. vLLM shares exactly one
                     pool on purpose -- it captures largest-first and PIECEWISE
                     before FULL so later activations land in buffers earlier
                     graphs already allocated (`cudagraph_utils.py:355`). N
                     pools give up that reuse, and vLLM's own
                     "Graph capturing finished in %d secs, took %.2f GiB" log
                     line then reports what it cost. This is the memory bill any
                     concurrency scheme pays, isolated from every timing effect.

    CS_CGDETAIL=1    span every `graph.__enter__` and `__exit__` with no duration
                     floor, and tag every capture with the line that built it.
                     Diagnostic only, and deliberately separate from the two
                     knobs above so a matrix arm is never measured with extra
                     probe work the arm it is compared against did not pay.
                     This is what shows that `cudagraph.capture_begin` in the
                     committed traces counts *captures over 5ms*, not captures.

    CS_CGSYNC=stream|none
                     `torch.cuda.graph.__enter__` calls `torch.cuda.synchronize()`
                     unconditionally (torch/cuda/graphs.py:437, "Free as much
                     memory as we can for the graph"). At 32B that is 1.6s of the
                     phase across 102 captures -- 96% of `cudagraph.capture_begin`
                     in runs/w6-base-c{1,2,3}. It is also a hard barrier: a device
                     sync issued while another thread is capturing raises, and
                     under `capture_error_mode="thread_local"` it *poisons* the
                     in-flight capture. So the sync has to go before any thread
                     can help, and this knob asks the prior question -- what does
                     removing it cost or save on its own?
                       stream: sync only the stream capture is about to start on
                       none:   skip it (empty_cache() still runs -- CS_CGEMPTY)

    CS_CGEMPTY=once  the arm that matters for an upstream patch: skip the
                     per-capture `empty_cache()` but do one `synchronize()` +
                     `empty_cache()` when `capture_model` returns. If the 3366
                     calls exist to keep the graph pool tight, hoisting them out
                     of the loop should keep the time saving *and* the memory,
                     which `CS_CGEMPTY=0` does not.

    CS_CGEMPTY=0     `__enter__` also calls `torch.cuda.empty_cache()` and
                     `torch._C._host_emptyCache()` for the same reason. `cudaFree`
                     is itself device-synchronizing, so CS_CGSYNC=none on its own
                     leaves an implicit barrier behind -- which is what the
                     measured result showed. This knob removes the other half, and
                     the two together price the whole per-capture prologue: 3.20s
                     of `__enter__` + 0.65s of `__exit__` across 3366 captures
                     (runs/cc-detail2, analysis/cudagraph_graph_census.py).

Both knobs are `N=1`/`device`-safe: the defaults reproduce upstream behaviour
through the same code path, which is what the control arm should use.

Neither is a shipping configuration. This probe exists to put numbers on an
upstream change, which is why it is opt-in twice over (a knob *and* the probe
list) and why it writes what it did to stderr and to the trace.
"""

import sys
import threading

import cs_patch
import cs_trace as T

_installed = False
_npools = 0
_sync = "device"
_empty = True
_empty_once = False
_detail = False

_lock = threading.Lock()
_pools = []            # graph pool handles, created lazily; refs held here
_next = 0              # round-robin cursor
_captures = 0
_mem0 = None           # (reserved, free) at the first capture


def install():
    global _installed, _npools, _sync, _empty, _detail
    _detail = T.env_flag("CGDETAIL")
    global _empty_once
    raw_empty = (T.env("CGEMPTY") or "").strip().lower()
    _empty_once = raw_empty == "once"
    _empty = not (_empty_once or raw_empty in ("0", "none", "no"))
    raw = T.env("CGPOOLS")
    try:
        _npools = int(str(raw).strip()) if raw not in (None, "") else 0
    except (TypeError, ValueError) as e:
        T.tracer.error("cgcapture.pools", e)
        _npools = 0
    _sync = (T.env("CGSYNC", "device") or "device").strip().lower()
    if _sync not in ("device", "stream", "none"):
        T.tracer.error("cgcapture.sync", ValueError("CS_CGSYNC=%r" % _sync))
        _sync = "device"

    # Nothing here runs by default: CS_PROBES defaults to "all", so T.enabled()
    # alone would change how every run captures graphs.
    if (_npools <= 0 and _sync == "device" and _empty and not _detail) \
            or not T.enabled("cgcapture"):
        return
    if _installed:
        return
    _installed = True
    cs_patch.after_import("torch.cuda.graphs", _wrap)
    if _empty_once:
        # v2 runner only: this repo measures v2, and patching a class that the
        # run never imports would fail silently rather than loudly.
        cs_patch.after_import("vllm.v1.worker.gpu.model_runner", _wrap_capture_model)
    T.tracer.at_finish(_summarize)


def _pool_for_next_capture():
    """Round-robin handle. Round-robin (not blocked) on purpose: vLLM captures
    in descending size order, so every pool still sees its own sizes
    largest-first and keeps the intra-pool reuse. Only cross-pool reuse is lost,
    which is exactly the effect being priced."""
    global _next
    import torch
    with _lock:
        while len(_pools) < _npools:
            _pools.append(torch.cuda.graph_pool_handle())
        idx = _next % _npools
        _next += 1
        return _pools[idx], idx


def _wrap(module):
    cls = getattr(module, "graph", None)
    if cls is None:
        T.tracer.error("cgcapture.graph",
                       AttributeError("torch.cuda.graphs.graph"))
        return
    if getattr(cls.__enter__, "_cs_cgcapture", False):
        return

    import torch

    orig_init = cls.__init__
    orig_enter = cls.__enter__          # already wrapped by cs_torch's span
    orig_exit = cls.__exit__

    def __init__(self, *a, **kw):
        orig_init(self, *a, **kw)
        # Only redirect captures that asked for a shared pool. A capture with
        # pool=None is torch allocating a fresh pool per graph already, and
        # rewriting it would change a different thing than the one being priced.
        if _npools > 0 and self.pool:
            h, idx = _pool_for_next_capture()
            self.pool = (h,)
            # One instant per capture. EngineCore never exits before the trace
            # is collected, so its at_finish meta is never written -- these are
            # what proves the arm engaged, and how many graphs each pool holds.
            args = {"pool": idx}
            if _detail:
                args["src"] = _caller()
            T.tracer.instant("cgcapture.pool", cat="cudagraph", **args)
            # vLLM sets this process-global just before constructing us
            # (pynccl_allocator.set_graph_pool_id, read by use_symmetric_memory);
            # ours has to be the one that wins.
            try:
                from vllm.distributed.device_communicators.pynccl_allocator \
                    import set_graph_pool_id
                set_graph_pool_id(h)
            except Exception:
                pass
        elif _detail:
            T.tracer.instant("cgcapture.pool", cat="cudagraph", pool=None,
                             src=_caller(), shared=bool(self.pool))

    def __enter__(self):
        global _captures, _mem0
        _captures += 1
        if _mem0 is None:
            _mem0 = _mem()
        if not _detail:
            return _enter_inner(self)
        # min_dur=0.0: the point of this arm is the captures the 5ms floor on
        # cudagraph.capture_begin hides.
        tok = T.tracer.begin("cgcapture.enter", "cudagraph")
        try:
            return _enter_inner(self)
        finally:
            T.tracer.end(tok, min_dur=0.0)

    def __exit__(self, *exc):
        if not _detail:
            return orig_exit(self, *exc)
        # cs_torch floors cudagraph.capture_end at 5ms too, so the phase
        # accounting is missing however much of it lands under that floor.
        tok = T.tracer.begin("cgcapture.exit", "cudagraph")
        try:
            return orig_exit(self, *exc)
        finally:
            T.tracer.end(tok, min_dur=0.0)

    def _enter_inner(self):
        if _sync == "device" and _empty:
            return orig_enter(self)
        # torch.cuda.graphs looks these up on the module at call time, so swapping
        # the attributes for the duration of __enter__ is enough. Capture is
        # single-threaded here by construction, which is the whole point.
        saved = {}
        if _sync != "device":
            saved["synchronize"] = torch.cuda.synchronize
            torch.cuda.synchronize = _presync
        if not _empty:
            saved["empty_cache"] = torch.cuda.empty_cache
            torch.cuda.empty_cache = _noempty
            # _host_emptyCache is reached as torch._C._host_emptyCache(), a C
            # extension attribute; leave it alone rather than monkeypatching
            # torch._C, and say so instead of implying the prologue is empty.
        try:
            return orig_enter(self)
        finally:
            for k, v in saved.items():
                setattr(torch.cuda, k, v)

    def _noempty(*a, **kw):
        T.tracer.instant("cgcapture.noempty", cat="cudagraph")

    def _presync(*a, **kw):
        if _sync == "none":
            # Same reason as cgcapture.pool: this is the only in-trace evidence
            # that the device sync was actually suppressed.
            T.tracer.instant("cgcapture.nosync", cat="cudagraph")
            return
        with T.span("cgcapture.presync", cat="cudagraph", scope="stream"):
            torch.cuda.current_stream().synchronize()

    __enter__._cs_cgcapture = True
    cls.__init__ = __init__
    cls.__enter__ = __enter__
    cls.__exit__ = __exit__

    sys.stderr.write("[cs_cgcapture] pools=%s sync=%s empty_cache=%s detail=%s\n"
                     % (_npools or "upstream(1, shared)", _sync,
                        "on" if _empty else "off", int(_detail)))
    sys.stderr.flush()


def _wrap_capture_model(module):
    """One synchronize()+empty_cache() after the whole capture loop, in place of
    the 3366 inside it. Wrapping capture_model rather than the last __exit__
    because there is no way to know which capture is last."""
    cls = getattr(module, "GPUModelRunner", None)
    orig = getattr(cls, "capture_model", None)
    if orig is None or getattr(orig, "_cs_cgcapture", False):
        T.tracer.error("cgcapture.capture_model",
                       AttributeError("GPUModelRunner.capture_model"))
        return

    def capture_model(self, *a, **kw):
        try:
            return orig(self, *a, **kw)
        finally:
            import torch
            with T.span("cgcapture.empty_once", cat="cudagraph"):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

    capture_model._cs_cgcapture = True
    cls.capture_model = capture_model


def _caller():
    """Which vLLM line constructed this capture. Two capture paths exist (the
    manager's FULL path and CUDAGraphWrapper's piecewise path) and they do not
    build the same number of graphs per size, which is the whole reason to ask."""
    import sys as _sys
    try:
        f = _sys._getframe(2)
        for _ in range(6):
            if f is None:
                break
            name = f.f_code.co_filename
            if "torch/cuda/graphs.py" not in name and "cs_cgcapture" not in name:
                return "%s:%d" % (name.rsplit("/vllm/", 1)[-1], f.f_lineno)
            f = f.f_back
    except Exception:
        pass
    return "?"


def _mem():
    try:
        import torch
        if not torch.cuda.is_initialized():
            return None
        return {"reserved_MiB": round(torch.cuda.memory_reserved() / 2**20, 1),
                "free_MiB": round(torch.cuda.mem_get_info()[0] / 2**20, 1)}
    except Exception:
        return None


def _summarize():
    # captures/pools proves the arm engaged; the memory pair brackets the phase
    # from the allocator's side, next to vLLM's own driver-side "took X GiB".
    T.tracer.meta("cgcapture.summary", pools=_npools, sync=_sync,
                  empty_cache=_empty,
                  captures=_captures, pools_created=len(_pools),
                  mem_at_first_capture=_mem0, mem_at_finish=_mem())
