"""forkserver: pay the import cost once, in a process that is safe to fork.

Under ``vllm serve`` the engine's imports are paid **twice**.
``entrypoints/serve/utils/api_utils.py:164`` forces ``spawn`` whenever
``VLLM_WORKER_MULTIPROC_METHOD`` is unset, so ``EngineCore`` is a fresh
interpreter that re-imports torch and vllm from scratch -- 15.5s of a 55.3s run
on ``20260901-191644-r2``. ``fork`` avoids that but forks from a multi-threaded,
torch-loaded parent, which Python warns can deadlock the child.

``forkserver`` is the mechanism that gets the win safely: one single-threaded,
CUDA-free process pre-imports the stack, and every child is a cheap fork from a
process that was never multi-threaded. vLLM already has the code for it
(``api_server.py:108``) but it cannot be reached -- ``envs.py:937`` restricts
the variable to ``spawn``/``fork``, so the value that switches the feature on
makes ``get_mp_context()`` raise ``ValueError``. And where the code sits it
would not pay anyway: ``ensure_running()`` only *starts* the preload, which the
first ``Process.start()`` then blocks on, and upstream starts it ~1.7s before
that first fork.

So this probe does two things the env var cannot:

  * arms the forkserver at **probe-install time** (t~0, before vLLM is imported)
    so the preload overlaps the parent's own ~15.7s of imports instead of
    landing on the critical path;
  * hands ``get_mp_context()`` a forkserver context directly, sidestepping the
    validator -- which is what makes this measurable with **no vLLM edit**.

It is deliberately opt-in twice over (``CS_FORKSERVER=1`` *and* the probe list),
because changing how vLLM starts its engine is not something a measurement probe
may do by default.

Knobs:
    CS_FORKSERVER=1            enable (required; nothing here runs without it)
    CS_FORKSERVER_MODE=early   early: arm at t~0, preload overlaps the parent's
                               imports. late: set the preload but let the first
                               Process.start() start the server, reproducing
                               upstream's placement. The two together are the
                               A/B that shows placement is the whole lever.
    CS_FORKSERVER_PRELOAD=...  comma list of modules the server pre-imports.

What this does *not* do: it never overrides a genuine veto. If CUDA is already
initialised, Ray owns process creation, ``--numa-bind`` hijacks the executable,
or we are on WSL, we defer to vLLM's own choice and say so in the trace
(``forkserver.declined``) -- those reasons are correctness, not conservatism.
"""

import multiprocessing
import os
import sys
import time

import cs_patch
import cs_trace as T

_installed = False

# Upstream preloads ``vllm.v1.engine.async_llm``, but that is the *frontend*
# class. What actually gets forked is EngineCoreProc, and at TP>1 WorkerProc, so
# a preload naming async_llm leaves real import cost in the child. This is the
# one tunable worth sweeping.
_DEFAULT_PRELOAD = "vllm.v1.engine.core"

# Every one of these does ``from vllm.utils.system_utils import get_mp_context``,
# which binds the function object at *that* module's import time. Patching only
# system_utils would miss v1/engine/utils.py:139 -- the EngineCore launch path
# this whole exercise is about.
_CALLERS = (
    "vllm.utils.system_utils",
    "vllm.v1.engine.utils",
    "vllm.v1.executor.multiproc_executor",
    "vllm.v1.engine.coordinator",
)

_ctx = None            # the forkserver context, once armed
_orig = None           # vLLM's own get_mp_context, for when we decline
_preload = ()
_served = 0            # times we handed out the forkserver context
_declined = []         # reasons we did not, in order


def install():
    global _installed, _ctx, _preload
    if _installed:
        return
    # CS_PROBES defaults to "all", so T.enabled() alone would switch this on for
    # every run. Require the explicit flag too.
    if not (T.env_flag("FORKSERVER") and T.enabled("forkserver")):
        return
    _installed = True

    # The span that contains the block-on-preload wait: _launch calls
    # connect_to_new_process, which calls ensure_running and then waits for the
    # server to accept. If the preload was not overlapped, the cost shows up
    # here -- which is exactly the thing being measured.
    cs_patch.patch("multiprocessing.popen_forkserver", "Popen._launch",
                   name="forkserver.launch_child", cat="engine", min_dur=0.0)

    if T.env_flag("FORKSERVER_ACTIVE"):
        _install_server_side()
        return

    _preload = _preload_list()
    mode = (T.env("FORKSERVER_MODE", "early") or "early").strip().lower()
    if mode not in ("early", "late"):
        # The two arms differ by this one string, so a typo would quietly run
        # `late` and report it as `early` -- a fabricated "no difference".
        # Refuse: no forkserver.armed in the trace, one probe.error saying why.
        T.tracer.error("forkserver.mode",
                       ValueError("CS_FORKSERVER_MODE=%r; want early|late"
                                  % mode))
        return

    try:
        multiprocessing.set_forkserver_preload(list(_preload))
    except Exception as e:
        T.tracer.error("forkserver.set_preload", e)
        return

    # The forkserver process and everything forked from it inherit this, so set
    # it *before* the server starts: it is what keeps a child from arming a
    # second, cold forkserver of its own.
    os.environ["CS_FORKSERVER_ACTIVE"] = "1"

    if mode == "early" and not _arm():
        os.environ.pop("CS_FORKSERVER_ACTIVE", None)
        return

    try:
        _ctx = multiprocessing.get_context("forkserver")
    except Exception as e:
        T.tracer.error("forkserver.get_context", e)
        return

    for name in _CALLERS:
        cs_patch.after_import(name, _rebind)

    T.tracer.at_finish(_summarize)
    T.tracer.meta("forkserver.armed", mode=mode, preload=list(_preload),
                  python=sys.version.split()[0])


def _preload_list():
    """The modules the server should pre-import, from CS_FORKSERVER_PRELOAD.

    Read the same way in the parent and in the server process, so the server can
    report which of them actually landed.
    """
    raw = T.env("FORKSERVER_PRELOAD", _DEFAULT_PRELOAD) or ""
    return tuple(m for m in (s.strip() for s in raw.split(",")) if m)


def _arm():
    """Start the server now, so its preload runs against our own imports.

    ``ensure_running()`` returns as soon as it has ``listen()``ed and spawned the
    interpreter; the preload runs in that child before it enters its accept
    loop. So this is a background task started here and joined at the first
    fork, and the span below measures only the launch, not the preload.
    """
    ts, t0 = time.time(), time.monotonic()
    try:
        from multiprocessing import forkserver
        forkserver.ensure_running()
    except Exception as e:
        # A forkserver that will not start is not a fallback we can paper over:
        # every Process.start() would retry the same failure. Leave vLLM's own
        # start method alone and let the run stand as a normal one.
        T.tracer.error("forkserver.ensure_running", e)
        return False
    T.tracer.emit_span("forkserver.arm", "engine", ts, time.monotonic() - t0,
                       args={"preload": list(_preload)})
    return True


def _install_server_side():
    """Runs in the forkserver process and in everything it forks.

    Nothing is armed here -- the server's own imports are already traced by
    cs_imports, and re-arming would nest a cold forkserver. What we do add is
    the safety invariant: the whole argument for forkserver over fork is that
    the process children come from is single-threaded and CUDA-free, so record
    whether that actually held.
    """
    if T.tracer.role != "forkserver":
        # A resource_tracker, or a child of a parent whose arm was declined.
        # The parent's forkserver.summary already says which; an event here
        # would just be noise in every process in the tree.
        return
    T.tracer.instant("forkserver.inherited", cat="engine")

    def _check():
        # sys.modules, never `import torch`: importing it here would add ~5s to
        # this process at teardown and make the trace lie about what the
        # preload actually loaded. None means "torch was not preloaded".
        torch = sys.modules.get("torch")
        cuda = None
        try:
            if torch is not None:
                cuda = bool(torch.cuda.is_initialized())
        except Exception:
            pass
        # forkserver.main() swallows ImportError on every preload entry, so a
        # typo costs the whole win and reports nothing. Name what did not land.
        want = _preload_list()
        st = T.read_proc_stat() or {}
        T.tracer.meta("forkserver.server_summary",
                      cuda_initialized=cuda,
                      num_threads=st.get("num_threads"),
                      modules=len(sys.modules),
                      preloaded=[m for m in want if m in sys.modules],
                      preload_missing=[m for m in want if m not in sys.modules])
    T.tracer.at_finish(_check)


def _rebind(module):
    """Point one caller module's own ``get_mp_context`` name at our wrapper."""
    global _orig
    fn = getattr(module, "get_mp_context", None)
    if fn is None or getattr(fn, "_cs_forkserver", False):
        return
    if _orig is None or module.__name__ == "vllm.utils.system_utils":
        _orig = fn      # prefer the genuine, unwrapped original
    setattr(module, "get_mp_context", _wrapper(module.__name__))


def _wrapper(where):
    def get_mp_context():
        global _served
        reason = _veto_reason()
        if reason is not None:
            _declined.append(reason)
            T.tracer.instant("forkserver.declined", cat="engine",
                             where=where, reason=reason)
            return _orig()
        su = sys.modules.get("vllm.utils.system_utils")
        try:
            su._sync_visible_devices_env_vars()
        except Exception as e:
            T.tracer.error("forkserver.sync_visible_devices", e)
        _served += 1
        T.tracer.instant("forkserver.context", cat="engine", where=where)
        return _ctx
    get_mp_context._cs_forkserver = True
    return get_mp_context


def _veto_reason():
    """``_maybe_force_spawn``'s reasons, minus its side effect.

    The real function writes ``VLLM_WORKER_MULTIPROC_METHOD=spawn`` and returns
    nothing, so we cannot ask it. Read the predicates out of system_utils' own
    namespace rather than reimplementing them: if upstream renames one, that
    shows up here as a decline, not as a stale copy of the rule still saying yes.
    """
    su = sys.modules.get("vllm.utils.system_utils")
    if su is None:
        return "vllm.utils.system_utils not imported"
    if "--numa-bind" in sys.argv:
        return "NUMA binding requires spawn"
    for attr, why in (("cuda_is_initialized", "CUDA is initialized"),
                      ("xpu_is_initialized", "XPU is initialized"),
                      ("in_wsl", "WSL detected"),
                      ("is_in_ray_actor", "in a Ray actor")):
        fn = getattr(su, attr, None)
        if fn is None:
            return "cannot evaluate %s" % attr
        try:
            if fn():
                return why
        except Exception as e:
            return "%s raised %s" % (attr, type(e).__name__)
    return None


def _summarize():
    T.tracer.meta("forkserver.summary", served=_served,
                  declined=_declined[:8], preload=list(_preload))
