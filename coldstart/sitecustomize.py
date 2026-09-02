"""Injection point for the llm-d cold-start probe.

Python's ``site`` module imports ``sitecustomize`` automatically during
interpreter startup, before any user code runs. Putting this directory on
``PYTHONPATH`` therefore instruments **every** Python process in the vLLM tree
-- the API server, EngineCore, and each worker -- with no change to the image,
no wrapper script, and no fork of vLLM. Child processes inherit ``PYTHONPATH``,
so ``spawn``-ed children are covered too, and forked children are re-anchored
via ``os.register_at_fork``.

Enable it with:

    PYTHONPATH=/opt/coldstart CS_TRACE_DIR=/var/log/coldstart/<run> vllm serve ...

Everything here is best-effort: a probe failure must never keep vLLM from
starting, so every step is wrapped and errors are recorded as trace events
rather than raised.

Knobs (all optional):
    CS_DISABLE=1            turn the probe off entirely
    CS_TRACE_DIR=<dir>      output directory (default /var/log/coldstart)
    CS_RUN_ID=<id>          label for this run
    CS_PROBES=all           or a comma list: imports,net,io,torch,vllm,sampler,env
                            ("all,-net" disables one)
    CS_FORKSERVER=1         start EngineCore from a pre-imported forkserver
                            instead of spawn/fork (off by default; see
                            cs_forkserver.py for CS_FORKSERVER_MODE/_PRELOAD)
    CS_MIN_DUR_MS=1         drop spans shorter than this
    CS_IMPORT_MIN_MS=15     drop import spans shorter than this
    CS_SAMPLE_MS=100        resource sampling interval
    CS_MAX_EVENTS=200000    per-process event cap
"""

import os
import sys

_ALREADY = "_cs_probe_installed"


def _chain_other_sitecustomize(self_dir):
    """Respect a sitecustomize.py the image already shipped: we shadow it by
    being first on sys.path, so run it after we are set up."""
    import importlib.util
    for entry in sys.path:
        try:
            if not entry or os.path.realpath(entry) == self_dir:
                continue
            cand = os.path.join(entry, "sitecustomize.py")
            if not os.path.isfile(cand):
                continue
            spec = importlib.util.spec_from_file_location(
                "_cs_chained_sitecustomize", cand)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return cand
        except Exception:
            continue
    return None


def _install():
    if getattr(sys, _ALREADY, False):
        return
    setattr(sys, _ALREADY, True)

    if os.environ.get("CS_DISABLE", "").strip().lower() not in ("", "0",
                                                               "false", "no"):
        return

    self_dir = os.path.dirname(os.path.realpath(__file__))
    if self_dir not in sys.path:
        sys.path.insert(0, self_dir)

    import time
    t_start = time.monotonic()

    import cs_trace as T
    if T.tracer.disabled:
        return

    installed = []
    failed = {}

    def step(name, fn):
        try:
            fn()
            installed.append(name)
        except Exception as e:
            failed[name] = "%s: %s" % (type(e).__name__, e)
            try:
                T.tracer.error("install:" + name, e)
            except Exception:
                pass

    # Order matters: the import hook must be in place before anything else is
    # imported, and cs_patch's hook must exist before probes register patches.
    import cs_patch
    step("hook", cs_patch.install_hook)

    import cs_imports
    step("imports", cs_imports.install)

    import cs_net
    step("net", cs_net.install)

    import cs_io
    step("io", cs_io.install)

    import cs_torch
    step("torch", cs_torch.install)

    import cs_vllm
    step("vllm", cs_vllm.install)

    import cs_sampler
    step("sampler", cs_sampler.install)

    # Opt-in twice over (CS_FORKSERVER=1 *and* the probe list): unlike every
    # other probe this one changes how vLLM starts its engine, so it must never
    # be on by default. Installed late so its get_mp_context patch lands after
    # cs_vllm's spans are registered.
    import cs_forkserver
    step("forkserver", cs_forkserver.install)

    # Opt-in twice over (CS_FST=1); sizes the unreachable fastsafetensors
    # parameters for an upstream change. See coldstart/cs_fst.py.
    import cs_fst
    step("fst", cs_fst.install)

    import cs_env
    step("env", cs_env.capture)
    step("proctitle", cs_env.install_title_probe)

    def _final():
        try:
            cs_io.summarize()
        except Exception:
            pass
        try:
            T.tracer.meta("probe.coverage", **cs_patch.coverage())
        except Exception:
            pass
    T.tracer.at_finish(_final)

    overhead = time.monotonic() - t_start
    T.tracer.meta("probe.installed", probes=installed, failed=failed,
                  overhead_s=round(overhead, 4),
                  trace_dir=T.tracer.dir, role=T.tracer.role,
                  min_dur_ms=T.tracer.min_dur * 1000.0,
                  note="time spent here is instrumentation overhead and is "
                       "subtracted from no phase; keep it small")

    chained = _chain_other_sitecustomize(self_dir)
    if chained:
        T.tracer.meta("probe.chained_sitecustomize", path=chained)


try:
    _install()
except Exception:
    # Never break the process being measured.
    if os.environ.get("CS_DEBUG"):
        import traceback
        traceback.print_exc()
