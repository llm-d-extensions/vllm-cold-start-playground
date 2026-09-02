"""fastsafetensors: force the transfer path vLLM cannot be told to use.

``--load-format fastsafetensors`` is the largest single win this harness has
measured outside imports (-12.4s at 32B), but the three parameters that matter
after that are all unreachable from a config file or an env var:

  * ``nogds`` -- ``weight_utils.py:1043`` sets ``nogds = pg.size() > 1``, so at
    TP=1 vLLM always asks for GPUDirect Storage. On a cluster with no GDS
    (no ``nvidia_fs``, no ``libcufile``; the GPU-operator ClusterPolicy here sets
    ``gds: {enabled: false}``) fastsafetensors degrades *internally* rather than
    raising, so the ``nogds=True`` fallback in that same function never fires and
    the cost is silent: **2.44s of per-process setup**, paid by every EngineCore,
    probing for a feature that is not installed.
  * ``max_threads`` / ``bbuf_size_kb`` -- ``fastsafetensors_weights_iterator``
    constructs ``ParallelLoader`` directly and passes neither, and neither
    appears in fastsafetensors' own ``LoaderConfig``, so ``FASTSAFETENSORS_CONFIG``
    cannot reach them on vLLM's path either.

So this probe overrides them at the one seam that works regardless of how vLLM
refers to the class: ``ParallelLoader.__init__`` itself. It exists to size an
upstream change, not to be a shipping configuration -- which is why it is opt-in
twice over (``CS_FST=1`` *and* the probe list) and why it records every value it
overrode in the trace.

Knobs:
    CS_FST=1                enable (required; nothing here runs without it)
    CS_FST_NOGDS=auto|1|0   auto (default): force nogds=True only when this host
                            has no GDS, which is the behaviour being proposed
                            upstream. 1/0 force unconditionally, for the A/B.
    CS_FST_MAX_THREADS=n    override max_threads (upstream default 16; 8 measures
                            better under a CPU-limited cgroup)
    CS_FST_BBUF_KB=n        override bbuf_size_kb (upstream default 16384)
    CS_FST_QUEUE_SIZE=n     override queue_size, for completeness; measured null

Measured at 32B, one fresh process per config, interleaved over 3 passes:
``nogds=False`` 16.57s (upstream) / ``nogds=True`` 13.97s / plus 8 threads and a
32768 KiB buffer 12.77s.
"""

import glob
import os
import sys

import cs_patch
import cs_trace as T

_installed = False
_applied = []          # the overrides that actually fired, in order
_gds = None            # cached GDS availability


def install():
    global _installed
    if _installed:
        return
    # CS_PROBES defaults to "all", so T.enabled() alone would switch this on for
    # every run. Changing how vLLM moves 61 GiB is not a probe's default.
    if not (T.env_flag("FST") and T.enabled("fst")):
        return
    _installed = True
    cs_patch.after_import("fastsafetensors.parallel_loader", _wrap)
    T.tracer.at_finish(_summarize)


def gds_available():
    """Is GPUDirect Storage actually usable on this host?

    This is the predicate the upstream fix should use in place of ``pg.size() >
    1``. Deliberately cheap and side-effect free: no ``libcufile`` dlopen, no
    CUDA call, because this runs before vLLM has imported torch.
    """
    global _gds
    if _gds is None:
        _gds = bool(glob.glob("/dev/nvidia-fs*")) and os.path.exists(
            "/proc/driver/nvidia-fs")
    return _gds


def _overrides(kwargs):
    """What we would change, given the caller's kwargs. Empty dict = no-op."""
    out = {}

    want = (T.env("FST_NOGDS", "auto") or "auto").strip().lower()
    if want in ("auto", ""):
        # Only force it when GDS is genuinely absent -- the proposed upstream
        # rule. On a GDS-equipped host this probe becomes a no-op, which is the
        # correct behaviour and worth having in the trace.
        if not gds_available():
            out["nogds"] = True
    elif want in ("1", "true", "yes"):
        out["nogds"] = True
    elif want in ("0", "false", "no"):
        out["nogds"] = False
    else:
        T.tracer.error("fst.nogds", ValueError("CS_FST_NOGDS=%r" % want))

    for env_name, kw in (("FST_MAX_THREADS", "max_threads"),
                         ("FST_BBUF_KB", "bbuf_size_kb"),
                         ("FST_QUEUE_SIZE", "queue_size")):
        raw = T.env(env_name)
        if raw is None or str(raw).strip() == "":
            continue
        try:
            out[kw] = int(raw)
        except (TypeError, ValueError) as e:
            T.tracer.error("fst.%s" % kw, e)

    # Never report an override for a value that already matches: a no-op logged
    # as a change is how a null result gets mistaken for a working patch.
    return {k: v for k, v in out.items() if kwargs.get(k) != v}


def _wrap(module):
    cls = getattr(module, "ParallelLoader", None)
    if cls is None:
        # cs_patch records a missing target in probe.coverage; do the same here
        # rather than raising, so a fastsafetensors rename shows up as an
        # unapplied patch instead of a failed run.
        T.tracer.error("fst.ParallelLoader",
                       AttributeError("fastsafetensors.parallel_loader"
                                      ".ParallelLoader"))
        return
    orig = cls.__init__
    if getattr(orig, "_cs_fst", False):
        return

    def __init__(self, *a, **kw):
        changed = _overrides(kw)
        if changed:
            before = {k: kw.get(k) for k in changed}
            kw.update(changed)
            _applied.append(changed)
            T.tracer.instant("fst.override", cat="weights",
                             changed=changed, was=before,
                             gds_available=gds_available())
            # Also on stderr, so `grep cs_fst runs/*/vllm.log` verifies the arm
            # engaged without opening a trace. An arm that silently did nothing
            # reads exactly like "the patch does not help".
            sys.stderr.write(
                "[cs_fst] ParallelLoader override: %s (was %s, gds=%s)\n"
                % (changed, before, gds_available()))
            sys.stderr.flush()
        return orig(self, *a, **kw)

    __init__._cs_fst = True
    cls.__init__ = __init__


def _summarize():
    T.tracer.meta("fst.summary", loaders=len(_applied),
                  applied=_applied[:4], gds_available=gds_available())
