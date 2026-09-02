"""Filesystem probe: where the weight bytes come from and how much stat-ing
the startup path does.

Two distinct costs show up here and they have different fixes:

1. **Bulk read** of weight shards. Most of this is invisible at the Python
   level (safetensors mmaps the file and the bytes are faulted in by native
   code), so byte volume and throughput come from ``/proc/self/io`` +
   ``majflt`` sampling in :mod:`cs_sampler`. What *is* visible here is which
   files were opened, how large they are, and from which filesystem.
2. **Metadata chatter** -- ``stat``/``exists``/``listdir`` storms. Free on
   overlayfs, brutal on NFS/CSI-backed PVCs where each call is a round trip.
   Those are counted (not spanned) and reported as count + total seconds.
"""

import os
import time

import cs_patch
import cs_trace as T

_installed = False
_MIN = T.env_float("IO_MIN_MS", 5.0) / 1000.0
_WEIGHT_EXT = (".safetensors", ".bin", ".pt", ".pth", ".gguf", ".npz",
               ".ckpt", ".onnx")
_CONFIG_EXT = (".json", ".model", ".txt", ".py", ".yaml", ".yml", ".jinja")
_opened = {}          # path -> size (weights/config only)


def _interesting(path):
    if not isinstance(path, str):
        return None
    low = path.lower()
    if low.endswith(_WEIGHT_EXT):
        return "weight"
    if low.endswith(_CONFIG_EXT):
        return "config"
    return None


def install():
    global _installed
    if _installed or not T.enabled("io"):
        return
    _installed = True
    _patch_open()
    _patch_metadata_calls()
    _patch_loaders()


def _patch_open():
    import builtins
    import io as _io
    orig_open = builtins.open

    trace_dir = os.path.realpath(T.tracer.dir or "")

    def open_(file, *a, **kw):
        t = time.monotonic()
        fh = orig_open(file, *a, **kw)
        dt = time.monotonic() - t
        # The probe's own trace files are not part of what we are measuring,
        # and counting them would put probe I/O in the report.
        if isinstance(file, str) and trace_dir and file.startswith(trace_dir):
            return fh
        T.count("io.open", 1, dt)
        kind = _interesting(file if isinstance(file, str) else str(file))
        if kind is not None:
            size = None
            try:
                size = os.fstat(fh.fileno()).st_size
            except Exception:
                pass
            if isinstance(file, str) and file not in _opened:
                _opened[file] = size
            if kind == "weight" or dt >= _MIN:
                T.tracer.instant("file.open." + kind, "fileio",
                                 path=str(file)[:300], size=size,
                                 open_s=round(dt, 6))
        elif dt >= _MIN:
            T.tracer.instant("file.open.slow", "fileio",
                             path=str(file)[:300], open_s=round(dt, 6))
        return fh
    open_._cs_wrapped = True
    builtins.open = open_
    try:
        _io.open = open_
    except Exception:
        pass


def _wrap_counter(mod, attr, key):
    orig = getattr(mod, attr, None)
    if orig is None or getattr(orig, "_cs_wrapped", False):
        return

    def wrapper(*a, **kw):
        t = time.monotonic()
        try:
            return orig(*a, **kw)
        finally:
            T.count(key, 1, time.monotonic() - t)
    wrapper._cs_wrapped = True
    try:
        setattr(mod, attr, wrapper)
    except Exception:
        pass


def _patch_metadata_calls():
    for attr in ("stat", "lstat", "listdir", "scandir", "access", "readlink"):
        _wrap_counter(os, attr, "fs.os." + attr)
    for attr in ("exists", "isfile", "isdir", "getsize", "islink"):
        _wrap_counter(os.path, attr, "fs.path." + attr)


def _patch_loaders():
    """Native/framework entry points where weight bytes are actually read."""
    cs_patch.patch("safetensors.torch", "load_file", name="weights.safetensors_load_file",
                   cat="weights", argfn=lambda *a, **kw: {
                       "file": str(a[0])[:300] if a else ""}, min_dur=0.0)
    cs_patch.patch("safetensors", "safe_open", name="weights.safe_open",
                   cat="weights", min_dur=0.002)
    cs_patch.patch("torch.serialization", "load", name="weights.torch_load",
                   cat="weights", min_dur=0.002)
    cs_patch.patch("mmap", "mmap.__init__", name="mmap", cat="fileio",
                   min_dur=0.005)


def summarize():
    if not _opened:
        return
    weights = {p: s for p, s in _opened.items()
               if _interesting(p) == "weight"}
    total = sum(s for s in weights.values() if s)
    T.tracer.meta("io.summary",
                  weight_files=len(weights),
                  weight_bytes=total,
                  weight_gib=round(total / (1 << 30), 3) if total else 0,
                  files=[{"path": p, "size": s} for p, s in
                         sorted(weights.items())][:200],
                  config_files=sum(1 for p in _opened
                                   if _interesting(p) == "config"))
