"""Run context capture: everything you need to know before comparing two runs.

Cold-start numbers are meaningless without the context that produced them. This
records the facts that most often explain a difference between two otherwise
identical runs:

* **CPU budget** -- cgroup ``cpu.max``. Model loading, safetensors conversion
  and Inductor compilation are all CPU-heavy; a 2-core limit can double
  time-to-ready versus 16 cores on the same GPU.
* **Filesystem behind the model** -- overlayfs vs NFS vs CSI block vs tmpfs,
  taken from ``/proc/self/mountinfo`` for the actual resolved path.
* **Cache state** -- size and file count of the HF cache, the vLLM cache root,
  the Inductor FX cache and the Triton cache. A "cold start" with a populated
  Inductor cache is a different experiment from one without.
* **Page cache** -- how much of the node's RAM holds file pages, which decides
  whether weight reads come from memory or from storage.
* **Env knobs** -- ``VLLM_*``, ``HF_*``, ``TORCH*``, ``NCCL_*``, ``OMP_*``, with
  secret-looking values redacted.
"""

import os
import sys
import threading
import time

import cs_trace as T

_PREFIXES = ("VLLM_", "HF_", "HUGGING", "TRANSFORMERS_", "TORCH", "CUDA",
             "NCCL_", "NVIDIA_", "TRITON_", "OMP_", "MKL_", "SAFETENSORS_",
             "TOKENIZERS_", "PYTHON", "LD_", "CS_", "RAY_", "GLOO_",
             "MODEL", "POD_", "NODE_", "KUBERNETES_", "XDG_", "HOME",
             "VLLM_CACHE_ROOT", "TORCHINDUCTOR_", "TORCHDYNAMO_")
_SECRET = ("TOKEN", "SECRET", "PASSWORD", "PASSWD", "KEY", "CRED", "AUTH")


def _redact(k, v):
    up = k.upper()
    if any(s in up for s in _SECRET) and v:
        return "<redacted:%d chars>" % len(v)
    return v[:400]


def _env_snapshot():
    return {k: _redact(k, v) for k, v in sorted(os.environ.items())
            if k.upper().startswith(_PREFIXES)}


def _read(path, limit=4096):
    try:
        with open(path, "r") as fh:
            return fh.read(limit).strip()
    except Exception:
        return None


def _cgroup_limits():
    out = {}
    v2 = _read("/sys/fs/cgroup/cpu.max")
    if v2:
        parts = v2.split()
        try:
            quota = parts[0]
            period = int(parts[1]) if len(parts) > 1 else 100000
            out["cpu_max"] = v2
            out["cpu_limit_cores"] = (None if quota == "max"
                                      else round(int(quota) / period, 3))
        except Exception:
            pass
        out["memory_max"] = _read("/sys/fs/cgroup/memory.max")
        out["memory_current"] = _read("/sys/fs/cgroup/memory.current")
        out["cgroup_version"] = 2
    else:
        q = _read("/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_quota_us")
        p = _read("/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_period_us")
        try:
            if q and p and int(q) > 0:
                out["cpu_limit_cores"] = round(int(q) / int(p), 3)
        except Exception:
            pass
        out["memory_max"] = _read(
            "/sys/fs/cgroup/memory/memory.limit_in_bytes")
        out["cgroup_version"] = 1
    return out


def _mounts():
    """path prefix -> (fstype, source, options) from /proc/self/mountinfo."""
    out = []
    try:
        with open("/proc/self/mountinfo", "r") as fh:
            for line in fh:
                f = line.split()
                try:
                    sep = f.index("-")
                except ValueError:
                    continue
                out.append((f[4], f[sep + 1], f[sep + 2],
                            f[5] if len(f) > 5 else ""))
    except Exception:
        pass
    return out


def fs_for_path(path, mounts=None):
    """Filesystem type serving ``path`` (longest matching mount point)."""
    mounts = mounts or _mounts()
    try:
        real = os.path.realpath(path)
    except Exception:
        real = path
    best = None
    for mp, fstype, source, opts in mounts:
        if real == mp or real.startswith(mp.rstrip("/") + "/"):
            if best is None or len(mp) > len(best[0]):
                best = (mp, fstype, source, opts)
    if best is None:
        return None
    return {"mountpoint": best[0], "fstype": best[1], "source": best[2],
            "options": best[3]}


def dir_stats(path, deadline_s=0.5, max_entries=200000):
    """Bounded size/count of a cache directory. Bounded because a populated HF
    cache can hold hundreds of thousands of files and we must not spend the
    measurement budget measuring."""
    if not path or not os.path.isdir(path):
        return None
    t_end = time.monotonic() + deadline_s
    files = 0
    size = 0
    truncated = False
    for root, dirs, names in os.walk(path):
        for n in names:
            files += 1
            try:
                size += os.lstat(os.path.join(root, n)).st_size
            except Exception:
                pass
            if files >= max_entries:
                truncated = True
                break
        if truncated or time.monotonic() > t_end:
            truncated = True
            break
    return {"path": path, "files": files, "bytes": size,
            "gib": round(size / (1 << 30), 3), "truncated": truncated,
            "fs": fs_for_path(path)}


def _cache_dirs():
    home = os.environ.get("HOME", "/root")
    hf = os.environ.get("HF_HOME") or os.path.join(home, ".cache/huggingface")
    return {
        "hf_home": hf,
        "hf_hub": os.environ.get("HF_HUB_CACHE") or os.path.join(hf, "hub"),
        "vllm_cache": os.environ.get("VLLM_CACHE_ROOT")
        or os.path.join(home, ".cache/vllm"),
        "inductor_cache": os.environ.get("TORCHINDUCTOR_CACHE_DIR")
        or os.path.join("/tmp", "torchinductor_" + _user()),
        "triton_cache": os.environ.get("TRITON_CACHE_DIR")
        or os.path.join(home, ".triton/cache"),
        # FlashInfer has used both names across releases, and the manifests set
        # both; honour either so the cache is never reported as absent when it
        # is merely under the other variable.
        "flashinfer_cache": os.environ.get("FLASHINFER_CACHE_DIR")
        or os.environ.get("FLASHINFER_WORKSPACE_BASE")
        or os.path.join(home, ".cache/flashinfer"),
    }


def _user():
    try:
        import getpass
        return getpass.getuser()
    except Exception:
        return str(os.getuid())


def _gpu_facts():
    out = {}
    drv = _read("/proc/driver/nvidia/version")
    if drv:
        out["nvidia_driver"] = drv.splitlines()[0]
    try:
        devs = sorted(d for d in os.listdir("/dev")
                      if d.startswith("nvidia") and d[6:].isdigit())
        out["dev_nodes"] = devs
    except Exception:
        pass
    for k in ("NVIDIA_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        if k in os.environ:
            out[k] = os.environ[k]
    return out


def _cpu_facts():
    out = {"cpu_count": os.cpu_count()}
    try:
        out["affinity"] = len(os.sched_getaffinity(0))
    except Exception:
        pass
    info = _read("/proc/cpuinfo", 65536) or ""
    for line in info.splitlines():
        if line.lower().startswith("model name"):
            out["model"] = line.split(":", 1)[1].strip()
            break
    out["loadavg"] = _read("/proc/loadavg")
    return out


def capture(full=None):
    """Emit the run context.

    Only the ``os.environ`` snapshot is taken synchronously -- vLLM mutates its
    own environment during startup (worker multiproc method, per-worker
    ``CUDA_VISIBLE_DEVICES``), so a late snapshot would not be the environment
    the process actually started with. Everything else (mountinfo parsing,
    cgroup and cpuinfo reads, cache directory scans) runs on a daemon thread to
    keep instrumentation off the critical path.

    ``full`` (cache directory scans) defaults to only the first process on the
    node, elected with an O_EXCL lock file.
    """
    if not T.enabled("env"):
        return
    try:
        env_snap = _env_snapshot()
        if full is None:
            full = _elect()
        threading.Thread(target=_capture_slow, args=(env_snap, full),
                         name="cs-env", daemon=True).start()
    except Exception as e:
        T.tracer.error("env.capture", e)


def _capture_slow(env_snap, full):
    try:
        mounts = _mounts()
        model = os.environ.get("CS_MODEL") or os.environ.get("MODEL_PATH")
        ctx = {
            "run_id": T.tracer.run_id,
            "kernel": " ".join(os.uname()) if hasattr(os, "uname") else "",
            "cpu": _cpu_facts(),
            "cgroup": _cgroup_limits(),
            "gpu": _gpu_facts(),
            "python": {"version": sys.version.split()[0],
                       "executable": sys.executable,
                       "dont_write_bytecode": bool(sys.dont_write_bytecode),
                       "path_entries": len(sys.path),
                       "pythonpath": os.environ.get("PYTHONPATH", "")},
            "env": env_snap,
            "page_cache_bytes": T.read_meminfo_cached(),
            "mounts_of_interest": {},
        }
        for label, path in _cache_dirs().items():
            ctx["mounts_of_interest"][label] = fs_for_path(path, mounts)
        if model:
            ctx["mounts_of_interest"]["model"] = fs_for_path(model, mounts)
        T.tracer.meta("run.context", **ctx)
        if full:
            _capture_caches(model)
    except Exception as e:
        T.tracer.error("env.capture_slow", e)


def _capture_caches(model=None):
    try:
        t0 = time.monotonic()
        caches = {label: dir_stats(path)
                  for label, path in _cache_dirs().items()}
        if model and os.path.isdir(model):
            caches["model"] = dir_stats(model)
        T.tracer.meta("run.caches", caches=caches,
                      scan_s=round(time.monotonic() - t0, 3),
                      note="measured on a background thread after startup "
                           "began; sizes are as of scan time, not t0")
    except Exception as e:
        T.tracer.error("env.capture_caches", e)


def install_title_probe():
    """vLLM calls ``setproctitle`` to name its subprocesses (``VLLM::EngineCore``,
    ``VLLM::Worker_TP0``, ...) *after* they start. Capturing that gives the
    report real process labels instead of "spawned"."""
    import cs_patch

    def _apply(module):
        orig = getattr(module, "setproctitle", None)
        if orig is None or getattr(orig, "_cs_wrapped", False):
            return

        def setproctitle(title, *a, **kw):
            try:
                T.tracer.meta("process.title", title=str(title)[:200])
                T.tracer.role = _role_from_title(str(title)) or T.tracer.role
            except Exception:
                pass
            return orig(title, *a, **kw)
        setproctitle._cs_wrapped = True
        try:
            module.setproctitle = setproctitle
        except Exception:
            pass

    cs_patch.after_import("setproctitle", _apply)


def _role_from_title(title):
    t = title.lower()
    if "enginecore" in t:
        return "engine_core"
    if "worker" in t:
        return "worker"
    if "vllm::" in t:
        return t.split("vllm::", 1)[1].split()[0][:40]
    return None


def _elect():
    try:
        os.makedirs(T.tracer.dir, exist_ok=True)
        fd = os.open(os.path.join(T.tracer.dir, ".env-capture.lock"),
                     os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        os.close(fd)
        return True
    except Exception:
        return False
