"""Background resource sampler: the timeseries that explains the spans.

A span tells you *that* weight loading took 40s; the sampler tells you whether
those 40s were disk-bound (read_bytes climbing at 300 MB/s), network-bound
(netns rx climbing), CPU-bound (one core pegged, the rest idle -- typical of
single-threaded safetensors conversion), or cgroup-throttled (``nr_throttled``
climbing, the single most common surprise in Kubernetes).

Samples are emitted as Chrome-trace counter events so they render as tracks
under the spans, and are also usable as plain columns for regression plots.

Per-process (every Python process in the tree samples itself):
    cpu_utime/stime, rss, threads, minflt/majflt, /proc/self/io byte counters,
    netns rx/tx, cgroup cpu usage + throttling, page cache size.

Per-node (one elected process only):
    per-GPU utilization, memory used, SM clock, power, temperature via NVML.
"""

import os
import threading
import time

import cs_trace as T

_INTERVAL = T.env_float("SAMPLE_MS", 100.0) / 1000.0
_GPU_INTERVAL = T.env_float("GPU_SAMPLE_MS", 250.0) / 1000.0
_thread = None
_stop = threading.Event()


def install():
    if not T.enabled("sampler"):
        return
    _start()
    T.on_fork(_restart_after_fork)


def _start():
    global _thread
    if _thread is not None and _thread.is_alive():
        return
    _stop.clear()
    _thread = threading.Thread(target=_loop, name="cs-sampler", daemon=True)
    _thread.start()


def _restart_after_fork():
    global _thread
    _thread = None
    _start()


def _cgroup_cpu():
    """cgroup v2 then v1. ``throttled_usec`` > 0 means the container is being
    stopped mid-startup by its CPU limit."""
    try:
        with open("/sys/fs/cgroup/cpu.stat", "r") as fh:
            out = {}
            for line in fh:
                k, _, v = line.partition(" ")
                if k in ("usage_usec", "nr_throttled", "throttled_usec",
                         "nr_periods"):
                    out[k] = int(v)
            if out:
                return out
    except Exception:
        pass
    try:
        out = {}
        with open("/sys/fs/cgroup/cpu,cpuacct/cpu.stat", "r") as fh:
            for line in fh:
                k, _, v = line.partition(" ")
                if k in ("nr_throttled", "throttled_time", "nr_periods"):
                    out[k] = int(v)
        with open("/sys/fs/cgroup/cpu,cpuacct/cpuacct.usage", "r") as fh:
            out["usage_usec"] = int(fh.read().strip()) // 1000
        return out or None
    except Exception:
        return None


def _rss_bytes():
    try:
        with open("/proc/self/statm", "r") as fh:
            return int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except Exception:
        return None


def _elect_gpu_sampler():
    """One process per node samples NVML; the rest skip it. O_EXCL on a file in
    the shared trace dir is enough since all processes share the volume."""
    if not T.env_flag("GPU_SAMPLE", True):
        return False
    try:
        os.makedirs(T.tracer.dir, exist_ok=True)
        path = os.path.join(T.tracer.dir, ".gpu-sampler.lock")
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return True
    except Exception:
        return False


def _nvml_open():
    for mod in ("pynvml", "vllm.third_party.pynvml"):
        try:
            nvml = __import__(mod, fromlist=["*"])
            nvml.nvmlInit()
            n = nvml.nvmlDeviceGetCount()
            handles = [nvml.nvmlDeviceGetHandleByIndex(i) for i in range(n)]
            info = []
            for i, h in enumerate(handles):
                name = nvml.nvmlDeviceGetName(h)
                if isinstance(name, bytes):
                    name = name.decode()
                mem = nvml.nvmlDeviceGetMemoryInfo(h)
                info.append({"index": i, "name": name,
                             "memory_total": getattr(mem, "total", None)})
            T.tracer.meta("gpu.inventory", devices=info, source=mod)
            return nvml, handles
        except Exception:
            continue
    return None, []


def _loop():
    sample_gpu = _elect_gpu_sampler()
    nvml, handles = (_nvml_open() if sample_gpu else (None, []))
    next_gpu = 0.0
    while not _stop.wait(_INTERVAL):
        now = time.monotonic()
        try:
            vals = {}
            st = T.read_proc_stat()
            if st:
                vals.update({"cpu_utime_s": round(st["utime"], 3),
                             "cpu_stime_s": round(st["stime"], 3),
                             "threads": st["num_threads"],
                             "minflt": st["minflt"], "majflt": st["majflt"]})
            rss = _rss_bytes()
            if rss:
                vals["rss_bytes"] = rss
            io = T.read_proc_io()
            if io:
                vals.update({"read_bytes": io.get("read_bytes"),
                             "write_bytes": io.get("write_bytes"),
                             "rchar": io.get("rchar"),
                             "wchar": io.get("wchar"),
                             "syscr": io.get("syscr")})
            net = T.read_net_dev()
            if net:
                vals.update(net)
            cg = _cgroup_cpu()
            if cg:
                vals.update({"cg_" + k: v for k, v in cg.items()})
            if vals:
                T.tracer.counter("proc", vals)
        except Exception:
            pass
        if nvml is not None and now >= next_gpu:
            next_gpu = now + _GPU_INTERVAL
            try:
                for i, h in enumerate(handles):
                    v = {}
                    try:
                        u = nvml.nvmlDeviceGetUtilizationRates(h)
                        v["gpu_util"] = u.gpu
                        v["mem_util"] = u.memory
                    except Exception:
                        pass
                    try:
                        m = nvml.nvmlDeviceGetMemoryInfo(h)
                        v["mem_used"] = m.used
                    except Exception:
                        pass
                    for fn, key in (("nvmlDeviceGetPowerUsage", "power_mw"),
                                    ("nvmlDeviceGetTemperature", "temp_c")):
                        try:
                            f = getattr(nvml, fn)
                            v[key] = f(h, 0) if key == "temp_c" else f(h)
                        except Exception:
                            pass
                    if v:
                        T.tracer.counter("gpu%d" % i, v, cat="gpu")
            except Exception:
                pass


def stop():
    _stop.set()
