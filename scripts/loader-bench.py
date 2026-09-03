"""Time each vLLM-available weight loader end-to-end to GPU, outside vLLM.

One arm per process. Reports GiB/s for the loader's own work: file -> tensors
resident on cuda:0. No vLLM model, no load_weights loop, so this is the
loader's ceiling, not the phase cost.
"""
import glob, os, sys, time
GIB = 1 << 30
SNAP = glob.glob("/cache/hf/hub/models--Qwen--Qwen3-32B/snapshots/*/")[0]
FILES = sorted(glob.glob(os.path.join(SNAP, "*.safetensors")))
TOTAL = sum(os.stat(os.path.realpath(f)).st_size for f in FILES)
arm = sys.argv[1]

import torch
torch.cuda.init()
dev = torch.device("cuda:0")

def report(label, dt, nbytes, ntensors, extra=""):
    print(f"RESULT\t{label}\t{dt:.2f}s\t{nbytes/GIB/dt:.2f} GiB/s\t"
          f"{ntensors} tensors\t{extra}", flush=True)

def natkey(p):
    import re
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p)]

# ---------------------------------------------------------------- H2D ceiling
if arm == "h2d":
    for pin in (True, False):
        buf = torch.empty(2 << 30, dtype=torch.uint8, pin_memory=pin)
        g = torch.empty(2 << 30, dtype=torch.uint8, device=dev)
        g.copy_(buf); torch.cuda.synchronize()          # warm
        t = time.perf_counter()
        for _ in range(4):
            g.copy_(buf)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t
        n = 4 * (2 << 30)
        print(f"RESULT\th2d_{'pinned' if pin else 'pageable'}\t{dt:.2f}s\t"
              f"{n/GIB/dt:.2f} GiB/s", flush=True)
        del buf, g; torch.cuda.empty_cache()
    sys.exit(0)

# --------------------------------------------------- fastsafetensors (current)
if arm.startswith("fst"):
    from fastsafetensors.parallel_loader import ParallelLoader
    from fastsafetensors import SingleGroup
    _, th, bb = arm.split(":")           # fst:<max_threads>:<bbuf_kb>
    t0 = time.perf_counter()
    pl = ParallelLoader(pg=SingleGroup(), hf_weights_files=sorted(FILES, key=natkey),
                        queue_size=2, use_tqdm_on_load=False, device=str(dev),
                        nogds=True, max_threads=int(th), bbuf_size_kb=int(bb))
    setup = time.perf_counter() - t0
    n = cnt = 0
    t = time.perf_counter()
    for name, tensor in pl.iterate_weights():
        assert tensor.device.type == "cuda", tensor.device
        n += tensor.numel() * tensor.element_size(); cnt += 1
    torch.cuda.synchronize()
    dt = time.perf_counter() - t
    pl.close()
    report(arm, dt, n, cnt, f"setup={setup:.2f}s dst=cuda")
    sys.exit(0)

# ------------------------------------------------------------- runai streamer
if arm.startswith("runai"):
    from runai_model_streamer import SafetensorsStreamer
    device = arm.split(":")[1]           # runai:cpu | runai:cuda
    parts = arm.split(":")               # runai:<dev>[:conc[:chunk_kib]]
    conc = parts[2] if len(parts) > 2 and parts[2] else None
    chunk = parts[3] if len(parts) > 3 and parts[3] else None
    if conc: os.environ["RUNAI_STREAMER_CONCURRENCY"] = conc
    if chunk: os.environ["RUNAI_STREAMER_CHUNK_BYTESIZE"] = str(int(chunk) * 1024)
    os.environ.setdefault("RUNAI_STREAMER_MEMORY_LIMIT", "-1")
    dst = "cpu" if device == "cpu" else "cuda:0"
    n = cnt = 0
    t = time.perf_counter()
    with SafetensorsStreamer() as s:
        s.stream_files(FILES, device=dst, is_distributed=False)
        for name, tensor in s.get_tensors():
            c = tensor if os.environ.get("NOCLONE") else tensor.clone()
            n += c.numel() * c.element_size(); cnt += 1
            del c
    torch.cuda.synchronize()
    dt = time.perf_counter() - t
    report(arm, dt, n, cnt, f"dst={dst} conc={conc or 'default'} chunk={chunk or 'default'}KiB")
    sys.exit(0)

# --------------------------------------------- stock safetensors (auto / hf)
if arm.split(":")[0] == "st":
    from safetensors.torch import safe_open, load_file
    mode = arm.split(":")[1]             # st:lazy | st:eager
    n = cnt = 0
    t = time.perf_counter()
    if mode == "lazy":
        for f in FILES:
            with safe_open(f, framework="pt") as h:
                for k in h.keys():
                    x = h.get_tensor(k).to(dev, non_blocking=True)
                    n += x.numel() * x.element_size(); cnt += 1
                    del x
    else:
        for f in FILES:
            sd = load_file(f, device="cpu")
            for k in list(sd):
                x = sd.pop(k).to(dev, non_blocking=True)
                n += x.numel() * x.element_size(); cnt += 1
                del x
    torch.cuda.synchronize()
    dt = time.perf_counter() - t
    report(arm, dt, n, cnt, f"safetensors {mode} -> cuda")
    sys.exit(0)


# ------------------------------------- vLLM's multi-thread lazy safetensors arm
# What `--load-format auto --model-loader-extra-config '{"enable_multithread_load":
# true, "num_threads": N}'` runs: weight_utils.multi_thread_safetensors_weights_
# iterator, i.e. safetensors.torch.load_file(device="cpu") over a ThreadPool,
# yielding host tensors. The H2D is then the model's own per-tensor
# param.data.copy_(), pageable -- so the honest arm is threaded read + .to(dev).
if arm.startswith("mtst:"):
    import concurrent.futures
    from safetensors.torch import load_file
    nw = int(arm.split(":")[1])
    t = time.perf_counter(); n = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=nw) as ex:
        futs = [ex.submit(load_file, f, device="cpu") for f in FILES]
        for fu in concurrent.futures.as_completed(futs):
            sd = fu.result()
            for k in list(sd):
                g = sd.pop(k).to(dev, non_blocking=True); n += 1
    torch.cuda.synchronize()
    report(arm, time.perf_counter() - t, TOTAL, n, f"threads={nw} -> cuda (pageable)")

# ------------------------------------------ safetensors prefetch strategy arm
# `--safetensors-load-strategy prefetch`: a daemon thread reads the files into
# the page cache while the lazy mmap iterator walks them. On this pod the cache
# is already 100% warm, so this measures the strategy's overhead, not its win.
if arm.startswith("stpf:"):
    from safetensors import safe_open
    nth, blk = arm.split(":")[1:3]
    from vllm.model_executor.model_loader.weight_utils import (
        safetensors_weights_iterator)
    t = time.perf_counter(); n = 0
    for name, w in safetensors_weights_iterator(
            FILES, False, "prefetch",
            safetensors_prefetch_num_threads=int(nth),
            safetensors_prefetch_block_size=int(blk) * 1024):
        g = w.to(dev, non_blocking=True); n += 1
    torch.cuda.synchronize()
    report(arm, time.perf_counter() - t, TOTAL, n,
           f"prefetch threads={nth} blk={blk}KiB -> cuda")

# --------------------------------------------------------- bit-level verify arm
# `verify:<loader-arm>` re-runs a loader and fingerprints every tensor it
# yields, so two loaders can be compared for bit-identical output. first_token
# in the boot harness only proves the server answers; this proves the weights
# are the same bytes. Two independent byte sums (whole buffer, and a stride-1021
# subsample so a permuted/shifted buffer cannot alias) plus shape and dtype.
if arm.startswith("verify:"):
    import json, hashlib
    sub = arm.split(":", 1)[1]
    def fp(t):
        b = t.detach().view(torch.uint8).reshape(-1)
        return [int(b.to(torch.int64).sum()), int(b[::1021].to(torch.int64).sum())]
    out = {}
    if sub.startswith("fst"):
        _, th, bb = sub.split(":")
        from fastsafetensors.parallel_loader import ParallelLoader
        from fastsafetensors import SingleGroup
        pl = ParallelLoader(pg=SingleGroup(), hf_weights_files=sorted(FILES, key=natkey),
                            queue_size=2, use_tqdm_on_load=False, device=str(dev),
                            nogds=True, max_threads=int(th), bbuf_size_kb=int(bb))
        for name, w in pl.iterate_weights():
            out[name] = [list(w.shape), str(w.dtype)] + fp(w)
        pl.close()
    elif sub.startswith("runai"):
        dst = sub.split(":")[1]
        os.environ["RUNAI_STREAMER_MEMORY_LIMIT"] = "-1"
        from runai_model_streamer import SafetensorsStreamer
        with SafetensorsStreamer() as s:
            s.stream_files(FILES, device=(str(dev) if dst == "cuda" else "cpu"),
                           is_distributed=False)
            for name, w in s.get_tensors():
                out[name] = [list(w.shape), str(w.dtype)] + fp(w.clone())
    elif sub.startswith("it"):
        import instanttensor
        with instanttensor.safe_open(sorted(FILES, key=natkey), framework="pt",
                                     device=str(dev), process_group=None,
                                     copy=True) as f:
            for name, w in f.tensors():
                out[name] = [list(w.shape), str(w.dtype)] + fp(w)
    elif sub == "st":
        from safetensors.torch import load_file
        for f in FILES:
            for name, w in load_file(f, device="cpu").items():
                out[name] = [list(w.shape), str(w.dtype)] + fp(w)
    path = f"/tmp/verify-{sub.replace(':', '_')}.json"
    json.dump(out, open(path, "w"), sort_keys=True)
    blob = json.dumps(out, sort_keys=True).encode()
    print(f"RESULT\tverify\t{sub}\t{len(out)} tensors\t"
          f"sha256={hashlib.sha256(blob).hexdigest()[:16]}\t{path}", flush=True)

# ------------------------------------------------------------- instanttensor
# it:<backend>:<concurrency>:<chunk_kib>:<copy>
#
# The only loader here that was designed for this shape of problem: it picks an
# io_uring or libaio backend, keeps a ring buffer in GPU memory, and writes
# straight to the device with no host staging tensor. `backend` matters a lot in
# this pod -- the default candidate list is [URING, AIO], both O_DIRECT, and the
# floor measurement has O_DIRECT at 11.36 GiB/s against buffered's 28.52 on a
# page-cache-warm GPFS mount. BackendPolicy.BUFFERED is the interesting arm.
#
# copy=0 is the zero-copy path vLLM does NOT take: its iterator hardcodes
# copy=True because it yields tensors the caller keeps. The difference is one
# clone of all 61 GiB, so it is worth pricing separately.
if arm.split(":")[0] == "it":
    import instanttensor
    parts = (arm.split(":") + ["auto", "0", "0", "1"])[1:5]
    be, conc, chunk, cp = parts

    def _backend(tok):
        for holder in (instanttensor.Backend, instanttensor.BackendPolicy):
            if hasattr(holder, tok):
                return getattr(holder, tok)
        raise SystemExit(f"no such instanttensor backend/policy: {tok}")

    kw = {}
    if be != "auto":
        kw["backend"] = [_backend(t) for t in be.split("+")]
    if int(conc):
        kw["concurrency"] = int(conc)
    if int(chunk):
        kw["chunk_size"] = int(chunk) * 1024
    # IT_WORK=1 makes the consumer behave like vLLM's load_weights loop instead
    # of a pure accounting loop: one device-to-device copy_ into a preallocated
    # destination per tensor, which is what `weight_loader(param, loaded_weight)`
    # ultimately does.
    #
    # It was added to test a hypothesis that MEASUREMENT REFUTED, and it is kept
    # as the refutation. The hypothesis: tensors() calls a blanket
    # torch.cuda.current_stream().synchronize() before EVERY yield
    # (_impl.py:723), so consumer work on that stream cannot overlap the loader's
    # fill of the next tensor, and that serialisation should show up as a gap
    # between this arm and the plain one. It does not. An extra full 61 GiB of
    # device-to-device copies is free: 5.50s vs 5.45s, and 5.46s vs 5.54s on a
    # rerun -- inside the noise, in both directions. The per-tensor sync costs
    # nothing here because the copy is trivially fast next to the read.
    #
    # The gap it was meant to explain was my own accounting error: I had compared
    # this rig's *load* leg against vLLM's whole weight-load *phase*. Matched up
    # properly there is no gap -- lt1-it-c2 weights.load_weights is 6.26s against
    # 5.45s load + 0.71s setup = 6.16s here -- and the phase residue is
    # loader-independent runner_load_model (~1.0s) + initialize_model (~0.37s).
    work = os.environ.get("IT_WORK") == "1"
    n = cnt = 0
    t0 = time.perf_counter()
    with instanttensor.safe_open(sorted(FILES, key=natkey), framework="pt",
                                device=str(dev), process_group=None,
                                copy=(cp == "1"), **kw) as f:
        setup = time.perf_counter() - t0
        t = time.perf_counter()
        for name, tensor in f.tensors():
            assert tensor.device.type == "cuda", tensor.device
            if work:
                torch.empty_like(tensor).copy_(tensor)
            n += tensor.numel() * tensor.element_size(); cnt += 1
        torch.cuda.synchronize()
        dt = time.perf_counter() - t
    report(f"it_{be}_c{conc}_k{chunk}_copy{cp}{'_work' if work else ''}",
           dt, n, cnt, f"setup={setup:.2f}s total={TOTAL/GIB:.2f}GiB")
    sys.exit(0)

# ------------------------------------------------------------------ tensorizer
# tz:<path>  -- deserialize an already-serialized tensorizer artifact to cuda:0.
# Unlike every other arm this needs a conversion pass first (scripts/tensorize.py
# via vLLM's serialize_vllm_model), so the artifact path is an argument.
if arm.split(":")[0] == "tz":
    from tensorizer import TensorDeserializer
    path = arm.split(":", 1)[1]
    t0 = time.perf_counter()
    des = TensorDeserializer(path, device=dev, lazy_load=False,
                             num_readers=int(os.environ.get("TZ_READERS", "8")))
    setup = time.perf_counter() - t0
    t = time.perf_counter()
    n = cnt = 0
    for name in des.keys():
        w = des[name]
        assert w.device.type == "cuda", w.device
        n += w.numel() * w.element_size(); cnt += 1
    torch.cuda.synchronize()
    dt = time.perf_counter() - t
    des.close()
    report("tz", dt, n, cnt, f"setup={setup:.2f}s artifact={os.stat(path).st_size/GIB:.2f}GiB")
    sys.exit(0)

if not arm.split(":")[0] in ("h2d", "fst", "runai", "st", "mtst", "stpf",
                             "verify", "it", "tz"):
    raise SystemExit(f"unknown arm {arm}")
