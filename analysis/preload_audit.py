"""Which modules can a forkserver preload without poisoning its children?

Run inside the vLLM container (it needs the real GPU and the real wheel):

    python3 analysis/preload_audit.py                 # audit the default list
    python3 analysis/preload_audit.py --modules a,b   # audit specific modules
    python3 analysis/preload_audit.py --json out.json

The forkserver's whole safety argument is that it is CUDA-free, so every
candidate is judged twice, because the two answers can differ and the
difference matters:

  * ``predicate``  -- ``torch.cuda.is_initialized()``, which is what vLLM's own
    ``_maybe_force_spawn`` consults (``utils/system_utils.py``). If this is
    true, vLLM reverts to spawn and the preload is pointless.
  * ``ground_truth`` -- fork a child and make it allocate on the GPU. This is
    the property we actually need. A module can leave the torch predicate false
    and still have created a CUDA context through cupy, a ctypes driver call or
    a C++ extension's static initialiser -- in which case the fork *looks* safe
    and the child dies at first use.

When CUDA does get initialised we want the culprit, not just the verdict, so
``torch.cuda.init`` and ``torch.cuda._lazy_init`` are wrapped to record the
first stack that reaches them. That stack is the actionable output: it names
the import-time line to make lazy.

Each module is measured in a *fresh interpreter*, so costs are comparable and
one poisoned candidate cannot contaminate the next.
"""

import argparse
import json
import os
import subprocess
import sys
import time

# The child's own import graph, from runs/fs-early-r2 (see docs/experiments.md).
# Ordered as the chain nests, so a bisect down the list finds the exact link
# where CUDA appears rather than only that "gpu_worker" is unsafe.
CANDIDATES = [
    # --- known-good baseline, for calibration -------------------------------
    "vllm.v1.engine.core",
    "vllm.v1.engine.async_llm",
    # --- the eager chain: 5.36s of the child's 7.01s import wall -----------
    "vllm.v1.worker.gpu_worker",
    "vllm.model_executor.warmup.kernel_warmup",
    "vllm.model_executor.warmup.b12x_warmup",
    "vllm.model_executor.kernels.linear",
    "vllm.model_executor.kernels.linear.mixed_precision",
    "vllm.model_executor.kernels.linear.mxfp8.humming",
    "vllm.model_executor.kernels.linear.mxfp4.humming",
    "vllm.model_executor.kernels.linear.nvfp4.cutlass",
    "vllm.model_executor.warmup.deep_gemm_warmup",
    "vllm.model_executor.warmup.flashinfer_sparse_mla_warmup",
    "vllm.v1.worker.gpu.warmup",
    "vllm.v1.worker.gpu.model_runner",
    "vllm.v1.worker.utils",
    "vllm.v1.worker.gpu.attn_utils",
    "vllm.v1.executor.uniproc_executor",
    "vllm.model_executor.model_loader",
    "vllm.model_executor.layers.quantization.fp8",
    "vllm.v1.sample.ops.topk_topp_sampler",
    "vllm.lora.layers",
    "vllm._custom_ops",
    "vllm._aiter_ops",
    "vllm.compilation.backends",
    # --- the lazy tail: 1.66s paid during compile/warmup -------------------
    "tilelang",
    "tvm",
    "flashinfer",
    "vllm.models.kimi_k3",
    "torch.fx.experimental.validator",
    "sympy.tensor.tensor",
    "huggingface_hub.hf_file_system",
    # --- third-party leaves seen in the child's graph ----------------------
    "cupy",
    "deep_ep",
    "z3",
]

# Runs in the fresh interpreter. Kept as a string rather than a helper module
# so the audit needs nothing on PYTHONPATH inside the container.
CHILD = r'''
import json, os, sys, time, traceback

MOD = sys.argv[1]
first_cuda = []

def _instrument():
    """Wrap torch's CUDA-init entry points to capture the first caller."""
    t = sys.modules.get("torch")
    if t is None:
        return
    for owner, attr in ((t.cuda, "init"), (t.cuda, "_lazy_init")):
        fn = getattr(owner, attr, None)
        if fn is None or getattr(fn, "_audited", False):
            continue
        def wrap(fn=fn, attr=attr):
            def inner(*a, **k):
                if not first_cuda:
                    first_cuda.append({
                        "via": attr,
                        # drop our own wrapper frame; keep the vllm frames
                        "stack": [l.rstrip() for l in
                                  traceback.format_stack()[:-1][-14:]],
                    })
                return fn(*a, **k)
            inner._audited = True
            return inner
        setattr(owner, attr, wrap())

t0 = time.monotonic()
err = None
try:
    # Import torch first *only* to instrument it, then measure MOD's own cost
    # separately -- otherwise every candidate is charged torch's ~5s and the
    # numbers say nothing about the candidate.
    import torch
    _instrument()
    t_torch = time.monotonic() - t0
    n_torch = len(sys.modules)
    t1 = time.monotonic()
    __import__(MOD)
    t_mod = time.monotonic() - t1
except BaseException as e:
    err = "%s: %s" % (type(e).__name__, e)
    t_mod = time.monotonic() - t0
    t_torch = n_torch = None

predicate = None
try:
    predicate = bool(sys.modules["torch"].cuda.is_initialized())
except Exception:
    pass

# Ground truth: can a forked child still use the GPU? This is the property the
# forkserver needs; the predicate above is only vLLM's proxy for it.
ground = None
detail = None
if err is None:
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r)
        try:
            import torch as _t
            _t.zeros(8, device="cuda")
            _t.cuda.synchronize()
            os.write(w, b"ok")
        except BaseException as e:
            os.write(w, ("%s: %s" % (type(e).__name__, e))[:400].encode())
        finally:
            os._exit(0)
    os.close(w)
    out = b""
    while True:
        c = os.read(r, 4096)
        if not c:
            break
        out += c
    os.close(r)
    os.waitpid(pid, 0)
    detail = out.decode("utf-8", "replace")
    ground = (detail == "ok")

print("@@AUDIT@@" + json.dumps({
    "module": MOD, "error": err,
    "import_s": t_mod, "torch_s": t_torch,
    "modules_after_torch": n_torch, "modules": len(sys.modules),
    "cuda_predicate": predicate, "cuda_fork_ok": ground,
    "fork_detail": None if ground else detail,
    "first_cuda": first_cuda[0] if first_cuda else None,
}))
'''


def audit(module, timeout):
    t0 = time.monotonic()
    try:
        p = subprocess.run([sys.executable, "-c", CHILD, module],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"module": module, "error": "timeout after %ss" % timeout,
                "wall_s": time.monotonic() - t0}
    for line in p.stdout.decode("utf-8", "replace").splitlines():
        if line.startswith("@@AUDIT@@"):
            r = json.loads(line[len("@@AUDIT@@"):])
            r["wall_s"] = time.monotonic() - t0
            return r
    # A hard crash (segfault, C++ abort) never reaches the print. That is
    # itself a verdict: unusable as a preload.
    tail = p.stderr.decode("utf-8", "replace").strip().splitlines()[-3:]
    return {"module": module, "error": "no verdict (rc=%d)" % p.returncode,
            "stderr": tail, "wall_s": time.monotonic() - t0}


def verdict(r):
    if r.get("error"):
        return "ERROR"
    if r.get("cuda_fork_ok") is False:
        return "POISONS" if r.get("cuda_predicate") else "POISONS(silent)"
    if r.get("cuda_predicate"):
        return "CUDA-INIT"
    return "safe"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modules", help="comma list, overrides the default set")
    ap.add_argument("--json", help="write the full result here")
    ap.add_argument("--timeout", type=float, default=300.0)
    a = ap.parse_args()
    mods = ([m.strip() for m in a.modules.split(",") if m.strip()]
            if a.modules else CANDIDATES)

    out = []
    print("%-52s %-15s %8s %7s" % ("module", "verdict", "import", "modules"))
    print("-" * 88)
    for m in mods:
        r = audit(m, a.timeout)
        out.append(r)
        v = verdict(r)
        imp = r.get("import_s")
        print("%-52s %-15s %8s %7s" % (
            m[:52], v,
            "-" if imp is None else "%.2fs" % imp,
            r.get("modules") or "-"))
        if r.get("error"):
            print("     ! %s" % r["error"])
            for l in r.get("stderr") or []:
                print("       %s" % l[:110])
        if r.get("fork_detail") and not r.get("cuda_fork_ok"):
            print("     fork: %s" % r["fork_detail"].strip().splitlines()[0][:110])
        fc = r.get("first_cuda")
        if fc:
            # only the frames outside torch: those are the fixable ones
            frames = [l for l in fc["stack"]
                      if "/torch/" not in l and l.strip().startswith("File")]
            for l in (frames or fc["stack"])[-3:]:
                print("     %s" % l.strip()[:110])
        sys.stdout.flush()

    safe = [r["module"] for r in out if verdict(r) == "safe"]
    print("\n%d/%d safe to preload" % (len(safe), len(out)))
    for r in out:
        if verdict(r) != "safe":
            print("  %-14s %s" % (verdict(r), r["module"]))
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(out, fh, indent=1)
        print("\nwrote %s" % a.json)


if __name__ == "__main__":
    main()
