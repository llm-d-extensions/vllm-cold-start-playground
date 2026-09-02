"""PyTorch / accelerator probe: CUDA context creation, collectives init,
torch.compile + Inductor caching, Triton JIT, and CUDA graph capture.

These are the phases people most often mistake for "loading the model":

* ``cuda.lazy_init``  -- first CUDA call creates the context; cost scales with
  driver, MIG/MPS config and how many libraries register kernels. Sub-second on
  a warm node, multiple seconds on a fresh one.
* ``inductor.*``      -- ``torch.compile`` work. A populated FX graph cache
  turns minutes into seconds, so cache hit/miss is the headline number here.
* ``triton.compile``  -- kernel JIT, counted and totalled.
* ``cudagraph.capture`` -- vLLM's graph capture over the batch-size buckets;
  usually the largest single warmup item once compilation is cached.
"""

import cs_patch
import cs_trace as T

_installed = False


def install():
    global _installed
    if _installed or not T.enabled("torch"):
        return
    _installed = True

    cs_patch.after_import("torch", _on_torch)

    # --- CUDA context / device setup ------------------------------------
    cs_patch.patch_many("torch.cuda", [
        ("_lazy_init", "cuda.lazy_init", "gpu"),
        ("init", "cuda.init", "gpu"),
        ("set_device", "cuda.set_device", "gpu"),
        ("synchronize", "cuda.synchronize", "gpu"),
        ("device_count", "cuda.device_count", "gpu"),
        ("is_available", "cuda.is_available", "gpu"),
        ("empty_cache", "cuda.empty_cache", "gpu"),
        ("mem_get_info", "cuda.mem_get_info", "gpu"),
    ], cat="gpu")

    # --- collectives ----------------------------------------------------
    cs_patch.patch_many("torch.distributed.distributed_c10d", [
        ("init_process_group", "dist.init_process_group", "collective"),
        ("new_group", "dist.new_group", "collective"),
        ("barrier", "dist.barrier", "collective"),
        ("_new_process_group_helper", "dist.new_pg_helper", "collective"),
    ], cat="collective")
    cs_patch.patch("torch.distributed.rendezvous", "rendezvous",
                   name="dist.rendezvous", cat="collective")

    # --- compilation ----------------------------------------------------
    cs_patch.patch("torch._inductor.compile_fx", "compile_fx",
                   name="inductor.compile_fx", cat="compile", min_dur=0.0)
    cs_patch.patch("torch._inductor.compile_fx", "compile_fx_inner",
                   name="inductor.compile_fx_inner", cat="compile")
    for mod in ("torch._inductor.codecache", "torch._inductor.output_code"):
        cs_patch.patch(mod, "FxGraphCache.load", name="inductor.fx_cache_load",
                       cat="compile", min_dur=0.0)
        cs_patch.patch(mod, "FxGraphCache.load_with_key",
                       name="inductor.fx_cache_load_key", cat="compile",
                       min_dur=0.0)
    for mod in ("torch._inductor.async_compile", "torch._inductor.codecache"):
        cs_patch.patch(mod, "AsyncCompile.warm_pool",
                       name="inductor.warm_compile_pool", cat="compile")
        cs_patch.patch(mod, "AsyncCompile.wait",
                       name="inductor.wait_for_compile", cat="compile")
    cs_patch.patch("torch._dynamo.eval_frame", "_optimize",
                   name="dynamo.optimize", cat="compile")
    cs_patch.patch("torch._dynamo.convert_frame", "convert_frame",
                   name="dynamo.convert_frame", cat="compile", min_dur=0.05)
    cs_patch.patch("triton.compiler.compiler", "compile",
                   name="triton.compile", cat="compile", min_dur=0.05)
    cs_patch.patch("torch.utils.cpp_extension", "load",
                   name="cpp_extension.load", cat="compile")

    # --- CUDA graphs ----------------------------------------------------
    cs_patch.patch("torch.cuda.graphs", "graph.__enter__",
                   name="cudagraph.capture_begin", cat="cudagraph",
                   min_dur=0.005)
    cs_patch.patch("torch.cuda.graphs", "graph.__exit__",
                   name="cudagraph.capture_end", cat="cudagraph",
                   min_dur=0.005)
    cs_patch.patch("torch.cuda.graphs", "make_graphed_callables",
                   name="cudagraph.make_graphed_callables", cat="cudagraph")

    # --- nvml -----------------------------------------------------------
    for mod in ("pynvml", "nvidia_ml_py", "vllm.third_party.pynvml"):
        cs_patch.patch(mod, "nvmlInit", name="nvml.init", cat="gpu")


def _on_torch(torch):
    """Record build/runtime facts once torch is importable (no CUDA calls:
    touching the driver here would move the cost we are trying to measure)."""
    try:
        info = {
            "torch": getattr(torch, "__version__", "?"),
            "cuda_build": getattr(getattr(torch, "version", None), "cuda", None),
            "hip_build": getattr(getattr(torch, "version", None), "hip", None),
            "git_version": getattr(getattr(torch, "version", None),
                                   "git_version", None),
            "file": getattr(torch, "__file__", None),
        }
        T.tracer.meta("torch.info", **info)
    except Exception as e:
        T.tracer.error("torch.info", e)
