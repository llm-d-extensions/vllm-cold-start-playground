"""vLLM phase probe.

A declarative patch table maps vLLM internals to canonical cold-start phases.
Paths move between vLLM releases, so every entry is optional: whatever exists
in the installed version gets a span, the rest is reported as ``not_applied``
by ``cs_patch.coverage()`` so a run is never silently under-instrumented and
never fails because of the probe.

Both engine generations are covered (``vllm.v1.*`` and the legacy
``vllm.engine.llm_engine`` path), plus the pieces around the engine that are
easy to forget and routinely expensive:

* platform resolution and plugin discovery (entry-point scanning)
* HF config/tokenizer resolution (network round trips)
* ZMQ/shared-memory handshakes between API server, EngineCore and workers
* usage telemetry (an outbound HTTPS call on the startup path)
* uvicorn/FastAPI app construction and startup
"""

import cs_patch
import cs_trace as T

_installed = False

# (module, attribute path, span name, category, kind)
PATCHES = [
    # ---- entrypoint / API server -------------------------------------
    ("vllm.entrypoints.cli.main", "main", "cli.main", "entrypoint", None),
    ("vllm.entrypoints.openai.api_server", "run_server",
     "api.run_server", "entrypoint", None),
    ("vllm.entrypoints.openai.api_server", "run_server_worker",
     "api.run_server_worker", "entrypoint", None),
    ("vllm.entrypoints.openai.api_server", "build_async_engine_client",
     "api.build_engine_client", "engine", "acm"),
    ("vllm.entrypoints.openai.api_server",
     "build_async_engine_client_from_engine_args",
     "api.build_engine_client_from_args", "engine", "acm"),
    ("vllm.entrypoints.openai.api_server", "build_app",
     "api.build_app", "apiserver", None),
    ("vllm.entrypoints.openai.api_server", "init_app_state",
     "api.init_app_state", "apiserver", None),
    ("vllm.entrypoints.openai.api_server", "create_server_socket",
     "api.create_socket", "apiserver", None),
    ("vllm.entrypoints.launcher", "serve_http", "api.serve_http",
     "apiserver", None),
    ("vllm.entrypoints.openai.serving_models", "OpenAIServingModels.init_static_loras",
     "api.init_static_loras", "apiserver", None),

    # ---- config resolution -------------------------------------------
    ("vllm.engine.arg_utils", "EngineArgs.create_engine_config",
     "config.create_engine_config", "config", None),
    ("vllm.engine.arg_utils", "AsyncEngineArgs.create_engine_config",
     "config.create_async_engine_config", "config", None),
    ("vllm.transformers_utils.config", "get_config", "config.get_hf_config",
     "config", None),
    ("vllm.transformers_utils.config", "try_get_generation_config",
     "config.get_generation_config", "config", None),
    ("vllm.transformers_utils.config", "get_hf_image_processor_config",
     "config.get_image_processor_config", "config", None),
    ("vllm.transformers_utils.tokenizer", "get_tokenizer",
     "config.get_tokenizer", "config", None),
    ("vllm.platforms", "resolve_current_platform_cls_qualname",
     "platform.resolve", "platform", None),
    ("vllm.plugins", "load_general_plugins", "plugins.load", "platform", None),
    ("vllm.plugins", "load_plugins_by_group", "plugins.load_group",
     "platform", None),
    ("vllm.usage.usage_lib", "UsageMessage.report_usage",
     "telemetry.report_usage", "telemetry", None),
    ("vllm.usage.usage_lib", "UsageMessage._report_usage_worker",
     "telemetry.report_worker", "telemetry", None),

    # ---- v1 engine ---------------------------------------------------
    ("vllm.v1.engine.async_llm", "AsyncLLM.__init__", "engine.async_llm_init",
     "engine", None),
    ("vllm.v1.engine.async_llm", "AsyncLLM.from_vllm_config",
     "engine.async_llm_from_config", "engine", None),
    ("vllm.v1.engine.llm_engine", "LLMEngine.from_vllm_config",
     "engine.llm_engine_from_config", "engine", None),
    ("vllm.v1.engine.core", "EngineCore.__init__", "engine.core_init",
     "engine", None),
    ("vllm.v1.engine.core", "EngineCore._initialize_kv_caches",
     "kvcache.initialize", "kvcache", None),
    ("vllm.v1.engine.core", "EngineCoreProc.__init__", "engine.core_proc_init",
     "engine", None),
    ("vllm.v1.engine.core", "EngineCoreProc.run_engine_core",
     "engine.run_engine_core", "engine", None),
    ("vllm.v1.engine.core_client", "MPClient.__init__",
     "engine.mp_client_init", "ipc", None),
    ("vllm.v1.engine.core_client", "AsyncMPClient.__init__",
     "engine.async_mp_client_init", "ipc", None),
    ("vllm.v1.engine.utils", "launch_core_engines", "engine.launch_cores",
     "ipc", None),
    ("vllm.v1.engine.utils", "CoreEngineProcManager.__init__",
     "engine.core_proc_manager", "ipc", None),
    ("vllm.v1.engine.utils", "wait_for_engine_startup",
     "engine.wait_for_startup", "ipc", None),
    ("vllm.v1.engine.processor", "Processor.__init__", "engine.processor_init",
     "engine", None),

    # ---- executor / workers -----------------------------------------
    ("vllm.v1.executor.abstract", "Executor.get_class", "executor.get_class",
     "executor", None),
    ("vllm.v1.executor.abstract", "UniProcExecutor._init_executor",
     "executor.init_uniproc", "executor", None),
    ("vllm.v1.executor.multiproc_executor", "MultiprocExecutor._init_executor",
     "executor.init_multiproc", "executor", None),
    ("vllm.v1.executor.multiproc_executor", "WorkerProc.__init__",
     "worker.proc_init", "executor", None),
    ("vllm.v1.executor.multiproc_executor", "WorkerProc.make_worker_process",
     "worker.make_process", "executor", None),
    ("vllm.v1.executor.multiproc_executor", "WorkerProc.worker_main",
     "worker.main", "executor", None),
    ("vllm.executor.uniproc_executor", "UniProcExecutor._init_executor",
     "executor.init_uniproc_v0", "executor", None),

    ("vllm.v1.worker.gpu_worker", "Worker.init_device", "worker.init_device",
     "device", None),
    ("vllm.v1.worker.gpu_worker", "Worker.load_model", "weights.load_model",
     "weights", None),
    ("vllm.v1.worker.gpu_worker", "Worker.determine_available_memory",
     "kvcache.determine_memory", "kvcache", None),
    ("vllm.v1.worker.gpu_worker", "Worker.initialize_from_config",
     "kvcache.initialize_from_config", "kvcache", None),
    ("vllm.v1.worker.gpu_worker", "Worker.compile_or_warm_up_model",
     "warmup.compile_or_warm_up", "warmup", None),
    ("vllm.v1.worker.gpu_worker", "init_worker_distributed_environment",
     "device.init_distributed", "collective", None),

    ("vllm.v1.worker.gpu_model_runner", "GPUModelRunner.load_model",
     "weights.runner_load_model", "weights", None),
    ("vllm.v1.worker.gpu_model_runner", "GPUModelRunner.profile_run",
     "warmup.profile_run", "warmup", None),
    ("vllm.v1.worker.gpu_model_runner", "GPUModelRunner.capture_model",
     "cudagraph.capture_model", "cudagraph", None),
    ("vllm.v1.worker.gpu_model_runner", "GPUModelRunner.initialize_kv_cache",
     "kvcache.runner_initialize", "kvcache", None),
    ("vllm.v1.worker.gpu_model_runner", "GPUModelRunner._dummy_run",
     "warmup.dummy_run", "warmup", None),
    ("vllm.v1.worker.gpu_model_runner", "GPUModelRunner._dummy_sampler_run",
     "warmup.dummy_sampler_run", "warmup", None),

    # The v2 model runner, selected by ``Worker.use_v2_model_runner``. It lives
    # in a *package* (``vllm.v1.worker.gpu.model_runner``) while the legacy one
    # is a module (``vllm.v1.worker.gpu_model_runner``), and both ship in the
    # same release. Patching only the legacy name is how 10.9s of the warmup
    # phase went dark: the v1 module is never imported on a v2 build, so its
    # patches wait in ``_pending`` forever and the span has no children.
    ("vllm.v1.worker.gpu.model_runner", "GPUModelRunner.load_model",
     "weights.runner_load_model", "weights", None),
    ("vllm.v1.worker.gpu.model_runner", "GPUModelRunner.profile_run",
     "warmup.profile_run", "warmup", None),
    ("vllm.v1.worker.gpu.model_runner", "GPUModelRunner.capture_model",
     "cudagraph.capture_model", "cudagraph", None),
    ("vllm.v1.worker.gpu.model_runner", "GPUModelRunner.initialize_kv_cache",
     "kvcache.runner_initialize", "kvcache", None),
    ("vllm.v1.worker.gpu.model_runner", "GPUModelRunner._dummy_run",
     "warmup.dummy_run", "warmup", None),
    ("vllm.v1.worker.gpu.model_runner", "GPUModelRunner._dummy_sampler_run",
     "warmup.dummy_sampler_run", "warmup", None),
    ("vllm.v1.worker.gpu.model_runner", "GPUModelRunner._dummy_pooler_run",
     "warmup.dummy_pooler_run", "warmup", None),

    # ---- the steps inside compile_or_warm_up_model --------------------
    # Patched at the *binding* site: gpu_worker.py does
    # ``from vllm.model_executor.warmup.kernel_warmup import kernel_warmup``
    # (and the same for warmup_kernels/freeze_gc_heap), so the name it calls
    # lives in gpu_worker's module dict, not in the defining module.
    ("vllm.v1.worker.gpu_worker", "kernel_warmup", "warmup.kernel_warmup",
     "warmup", None),
    ("vllm.v1.worker.gpu_worker", "warmup_kernels", "warmup.warmup_kernels",
     "warmup", None),
    ("vllm.v1.worker.gpu_worker", "freeze_gc_heap", "warmup.freeze_gc_heap",
     "warmup", None),
    # These two are imported *inside* compile_or_warm_up_model, so here the
    # defining module is the right target.
    ("vllm.compilation.compiler_interface", "trigger_inductor_lazy_init",
     "warmup.inductor_lazy_init", "warmup", None),
    ("vllm.utils.jit_monitor", "activate", "warmup.jit_monitor_activate",
     "warmup", None),
    # v2 runs scheduler-realistic prefill + decode steps through the worker's
    # own entry points; without these the warmup_kernels span has no interior.
    ("vllm.v1.worker.gpu.warmup", "run_mixed_prefill_decode_warmup",
     "warmup.mixed_prefill_decode", "warmup", None),

    # kernel_warmup's chain of gated sub-warmups. Every one of these is
    # ``from ... import``-ed into kernel_warmup's namespace at module top, so
    # the defining modules are the wrong target -- patch the namespace that
    # actually resolves the name at call time.
    ("vllm.model_executor.warmup.kernel_warmup", "qwen_triton_warmup",
     "warmup.qwen_triton", "warmup", None),
    ("vllm.model_executor.warmup.kernel_warmup", "deepseek_v4_mhc_warmup",
     "warmup.deepseek_v4_mhc", "warmup", None),
    ("vllm.model_executor.warmup.kernel_warmup", "kimi_k3_triton_warmup",
     "warmup.kimi_k3_triton", "warmup", None),
    ("vllm.model_executor.warmup.kernel_warmup", "fa4_cutedsl_warmup",
     "warmup.fa4_cutedsl", "warmup", None),
    ("vllm.model_executor.warmup.kernel_warmup", "sparse_mla_triton_warmup",
     "warmup.sparse_mla_triton", "warmup", None),
    ("vllm.model_executor.warmup.kernel_warmup", "_warmup_ll_bf16_router_gemm",
     "warmup.ll_bf16_router_gemm", "warmup", None),
    ("vllm.model_executor.warmup.kernel_warmup", "cutedsl_warmup",
     "warmup.cutedsl", "warmup", None),
    ("vllm.model_executor.warmup.kernel_warmup",
     "flashinfer_sparse_mla_decode_autotune_warmup",
     "warmup.flashinfer_sparse_mla_decode", "warmup", None),
    ("vllm.model_executor.warmup.kernel_warmup",
     "deepseek_v4_sparse_mla_attention_warmup",
     "warmup.deepseek_v4_sparse_mla", "warmup", None),
    ("vllm.model_executor.warmup.kernel_warmup", "deep_gemm_warmup",
     "warmup.deep_gemm", "warmup", None),
    ("vllm.model_executor.warmup.kernel_warmup", "b12x_warmup",
     "warmup.b12x", "warmup", None),
    ("vllm.model_executor.warmup.kernel_warmup", "flashinfer_autotune",
     "warmup.flashinfer_autotune", "warmup", None),
    ("vllm.model_executor.warmup.kernel_warmup", "warm_v1_block_table_kernels",
     "warmup.v1_block_table", "warmup", None),
    # imported lazily inside kernel_warmup(), so its own module is the target
    ("vllm.model_executor.warmup.minimax_m3_msa_warmup", "minimax_m3_msa_warmup",
     "warmup.minimax_m3_msa", "warmup", None),

    # ---- model construction + weight loading -------------------------
    ("vllm.model_executor.model_loader", "get_model", "weights.get_model",
     "weights", None),
    ("vllm.model_executor.model_loader", "get_model_loader",
     "weights.get_loader", "weights", None),
    ("vllm.model_executor.model_loader.base_loader", "BaseModelLoader.load_model",
     "weights.loader_load_model", "weights", None),
    ("vllm.model_executor.model_loader.default_loader",
     "DefaultModelLoader.download_model", "weights.download_model",
     "weights", None),
    ("vllm.model_executor.model_loader.default_loader",
     "DefaultModelLoader.load_weights", "weights.load_weights",
     "weights", None),
    ("vllm.model_executor.model_loader.default_loader",
     "DefaultModelLoader._prepare_weights", "weights.prepare_weights",
     "weights", None),
    ("vllm.model_executor.model_loader.loader",
     "DefaultModelLoader.load_model", "weights.loader_load_model_v0",
     "weights", None),
    ("vllm.model_executor.model_loader.loader",
     "DefaultModelLoader._prepare_weights", "weights.prepare_weights_v0",
     "weights", None),
    ("vllm.model_executor.model_loader.weight_utils",
     "download_weights_from_hf", "weights.download_from_hf", "network", None),
    ("vllm.model_executor.model_loader.weight_utils",
     "safetensors_weights_iterator", "weights.safetensors_iterator",
     "weights", None),
    ("vllm.model_executor.model_loader.weight_utils",
     "np_cache_weights_iterator", "weights.np_cache_iterator",
     "weights", None),
    ("vllm.model_executor.model_loader.utils", "initialize_model",
     "weights.initialize_model", "weights", None),
    ("vllm.model_executor.model_loader.utils", "get_model_architecture",
     "weights.get_architecture", "weights", None),
    ("vllm.model_executor.models.registry", "ModelRegistry.resolve_model_cls",
     "weights.resolve_model_cls", "weights", None),
    ("vllm.model_executor.models.registry", "_run_in_subprocess",
     "weights.registry_subprocess", "weights", None),

    # ---- distributed / IPC ------------------------------------------
    ("vllm.distributed.parallel_state", "init_distributed_environment",
     "collective.init_env", "collective", None),
    ("vllm.distributed.parallel_state", "initialize_model_parallel",
     "collective.init_model_parallel", "collective", None),
    ("vllm.distributed.parallel_state", "GroupCoordinator.__init__",
     "collective.group_coordinator", "collective", None),
    ("vllm.distributed.device_communicators.pynccl",
     "PyNcclCommunicator.__init__", "collective.pynccl_init",
     "collective", None),
    ("vllm.distributed.device_communicators.custom_all_reduce",
     "CustomAllreduce.__init__", "collective.custom_allreduce_init",
     "collective", None),
    ("vllm.distributed.device_communicators.shm_broadcast",
     "MessageQueue.__init__", "ipc.message_queue_init", "ipc", None),
    ("vllm.distributed.device_communicators.shm_broadcast",
     "MessageQueue.wait_until_ready", "ipc.message_queue_wait",
     "ipc", None),
    ("vllm.distributed.utils", "StatelessProcessGroup.create",
     "ipc.stateless_pg_create", "ipc", None),

    # ---- compilation -------------------------------------------------
    ("vllm.compilation.backends", "VllmBackend.__call__",
     "compile.vllm_backend", "compile", None),
    ("vllm.compilation.backends", "compile_or_warm_up",
     "compile.or_warm_up", "compile", None),
    ("vllm.compilation.compiler_interface", "InductorAdaptor.compile",
     "compile.inductor_adaptor", "compile", None),
    ("vllm.compilation.compiler_interface", "InductorAdaptor.initialize_cache",
     "compile.inductor_cache_init", "compile", None),
    ("vllm.compilation.compiler_interface", "InductorStandaloneAdaptor.compile",
     "compile.inductor_standalone", "compile", None),
    ("vllm.compilation.wrapper",
     "TorchCompileWrapperWithCustomDispatcher.__init__",
     "compile.wrapper_init", "compile", None),
    ("vllm.compilation.decorators", "_support_torch_compile",
     "compile.support_decorator", "compile", None),

    # ---- legacy (v0) engine -----------------------------------------
    ("vllm.engine.llm_engine", "LLMEngine.__init__", "engine.v0_llm_init",
     "engine", None),
    ("vllm.engine.async_llm_engine", "AsyncLLMEngine.from_engine_args",
     "engine.v0_from_engine_args", "engine", None),
    ("vllm.worker.worker", "Worker.init_device", "worker.v0_init_device",
     "device", None),
    ("vllm.worker.worker", "Worker.load_model", "weights.v0_load_model",
     "weights", None),
    ("vllm.worker.model_runner", "GPUModelRunnerBase.load_model",
     "weights.v0_runner_load_model", "weights", None),
    ("vllm.worker.model_runner", "ModelRunner.capture_model",
     "cudagraph.v0_capture_model", "cudagraph", None),

    # ---- HTTP server -------------------------------------------------
    # Server.startup is instrumented in _instrument_ready_marker() instead,
    # so it can also emit the "api.startup_complete" readiness edge.
    ("uvicorn.server", "Server.serve", "apiserver.uvicorn_serve",
     "apiserver", None),
    ("uvicorn.config", "Config.load", "apiserver.uvicorn_config_load",
     "apiserver", None),
    ("starlette.routing", "Router.startup", "apiserver.starlette_startup",
     "apiserver", None),
]


def install():
    global _installed
    if _installed or not T.enabled("vllm"):
        return
    _installed = True
    for module, path, name, cat, kind in PATCHES:
        cs_patch.patch(module, path, name=name, cat=cat, kind=kind)
    cs_patch.after_import("vllm", _on_vllm)
    cs_patch.after_import("vllm.version", _on_vllm_version)
    _instrument_config_capture()
    _instrument_ready_marker()
    _instrument_cudagraph_capture()


def _on_vllm(vllm):
    T.tracer.meta("vllm.info",
                  version=getattr(vllm, "__version__", "?"),
                  commit=getattr(vllm, "__version_tuple__", None),
                  file=getattr(vllm, "__file__", None))


def _on_vllm_version(mod):
    T.tracer.meta("vllm.version", version=getattr(mod, "__version__", "?"))


def _config_summary(cfg):
    """Flatten the interesting knobs of a VllmConfig for the run metadata.

    These are exactly the fields that change cold-start behaviour, so a report
    is only comparable against another run with the same values.
    """
    def g(obj, *names):
        for n in names:
            obj = getattr(obj, n, None)
            if obj is None:
                return None
        return obj

    out = {}
    try:
        mc = getattr(cfg, "model_config", None)
        pc = getattr(cfg, "parallel_config", None)
        cc = getattr(cfg, "cache_config", None)
        comp = getattr(cfg, "compilation_config", None)
        lc = getattr(cfg, "load_config", None)
        sc = getattr(cfg, "scheduler_config", None)
        out.update({
            "model": str(g(mc, "model")),
            "served_model_name": str(g(mc, "served_model_name")),
            "dtype": str(g(mc, "dtype")),
            "quantization": str(g(mc, "quantization")),
            "max_model_len": g(mc, "max_model_len"),
            "enforce_eager": g(mc, "enforce_eager"),
            "trust_remote_code": g(mc, "trust_remote_code"),
            "seed": g(mc, "seed"),
            "tensor_parallel_size": g(pc, "tensor_parallel_size"),
            "pipeline_parallel_size": g(pc, "pipeline_parallel_size"),
            "data_parallel_size": g(pc, "data_parallel_size"),
            "distributed_executor_backend":
                str(g(pc, "distributed_executor_backend")),
            "worker_cls": str(g(pc, "worker_cls")),
            "gpu_memory_utilization": g(cc, "gpu_memory_utilization"),
            "block_size": g(cc, "block_size"),
            "cache_dtype": str(g(cc, "cache_dtype")),
            "load_format": str(g(lc, "load_format")),
            "max_num_seqs": g(sc, "max_num_seqs"),
            "max_num_batched_tokens": g(sc, "max_num_batched_tokens"),
        })
        if comp is not None:
            out["compilation"] = {
                "level": g(comp, "level"),
                "mode": str(g(comp, "mode")),
                "backend": str(g(comp, "backend")),
                "cudagraph_mode": str(g(comp, "cudagraph_mode")),
                "use_cudagraph": g(comp, "use_cudagraph"),
                "full_cuda_graph": g(comp, "full_cuda_graph"),
                "cache_dir": str(g(comp, "cache_dir")),
                "n_cudagraph_sizes": len(g(comp, "cudagraph_capture_sizes")
                                         or []),
                "custom_ops": g(comp, "custom_ops"),
            }
    except Exception as e:
        out["error"] = repr(e)
    return out


def _instrument_config_capture():
    """Emit the resolved VllmConfig as run metadata (needs the *return* value,
    which the generic span wrapper does not see)."""

    def _apply(module):
        for cls_name in ("EngineArgs", "AsyncEngineArgs"):
            cls = getattr(module, cls_name, None)
            if cls is None:
                continue
            orig = getattr(cls, "create_engine_config", None)
            if orig is None or getattr(orig, "_cs_config_capture", False):
                continue

            def make(orig=orig):
                def wrapper(*a, **kw):
                    cfg = orig(*a, **kw)
                    try:
                        T.tracer.meta("vllm.config", **_config_summary(cfg))
                    except Exception:
                        pass
                    return cfg
                wrapper._cs_config_capture = True
                try:
                    wrapper.__name__ = getattr(orig, "__name__", "wrapper")
                except Exception:
                    pass
                return wrapper
            setattr(cls, "create_engine_config", make())

    cs_patch.after_import("vllm.engine.arg_utils", _apply)


def _instrument_ready_marker():
    """Mark the in-process moment uvicorn finishes startup, i.e. the first
    instant the socket can answer ``/health``. The sidecar poller measures the
    same edge from outside; having both separates server-side readiness from
    the client-visible one."""

    def _apply(module):
        server = getattr(module, "Server", None)
        if server is None:
            return
        orig = getattr(server, "startup", None)
        if orig is None or getattr(orig, "_cs_ready", False):
            return
        import functools

        @functools.wraps(orig)
        async def startup(self, *a, **kw):
            tok = T.tracer.begin("apiserver.uvicorn_startup", "apiserver")
            try:
                return await orig(self, *a, **kw)
            finally:
                T.tracer.end(tok, min_dur=0.0)
                T.tracer.instant("api.startup_complete", "ready",
                                 note="uvicorn startup handlers done; "
                                      "server accepting requests")
        startup._cs_ready = True
        setattr(server, "startup", startup)

    cs_patch.after_import("uvicorn.server", _apply)


def _instrument_cudagraph_capture():
    """Split CUDA graph capture by mode, size, and warmup-vs-captured forward.

    ``CudaGraphManager.capture`` (v1/worker/gpu/cudagraph_utils.py) loops
    ``[PIECEWISE, FULL]`` and, per batch descriptor, calls
    ``create_forward_fn(desc, warmup=True)`` then ``forward_fn(NONE)`` -- an
    *eager* forward -- before capturing a second one. With 51 capture sizes and
    two modes that is ~200 model forwards, and it is the single largest item in
    the warmup phase. None of that structure is visible from outside the call:
    one span around ``capture()`` gives a total, and the mode is a loop variable.

    The seam is the ``create_forward_fn`` argument. Wrapping it -- and the
    ``forward_fn`` it returns -- attributes every forward to a (mode, num_tokens)
    pair without reimplementing any of the capture logic. Only the base class is
    patched: ``ModelCudaGraphManager.capture`` builds its own closure and reaches
    this one through ``super()``.

    The warmup-vs-captured split cannot be read off the ``run_mode`` argument:
    for FULL, and for breakable PIECEWISE, the capture pass also calls
    ``forward_fn(CUDAGraphMode.NONE)`` -- inside ``torch.cuda.graph()`` -- so
    both passes look identical from there. The authoritative signal is the
    ``warmup=`` flag ``create_forward_fn`` was called with: a forward is a
    capture if it came from a ``warmup=False`` factory call, or if it is run in
    a real graph mode. Non-breakable PIECEWISE reuses the ``warmup=True``
    factory for its capture and is caught by the second half of that rule.

    ~300 spans is a rounding error against a 60s trace, so every forward and
    every input prep is recorded exactly; ``CS_CUDAGRAPH_MIN_DUR`` can trim the
    short ones if a much larger capture list ever makes that worthwhile.
    """

    def _apply(module):
        cls = getattr(module, "CudaGraphManager", None)
        if cls is None:
            T.tracer.error("cudagraph.CudaGraphManager",
                           AttributeError("vllm.v1.worker.gpu.cudagraph_utils"
                                          ".CudaGraphManager"))
            return
        orig = getattr(cls, "capture", None)
        if orig is None or getattr(orig, "_cs_cudagraph", False):
            return
        min_dur = T.env_float("CUDAGRAPH_MIN_DUR", 0.0)

        def capture(self, create_forward_fn, *a, **kw):
            counts = {}

            def wrapped(desc, warmup=False, *da, **dkw):
                mode = _desc_mode(desc)
                tokens = getattr(desc, "num_tokens", None)
                tok = T.tracer.begin("cudagraph.prepare_inputs", "cudagraph",
                                     mode=mode, num_tokens=tokens,
                                     warmup=bool(warmup))
                try:
                    fn = create_forward_fn(desc, warmup, *da, **dkw)
                finally:
                    T.tracer.end(tok, min_dur=min_dur)

                # One factory call can serve both passes (non-breakable
                # PIECEWISE), so the counter is per closure, not per descriptor.
                seen = []

                def forward(run_mode, *fa, **fkw):
                    # A capture pass either runs in a real graph mode, or comes
                    # from a warmup=False factory call. Everything else is the
                    # eager warmup forward, which is unconditional on this path
                    # -- see cudagraph_num_of_warmups, read only by v1.
                    graphed = _mode_name(run_mode) != "NONE"
                    eager = not graphed and warmup and not seen
                    seen.append(1)
                    key = (mode, "warmup" if eager else "capture")
                    counts[key] = counts.get(key, 0) + 1
                    name = ("cudagraph.warmup_forward" if eager
                            else "cudagraph.capture_forward")
                    tok = T.tracer.begin(name, "cudagraph", mode=mode,
                                         num_tokens=tokens, graphed=graphed)
                    try:
                        return fn(run_mode, *fa, **fkw)
                    finally:
                        T.tracer.end(tok, min_dur=0.0)

                return forward

            tok = T.tracer.begin("cudagraph.manager_capture", "cudagraph")
            try:
                return orig(self, wrapped, *a, **kw)
            finally:
                T.tracer.end(tok, min_dur=0.0)
                try:
                    T.tracer.instant(
                        "cudagraph.forward_counts", cat="cudagraph",
                        counts={"%s/%s" % k: v for k, v in counts.items()},
                        total=sum(counts.values()))
                except Exception:
                    pass

        capture._cs_cudagraph = True
        try:
            capture.__name__ = getattr(orig, "__name__", "capture")
        except Exception:
            pass
        setattr(cls, "capture", capture)

    cs_patch.after_import("vllm.v1.worker.gpu.cudagraph_utils", _apply)


def _mode_name(mode):
    """CUDAGraphMode -> a short string, without importing vllm.config."""
    name = getattr(mode, "name", None)
    if isinstance(name, str):
        return name
    return str(mode)


def _desc_mode(desc):
    return _mode_name(getattr(desc, "cg_mode", None))
