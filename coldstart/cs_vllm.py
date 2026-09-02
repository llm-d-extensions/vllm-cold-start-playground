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
