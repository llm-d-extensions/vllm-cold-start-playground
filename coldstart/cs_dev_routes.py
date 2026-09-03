"""Expose HTTP routes for engine machinery that has no route of its own.

Three pieces already exist inside vLLM and are reachable via
``collective_rpc``, but ``vllm/entrypoints/serve/dev/sleep/api_router.py``
never grew a route for them:

  * ``Worker.checkpoint_prepare`` / ``checkpoint_restore``
    (``v1/worker/gpu_worker.py``) -- detach / re-attach device-communicator
    state around a ``cuda-checkpoint`` checkpoint/restore of a TP>1 engine's
    ``WorkerProc``s. Checkpointing a TP>1 engine without this freezes live
    cross-rank state that ``cuda-checkpoint`` was never designed to freeze.

    Do not read the name as "tears down NCCL": it does not, and measuring that
    is what `scripts/tp2-park-probe.sh` is for. On vLLM 0.28.0
    ``CudaCommunicator.checkpoint_prepare`` (``cuda_communicator.py:588``)
    releases only the FlashInfer all-reduce workspace and the FlashInfer
    all2all manager -- its own comment says "Only FlashInfer all-reduce and
    FlashInfer all2all are supported for now". ``pynccl_comm`` is built
    unconditionally whenever ``world_size > 1`` and is never touched, and the
    stock TP dispatch chain on an NVLink H100 pair is
    ``['CUSTOM', 'SYMM_MEM', 'PYNCCL']`` -- none of which this method releases.
    So with default backends the call returns in ~0.06s having freed nothing
    that matters, and the checkpoint that follows hangs. See
    ``docs/sleep-mode.md`` for the isolating matrix.
  * ``Worker.reload_weights`` -> ``GPUModelRunner.reload_weights``
    (``v1/worker/gpu_model_runner.py``) -- reload the original checkpoint from
    disk straight into the existing, already-mapped parameter tensors, in
    place. This is the "reload from disk" that sleep level=2's own docstring
    promises on wake_up ("level 2 discards weights (reloaded from the model
    source on resume)") but ``CuMemBackend.resume()`` never implements: it
    only remaps virtual addresses, and copies back data for tags it actually
    backed up to host RAM -- which level=2 deliberately does not do for
    weights. Calling this after ``/wake_up`` is the fix.

This probe attaches all three as POST routes on a new router, following
``dev/sleep/api_router.py``'s own ``engine_client(request).<method>()``
pattern. The seam is ``vllm.entrypoints.serve.register_vllm_dev_api_routers``
-- the one function ``entrypoints/openai/api_server.py`` calls (gated on
``VLLM_SERVER_DEV_MODE``) to attach every dev-only router. Wrapping it means
the new routes appear exactly where ``/sleep`` and ``/wake_up`` do, with no
other code path touched.

Knobs:
    CS_DEV_ROUTES=1   enable (required; nothing here runs without it)
"""

import cs_patch
import cs_trace as T

_installed = False


def install():
    global _installed
    if _installed:
        return
    # Opt-in twice over, like cs_forkserver/cs_fst: this adds engine-control
    # HTTP surface, which is not something a probe should ever do by default.
    if not (T.env_flag("DEV_ROUTES") and T.enabled("dev_routes")):
        return
    _installed = True
    cs_patch.after_import("vllm.entrypoints.serve", _wrap)


def _wrap(module):
    orig = getattr(module, "register_vllm_dev_api_routers", None)
    if orig is None:
        T.tracer.error(
            "dev_routes.register_vllm_dev_api_routers",
            AttributeError("vllm.entrypoints.serve"
                            ".register_vllm_dev_api_routers"))
        return
    if getattr(orig, "_cs_dev_routes", False):
        return

    def register_vllm_dev_api_routers(app):
        orig(app)
        _attach(app)

    register_vllm_dev_api_routers._cs_dev_routes = True
    module.register_vllm_dev_api_routers = register_vllm_dev_api_routers


def _attach(app):
    from fastapi import APIRouter, Request, Response

    def engine_client(request):
        return request.app.state.engine_client

    router = APIRouter()

    @router.post("/checkpoint_prepare")
    async def checkpoint_prepare(raw_request: Request):
        await engine_client(raw_request).collective_rpc("checkpoint_prepare")
        return Response(status_code=200)

    @router.post("/checkpoint_restore")
    async def checkpoint_restore(raw_request: Request):
        await engine_client(raw_request).collective_rpc("checkpoint_restore")
        return Response(status_code=200)

    @router.post("/reload_weights")
    async def reload_weights(raw_request: Request):
        await engine_client(raw_request).collective_rpc("reload_weights")
        return Response(status_code=200)

    app.include_router(router)
    T.tracer.instant("dev_routes.attached", cat="probe",
                      routes=["/checkpoint_prepare", "/checkpoint_restore",
                              "/reload_weights"])
