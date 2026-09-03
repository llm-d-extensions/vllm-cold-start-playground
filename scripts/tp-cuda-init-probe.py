#!/usr/bin/env python3
"""Name the call that initialises CUDA in the process that launches EngineCore.

At TP=2 both cold-start levers that depend on `fork` are vetoed, because CUDA is
already initialised by the time vLLM picks a multiprocessing context:
`_maybe_force_spawn()` reverts `fork` to `spawn` (`utils/system_utils.py:157`)
and `coldstart/cs_forkserver.py` declines with `served=0`. The traces put the
`cuda.lazy_init` 3.5-5.5s *before* `engine.core_proc_manager` and inside
`config.create_engine_config`, but a trace span cannot name the line.

Two candidates were eliminated by reading the source: `has_flashinfer()` uses
`importlib.util.find_spec` specifically to avoid this, and
`get_device_capability()` resolves to the NVML implementation
(`platforms/cuda.py:738`), which does not touch the CUDA runtime.

So ask torch. `torch.cuda._lazy_init` is the single funnel through which a CUDA
context gets created, so wrapping it and dumping the Python stack names the
caller exactly. Config creation only -- no engine, no weights, no GPU memory
beyond the context itself -- so this costs one torch+vllm import, not a boot.

Run both arms: the difference between them *is* the answer, because TP=1 never
initialises CUDA here.

    python3 tp-cuda-init-probe.py --tp 2
    python3 tp-cuda-init-probe.py --tp 1
"""

import argparse
import sys
import traceback


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--model", default="Qwen/Qwen3-32B")
    ap.add_argument("--max-hits", type=int, default=3,
                    help="stacks to print; the first is the one that matters")
    a = ap.parse_args()

    # Patch before vllm is imported, so an import-time initialisation is caught
    # too rather than being mistaken for a config-time one.
    import torch

    print(f"torch {torch.__version__}  cuda_initialized_at_import="
          f"{torch.cuda.is_initialized()}")

    hits = []
    orig = torch.cuda._lazy_init

    def traced(*args, **kw):
        if len(hits) < a.max_hits and not torch.cuda.is_initialized():
            hits.append(traceback.extract_stack()[:-1])
        return orig(*args, **kw)

    torch.cuda._lazy_init = traced

    import vllm  # noqa: F401
    from vllm.engine.arg_utils import EngineArgs

    print(f"vllm {vllm.__version__}  after_vllm_import="
          f"{torch.cuda.is_initialized()}  hits={len(hits)}")

    args = EngineArgs(model=a.model, tensor_parallel_size=a.tp,
                      max_model_len=8192, enforce_eager=False)
    cfg = args.create_engine_config()
    print(f"tp={a.tp}  after_create_engine_config="
          f"{torch.cuda.is_initialized()}  hits={len(hits)}")
    print(f"  (config built ok: tp={cfg.parallel_config.tensor_parallel_size} "
          f"fusions={getattr(cfg.compilation_config.pass_config, 'enable_allreduce_rms_fusion', '?')})")

    if not hits:
        print("\nNo _lazy_init during config creation at this TP. "
              "If this is the --tp 1 arm, that is the expected contrast.")
        return 0

    for i, st in enumerate(hits):
        print(f"\n=== stack {i + 1}: what called torch.cuda._lazy_init ===")
        # Print the whole stack, but mark the frames that are vLLM's or a
        # third party's -- the boundary between them is the line to report.
        for f in st:
            tag = ""
            if "/vllm/" in f.filename:
                tag = "  <-- vllm"
            elif "/flashinfer/" in f.filename or "/tilelang/" in f.filename:
                tag = "  <-- third party"
            print(f"  {f.filename}:{f.lineno} in {f.name}{tag}")
            if f.line:
                print(f"      {f.line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
