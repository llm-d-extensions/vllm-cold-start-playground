#!/usr/bin/env python3
"""Is vLLM's model-registry inspection a cold-start cost, and does the
``modelinfos`` cache remove it?

Every committed run in ``runs/`` reports a "cold start" that is warm in one
specific place nobody was looking: ``_LazyRegisteredModel.inspect_model_cls``
(``vllm/model_executor/models/registry.py:997``). It answers questions the
config layer asks about an architecture -- is it a text-generation model, is it
multimodal, does it support LoRA -- and it answers them by **importing the model
class in a throwaway subprocess**, because doing it in-process would initialise
CUDA in the API server:

    _SUBPROCESS_COMMAND = [sys.executable, "-m", "vllm.model_executor.models.registry"]

That subprocess is a fresh interpreter doing a full ``import vllm`` (and so a
full ``import torch``). Since some version it is cached: ``inspect_model_cls``
hashes the model module's bytes and looks for
``$VLLM_CACHE_ROOT/modelinfos/<module>-<class>.json``, and only spawns on a miss
(then writes the file). So the cost is paid **once per (vLLM build, model
module) per cache volume** and is invisible on every run after it -- which is
every run in ``runs/`` except three.

This probe measures the two paths directly, without booting vLLM: no GPU, no
weights, ~15s per cold trial instead of ~90s per boot. It calls
``inspect_model_cls`` on the real registry entry, with the cache file removed
(``cold``) or present (``warm``), one fresh interpreter per trial so neither the
``lru_cache`` on ``_try_inspect_model_cls`` nor an already-imported model module
can make a repeat free.

Arms are interleaved (cold, warm, cold, warm, ...) rather than blocked, so drift
in the node cannot line up with an arm -- the house rule from
docs/measurement.md.

Usage, inside the pod:
    python3 registry-cache-probe.py --arch Qwen3ForCausalLM --repeat 5
"""

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path


def resolve(arch):
    """(model, cache_path) for one architecture, or exit with why not."""
    from vllm.model_executor.models.registry import ModelRegistry

    model = ModelRegistry.models.get(arch)
    if model is None:
        sys.exit("registry-cache-probe: %r is not a registered architecture" % arch)
    if not hasattr(model, "_get_cache_filename"):
        # A non-lazy entry (already-imported class) has no subprocess path and
        # no cache; saying so is more useful than measuring 0.0 three times.
        sys.exit("registry-cache-probe: %r is a %s, not a _LazyRegisteredModel: "
                 "it has no subprocess/cache path to measure"
                 % (arch, type(model).__name__))
    cache_dir = Path(model._get_cache_dir())
    return model, cache_dir / model._get_cache_filename()


def _module_path(model):
    """The .py whose bytes the cache key hashes -- vLLM's own branch, verbatim.

    Resolved off the already-imported registry module rather than
    ``find_spec(model.module_name)``, which would import
    ``vllm.model_executor.models`` (~5s) just to learn a filename.
    """
    import vllm.model_executor.models.registry as reg

    if model.module_name.startswith("vllm.model_executor.models."):
        return Path(reg.__file__).parent / ("%s.py" % model.module_name.split(".")[-1])
    import importlib.util
    try:
        spec = importlib.util.find_spec(model.module_name)
    except (ImportError, ValueError):
        return None
    return Path(spec.origin) if spec is not None and spec.origin else None


def trial(arch, arm):
    """One measurement, in this (fresh) interpreter. Prints one JSON line.

    ``import vllm`` is deliberately outside the timed region: the process that
    pays this in production (the API server) has already imported vllm when it
    reaches config resolution. What is being timed is only the registry call.
    """
    model, cache_path = resolve(arch)

    if arm == "cold":
        # Remove the whole directory, not just this file: a sibling entry proves
        # nothing about this one, and leaving it makes "cold" ambiguous.
        shutil.rmtree(cache_path.parent, ignore_errors=True)
    elif arm == "warm":
        if not cache_path.exists():
            # Populate it, untimed, so the timed call below is a real cache hit
            # rather than a miss mislabelled as one.
            model.inspect_model_cls()
        if not cache_path.exists():
            sys.exit("registry-cache-probe: warm arm could not populate %s" % cache_path)

    existed = cache_path.exists()

    t = time.perf_counter()
    mi = model.inspect_model_cls()
    total = time.perf_counter() - t

    # Break the warm path down: it is not free either -- the hash reads the
    # model module's bytes off the cache volume (and for a package entry point,
    # every .py under it) before the JSON is even opened.
    hash_s = load_s = None
    if existed:
        model_path = _module_path(model)
        if model_path is not None and model_path.exists():
            t = time.perf_counter()
            h = model._get_modelinfo_module_hash(model_path)
            hash_s = time.perf_counter() - t
            t = time.perf_counter()
            model._load_modelinfo_from_cache(h)
            load_s = time.perf_counter() - t

    print(json.dumps({
        "arm": arm,
        "arch": arch,
        "cache_existed": existed,
        "spawned_subprocess": not existed,
        "total_s": total,
        "hash_s": hash_s,
        "cache_load_s": load_s,
        "cache_path": str(cache_path),
        "cache_bytes": cache_path.stat().st_size if cache_path.exists() else None,
        "is_text_generation": getattr(mi, "is_text_generation_model", None),
    }))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="Qwen3ForCausalLM")
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--trial", choices=("cold", "warm"),
                    help="internal: run one measurement in this interpreter")
    args = ap.parse_args()

    if args.trial:
        trial(args.arch, args.trial)
        return

    # --- driver: fresh interpreter per trial, arms interleaved -------------
    print("registry-cache-probe: arch=%s repeat=%d" % (args.arch, args.repeat))
    print("VLLM_CACHE_ROOT=%s" % os.environ.get("VLLM_CACHE_ROOT", "(unset)"))
    import vllm
    print("vllm=%s python=%s" % (vllm.__version__, sys.version.split()[0]))
    _, cache_path = resolve(args.arch)
    print("cache file=%s (exists=%s)\n" % (cache_path, cache_path.exists()))

    rows = []
    for i in range(args.repeat):
        for arm in ("cold", "warm"):
            r = subprocess.run(
                [sys.executable, os.path.abspath(__file__),
                 "--arch", args.arch, "--trial", arm],
                capture_output=True, text=True)
            if r.returncode != 0:
                print("  trial %d %s FAILED rc=%d\n%s" % (i + 1, arm, r.returncode,
                                                          r.stderr[-2000:]))
                continue
            rec = json.loads(r.stdout.strip().splitlines()[-1])
            rec["repeat"] = i + 1
            rows.append(rec)
            print("  %-5s r%d  total=%7.3fs  spawned=%-5s  hash=%-8s cache_load=%s"
                  % (arm, i + 1, rec["total_s"], rec["spawned_subprocess"],
                     "%.4fs" % rec["hash_s"] if rec["hash_s"] is not None else "-",
                     "%.4fs" % rec["cache_load_s"] if rec["cache_load_s"] is not None else "-"))

    print("\n%-6s %3s  %9s %9s %9s   %s" % ("arm", "n", "median", "min", "max", "spawned"))
    med = {}
    for arm in ("cold", "warm"):
        xs = [r["total_s"] for r in rows if r["arm"] == arm]
        if not xs:
            continue
        med[arm] = statistics.median(xs)
        sp = sum(1 for r in rows if r["arm"] == arm and r["spawned_subprocess"])
        print("%-6s %3d  %8.3fs %8.3fs %8.3fs   %d/%d"
              % (arm, len(xs), med[arm], min(xs), max(xs), sp, len(xs)))

    if "cold" in med and "warm" in med:
        d = med["cold"] - med["warm"]
        print("\nVERDICT: registry inspection costs %.2fs on a cold modelinfos cache "
              "and %.3fs warm" % (med["cold"], med["warm"]))
        print("         delta = %.2fs (%.0fx). %s" % (
            d, med["cold"] / med["warm"] if med["warm"] else float("inf"),
            "Cold and warm start DO differ: this belongs in the phase table."
            if d > 0.5 else "No material difference."))

    Path("registry-cache-probe.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows))
    print("\nrows -> registry-cache-probe.jsonl")


if __name__ == "__main__":
    main()
