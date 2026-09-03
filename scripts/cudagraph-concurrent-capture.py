#!/usr/bin/env python3
"""Is CUDA graph capture parallelizable inside one process?

The README prices `cudagraph capture` at 11.4s of a 43.3s time-to-ready for
Qwen3-32B on one H100 and lists "overlap capture across sizes" as an
unquantified upstream option. This script answers the *primitive* half of that
question, with no vLLM involved:

  1. does CUDA/PyTorch permit two captures to be underway at once,
  2. does a second thread make capture finish sooner, and
  3. if not, what serializes it?

Capture records kernels, it does not run them, so the phase is pure CPU work:
Python dispatch, ATen, the caching allocator, and driver-side graph node
insertion. Arms are built to separate those.

Workload: a stack of small bf16 matmuls, sized so each op is dispatch-bound
rather than GPU-bound -- the regime vLLM's PIECEWISE capture is in (124ms/size
at 1 token, 174ms at 512, i.e. nearly flat in batch size).

Arms:
  serial                  K captures back to back, one stream, one shared pool
                          (what vLLM does today)
  serial-globalmode       same, capture_error_mode="global" (torch's default)
  threads-N-ownpool       K captures over N threads, one mempool per thread
  threads-N-sharedpool    K captures over N threads, all into one mempool
  threads-N-noalloc       N threads, own pools, and a forward that allocates
                          nothing (all `out=` variants) -- isolates the
                          caching allocator's per-device lock from the driver
  serial-noalloc          the no-allocation forward, captured serially
  threads-2-globalmode    two threads with torch's default capture mode

Every arm replays what it captured and compares against an eager reference, so
"it did not crash" is never mistaken for "it worked". `--pool-probe` separately
measures what N mempools cost in graph-pool bytes, which is the price of the
only concurrency shape that works.

Usage:
  python3 cudagraph-concurrent-capture.py --graphs 32 --reps 5
  python3 cudagraph-concurrent-capture.py --pool-probe
"""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time

import torch

DEV = "cuda"


# --------------------------------------------------------------------------- #
# workload
# --------------------------------------------------------------------------- #
class Work:
    """One capturable unit: static input/output buffers plus a forward.

    Buffers are per-unit because a captured graph bakes in pointer values; two
    concurrent captures sharing an input buffer would be recording reads of the
    same address, which is exactly the aliasing question we want to keep out of
    the timing result.
    """

    def __init__(self, layers: int, hidden: int, tokens: int, seed: int, share_w=None):
        g = torch.Generator(device=DEV).manual_seed(seed)
        # Weights can be shared across units: they are read-only during capture
        # and dominate the memory footprint otherwise.
        self.w = (
            share_w
            if share_w is not None
            else [
                torch.randn(
                    hidden, hidden, device=DEV, dtype=torch.bfloat16, generator=g
                )
                / hidden**0.5
                for _ in range(layers)
            ]
        )
        self.x = torch.randn(
            tokens, hidden, device=DEV, dtype=torch.bfloat16, generator=g
        )
        self.out = torch.empty_like(self.x)
        # scratch for the no-allocation variant
        self.t1 = torch.empty_like(self.x)
        self.t2 = torch.empty_like(self.x)
        self.graph: torch.cuda.CUDAGraph | None = None
        self.ref: torch.Tensor | None = None

    # allocating forward: every op produces a fresh tensor, so every op takes
    # the caching allocator's per-device lock. This is what a real model does.
    def forward(self) -> torch.Tensor:
        h = self.x
        for w in self.w:
            h = torch.nn.functional.silu(torch.mm(h, w))
        return h

    def run_into_out(self) -> None:
        self.out.copy_(self.forward())

    # non-allocating forward: same kernels, same node count, zero allocator
    # traffic. Ping-pongs between two scratch buffers.
    def run_noalloc(self) -> None:
        src, dst = self.x, self.t1
        for i, w in enumerate(self.w):
            torch.mm(src, w, out=dst)
            torch.nn.functional.silu(dst, inplace=True)
            src, dst = dst, (self.t2 if src is self.t1 else self.t1)
        self.out.copy_(src)

    def body(self, noalloc: bool):
        return self.run_noalloc if noalloc else self.run_into_out

    def warmup(self, stream: torch.cuda.Stream, noalloc: bool, iters: int = 3) -> None:
        """Must run in the *capturing thread*: cuBLAS workspaces and other lazy
        per-thread state have to be paid before capture starts."""
        fn = self.body(noalloc)
        with torch.cuda.stream(stream):
            for _ in range(iters):
                fn()
        stream.synchronize()

    def capture(self, stream, pool, mode: str, noalloc: bool) -> None:
        graph = torch.cuda.CUDAGraph()
        fn = self.body(noalloc)
        with torch.cuda.stream(stream):
            # Deliberately NOT torch.cuda.graph(): its __enter__ calls
            # torch.cuda.synchronize() + empty_cache(), both process-global and
            # hostile to a concurrent capture. That is itself a finding.
            if pool is None:
                graph.capture_begin(capture_error_mode=mode)
            else:
                graph.capture_begin(pool=pool, capture_error_mode=mode)
            fn()
            graph.capture_end()
        self.graph = graph

    def verify(self) -> float:
        self.out.zero_()
        self.graph.replay()
        torch.cuda.synchronize()
        return (self.out.float() - self.ref.float()).abs().max().item()


def make_units(n: int, args, noalloc: bool) -> list[Work]:
    g = torch.Generator(device=DEV).manual_seed(7)
    shared_w = [
        torch.randn(
            args.hidden, args.hidden, device=DEV, dtype=torch.bfloat16, generator=g
        )
        / args.hidden**0.5
        for _ in range(args.layers)
    ]
    units = [
        Work(args.layers, args.hidden, args.tokens, seed=1000 + i, share_w=shared_w)
        for i in range(n)
    ]
    for u in units:
        u.body(noalloc)()
        u.ref = u.out.clone()
    torch.cuda.synchronize()
    return units


# --------------------------------------------------------------------------- #
# arms
# --------------------------------------------------------------------------- #
def arm_serial(units, mode: str, share_pool: bool, noalloc: bool) -> dict:
    stream = torch.cuda.Stream()
    pool = torch.cuda.graph_pool_handle() if share_pool else None
    for u in units:
        u.warmup(stream, noalloc)
    torch.cuda.synchronize()
    base = torch.cuda.memory_reserved()
    t0, c0 = time.perf_counter(), time.process_time()
    for u in units:
        u.capture(stream, pool, mode, noalloc)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    return {"wall": wall, "cpu": time.process_time() - c0, "pool_base": base}


def arm_threads(units, nthreads: int, mode: str, share_pool: bool, noalloc) -> dict:
    shared = torch.cuda.graph_pool_handle() if share_pool else None
    chunks = [units[i::nthreads] for i in range(nthreads)]
    spans: list[tuple[float, float]] = []
    errors: list[str] = []
    lock = threading.Lock()
    start = threading.Barrier(nthreads + 1, timeout=600)

    def worker(mine) -> None:
        try:
            stream = torch.cuda.Stream()
            pool = shared if share_pool else torch.cuda.graph_pool_handle()
            for u in mine:
                u.warmup(stream, noalloc)
            stream.synchronize()
            start.wait()
            t0 = time.perf_counter()
            for u in mine:
                u.capture(stream, pool, mode, noalloc)
            stream.synchronize()
            t1 = time.perf_counter()
            with lock:
                spans.append((t0, t1))
        except Exception as exc:  # noqa: BLE001 - the failure IS the result
            with lock:
                errors.append(f"{type(exc).__name__}: {exc}")
            try:
                start.abort()
            except Exception:
                pass

    threads = [threading.Thread(target=worker, args=(c,), daemon=True) for c in chunks]
    for t in threads:
        t.start()
    try:
        start.wait()
    except Exception:
        pass
    base = torch.cuda.memory_reserved()
    t0, c0 = time.perf_counter(), time.process_time()
    for t in threads:
        t.join(timeout=600)
    wall, cpu = time.perf_counter() - t0, time.process_time() - c0
    try:
        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001
        # In "global" capture mode this is itself the result: a sync issued from
        # a thread that is not capturing is rejected while another thread's
        # capture is underway.
        errors.append(f"{type(exc).__name__} on host sync: {exc}".split("\n")[0])

    busy = sum(b - a for a, b in spans)
    return {
        "wall": wall,
        "cpu": cpu,
        "errors": errors,
        "concurrency": round(busy / wall, 2) if wall and spans else 0.0,
        "pool_base": base,
    }


# --------------------------------------------------------------------------- #
def mib(x: float) -> float:
    return round(x / 2**20, 1)


def pool_probe(args) -> dict:
    """What do N mempools cost, in graph-pool bytes?

    vLLM shares one pool across all 102 graphs precisely so that the activation
    buffers of graph n+1 land on top of graph n's -- which is why it captures
    largest-first and PIECEWISE before FULL. Concurrency forces one pool per
    thread, so the reuse is lost across pools. This measures how much.
    """
    out = {}
    for npools in args.pools:
        torch.cuda.empty_cache()
        units = make_units(args.graphs, args, noalloc=False)
        streams = [torch.cuda.Stream() for _ in range(npools)]
        pools = [torch.cuda.graph_pool_handle() for _ in range(npools)]
        for i, u in enumerate(units):
            u.warmup(streams[i % npools], False)
        torch.cuda.synchronize()
        base = torch.cuda.memory_reserved()
        # Capture serially -- this arm is about pool count, not concurrency, so
        # it isolates the memory effect from every timing effect.
        for i, u in enumerate(units):
            u.capture(streams[i % npools], pools[i % npools], "thread_local", False)
        torch.cuda.synchronize()
        grew = torch.cuda.memory_reserved() - base
        maxdiff = max(u.verify() for u in units)
        out[str(npools)] = {"graph_pool_MiB": mib(grew), "max_abs_diff": maxdiff}
        print(
            f"  {npools} pool(s) for {args.graphs} graphs: "
            f"{mib(grew):8.1f} MiB reserved for graph memory   maxdiff={maxdiff:g}"
        )
        for u in units:
            u.graph = None
        del units, streams, pools
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=200)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--graphs", type=int, default=32)
    ap.add_argument("--threads", type=int, nargs="*", default=[2, 4, 8])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--pools", type=int, nargs="*", default=[1, 2, 4, 8])
    ap.add_argument("--pool-probe", action="store_true")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    print(
        f"torch {torch.__version__}  cuda {torch.version.cuda}  "
        f"{torch.cuda.get_device_name(0)}"
    )
    print(
        f"workload: {args.graphs} graphs x {args.layers} layers x "
        f"({args.tokens},{args.hidden}) bf16, ~{2 * args.layers} nodes/graph\n"
    )

    results: dict[str, dict] = {}

    if args.pool_probe:
        print("graph-pool cost of N mempools (serial capture, memory only):")
        results["pool_probe"] = pool_probe(args)
        if args.json_out:
            with open(args.json_out, "w") as f:
                json.dump({"args": vars(args), "results": results}, f, indent=2)
        return

    def run(name: str, fn, noalloc: bool = False) -> None:
        walls, cpus, errs, extra = [], [], [], {}
        maxdiff = 0.0
        for _ in range(args.reps):
            torch.cuda.empty_cache()
            units = make_units(args.graphs, args, noalloc)
            r = fn(units)
            if r.get("errors"):
                errs = r["errors"]
                break
            walls.append(r["wall"])
            cpus.append(r["cpu"])
            captured = [u for u in units if u.graph is not None]
            if len(captured) != args.graphs:
                errs = [f"only {len(captured)}/{args.graphs} graphs captured"]
                break
            maxdiff = max(maxdiff, max(u.verify() for u in units))
            extra = {
                k: v for k, v in r.items() if k not in ("wall", "cpu", "errors", "pool_base")
            }
            extra["graphmem_MiB"] = mib(torch.cuda.memory_reserved() - r["pool_base"])
            for u in units:
                u.graph = None
            del units
        if errs:
            print(f"{name:26s} FAILED: {errs[0][:150]}")
            results[name] = {"error": errs[0]}
            return
        med = statistics.median(walls)
        print(
            f"{name:26s} {med * 1e3:8.1f} ms  "
            f"[{min(walls) * 1e3:.0f}-{max(walls) * 1e3:.0f}]  "
            f"cpu/wall={statistics.median(cpus) / med:4.2f}  "
            f"per-graph={med / args.graphs * 1e3:5.2f} ms  "
            f"maxdiff={maxdiff:g}  {extra}"
        )
        results[name] = {
            "median_s": med,
            "runs_s": walls,
            "cpu_s": cpus,
            "max_abs_diff": maxdiff,
            **extra,
        }

    run("serial", lambda u: arm_serial(u, "thread_local", True, False))
    run("serial-globalmode", lambda u: arm_serial(u, "global", True, False))
    for n in args.threads:
        run(f"threads-{n}-ownpool", lambda u, n=n: arm_threads(u, n, "thread_local", False, False))
    for n in args.threads:
        run(f"threads-{n}-sharedpool", lambda u, n=n: arm_threads(u, n, "thread_local", True, False))

    print()
    run("serial-noalloc", lambda u: arm_serial(u, "thread_local", True, True), True)
    for n in args.threads:
        run(
            f"threads-{n}-noalloc",
            lambda u, n=n: arm_threads(u, n, "thread_local", False, True),
            True,
        )

    # Last: a failed capture in "global" mode leaves the process with a capture
    # still underway, after which even empty_cache() raises. Nothing can follow
    # it, which is part of the finding.
    print()
    run("threads-2-globalmode", lambda u: arm_threads(u, 2, "global", False, False))

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"args": vars(args), "results": results}, f, indent=2)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
