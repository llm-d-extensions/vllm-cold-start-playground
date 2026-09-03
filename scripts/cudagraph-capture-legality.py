#!/usr/bin/env python3
"""What is legal while another thread is capturing a CUDA graph?

A threaded capture implementation does not get to choose the ambient calls its
process makes: PyTorch's own `torch.cuda.graph.__enter__` calls
`torch.cuda.synchronize()` and `empty_cache()`, and vLLM's capture loop sits
inside `graph_capture()`. This probe checks each of those against a capture that
is underway in another thread, under both capture error modes, and reports both
whether the call raises **and whether the capture survived it** -- which is the
fact a design needs, and the harsher of the two.

Each probe gets its own fresh capture (`--isolated`, the default). That matters:
the first thing worth probing is a device sync, and a device sync *invalidates*
the in-flight capture, so a single shared capture would leave every later probe
running against a capture that is already dead. `--sequential` reproduces that
weaker one-capture-for-all shape, and its rows are only trustworthy down to the
first RAISES.

Everything runs on one CUDA stream per thread; the capture is held open by a
barrier so the probe runs strictly inside the capture window.
"""

from __future__ import annotations

import argparse
import threading
import time

import torch

DEV = "cuda"


def _second_capture(own_pool: bool, pool=None) -> None:
    a = torch.randn(64, 64, device=DEV)
    b = torch.randn(64, 64, device=DEV)
    st = torch.cuda.Stream()
    with torch.cuda.stream(st):
        for _ in range(3):
            a @ b
    st.synchronize()
    g = torch.cuda.CUDAGraph()
    p = torch.cuda.graph_pool_handle() if own_pool else pool
    with torch.cuda.stream(st):
        g.capture_begin(pool=p, capture_error_mode="thread_local")
        c = a @ b  # noqa: F841
        g.capture_end()
    st.synchronize()
    g.replay()


def checks_for(x, w, pool):
    return {
        "torch.cuda.synchronize()": lambda: torch.cuda.synchronize(),
        "torch.cuda.empty_cache()": lambda: torch.cuda.empty_cache(),
        "torch.cuda.current_stream().synchronize()": (
            lambda: torch.cuda.current_stream().synchronize()
        ),
        "torch.zeros(1<<22, device=cuda)  # fresh cudaMalloc": (
            lambda: torch.zeros(1 << 22, device=DEV)
        ),
        "eager matmul on default stream": lambda: (x @ w).sum().item(),
        "torch.cuda.mem_get_info()": lambda: torch.cuda.mem_get_info(),
        "second capture, own pool": lambda: _second_capture(own_pool=True),
        "second capture, same pool": lambda: _second_capture(own_pool=False,
                                                             pool=pool),
    }


class Capture:
    """A graph capture held open in another thread until released."""

    def __init__(self, mode: str):
        self.mode = mode
        self.x = torch.randn(256, 256, device=DEV)
        self.w = torch.randn(256, 256, device=DEV)
        self.pool = torch.cuda.graph_pool_handle()
        self._stream = torch.cuda.Stream()
        self._graph = torch.cuda.CUDAGraph()
        self._inside = threading.Event()
        self._release = threading.Event()
        self.err: list[str] = []
        self._th = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        try:
            with torch.cuda.stream(self._stream):
                for _ in range(3):  # warm up in the capturing thread
                    self.x @ self.w
            self._stream.synchronize()
            with torch.cuda.stream(self._stream):
                self._graph.capture_begin(pool=self.pool,
                                          capture_error_mode=self.mode)
                y = self.x @ self.w  # noqa: F841
                self._inside.set()
                self._release.wait(30)
                self._graph.capture_end()
            self._stream.synchronize()
        except Exception as exc:  # noqa: BLE001
            self.err.append("%s: %s" % (type(exc).__name__,
                                        str(exc).splitlines()[0]))
            self._inside.set()

    def __enter__(self) -> "Capture":
        self._th.start()
        self._inside.wait(30)
        time.sleep(0.05)
        return self

    def __exit__(self, *exc) -> None:
        self._release.set()
        self._th.join(30)


def probe(mode: str, isolated: bool) -> None:
    print("\n--- another thread capturing with capture_error_mode=%r, %s ---"
          % (mode, "one fresh capture per probe" if isolated
             else "one capture shared by all probes"))
    print("  %-52s %-8s %s" % ("operation, from a non-capturing thread",
                               "probe", "capture survived"))

    if isolated:
        for name, _ in checks_for(None, None, None).items():
            with Capture(mode) as cap:
                fn = checks_for(cap.x, cap.w, cap.pool)[name]
                verdict, detail = _run_check(fn)
            print("  %-52s %-8s %s" % (name, verdict,
                                       "no  (%s)" % cap.err[0] if cap.err
                                       else "yes"))
            if detail:
                print("        %s" % detail)
        return

    with Capture(mode) as cap:
        for name, fn in checks_for(cap.x, cap.w, cap.pool).items():
            verdict, detail = _run_check(fn)
            print("  %-52s %-8s %s" % (name, verdict, "-"))
            if detail:
                print("        %s" % detail)
    print("  capturing thread: %s" % (cap.err[0] if cap.err
                                      else "finished cleanly"))


def _run_check(fn) -> tuple[str, str]:
    try:
        fn()
        return "ok", ""
    except Exception as exc:  # noqa: BLE001
        return "RAISES", "%s: %s" % (type(exc).__name__,
                                     str(exc).splitlines()[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequential", action="store_true",
                    help="one capture for all probes (the weaker shape; rows "
                         "after the first RAISES are not trustworthy)")
    args = ap.parse_args()

    print("torch %s  cuda %s  %s" % (torch.__version__, torch.version.cuda,
                                     torch.cuda.get_device_name(0)))
    torch.zeros(1, device=DEV)  # init context
    probe("thread_local", isolated=not args.sequential)
    # "global" is torch's default and poisons the process on failure, so it
    # goes last.
    probe("global", isolated=not args.sequential)


if __name__ == "__main__":
    main()
