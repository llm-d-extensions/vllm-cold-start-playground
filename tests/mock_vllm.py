#!/usr/bin/env python3
"""Synthetic stand-in for ``vllm serve``: a self-test for the cold-start tooling.

It reproduces the *shape* of a vLLM cold start -- serial phases across several
Python processes, a large sequential read, a few HTTP round trips, an HTTP
server that reports unhealthy until the engine is up -- without needing a GPU,
weights, or the vLLM image. Use it to check that the probe, the readiness
poller and the report agree end to end after changing any of them.

Phases are emitted with the same span names and categories the real probe
attaches to vLLM internals, so the report's phase attribution is exercised for
real. Everything else (imports, file reads, sockets, /proc sampling, readiness)
is genuinely happening.

    python tests/mock_vllm.py --port 8000 --weight-mb 512 --workers 1
"""

import argparse
import json
import multiprocessing as mp
import os
import socket
import sys
import threading
import time

sys.path.insert(0, os.environ.get("CS_PROBE_DIR", "/opt/coldstart"))
try:
    import cs_trace as T
except Exception:                                    # probe not installed
    class _N(object):
        def __getattr__(self, k):
            return lambda *a, **kw: _Null()
    class _Null(object):
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def set(self, **kw):
            pass
    T = _N()


def busy(seconds):
    """Burn CPU rather than sleeping: sleeping would hide cgroup throttling,
    which is one of the things this tool exists to surface."""
    end = time.monotonic() + seconds
    x = 0
    while time.monotonic() < end:
        x = (x * 31 + 7) % 1000003
    return x


def make_weights(path, mb):
    if os.path.exists(path) and os.path.getsize(path) >= mb * (1 << 20):
        return path
    chunk = os.urandom(1 << 20)
    with open(path, "wb") as fh:
        for _ in range(mb):
            fh.write(chunk)
    return path


def read_weights(path):
    n = 0
    with T.span("weights.load_weights", "weights", file=path):
        with open(path, "rb") as fh:
            while True:
                b = fh.read(8 << 20)
                if not b:
                    break
                n += len(b)
    return n


def worker_main(rank, weight_file, ready_q):
    """Runs in a *spawned* child: re-imports everything, re-inits, loads
    weights. This is the per-process cost multiplication the report calls out."""
    with T.span("worker.main", "executor", rank=rank):
        import hashlib, ssl, urllib.request, xml.etree.ElementTree  # noqa: F401
        with T.span("worker.init_device", "device", rank=rank):
            busy(0.4)
        with T.span("collective.init_env", "collective", rank=rank):
            busy(0.2)
        with T.span("weights.load_model", "weights", rank=rank):
            read_weights(weight_file)
        with T.span("compile.vllm_backend", "compile", rank=rank):
            busy(0.8)
        with T.span("cudagraph.capture_model", "cudagraph", rank=rank):
            busy(0.5)
        with T.span("kvcache.determine_memory", "kvcache", rank=rank):
            busy(0.2)
    ready_q.put(rank)


class Handler(object):
    pass


def serve(port, state):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _json(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._json(200 if state["ready"] else 503,
                           {"ready": state["ready"]})
            elif self.path == "/v1/models":
                if not state["ready"]:
                    self._json(503, {"error": "engine not ready"})
                else:
                    self._json(200, {"object": "list", "data": [
                        {"id": state["model"], "object": "model"}]})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0) or 0)
            self.rfile.read(length)
            if not state["ready"]:
                self._json(503, {"error": "not ready"})
                return
            # First request pays lazy warmup, exactly like a real first token.
            if not state["warm"]:
                with T.span("warmup.first_request_jit", "warmup"):
                    busy(state["first_request_s"])
                state["warm"] = True
            self._json(200, {"choices": [{"text": " pong"}]})

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("0.0.0.0", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--weight-mb", type=int, default=256)
    ap.add_argument("--weight-file", default="/tmp/mock-weights.safetensors")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--model", default="mock/tiny-1b")
    ap.add_argument("--first-request-s", type=float, default=1.0)
    ap.add_argument("--serve-seconds", type=float, default=25.0)
    args = ap.parse_args()

    with T.span("cli.main", "entrypoint"):
        # phase: config resolution, with the network round trips a real start
        # makes against the HF hub
        with T.span("config.create_engine_config", "config"):
            busy(0.3)
            try:
                socket.getaddrinfo("huggingface.co", 443)
            except Exception:
                pass
            with T.span("config.get_hf_config", "config"):
                busy(0.2)
            with T.span("config.get_tokenizer", "config"):
                busy(0.3)
        T.tracer.meta("vllm.config", model=args.model, dtype="bfloat16",
                      tensor_parallel_size=args.workers, enforce_eager=False,
                      compilation={"level": 3, "n_cudagraph_sizes": 8},
                      note="synthetic run from tests/mock_vllm.py")

        make_weights(args.weight_file, args.weight_mb)

        # phase: bind the socket early (like vLLM), long before ready
        with T.span("api.create_socket", "apiserver"):
            state = {"ready": False, "model": args.model, "warm": False,
                     "first_request_s": args.first_request_s}
            serve(args.port, state)

        # phase: engine + workers
        with T.span("engine.core_init", "engine"):
            with T.span("ipc.message_queue_init", "ipc"):
                busy(0.3)
            ctx = mp.get_context("spawn")
            q = ctx.Queue()
            procs = [ctx.Process(target=worker_main,
                                 args=(r, args.weight_file, q),
                                 name="MockWorker%d" % r)
                     for r in range(args.workers)]
            with T.span("worker.make_process", "executor"):
                for p in procs:
                    p.start()
            with T.span("engine.wait_for_startup", "ipc"):
                for _ in procs:
                    q.get()
            for p in procs:
                p.join()
            with T.span("kvcache.initialize", "kvcache"):
                busy(0.4)

        with T.span("apiserver.uvicorn_startup", "apiserver"):
            busy(0.2)
        state["ready"] = True
        T.tracer.instant("api.startup_complete", "ready",
                         note="mock engine ready")

    print("[mock_vllm] ready on :%d" % args.port, flush=True)
    time.sleep(args.serve_seconds)
    print("[mock_vllm] exiting", flush=True)


if __name__ == "__main__":
    main()
