#!/usr/bin/env python3
"""Readiness edge detector -- the outside view of "ready to serve".

Runs next to vLLM (sidecar container, or a background process in the same
container) and polls the HTTP surface at a tight interval, recording the first
instant each distinct readiness edge is crossed:

  ready.port_open    TCP connect succeeds -- the socket is bound and listening.
                     vLLM binds early, so this is *not* readiness; the gap
                     between this and health_200 is engine bring-up.
  ready.health_200   GET /health returns 200.
  ready.models_200   GET /v1/models returns 200 with the model listed.
  ready.first_token   a real 1-token completion succeeds. This is the edge that
                     matters for serving: /health can pass while the first
                     request still triggers lazy work (JIT kernels, sampler
                     warmup), and that latency is part of cold start even though
                     no readiness probe sees it.

Self-contained (stdlib only) so it can run in a sidecar image that has no vLLM
and no probe on PYTHONPATH. Emits the same JSONL event schema as the in-process
probe, so the analysis tool merges both without special-casing.
"""

import argparse
import json
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.request

STATES = ("port_open", "health_200", "models_200", "first_token")


class Writer(object):
    def __init__(self, trace_dir, run_id, role="ready"):
        self.path = None
        self.fh = None
        self.run_id = run_id
        self.role = role
        if trace_dir:
            try:
                os.makedirs(trace_dir, exist_ok=True)
                self.path = os.path.join(
                    trace_dir, "events.%s.%d.jsonl" % (role, os.getpid()))
                self.fh = open(self.path, "a", buffering=1)
            except Exception as e:
                sys.stderr.write("[cs_ready] cannot write trace: %s\n" % e)

    def event(self, ph, name, cat, ts, **kw):
        ev = {"ph": ph, "name": name, "cat": cat, "ts": ts,
              "pid": os.getpid(), "tid": threading.get_ident(),
              "role": self.role}
        ev.update(kw)
        if self.fh:
            try:
                self.fh.write(json.dumps(ev, separators=(",", ":")) + "\n")
            except Exception:
                pass

    def instant(self, name, ts, **args):
        self.event("i", name, "ready", ts, args=args or None)

    def span(self, name, ts, dur, **args):
        self.event("X", name, "ready", ts, dur=dur, args=args or None)

    def meta(self, name, **args):
        self.event("M", name, "ready", time.time(), args=args)


def tcp_open(host, port, timeout):
    s = socket.socket()
    s.settimeout(timeout)
    try:
        s.connect((host, port))
        return True, None
    except Exception as e:
        return False, "%s: %s" % (type(e).__name__, e)
    finally:
        try:
            s.close()
        except Exception:
            pass


def http_get(url, timeout):
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read(4096)
    except urllib.error.HTTPError as e:
        return e.code, b""
    except Exception as e:
        return None, str(e).encode()[:200]


def http_post(url, payload, timeout):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read(65536)
    except urllib.error.HTTPError as e:
        try:
            return e.code, e.read(2048)
        except Exception:
            return e.code, b""
    except Exception as e:
        return None, str(e).encode()[:200]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default=os.environ.get("CS_READY_HOST",
                                                     "127.0.0.1"))
    ap.add_argument("--port", type=int,
                    default=int(os.environ.get("CS_READY_PORT", "8000")))
    ap.add_argument("--interval-ms", type=float,
                    default=float(os.environ.get("CS_READY_INTERVAL_MS", "20")))
    ap.add_argument("--timeout", type=float,
                    default=float(os.environ.get("CS_READY_TIMEOUT", "1800")),
                    help="give up after this many seconds")
    ap.add_argument("--trace-dir", default=os.environ.get("CS_TRACE_DIR",
                                                          "/var/log/coldstart"))
    ap.add_argument("--run-id", default=os.environ.get("CS_RUN_ID", "run"))
    ap.add_argument("--health-path", default="/health")
    ap.add_argument("--models-path", default="/v1/models")
    ap.add_argument("--first-token", action="store_true",
                    default=os.environ.get("CS_READY_FIRST_TOKEN", "") not in
                    ("", "0", "false"),
                    help="after /v1/models, send a 1-token completion and "
                         "record when it returns")
    ap.add_argument("--model", default=os.environ.get("CS_MODEL", ""),
                    help="model name for the first-token request "
                         "(default: first entry of /v1/models)")
    ap.add_argument("--t0", type=float,
                    default=float(os.environ.get("CS_T0", "0") or 0),
                    help="wall-clock epoch seconds of the vLLM exec, for live "
                         "elapsed reporting")
    ap.add_argument("--exit-after-ready", action="store_true", default=True)
    args = ap.parse_args(argv)

    base = "http://%s:%d" % (args.host, args.port)
    w = Writer(args.trace_dir, args.run_id)
    t_start = time.time()
    t0 = args.t0 or t_start
    w.meta("ready.poller_start", base=base, interval_ms=args.interval_ms,
           t0=t0, first_token=args.first_token, pid=os.getpid())

    seen = {}
    last_err = None
    interval = args.interval_ms / 1000.0
    deadline = t_start + args.timeout
    model = args.model or None

    def mark(state, **args_):
        now = time.time()
        seen[state] = now
        w.instant("ready." + state, now, elapsed_since_t0_s=round(now - t0, 4),
                  **args_)
        sys.stdout.write("[cs_ready] %-12s t+%.3fs\n" % (state, now - t0))
        sys.stdout.flush()

    def note_error(kind, detail):
        nonlocal last_err
        sig = "%s|%s" % (kind, str(detail)[:80])
        if sig != last_err:
            last_err = sig
            w.instant("ready.transition", time.time(), kind=kind,
                      detail=str(detail)[:200],
                      elapsed_since_t0_s=round(time.time() - t0, 4))

    rc = 1
    while time.time() < deadline:
        if "port_open" not in seen:
            ok, err = tcp_open(args.host, args.port, 1.0)
            if ok:
                mark("port_open")
            else:
                note_error("tcp", err)
                time.sleep(interval)
                continue

        if "health_200" not in seen:
            code, body = http_get(base + args.health_path, 2.0)
            if code == 200:
                mark("health_200")
            else:
                note_error("health", "code=%s %s" % (code, body[:80]))
                time.sleep(interval)
                continue

        if "models_200" not in seen:
            code, body = http_get(base + args.models_path, 5.0)
            if code == 200:
                ids = []
                try:
                    ids = [d.get("id") for d in json.loads(body).get("data", [])]
                except Exception:
                    pass
                if ids and not model:
                    model = ids[0]
                mark("models_200", models=ids)
            else:
                note_error("models", "code=%s" % code)
                time.sleep(interval)
                continue

        if args.first_token and "first_token" not in seen:
            t_req = time.time()
            code, body = http_post(
                base + "/v1/completions",
                {"model": model or "", "prompt": "ping", "max_tokens": 1,
                 "temperature": 0.0}, 300.0)
            dur = time.time() - t_req
            if code == 200:
                w.span("ready.first_request", t_req, dur, model=model,
                       note="latency of the first real request; lazy kernel "
                            "JIT and sampler warmup land here")
                mark("first_token", request_s=round(dur, 4), model=model)
            else:
                note_error("completion", "code=%s %s" % (code, body[:120]))
                time.sleep(max(interval, 0.2))
                continue

        rc = 0
        break

    total = time.time() - t0
    w.meta("ready.summary",
           reached={k: round(v - t0, 4) for k, v in seen.items()},
           timed_out=rc != 0, total_s=round(total, 4), t0=t0,
           poller_overhead_note="poll interval bounds the resolution of these "
                                "edges (%.0f ms)" % args.interval_ms)
    if rc == 0:
        sys.stdout.write("[cs_ready] READY after %.3fs from t0\n" % total)
    else:
        sys.stdout.write("[cs_ready] TIMEOUT after %.1fs (reached: %s)\n"
                         % (total, ",".join(sorted(seen)) or "nothing"))
    sys.stdout.flush()
    return rc


if __name__ == "__main__":
    sys.exit(main())
