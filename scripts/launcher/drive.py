#!/usr/bin/env python3
"""In-pod driver for one launcher "arm".

Starts a launcher (either the stock warm-import launcher.py or the cold-import
variant), waits for it to be healthy, then runs N sequential vLLM launches on
that *same* persistent launcher -- so launch #1 is the first launch on a fresh
launcher and launch #2..N reuse the warm process (and whatever the node/PVC
caches have warmed). Each launch is timed POST -> /health 200, its instance is
deleted before the next, and the launcher is torn down at the end.

Emits one JSON object: {arm, module, launches:[result, ...]}.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
import urllib.error

import bench


def launcher_healthy(port, deadline):
    url = f"http://127.0.0.1:{port}/health"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(0.2)
    return False


def delete_instance(launcher_port, instance_id):
    url = f"http://127.0.0.1:{launcher_port}/v2/vllm/instances/{instance_id}"
    try:
        req = urllib.request.Request(url, method="DELETE")
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status
    except Exception as e:
        return str(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, help="label for this arm")
    ap.add_argument("--module", default="launcher.py",
                    help="launcher.py (warm-import) or launcher_coldimport.py")
    ap.add_argument("--launcher-port", type=int, default=8001)
    ap.add_argument("--vllm-base-port", type=int, default=8100)
    ap.add_argument("--options", required=True, help="vLLM serve options string")
    ap.add_argument("--env", default="{}", help="per-instance env_vars JSON")
    ap.add_argument("--launches", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--logdir", default="/tmp")
    ap.add_argument("--srcdir", default="/opt/launcher")
    ap.add_argument("--launcher-log", default="/tmp/launcher-stdout.log")
    ap.add_argument("--settle", type=float, default=3.0,
                    help="seconds to wait after DELETE before next launch")
    ap.add_argument("--out", default="", help="also write result JSON to this path")
    args = ap.parse_args()

    env_vars = json.loads(args.env)

    # Start the launcher process.
    lp = open(args.launcher_log, "wb")
    t_launcher_start = time.monotonic()
    proc = subprocess.Popen(
        [sys.executable, os.path.join(args.srcdir, args.module),
         "--port", str(args.launcher_port), "--log-level", "info"],
        cwd=args.srcdir, stdout=lp, stderr=subprocess.STDOUT,
        env={**os.environ},
    )
    out = {"arm": args.arm, "module": args.module, "options": args.options,
           "env": env_vars, "launches": []}
    try:
        if not launcher_healthy(args.launcher_port, t_launcher_start + 300):
            out["error"] = "launcher did not become healthy in 300s"
            print(json.dumps(out))
            return
        out["launcher_ready_s"] = round(time.monotonic() - t_launcher_start, 3)

        for i in range(args.launches):
            vllm_port = args.vllm_base_port + i
            inst = f"{args.arm}-{i}"
            # The port must be unique per launch; drive.py owns it, so the
            # caller's --options must NOT contain --port.
            options = f"{args.options} --port {vllm_port}"
            res = bench.run_once(
                args.launcher_port, vllm_port, inst, options, env_vars,
                label=f"{args.arm}#{i}", logdir=args.logdir, timeout=args.timeout,
            )
            res["launch_index"] = i
            # Preserve the instance log before DELETE (launcher unlinks it on stop).
            src = res.get("log_path")
            if src and os.path.exists(src):
                keepdir = os.path.join(args.srcdir, "logs")
                os.makedirs(keepdir, exist_ok=True)
                dst = os.path.join(keepdir, f"{inst}.log")
                try:
                    with open(src, "rb") as fi, open(dst, "wb") as fo:
                        fo.write(fi.read())
                    res["kept_log"] = dst
                except OSError:
                    pass
            out["launches"].append(res)
            delete_instance(args.launcher_port, inst)
            time.sleep(args.settle)
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        lp.close()

    payload = json.dumps(out)
    if args.out:
        try:
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            with open(args.out, "w") as f:
                f.write(payload)
        except OSError:
            pass
    print(payload)


if __name__ == "__main__":
    main()
