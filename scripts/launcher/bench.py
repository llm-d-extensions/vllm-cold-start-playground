#!/usr/bin/env python3
"""Cold-start benchmark client for the vLLM launcher.

Runs *inside* the pod so all timing is local (no kube-apiserver round trips).

It measures the one number the experiment cares about: the wall-clock time from
the moment the launcher is asked to create a vLLM instance (POST /v2/vllm/
instances) to the moment that instance's OpenAI server answers /health with 200
-- i.e. "time for vLLM to boot from the instant the launcher issues process
creation". The launcher issues multiprocessing.Process.start() synchronously
inside the POST handler, so the POST send time is the process-creation instant
to within a millisecond.

It then reads the instance log and pulls out vLLM's *own* self-reported phase
durations (weight load, engine init, CUDA graph capture, server up), which are
far more reliable than diffing 1-second-resolution log timestamps.

Stdlib only -- nothing to install in the image.
"""
import argparse
import glob
import json
import re
import time
import urllib.request
import urllib.error


def http(method, url, body=None, timeout=5):
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read()


def poll_ready(vllm_port, deadline):
    """Poll the vLLM OpenAI server /health until 200. Return time of first 200."""
    url = f"http://127.0.0.1:{vllm_port}/health"
    while time.monotonic() < deadline:
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=2) as r:
                if r.status == 200:
                    return time.monotonic()
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(0.1)
    return None


# vLLM self-reported durations (embedded in the log, second/decimal precision).
PATTERNS = {
    # "Loading model weights took 12.34 GiB and 15.67 seconds"
    "weight_load_s": re.compile(r"Loading model weights took [\d.]+ GiB and ([\d.]+) seconds"),
    # "Model loading took X GiB and Y seconds" (alt phrasing)
    "model_loading_s": re.compile(r"Model loading took .*?([\d.]+) seconds"),
    # "init engine (profile, create kv cache, warmup model) took 20.12 seconds"
    "init_engine_s": re.compile(r"init engine \(.*?\) took ([\d.]+) seconds"),
    # "Graph capturing finished in 11 secs" / "in 11.4 secs"
    "cudagraph_capture_s": re.compile(r"Graph capturing finished in ([\d.]+) secs"),
    # "Capturing CUDA graph shapes: 100%" -> count only, not timed here
    # torch.compile: "Dynamo bytecode transform time: 0.25 s"
    "dynamo_s": re.compile(r"Dynamo bytecode transform time: ([\d.]+) s"),
    # "torch.compile takes X s in total"
    "compile_total_s": re.compile(r"torch.compile takes ([\d.]+) s in total"),
}

# Milestone lines: record the offset (s) of first occurrence from log start.
MILESTONES = {
    "api_server_start": re.compile(r"Starting vLLM API server"),
    "engine_core_started": re.compile(r"EngineCore.*(started|waiting for work)|Engine.*started"),
    "weights_start": re.compile(r"Starting to load model"),
    "capture_start": re.compile(r"Capturing CUDA graph"),
    "server_up": re.compile(r"(Application startup complete|Uvicorn running on)"),
}

TS_RE = re.compile(r"\b(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})")


def parse_log(text):
    out = {"durations": {}, "milestones": {}, "captured_shapes": None}
    for name, pat in PATTERNS.items():
        m = pat.search(text)
        if m:
            out["durations"][name] = float(m.group(1))
    # milestone offsets relative to first parseable timestamp
    lines = text.splitlines()
    base = None
    def ts_of(line):
        m = TS_RE.search(line)
        if not m:
            return None
        mo, d, h, mi, s = map(int, m.groups())
        return ((d * 24 + h) * 60 + mi) * 60 + s
    for line in lines:
        t = ts_of(line)
        if t is not None:
            base = t
            break
    if base is not None:
        for name, pat in MILESTONES.items():
            for line in lines:
                if pat.search(line):
                    t = ts_of(line)
                    if t is not None:
                        out["milestones"][name] = t - base
                    break
    m = re.findall(r"Capturing CUDA graph shapes:\s*100%.*?(\d+)/(\d+)", text)
    if m:
        out["captured_shapes"] = int(m[-1][1])
    return out


def run_once(launcher_port, vllm_port, instance_id, options, env_vars,
             label="", logdir="/tmp", timeout=600):
    """Create one instance, time it to /health 200, parse its log. Returns dict."""
    config = {"options": options, "env_vars": env_vars}
    # Create via PUT so we control the instance id (-> we know the log path).
    create_url = f"http://127.0.0.1:{launcher_port}/v2/vllm/instances/{instance_id}"

    t_request = time.monotonic()
    status, resp = http("PUT", create_url, config, timeout=30)
    t_created = time.monotonic()  # POST returned == fork issued
    if status not in (200, 201):
        return {"label": label, "error": f"create returned {status}",
                "resp": resp.decode()[:500]}

    deadline = t_request + timeout
    t_ready = poll_ready(vllm_port, deadline)

    result = {
        "label": label,
        "instance_id": instance_id,
        "options": options,
        "env": env_vars,
        "post_overhead_s": round(t_created - t_request, 3),
    }
    if t_ready is None:
        result["ready"] = False
        result["timeout_s"] = timeout
    else:
        result["ready"] = True
        result["total_s"] = round(t_ready - t_request, 3)

    pattern = f"{logdir}/launcher-*-vllm-{instance_id}.log"
    matches = glob.glob(pattern)
    if matches:
        try:
            with open(matches[0], "r", errors="replace") as f:
                text = f.read()
            result["log_path"] = matches[0]
            result["phases"] = parse_log(text)
        except OSError as e:
            result["log_error"] = str(e)
    else:
        result["log_error"] = f"no log matching {pattern}"
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--launcher-port", type=int, default=8001)
    ap.add_argument("--vllm-port", type=int, required=True)
    ap.add_argument("--instance-id", required=True)
    ap.add_argument("--options", required=True)
    ap.add_argument("--env", default="{}")
    ap.add_argument("--label", default="")
    ap.add_argument("--logdir", default="/tmp")
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()
    result = run_once(args.launcher_port, args.vllm_port, args.instance_id,
                      args.options, json.loads(args.env), args.label,
                      args.logdir, args.timeout)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
