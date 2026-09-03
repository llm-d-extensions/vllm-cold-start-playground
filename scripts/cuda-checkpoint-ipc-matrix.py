#!/usr/bin/env python3
"""Isolate *which* piece of cross-rank GPU state makes `cuda-checkpoint` hang.

`reports/demo-cold-start-tp2-*.txt` records a TP=2 park attempt where
`/sleep?level=1` and `/checkpoint_prepare` both succeed and
`cuda-checkpoint --action checkpoint` then hangs forever on the first rank.
Bisecting that with real vLLM boots costs ~96s per trial and mixes a dozen
variables. This reproducer removes vLLM entirely: two child processes, one GPU
each, holding exactly one kind of cross-process state, then the same
lock -> checkpoint -> restore -> unlock sequence the demo runs.

Each mode isolates one mechanism. `plain` is the control (no cross-process
state at all); the `_teardown` variants answer the question vLLM's
`Worker.checkpoint_prepare` is *trying* to answer -- if releasing the state
first makes the checkpoint complete, then an incomplete teardown is the bug and
it is fixable in vLLM; if it still hangs, the blocker is below vLLM.

    plain             one context per child, nothing shared            (control)
    nccl              live NCCL communicator (torch.distributed)
    nccl_teardown     ... destroy_process_group() before checkpoint
    nccl_graph        a captured CUDA graph containing an all-reduce, then
                      destroy_process_group() -- the sleep-level-1 shape, where
                      the graphs are deliberately *kept*. `pynccl.py:148` warns
                      that ncclCommAbort blocks until every graph that captured
                      a NCCL op on the comm is gone, so this measures whether
                      "keep the graphs" and "release the comm" can both hold.
    nccl_graph_free   ... drop the graph first, then destroy. If this is the
                      only one that works, a TP>1 park costs the capture phase
                      it was supposed to save.
    ipc               peer tensor mapped via cuIpcOpenMemHandle
                      -- this is what vLLM's CUSTOM all-reduce does
    ipc_teardown      ... importer frees the mapping before checkpoint
    symm              torch symmetric memory
                      -- cuMemExportToShareableHandle, which cuda-checkpoint's
                         own README lists as unsupported

Ordering matters to the driver: the README says "cuda-checkpoint must be
invoked on the processes in a job sequentially", so serial is the documented
shape -- but that sentence belongs to the driver-610 job-file feature. --parallel
tests the other reading (both ranks mid-checkpoint at once), which is what the
demo's serial for-loop could not do.

Usage (inside the GPU pod, cuda-checkpoint staged at $CC):
    python3 cuda-checkpoint-ipc-matrix.py                  # every mode, serial
    python3 cuda-checkpoint-ipc-matrix.py --mode ipc --parallel
    python3 cuda-checkpoint-ipc-matrix.py --json /tmp/matrix.json
"""

import argparse
import json
import os
import pickle
import subprocess
import sys
import time
import multiprocessing as mp

CC = os.environ.get("CC", "/tmp/cuda-checkpoint")
MODES = ["plain", "nccl", "nccl_teardown", "nccl_graph", "nccl_graph_free",
         "ipc", "ipc_teardown", "symm"]
# Modes whose cross-rank state is released on command before the checkpoint.
TEARDOWN_MODES = {"nccl_teardown", "ipc_teardown", "nccl_graph", "nccl_graph_free"}
NBYTES = 64 << 20  # 64 MiB per buffer: big enough to be a real mapping, small
                   # enough that allocation is instant.


# --------------------------------------------------------------------------
# child: hold one kind of cross-process state, then obey commands via files.
# Files rather than pipes so that a child wedged inside the CUDA driver still
# leaves an inspectable trace of how far it got.
# --------------------------------------------------------------------------
def child(mode: str, rank: int, d: str) -> None:
    def say(stage, **kw):
        with open(f"{d}/log-{rank}", "a") as f:
            f.write(json.dumps({"t": round(time.time(), 3), "stage": stage, **kw}) + "\n")

    try:
        import torch
        import torch.distributed as dist

        torch.cuda.set_device(rank)
        n = NBYTES // 2  # bfloat16 elements
        say("cuda_init", torch=torch.__version__, dev=torch.cuda.get_device_name(rank))

        local = torch.ones(n, dtype=torch.bfloat16, device=f"cuda:{rank}")
        peer = None
        pg = False

        if mode.startswith("nccl") or mode == "symm":
            dist.init_process_group(
                backend="nccl",
                init_method=f"file://{d}/store",
                rank=rank,
                world_size=2,
                device_id=torch.device(f"cuda:{rank}"),
            )
            pg = True
            dist.all_reduce(local)
            torch.cuda.synchronize()
            say("nccl_ready", val=float(local[0]))

        graph = None
        if mode.startswith("nccl_graph"):
            # Capture an all-reduce into a CUDA graph, the way vLLM's warmup
            # captures 51 sizes x 2 modes with the TP all-reduce inside them.
            gstream = torch.cuda.Stream()
            gstream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(gstream):
                dist.all_reduce(local)
            torch.cuda.current_stream().wait_stream(gstream)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                dist.all_reduce(local)
            graph.replay()
            torch.cuda.synchronize()
            say("graph_captured", val=float(local[0]))

        if mode == "symm":
            # cuMemCreate + cuMemExportToShareableHandle under the hood -- the
            # exact allocation shape cuda-checkpoint's README says it does not
            # support.
            from torch.distributed._symmetric_memory import empty as symm_empty
            from torch.distributed._symmetric_memory import rendezvous

            symm = symm_empty(n, dtype=torch.bfloat16, device=f"cuda:{rank}")
            symm.fill_(1)
            hdl = rendezvous(symm, group=dist.group.WORLD)
            # Touch a peer buffer so the mapping is live, not merely created.
            pbuf = hdl.get_buffer(1 - rank, (n,), torch.bfloat16)
            torch.cuda.synchronize()
            say("symm_ready", peer_val=float(pbuf[0]), world=hdl.world_size)
            peer = pbuf

        if mode.startswith("ipc"):
            # cuIpcGetMemHandle / cuIpcOpenMemHandle, via torch's own reducer.
            # This is mechanically what vLLM's CUSTOM all-reduce does with its
            # per-rank signal and data buffers.
            from torch.multiprocessing.reductions import reduce_tensor

            with open(f"{d}/h-{rank}.pkl", "wb") as f:
                pickle.dump(reduce_tensor(local), f)
            os.rename(f"{d}/h-{rank}.pkl", f"{d}/handle-{rank}.pkl")
            other = f"{d}/handle-{1 - rank}.pkl"
            for _ in range(600):
                if os.path.exists(other):
                    break
                time.sleep(0.1)
            with open(other, "rb") as f:
                fn, args = pickle.load(f)
            peer = fn(*args)
            # Read it: an unread mapping may not be materialised.
            got = float(peer[0])
            torch.cuda.synchronize()
            say("ipc_ready", peer_val=got, peer_dev=str(peer.device))

        open(f"{d}/ready-{rank}", "w").close()
        say("ready")

        # Command loop. `teardown` releases whatever cross-process state this
        # mode holds -- the probe for "is an incomplete teardown the bug?".
        # `verify` proves the process still computes after a restore.
        while True:
            cmd = f"{d}/cmd-{rank}"
            if os.path.exists(cmd):
                what = open(cmd).read().strip()
                os.remove(cmd)
                if what == "teardown":
                    import gc
                    import threading

                    if mode == "nccl_graph_free" and graph is not None:
                        # Drop the graphs first: this is the trade the park
                        # would have to make at TP>1.
                        del graph
                        graph = None
                        gc.collect()
                        say("graph_dropped")
                    if peer is not None:
                        del peer
                        peer = None
                    if pg:
                        # destroy_process_group() calls into ncclCommAbort/
                        # Destroy, which can block indefinitely. Time it in a
                        # thread so a block is recorded rather than inherited.
                        box = {}
                        t0 = time.time()

                        def _destroy():
                            try:
                                dist.destroy_process_group()
                                box["ok"] = True
                            except Exception as e:  # noqa: BLE001
                                box["err"] = f"{type(e).__name__}: {e}"

                        th = threading.Thread(target=_destroy, daemon=True)
                        th.start()
                        th.join(timeout=30.0)
                        say("destroy_pg",
                            returned=not th.is_alive(),
                            s=round(time.time() - t0, 2), **box)
                        pg = th.is_alive()  # still held if the abort blocked
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    say("torn_down")
                    open(f"{d}/torn-{rank}", "w").close()
                elif what == "verify":
                    out = float((local * 2).sum())
                    say("verified", sum=out)
                    with open(f"{d}/verified-{rank}", "w") as f:
                        f.write(repr(out))
            time.sleep(0.1)
    except Exception as e:  # noqa: BLE001 - a child crash must be visible, not silent
        say("error", err=f"{type(e).__name__}: {e}")
        open(f"{d}/failed-{rank}", "w").close()
        raise


# --------------------------------------------------------------------------
# parent: drive cuda-checkpoint and time every action.
# --------------------------------------------------------------------------
def cc(action: str, pid: int, wall: float, extra=()) -> dict:
    """One cuda-checkpoint invocation, wall-clock bounded.

    A hang is the result we are hunting, so a timeout is data, not an error:
    record it and SIGKILL the client (which is what the demo run had to do by
    hand).
    """
    argv = [CC, "--action", action, "--pid", str(pid), *extra]
    t0 = time.time()
    p = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        out = p.communicate(timeout=wall)[0]
        return {"action": action, "pid": pid, "ok": p.returncode == 0,
                "rc": p.returncode, "s": round(time.time() - t0, 2),
                "out": (out or "").strip()[:400]}
    except subprocess.TimeoutExpired:
        p.kill()
        p.communicate()
        return {"action": action, "pid": pid, "ok": False, "hung": True,
                "s": round(time.time() - t0, 2),
                "out": f"HUNG: no return in {wall}s (client SIGKILLed)"}


def state(pid: int, wall: float = 10.0) -> str:
    """`--get-state`, bounded. It blocking is itself a finding: it means the
    driver's per-process checkpoint lock is held by the wedged call."""
    try:
        r = subprocess.run([CC, "--get-state", "--pid", str(pid)],
                           capture_output=True, text=True, timeout=wall)
        return (r.stdout or r.stderr).strip() or f"rc={r.returncode}"
    except subprocess.TimeoutExpired:
        return "BLOCKED"


def mem() -> str:
    r = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used",
                        "--format=csv,noheader"], capture_output=True, text=True)
    return " | ".join(r.stdout.split("\n")[:2]).strip()


def wchan(pid: int) -> str:
    try:
        return open(f"/proc/{pid}/wchan").read().strip() or "?"
    except OSError:
        return "gone"


def run_mode(mode: str, parallel: bool, wall: float) -> dict:
    import shutil
    import tempfile

    d = tempfile.mkdtemp(prefix=f"ccipc-{mode}-")
    res = {"mode": mode, "parallel": parallel, "idle_mem_before": mem(), "steps": []}
    ctx = mp.get_context("spawn")  # never fork: the parent must stay CUDA-free
    procs = [ctx.Process(target=child, args=(mode, r, d), daemon=True) for r in (0, 1)]
    for p in procs:
        p.start()
    pids = [p.pid for p in procs]
    res["pids"] = pids
    print(f"\n=== mode={mode} parallel={parallel} pids={pids}")
    print(f"    idle before: {res['idle_mem_before']}")

    try:
        t0 = time.time()
        while time.time() - t0 < 240:
            if all(os.path.exists(f"{d}/ready-{r}") for r in (0, 1)):
                break
            if any(os.path.exists(f"{d}/failed-{r}") for r in (0, 1)):
                res["error"] = "child failed during setup"
                break
            if any(not p.is_alive() for p in procs):
                res["error"] = "child exited during setup"
                break
            time.sleep(0.2)
        else:
            res["error"] = "children never became ready"

        if "error" in res:
            for r in (0, 1):
                lg = f"{d}/log-{r}"
                if os.path.exists(lg):
                    res.setdefault("child_log", {})[r] = open(lg).read().strip().split("\n")
            print(f"    ERROR: {res['error']}")
            return res

        res["mem_ready"] = mem()
        print(f"    both ready, mem: {res['mem_ready']}")

        if mode in TEARDOWN_MODES:
            for r in (0, 1):
                with open(f"{d}/cmd-{r}", "w") as f:
                    f.write("teardown")
            t0 = time.time()
            while time.time() - t0 < 60 and not all(
                os.path.exists(f"{d}/torn-{r}") for r in (0, 1)
            ):
                time.sleep(0.1)
            torn = all(os.path.exists(f"{d}/torn-{r}") for r in (0, 1))
            res["steps"].append({"action": "teardown", "ok": torn,
                                 "s": round(time.time() - t0, 2)})
            res["mem_after_teardown"] = mem()
            print(f"    teardown ok={torn}  mem: {res['mem_after_teardown']}")
            # The destroy_pg line is the interesting one: did the abort return?
            for r in (0, 1):
                lg = f"{d}/log-{r}"
                if os.path.exists(lg):
                    for ln in open(lg):
                        if '"destroy_pg"' in ln or '"graph_' in ln:
                            print(f"      rank{r}: {ln.strip()}")

        for pid in pids:
            r = cc("lock", pid, wall=30, extra=["--timeout", "20000"])
            res["steps"].append(r)
            print(f"    lock {pid}: ok={r['ok']} {r['s']}s {r['out']}")
            if not r["ok"]:
                res["verdict"] = "lock failed"
                return res

        if parallel:
            t0 = time.time()
            clients = [
                subprocess.Popen([CC, "--action", "checkpoint", "--pid", str(pid)],
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True)
                for pid in pids
            ]
            for pid, c in zip(pids, clients):
                try:
                    out = c.communicate(timeout=max(1.0, wall - (time.time() - t0)))[0]
                    r = {"action": "checkpoint(par)", "pid": pid, "ok": c.returncode == 0,
                         "rc": c.returncode, "s": round(time.time() - t0, 2),
                         "out": (out or "").strip()[:400]}
                except subprocess.TimeoutExpired:
                    c.kill()
                    c.communicate()
                    r = {"action": "checkpoint(par)", "pid": pid, "ok": False,
                         "hung": True, "s": round(time.time() - t0, 2),
                         "out": f"HUNG: no return in {wall}s"}
                res["steps"].append(r)
                print(f"    checkpoint(par) {pid}: ok={r['ok']} {r['s']}s {r['out']}")
        else:
            for pid in pids:
                r = cc("checkpoint", pid, wall=wall)
                res["steps"].append(r)
                print(f"    checkpoint {pid}: ok={r['ok']} {r['s']}s {r['out']}")
                if not r["ok"]:
                    break

        res["states"] = {pid: state(pid) for pid in pids}
        res["wchan"] = {pid: wchan(pid) for pid in pids}
        res["mem_after_checkpoint"] = mem()
        print(f"    states: {res['states']}")
        print(f"    wchan:  {res['wchan']}")
        print(f"    mem after checkpoint: {res['mem_after_checkpoint']}")

        ckpt_ok = [s for s in res["steps"] if s["action"].startswith("checkpoint")]
        if ckpt_ok and all(s["ok"] for s in ckpt_ok):
            for pid in pids:
                r = cc("restore", pid, wall=wall)
                res["steps"].append(r)
                print(f"    restore {pid}: ok={r['ok']} {r['s']}s {r['out']}")
            for pid in pids:
                r = cc("unlock", pid, wall=30)
                res["steps"].append(r)
                print(f"    unlock {pid}: ok={r['ok']} {r['s']}s {r['out']}")
            ok = all(s["ok"] for s in res["steps"] if s["action"] in ("restore", "unlock"))
            print(f"    restore+unlock ok={ok}  mem: {mem()}")
            if ok:
                for r in (0, 1):
                    with open(f"{d}/cmd-{r}", "w") as f:
                        f.write("verify")
                t0 = time.time()
                while time.time() - t0 < 60 and not all(
                    os.path.exists(f"{d}/verified-{r}") for r in (0, 1)
                ):
                    time.sleep(0.1)
                vok = all(os.path.exists(f"{d}/verified-{r}") for r in (0, 1))
                res["verified_after_restore"] = vok
                print(f"    compute after restore: {'OK' if vok else 'FAILED'}")
            res["verdict"] = "round trip OK" if ok else "restore/unlock failed"
        else:
            res["verdict"] = "checkpoint HUNG" if any(
                s.get("hung") for s in ckpt_ok
            ) else "checkpoint failed"
        return res
    finally:
        for r in (0, 1):
            lg = f"{d}/log-{r}"
            if os.path.exists(lg):
                res.setdefault("child_log", {})[r] = open(lg).read().strip().split("\n")
        # A child wedged in the driver ignores SIGTERM; the demo needed SIGKILL
        # and so do we.
        for p in procs:
            if p.is_alive():
                p.kill()
        for p in procs:
            p.join(timeout=15)
        time.sleep(3)
        res["idle_mem_after_kill"] = mem()
        print(f"    idle after cleanup: {res['idle_mem_after_kill']}")
        shutil.rmtree(d, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", action="append", choices=MODES,
                    help="repeatable; default is every mode in order")
    ap.add_argument("--parallel", action="store_true",
                    help="start both ranks' checkpoints at once instead of serially")
    ap.add_argument("--wall", type=float, default=60.0,
                    help="seconds to wait for one checkpoint before calling it hung")
    ap.add_argument("--json", help="write the full result matrix here")
    a = ap.parse_args()

    if not os.access(CC, os.X_OK):
        print(f"no cuda-checkpoint at {CC} (set $CC)", file=sys.stderr)
        return 2

    drv = subprocess.run(["nvidia-smi", "--query-gpu=driver_version",
                          "--format=csv,noheader", "-i", "0"],
                         capture_output=True, text=True).stdout.strip()
    print(f"driver={drv}  cc={CC}  wall={a.wall}s  parallel={a.parallel}")
    print("NOTE: cuIpcGetMemHandle-based checkpoint support is a driver-610 "
          "feature; cuMemExportToShareableHandle memory is documented "
          "unsupported outright.")

    out = {"driver": drv, "parallel": a.parallel, "wall": a.wall, "results": []}
    for mode in a.mode or MODES:
        out["results"].append(run_mode(mode, a.parallel, a.wall))

    print("\n=== summary" + (" (parallel)" if a.parallel else " (serial)"))
    for r in out["results"]:
        print(f"  {r['mode']:16s} {r.get('verdict', r.get('error', '?'))}")

    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
