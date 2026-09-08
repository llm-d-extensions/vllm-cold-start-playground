#!/usr/bin/env python3
"""Run a command with PR_SET_CHILD_SUBREAPER so criu can restore the pids it dumped.

Why this exists. `criu dump` SIGKILLs the tree it dumped, and `criu restore`
then recreates it *at the same pids* (clone3 with set_tid, via
/proc/sys/kernel/ns_last_pid). A pid that is still occupied by a **zombie**
therefore breaks the restore outright:

    Error (criu/cr-restore.c:1230): Can't fork for 1074: File exists

In this pod that is the default outcome, not an edge case. Pid 1 in the
container is `sleep infinity`, which never calls wait(), so every process that
outlives its parent -- exactly what happens when criu kills a multi-process tree
from the root down -- is reparented to pid 1 and stays a zombie forever. vLLM's
EngineCore and WorkerProcs are grandchildren of the process we dump, so this hits
every arm with TP>=1.

Setting PR_SET_CHILD_SUBREAPER on this process makes the kernel reparent orphaned
descendants *here* instead of to pid 1, and the loop below reaps them. Run the
whole probe under it:

    python3 criu-reaper.py bash criu-vllm-probe.sh --arm ckpt --tp 1

Exits with the wrapped command's status. Reaped orphans are logged to stderr so a
transcript shows what was cleaned up and when.
"""

import ctypes
import os
import sys
import time

PR_SET_CHILD_SUBREAPER = 36


def main(argv):
    if len(argv) < 2:
        sys.stderr.write("usage: criu-reaper.py <cmd> [args...]\n")
        return 2
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    if libc.prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
        err = ctypes.get_errno()
        sys.stderr.write("[reaper] prctl(PR_SET_CHILD_SUBREAPER) failed: %s -- "
                         "orphans will land on pid 1 and block criu restore\n"
                         % os.strerror(err))
    else:
        sys.stderr.write("[reaper] subreaper armed, pid=%d\n" % os.getpid())
    sys.stderr.flush()

    child = os.fork()
    if child == 0:
        os.execvp(argv[1], argv[1:])
        os._exit(127)

    rc = 0
    while True:
        try:
            pid, status = os.waitpid(-1, 0)
        except ChildProcessError:
            break
        except InterruptedError:
            continue
        if pid == child:
            rc = os.waitstatus_to_exitcode(status)
            # The command is done, but descendants it deliberately left running
            # (a restored tree, say) may still be alive. Drain what is already
            # dead and go, rather than blocking on them.
            deadline = time.time() + 5
            while time.time() < deadline:
                try:
                    p, _ = os.waitpid(-1, os.WNOHANG)
                except ChildProcessError:
                    break
                if p == 0:
                    break
                sys.stderr.write("[reaper] reaped orphan %d\n" % p)
            break
        sys.stderr.write("[reaper] reaped orphan %d\n" % pid)
        sys.stderr.flush()
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv))
