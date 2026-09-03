#!/usr/bin/env bash
# Regression test for coldstart-run.sh's exclusivity guard.
#
# The guard refuses to run a second harness in one container, because two vLLMs
# on one GPU produce numbers that quietly are not comparable. It did that job,
# but it also refused when nothing else was running: `$(pgrep ...)` forks a
# subshell that has not exec'd yet, so it still carries the script's own cmdline
# and pgrep matches it. Whether the /proc walk sees that fork is a race, so runs
# failed intermittently -- three boots in a row, then none, with the pid in the
# message a few above the script's own and gone by the time anyone looked.
#
# Both directions matter and only one of them was ever exercised by hand, so
# both are asserted here: alone it must allow (repeatedly, since the bug was a
# race), and with a genuine sibling it must still refuse.
#
# Needs /proc, so it runs in the same Linux container as the rest of the
# self-test. Invoked by tests/selftest.sh; runnable alone for a quick check.
set -uo pipefail

FAIL=0
# Split literals: this script's own cmdline must not match the pattern the guard
# greps for, or the test would be its own second harness. That is precisely the
# contamination that made a first attempt at this test report false failures.
N="coldstart-""run.sh"
S="sib-coldstart-""run.sh"
TD="$(mktemp -d)"
trap 'rm -rf "$TD"' EXIT

# The guard is built on pgrep, and python:3.12-slim has no procps -- so without
# this every scan comes back empty, the guard allows everything, and "10/10
# allowed" passes while testing nothing at all. A test that cannot fail is worse
# than no test, so install it, and if that is impossible say SKIPPED rather than
# reporting a pass.
if ! command -v pgrep >/dev/null 2>&1; then
  echo "== guard: installing procps (pgrep absent; the guard needs it)"
  if ! { apt-get -qq update >/dev/null 2>&1 \
         && apt-get -qq install -y procps >/dev/null 2>&1; } \
     || ! command -v pgrep >/dev/null 2>&1; then
    echo "GUARD SKIPPED: no pgrep and procps could not be installed."
    echo "   The guard cannot be exercised here. This is NOT a pass."
    exit 0
  fi
fi

GUARD_BODY="${GUARD_BODY:-/work/scripts/coldstart-run.sh}"
sed -n '/^preflight_exclusive() {/,/^}/p' "$GUARD_BODY" > "$TD/body.sh"
if ! grep -q "preflight_exclusive" "$TD/body.sh"; then
  echo "!! could not extract preflight_exclusive from $GUARD_BODY" >&2
  exit 2
fi
cp "$TD/body.sh" "$TD/$N"; echo 'preflight_exclusive' >> "$TD/$N"
cp "$TD/body.sh" "$TD/$S"; echo 'sleep 30'            >> "$TD/$S"

echo "== guard: allows a lone run (10x, the bug was a race)"
for i in $(seq 1 10); do
  if ! bash "$TD/$N" 2>"$TD/err"; then
    echo "   !! FALSE POSITIVE on run $i:"; sed 's/^/      /' "$TD/err"; FAIL=1; break
  fi
done
[[ "$FAIL" == 0 ]] && echo "   ok: 10/10 allowed"

echo "== guard: still refuses a real sibling"
# No setsid needed: the sibling is a child of THIS script, not of the guard
# process, so the guard's "skip my own children" rule does not hide it.
bash "$TD/$S" >/dev/null 2>&1 &
SIB=$!
sleep 1
bash "$TD/$N" >/dev/null 2>&1; rc=$?
if [[ "$rc" == 3 ]]; then
  echo "   ok: refused with exit 3"
else
  echo "   !! FAIL: expected exit 3 with a sibling running, got $rc"; FAIL=1
fi
kill -9 "$SIB" 2>/dev/null || true
wait "$SIB" 2>/dev/null || true

echo "== guard: allows again once the sibling is gone"
sleep 1
if bash "$TD/$N" 2>"$TD/err"; then
  echo "   ok: allowed"
else
  echo "   !! FAIL: still refusing after the sibling exited:"; sed 's/^/      /' "$TD/err"; FAIL=1
fi

if [[ "$FAIL" == 0 ]]; then echo "GUARD OK"; else echo "GUARD FAILED"; fi
exit "$FAIL"
