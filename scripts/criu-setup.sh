#!/usr/bin/env bash
# Make the hand-built criu + CUDA plugin + cuda-checkpoint environment
# reproducible, so a second pod can be given the SAME toolchain as the first.
#
# Every arm in reports/criu-cuda-checkpoint.txt ran in one pod whose criu was
# built by hand, interactively, and never scripted:
#
#   criu           4.1.1-1build1, Ubuntu noble pool. 4.2 was tried and
#                  REJECTED at install time -- it depends on libnet9, which
#                  this image does not carry and which the person doing the
#                  build declined to pull in. Do not "helpfully" upgrade this:
#                  a newer criu means a newer plugin ABI, and the whole point
#                  of this script is that pod A and pod B run the identical
#                  binary. (reports/criu-raw/environment.txt:11)
#   cuda_plugin.so criu 4.1.1's own source ships plugins/cuda/cuda_plugin.c;
#                  Ubuntu's criu package ships an amdgpu plugin MAN PAGE and no
#                  plugin .so of any kind (reports/criu-raw/environment.txt:16-24).
#                  Built from the matching v4.1.1 source tag, installed to
#                  /usr/lib/criu/cuda_plugin.so (49768 bytes, mode 755 on the
#                  original build), /usr/sbin/criu NEVER replaced.
#   cuda-checkpoint 580.105.08, made reachable on $PATH as /usr/local/bin/
#                  cuda-checkpoint. This is load-bearing, not cosmetic:
#                  cuda_plugin.c resolves the helper by NAME at dump/restore
#                  time (checkpoint-restore/criu plugins/cuda/cuda_plugin.c),
#                  so "on $PATH" is a functional requirement, not a convenience.
#
# None of that was a script. This one is, and it exists for exactly one reason:
# the cross-pod restore test (does an image built in pod A restore in pod B?)
# is meaningless unless pod B's criu + plugin + cuda-checkpoint are the SAME
# bytes as pod A's, and "SSH in and remember the same nine commands" is not a
# control on that. A real deployment would bake this into the container image;
# this script's PVC-backed cache is the stand-in for that -- build once, and
# every pod after the first installs from cache in seconds, which is also what
# makes pod A and pod B's setups *comparable* rather than just both "working".
#
# Two extra build deps beyond what this image already carries, discovered the
# hard way in the original session (reports/criu-raw/environment.txt:81-85):
# libprotobuf-dev (NOT libprotobuf-c-dev alone -- criu's own descriptor.proto
# needs the C++ protobuf compiler) and uuid-dev (-luuid). The rest of the
# package list below is criu's own published build dependency list
# (criu.org/Installation#Dependencies) for whatever this image did not already
# have; environment.txt only names the two that actually bit someone, so this
# script installs the standard superset and lets apt no-op on what already
# exists rather than guessing which subset was already present.
#
# THE criu check TRAP, reproduced here because it will otherwise cost someone
# a day: cr_plugin_init() only runs from the dump/pre-dump/restore stages, so
# `criu check --all` prints "Looks good." with an EMPTY plugin directory and
# with a working plugin alike (reports/criu-raw/environment.txt:99-101,
# reports/criu-cuda-checkpoint.txt:87-91). This script runs `criu check --all`
# for the record but treats its result as informational only, never as
# verification -- see VERIFY below for what actually proves the plugin loads.
#
# The plugin build has one more sharp edge worth carrying: the *top-level*
# `make` builds cuda_plugin as part of its default target (criu's own
# Makefile: `all: criu lib crit cuda_plugin`) but ALSO tries to build
# lib/pycriu, which needs a C++ protoc this script does not install, and dies
# there. That failure is harmless and expected -- plugins/cuda/cuda_plugin.so
# is already sitting on disk by the time make gets that far
# (reports/criu-raw/environment.txt:96-97) -- so this script does not treat a
# non-zero top-level `make` exit as fatal. It treats a MISSING .so as fatal.
#
# Usage (no argument is required):
#
#   bash criu-setup.sh                 install/build, print + save provenance
#   bash criu-setup.sh --verify         same, then run the ONE real proof
#                                       (see VERIFY below) that the plugin
#                                       actually loads at dump time
#   bash criu-setup.sh --force-rebuild  ignore the PVC cache, rebuild, refresh it
#   bash criu-setup.sh -h|--help        print this header
#
# Env overrides: CACHE_ROOT (default /cache/criu-toolchain, expected to be the
# PVC mounted at /cache -- see manifests/cache-pvc.yaml), CRIU_VER (default
# 4.1.1, see the warning above before touching this), CUDA_CKPT (default
# cuda-checkpoint, resolved on $PATH -- matches the variable name every other
# criu script in this directory uses, and NOT "CC": see the CC/compiler-
# variable incident at reports/criu-raw/environment.txt:103-121),
# PROVENANCE_OUT (default /var/log/coldstart/criu-setup-provenance.txt, i.e.
# the "traces" emptyDir every other script in this repo already pulls out with
# `make fetch` -- see manifests/pod-exec-criu.yaml).
#
# VERIFY (--verify). `criu check` cannot prove the plugin loads (see the trap
# above), so this defers to the one arm in this repo that already does a real,
# small, cheap dump/restore/replay of an actual CUDA process with the plugin
# doing the work: scripts/criu-matrix.sh arm `plugin` (512 MiB tensors + one
# captured CUDA graph, ~3s dump, ~3s restore, correctness checked by replaying
# the graph and comparing the value -- reports/criu-cuda-checkpoint.txt:105-143).
# This script does not reimplement that probe; it runs it under the subreaper
# (scripts/criu-reaper.py, required or the restore fails on a zombied pid --
# see that file's header) and greps its dump.log for the line that is the only
# real proof there is: "cuda_plugin: finished cuda_plugin stage 0 err 0".
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- fixed by design, not by convenience -- see the header before changing ---
CRIU_VER="${CRIU_VER:-4.1.1}"
CRIU_PKG_VER="${CRIU_PKG_VER:-4.1.1-1build1}"   # exact Ubuntu noble pool version
PLUGIN_DIR="/usr/lib/criu"                       # the ONLY dir criu searches
PLUGIN_SO="$PLUGIN_DIR/cuda_plugin.so"
CRIU_BIN="$(command -v criu 2>/dev/null || echo /usr/sbin/criu)"

CACHE_ROOT="${CACHE_ROOT:-/cache/criu-toolchain}"
DL_CACHE="$CACHE_ROOT/dl"
CUDA_CKPT_NAME="cuda-checkpoint"
PROVENANCE_OUT="${PROVENANCE_OUT:-/var/log/coldstart/criu-setup-provenance.txt}"

DO_VERIFY=0
FORCE_REBUILD=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --verify) DO_VERIFY=1; shift ;;
    --force-rebuild) FORCE_REBUILD=1; shift ;;
    -h|--help) sed -n '2,74p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

say() { echo "[$(date -u +%H:%M:%S)] $*"; }
die() { echo "[$(date -u +%H:%M:%S)] FATAL: $*" >&2; exit 1; }

[[ "$(id -u)" == 0 ]] || die "must run as root inside the vLLM container (see manifests/pod-exec-criu.yaml securityContext)"

mkdir -p "$CACHE_ROOT" "$DL_CACHE" || die "cannot create $CACHE_ROOT -- is /cache (the PVC) mounted?"
mkdir -p "$(dirname "$PROVENANCE_OUT")" 2>/dev/null || PROVENANCE_OUT=/tmp/criu-setup-provenance.txt

# ---------------------------------------------------------------------------
# 1. criu itself: install the pinned package, never a newer one.
# ---------------------------------------------------------------------------
current_criu_ver() { dpkg-query -W -f='${Version}' criu 2>/dev/null || true; }

ensure_criu() {
  local have; have="$(current_criu_ver)"
  if [[ "$have" == "$CRIU_PKG_VER" ]]; then
    say "criu $have already installed (matches pin) -- skipping apt"
    return 0
  fi
  if [[ -n "$have" ]]; then
    say "criu $have installed, pin wants $CRIU_PKG_VER -- reinstalling the pinned version"
  else
    say "criu not installed -- installing $CRIU_PKG_VER"
  fi
  apt-get update -qq || die "apt-get update failed"
  # Exact version pin. Do NOT drop the =version: an unpinned `apt-get install
  # criu` on a pool that has moved on could hand you 4.2, which this image
  # cannot even accept -- 4.2 needs libnet9, which was tried and rejected here
  # (reports/criu-raw/environment.txt:11) -- and a plugin built against 4.1.1
  # headers does not necessarily load into a 4.2 binary. If this exact version
  # string is no longer in the pool by the time you read this, STOP and treat
  # it as a real finding, not a nuisance to work around by relaxing the pin.
  if ! apt-get install -y --no-install-recommends "criu=${CRIU_PKG_VER}"; then
    # Observed 2026-09-08: this is not the "pool moved on to 4.2" case the
    # comment above warns about -- `apt-cache madison criu` and the raw
    # Packages.gz for both noble and noble-updates universe had ZERO criu
    # entries, pinned or not (`apt-get install criu` with no version fails
    # the same way: "Package 'criu' has no installation candidate"). The
    # .deb bytes for this exact pin were still sitting in the pool, though:
    # http://archive.ubuntu.com/ubuntu/pool/universe/c/criu/ served both
    # criu_${CRIU_PKG_VER}_amd64.deb and the libcompel1 build it hard-depends
    # on (= exact version). Fall back to fetching those directly, then let
    # apt resolve the ordinary shared-library deps (libnet1, libnftables1,
    # libprotobuf-c1 -- generic Ubuntu libs, not part of what went missing)
    # from the still-functional generic index.
    say "criu=${CRIU_PKG_VER} has no apt candidate (index drift, not a pin miss) -- trying the pool .deb directly"
    local pool="http://archive.ubuntu.com/ubuntu/pool/universe/c/criu"
    local deb="criu_${CRIU_PKG_VER}_amd64.deb"
    local compel_deb="libcompel1_${CRIU_PKG_VER}_amd64.deb"
    ( cd "$DL_CACHE" \
      && curl -fsSLO "${pool}/${deb}" \
      && curl -fsSLO "${pool}/${compel_deb}" ) \
      || die "direct pool fetch of $deb / $compel_deb failed -- see comment above before relaxing the pin"
    dpkg -i "$DL_CACHE/$compel_deb" "$DL_CACHE/$deb" || true
    apt-get install -y -f \
      || die "apt-get install -f could not resolve criu's runtime deps after the pool-.deb install"
  fi
  apt-mark hold criu >/dev/null 2>&1 || true
  have="$(current_criu_ver)"
  [[ "$have" == "$CRIU_PKG_VER" ]] || die "installed criu is $have, expected $CRIU_PKG_VER"
  CRIU_BIN="$(command -v criu)"
}

# ---------------------------------------------------------------------------
# 2. Build-time deps for the plugin. Only libprotobuf-dev and uuid-dev are
# confirmed-by-incident (environment.txt:81-85); the rest is criu's own
# published dependency list for anything this image does not already carry.
# apt no-ops on anything already installed, so this is safe to run every time
# and cheap once the fast (cache-hit) path below is taken instead.
# ---------------------------------------------------------------------------
ensure_build_deps() {
  say "ensuring plugin build deps (libprotobuf-dev, uuid-dev, + criu's standard build list)"
  apt-get update -qq || die "apt-get update failed"
  apt-get install -y --no-install-recommends \
    build-essential pkg-config \
    libprotobuf-dev libprotobuf-c-dev protobuf-c-compiler protobuf-compiler python3-protobuf \
    libnl-3-dev libnet1-dev libcap-dev uuid-dev \
    curl ca-certificates \
    || die "apt-get install (plugin build deps) failed"
}

# ---------------------------------------------------------------------------
# 3. Cache key: criu version AND driver version, per the task this script
# exists for -- a plugin cached under one driver's key is never silently
# reused under another's, even though the .so itself does not link against
# the driver (plugins/cuda/Makefile links against nothing but criu's own
# headers and shells out to cuda-checkpoint at runtime by name). Keying on the
# driver is what makes "pod A's toolchain" and "pod B's toolchain" a
# comparable statement instead of an assumption.
# ---------------------------------------------------------------------------
driver_version() {
  nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | tr -d ' \r'
}

DRIVER_VER="$(driver_version)"
[[ -n "$DRIVER_VER" ]] || { say "WARNING: nvidia-smi did not report a driver version (no GPU visible?) -- caching under 'nodriver'"; DRIVER_VER="nodriver"; }
CACHE_KEY="criu${CRIU_VER}_driver${DRIVER_VER}"
KEY_DIR="$CACHE_ROOT/$CACHE_KEY"
CACHED_SO="$KEY_DIR/cuda_plugin.so"
CACHED_MANIFEST="$KEY_DIR/manifest.txt"

# ---------------------------------------------------------------------------
# 4. ABI cross-check (the (b) verification the task asks for): every UNDEFINED
# dynamic symbol the plugin needs (fi_strategy, opts, add_inventory_plugin,
# compel_wait_task, ... -- reports/criu-raw/environment.txt:90-92) must be
# something the installed criu binary actually EXPORTS. This is necessary but
# not sufficient -- it catches "this .so was built for a different criu" but
# cannot prove cr_plugin_init() runs cleanly, which is what --verify is for.
#
# The exports set has to include criu's shared-library dependencies, not just
# the criu binary itself: a dlopen()'d plugin resolves against the WHOLE
# process's global symbol table (RTLD_DEFAULT), which is criu's own exports
# PLUS everything already loaded alongside it -- libc above all. Checking
# `nm -D "$CRIU_BIN"` alone therefore reports every ordinary libc call the
# plugin makes (atoi, close, fork, malloc, ptrace, ...) as "missing", which is
# a false failure, not an ABI mismatch: criu.org's own environment.txt run
# only eyeballed the criu-internal names (fi_strategy et al.) and never hit
# this, but an automated diff over the full undefined-symbol list does.
#
# Second false-failure source, same symptom: `nm -D` prints an UNDEFINED
# reference as `name@VERSION` (one @) but a DEFAULT-VERSION EXPORT as
# `name@@VERSION` (two @s) -- e.g. the plugin needs `atoi@GLIBC_2.2.5` and
# libc.so.6 exports `atoi@@GLIBC_2.2.5`. Byte-for-byte these never match, so
# comparing raw nm output always "finds" every versioned libc symbol missing.
# Strip everything from the first @ on both sides -- the dynamic linker's own
# version negotiation at actual load time is the real gate; this check only
# needs to know an object providing the base symbol exists somewhere criu
# will have loaded.
# ---------------------------------------------------------------------------
resolvable_exports() {
  # criu's own exports, plus the exports of every .so it is dynamically
  # linked against -- the actual resolution set a dlopen()'d plugin sees.
  local bin="$1" lib
  { nm -D "$bin" 2>/dev/null
    ldd "$bin" 2>/dev/null | awk '{print $3}' | while read -r lib; do
      [[ -n "$lib" && -f "$lib" ]] && nm -D "$lib" 2>/dev/null
    done
  } | awk '$2 ~ /^[A-Za-z]$/ {print $NF}' | sed 's/@.*//' | sort -u
}

abi_check() {
  local so="$1"
  command -v nm >/dev/null || { say "  no nm on PATH (binutils) -- skipping ABI cross-check, cannot confirm"; return 1; }
  command -v ldd >/dev/null || { say "  no ldd on PATH -- skipping ABI cross-check, cannot confirm"; return 1; }
  local undef criu_exports missing
  undef=$(nm -D --undefined-only "$so" 2>/dev/null | awk '{print $NF}' | sed 's/@.*//' | sort -u)
  [[ -n "$undef" ]] || { say "  plugin exposes no undefined dynamic symbols? unexpected -- treating as ABI-check failure"; return 1; }
  criu_exports=$(resolvable_exports "$CRIU_BIN")
  [[ -n "$criu_exports" ]] || { say "  nm -D produced nothing for $CRIU_BIN (+ its shared libs) -- cannot cross-check"; return 1; }
  missing=$(comm -23 <(echo "$undef") <(echo "$criu_exports"))
  if [[ -n "$missing" ]]; then
    say "  ABI cross-check FAILED -- symbols the plugin needs that $CRIU_BIN (+ its shared libs) does not export:"
    echo "$missing" | sed 's/^/      /'
    return 1
  fi
  say "  ABI cross-check ok: every undefined symbol in the plugin ($(echo "$undef" | wc -l | tr -d ' ')) is exported by $CRIU_BIN or a library it links"
  return 0
}

static_checks() {
  local so="$1"
  [[ -f "$so" ]] || { say "  no .so at $so"; return 1; }
  local size mode
  size=$(stat -c '%s' "$so" 2>/dev/null || echo 0)
  mode=$(stat -c '%a' "$so" 2>/dev/null || echo 000)
  say "  $so : $size bytes, mode $mode (original build was 49768 bytes, mode 755 -- size will drift with compiler/libc, mode should not)"
  [[ "$size" -gt 1000 ]] || { say "  .so implausibly small -- truncated build?"; return 1; }
  abi_check "$so"
}

# ---------------------------------------------------------------------------
# 5. Build from source. Only reached on a cache miss.
# ---------------------------------------------------------------------------
build_plugin() {
  ensure_build_deps
  local tarball="$DL_CACHE/criu-${CRIU_VER}.tar.gz"
  if [[ -s "$tarball" ]]; then
    say "criu-$CRIU_VER source tarball already cached at $tarball -- skipping fetch"
  else
    say "fetching criu $CRIU_VER source from github.com/checkpoint-restore/criu (tag v$CRIU_VER) -- needs network egress"
    curl -fL --retry 3 -o "$tarball.tmp" \
      "https://github.com/checkpoint-restore/criu/archive/refs/tags/v${CRIU_VER}.tar.gz" \
      || { rm -f "$tarball.tmp"; die "source fetch failed -- check egress to github.com/codeload.github.com"; }
    mv "$tarball.tmp" "$tarball"
  fi

  local work="/tmp/criu-build-$$"
  rm -rf "$work"; mkdir -p "$work"
  tar -xzf "$tarball" -C "$work" --strip-components=1 || die "corrupt criu source tarball"

  say "building criu $CRIU_VER from source (top-level 'make', NO relaxed flags, NO -Werror suppression)"
  say "  this is expected to fail LATE, at lib/pycriu (needs a C++ protoc this"
  say "  script does not install) -- that failure is harmless: plugins/cuda/"
  say "  cuda_plugin.so is already built by the time make gets there"
  ( cd "$work" && make ) >"$work/make.log" 2>&1
  local rc=$?
  if [[ ! -s "$work/plugins/cuda/cuda_plugin.so" ]]; then
    say "make exit=$rc AND no plugins/cuda/cuda_plugin.so was produced -- this is a real failure, not the expected late one"
    say "  tail of make.log:"
    tail -60 "$work/make.log" | sed 's/^/      /'
    rm -rf "$work"
    return 1
  fi
  say "make exit=$rc (non-zero here is the expected lib/pycriu failure -- see above); cuda_plugin.so is present"

  static_checks "$work/plugins/cuda/cuda_plugin.so" || { say "freshly-built plugin failed static checks -- refusing to install it"; rm -rf "$work"; return 1; }

  mkdir -p "$KEY_DIR"
  install -m 755 "$work/plugins/cuda/cuda_plugin.so" "$CACHED_SO"
  {
    echo "criu_ver=$CRIU_VER"
    echo "criu_pkg_ver=$CRIU_PKG_VER"
    echo "driver_ver=$DRIVER_VER"
    echo "built_utc=$(date -u +%FT%TZ)"
    echo "sha256=$(sha256sum "$CACHED_SO" | awk '{print $1}')"
    echo "size=$(stat -c '%s' "$CACHED_SO")"
    echo "gcc=$(gcc --version 2>/dev/null | head -1)"
  } > "$CACHED_MANIFEST"
  say "cached build product + manifest under $KEY_DIR"
  rm -rf "$work"
  return 0
}

# ---------------------------------------------------------------------------
# 6. Cache hit / miss decision, then install into the one directory criu
# searches.
# ---------------------------------------------------------------------------
install_plugin() {
  if [[ "$FORCE_REBUILD" == 0 && -f "$CACHED_SO" ]]; then
    say "cache hit at $KEY_DIR -- verifying before trusting it"
    if static_checks "$CACHED_SO"; then
      mkdir -p "$PLUGIN_DIR"
      install -m 755 "$CACHED_SO" "$PLUGIN_SO"
      say "installed $PLUGIN_SO from cache (no compile) -- $(stat -c '%s' "$PLUGIN_SO") bytes"
      PLUGIN_SOURCE="cache ($KEY_DIR)"
      return 0
    fi
    say "cached artifact failed static checks -- rebuilding instead of trusting a stale/foreign .so"
  fi
  build_plugin || die "plugin build failed -- see log above"
  mkdir -p "$PLUGIN_DIR"
  install -m 755 "$CACHED_SO" "$PLUGIN_SO"
  say "installed $PLUGIN_SO (freshly built) -- $(stat -c '%s' "$PLUGIN_SO") bytes"
  PLUGIN_SOURCE="compiled just now"
}

# ---------------------------------------------------------------------------
# 7. cuda-checkpoint. Check the image before assuming it must be fetched: it
# is NOT part of the CUDA toolkit or any apt package (confirmed against
# NVIDIA/cuda-checkpoint's own README -- the utility ships only as a binary in
# that repo's bin/ directory, one build per CPU arch, no distro packaging), so
# a vLLM serving image built from an nvidia/cuda base has no particular reason
# to carry it. This script still looks before it fetches, on the chance a
# base image someday bundles it.
# ---------------------------------------------------------------------------
find_cuda_checkpoint() {
  local c
  c="$(command -v "$CUDA_CKPT_NAME" 2>/dev/null)" && { echo "$c"; return 0; }
  for c in /usr/local/bin/cuda-checkpoint /usr/bin/cuda-checkpoint \
           /usr/local/cuda/bin/cuda-checkpoint /usr/local/cuda-*/bin/cuda-checkpoint \
           /opt/nvidia/cuda-checkpoint; do
    [[ -x "$c" ]] && { echo "$c"; return 0; }
  done
  # Bounded search of the two trees a CUDA toolkit or driver installer would
  # plausibly have put it under. Not /: this container has a full Ubuntu
  # filesystem and vLLM's own site-packages tree, and an unbounded find there
  # is slow for no benefit -- cuda-checkpoint is a driver-adjacent tool, not a
  # Python package.
  find /usr/local /opt -maxdepth 6 -type f -iname 'cuda-checkpoint' -perm -u+x 2>/dev/null | head -1
}

fetch_cuda_checkpoint() {
  # Caller does CUDA_CKPT_PATH="$(fetch_cuda_checkpoint)" -- ONLY the final
  # echo below is the return value. say() writes to stdout like everywhere
  # else in this script, so it has to be redirected to stderr in here
  # specifically, or every one of these status lines gets captured into
  # CUDA_CKPT_PATH right along with the real path, corrupting it silently.
  local dst="$DL_CACHE/cuda-checkpoint"
  if [[ -x "$dst" ]]; then
    say "cuda-checkpoint already cached at $dst -- skipping fetch" >&2
  else
    say "fetching cuda-checkpoint from github.com/NVIDIA/cuda-checkpoint (main, bin/x86_64_Linux) -- needs network egress" >&2
    say "  NVIDIA ships no versioned release for this repo; main is the only ref, and the" >&2
    say "  README states the binary tracks the DRIVER's own capability set, not its own" >&2
    say "  version scheme -- so 'the same binary' means 'the same file', pin by sha256" >&2
    say "  in the manifest below, not by a version string." >&2
    curl -fL --retry 3 -o "$dst.tmp" \
      "https://raw.githubusercontent.com/NVIDIA/cuda-checkpoint/main/bin/x86_64_Linux/cuda-checkpoint" \
      || { rm -f "$dst.tmp"; die "cuda-checkpoint fetch failed -- check egress to raw.githubusercontent.com"; }
    chmod +x "$dst.tmp"
    mv "$dst.tmp" "$dst"
  fi
  install -m 755 "$dst" "/usr/local/bin/${CUDA_CKPT_NAME}"
  echo "/usr/local/bin/${CUDA_CKPT_NAME}"
}

ensure_cuda_checkpoint() {
  local found; found="$(find_cuda_checkpoint)"
  if [[ -n "$found" && -x "$found" ]]; then
    say "cuda-checkpoint already present in the image at $found"
    if [[ "$(dirname "$found")" != "/usr/local/bin" ]] && ! command -v "$CUDA_CKPT_NAME" >/dev/null; then
      ln -sf "$found" "/usr/local/bin/${CUDA_CKPT_NAME}"
      say "  symlinked onto \$PATH at /usr/local/bin/${CUDA_CKPT_NAME} (the plugin resolves the NAME via \$PATH at dump time -- this is not optional)"
    fi
    CUDA_CKPT_PATH="$(command -v "$CUDA_CKPT_NAME")"
    CUDA_CKPT_ORIGIN="already in image"
  else
    # Bug (found in the cross-pod session, then found AGAIN one level deeper
    # in the first fix attempt): CUDA_CKPT_PATH="$(fetch_cuda_checkpoint)" is
    # a command substitution, which always runs fetch_cuda_checkpoint in a
    # SUBSHELL -- any variable that function sets (CUDA_CKPT_FETCH_ORIGIN,
    # in the first attempt) is invisible here once it returns, regardless of
    # local/global declaration, and under `set -u` that read used to crash
    # with "unbound variable" outright. Determine the origin label out here,
    # in this shell, from the same filesystem predicate fetch_cuda_checkpoint
    # itself checks (a pure read with no side effects, checked strictly
    # before that function can create/mv the file) -- no data needs to cross
    # the subshell boundary at all.
    if [[ -x "$DL_CACHE/cuda-checkpoint" ]]; then
      CUDA_CKPT_ORIGIN="cache ($DL_CACHE/cuda-checkpoint)"
    else
      CUDA_CKPT_ORIGIN="freshly fetched from github.com/NVIDIA/cuda-checkpoint (network)"
    fi
    CUDA_CKPT_PATH="$(fetch_cuda_checkpoint)"
  fi
  command -v "$CUDA_CKPT_NAME" >/dev/null || die "cuda-checkpoint installed but not resolvable via \$PATH by name -- the plugin will fail at dump time"
  CUDA_CKPT_VER="$("$CUDA_CKPT_PATH" --help 2>&1 | grep -oE 'Version [0-9.]+' | head -1)"
  say "cuda-checkpoint: $CUDA_CKPT_PATH ($CUDA_CKPT_ORIGIN) -- ${CUDA_CKPT_VER:-version string not found in --help output}"
  # Sanity smoke test that does not touch a real process: --get-state on our
  # own pid should return a clean "not a CUDA process" style answer rather
  # than an exec error, which is the cheapest proof the binary actually runs
  # on this kernel/driver.
  "$CUDA_CKPT_PATH" --get-state --pid "$$" >/tmp/criu-setup-ccsmoke.log 2>&1
  say "  smoke test (--get-state --pid \$\$): $(head -1 /tmp/criu-setup-ccsmoke.log 2>/dev/null || echo 'no output')"
}

# ---------------------------------------------------------------------------
# 8. Informational-only criu check. See the TRAP paragraph in the header:
# this NEVER gates success/failure, it is printed so a reader has it and does
# not mistake it for proof.
# ---------------------------------------------------------------------------
run_criu_check() {
  say "criu check --all (INFORMATIONAL ONLY -- see the trap in this script's header:"
  say "  cr_plugin_init() never runs here, so this cannot confirm the plugin loads)"
  local out; out="$("$CRIU_BIN" check --all 2>&1)"
  echo "$out" | sed 's/^/    /'
}

# ---------------------------------------------------------------------------
# 9. --verify: the one real proof. Defers to scripts/criu-matrix.sh arm
# `plugin` -- see this script's header for why that arm and not a new probe.
# ---------------------------------------------------------------------------
do_verify() {
  local reaper="$SCRIPT_DIR/criu-reaper.py"
  local matrix="$SCRIPT_DIR/criu-matrix.sh"
  [[ -f "$reaper" && -f "$matrix" ]] || die "--verify needs $reaper and $matrix next to this script"
  local imgroot
  if [[ -d /criu && -w /criu ]]; then imgroot=/criu/img/setup-verify
  else imgroot=/tmp/criu-setup-verify
  fi
  rm -rf "$imgroot"; mkdir -p "$imgroot"
  say "verify: running scripts/criu-matrix.sh arm=plugin under the subreaper (img dir $imgroot)"
  local out
  out="$(CUDA_CKPT="$CUDA_CKPT_PATH" CRIU="$CRIU_BIN" IMGROOT="$imgroot" \
        python3 "$reaper" bash "$matrix" --arms plugin --img-dir "$imgroot" 2>&1)"
  echo "$out" | sed 's/^/    /'
  local dumplog="$imgroot/plugin/dump.log"
  if grep -aq "cuda_plugin: finished cuda_plugin stage 0 err 0" "$dumplog" 2>/dev/null \
     && echo "$out" | grep -q "VERDICT: IDENTICAL"; then
    say "VERIFY PASSED: plugin ran ('finished cuda_plugin stage 0 err 0' in $dumplog) and the round-trip was correct (VERDICT: IDENTICAL)"
    VERIFY_RESULT="PASSED"
  else
    say "VERIFY FAILED: did not find both the plugin-stage-0 success line in $dumplog and VERDICT: IDENTICAL in the arm's own output -- see the transcript above"
    VERIFY_RESULT="FAILED"
  fi
}

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
PLUGIN_SOURCE="unknown"
VERIFY_RESULT="not run (pass --verify)"
CUDA_CKPT_PATH=""
CUDA_CKPT_ORIGIN="unknown"
CUDA_CKPT_VER=""

say "criu-setup: driver=$DRIVER_VER  cache_key=$CACHE_KEY  cache_root=$CACHE_ROOT"
ensure_criu
install_plugin
ensure_cuda_checkpoint
run_criu_check
[[ "$DO_VERIFY" == 1 ]] && do_verify

# ---------------------------------------------------------------------------
# 10. Provenance block. Pasteable into a report, and written to disk so the
# experiment can pull one file per pod and diff them -- that diff (or its
# absence) is the entire proof that pod A and pod B ran the same toolchain.
# ---------------------------------------------------------------------------
{
  echo "=== criu-setup provenance ($(date -u +%FT%TZ)) ==="
  echo "host (pod)        : $(hostname)"
  echo "kernel             : $(uname -r)"
  echo "criu version       : $("$CRIU_BIN" --version 2>&1)"
  echo "criu path          : $CRIU_BIN (never replaced by this script)"
  echo "plugin path        : $PLUGIN_SO"
  if [[ -f "$PLUGIN_SO" ]]; then
    echo "plugin size        : $(stat -c '%s' "$PLUGIN_SO") bytes"
    echo "plugin mode        : $(stat -c '%a' "$PLUGIN_SO")"
    echo "plugin mtime       : $(stat -c '%y' "$PLUGIN_SO")"
    echo "plugin sha256      : $(sha256sum "$PLUGIN_SO" | awk '{print $1}')"
  else
    echo "plugin             : MISSING -- setup did not succeed"
  fi
  echo "plugin source      : $PLUGIN_SOURCE"
  echo "cache key          : $CACHE_KEY"
  echo "cuda-checkpoint    : $CUDA_CKPT_PATH ($CUDA_CKPT_ORIGIN)"
  echo "cuda-checkpoint ver: ${CUDA_CKPT_VER:-unknown}"
  echo "driver version     : $DRIVER_VER"
  echo "verify (--verify)  : $VERIFY_RESULT"
} | tee "$PROVENANCE_OUT"

say "provenance written to $PROVENANCE_OUT"
[[ -f "$PLUGIN_SO" ]] || exit 1
exit 0
