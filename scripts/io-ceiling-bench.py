import ctypes, glob, io, mmap, os, time
from concurrent.futures import ThreadPoolExecutor

SNAP = glob.glob("/cache/hf/hub/models--Qwen--Qwen3-32B/snapshots/*/")[0]
FILES = sorted(glob.glob(os.path.join(SNAP, "*.safetensors")))
TOTAL = sum(os.stat(os.path.realpath(f)).st_size for f in FILES)
GIB = 1 << 30
ALIGN = 4 << 20
PAGE = os.sysconf("SC_PAGE_SIZE")

libc = ctypes.CDLL("libc.so.6", use_errno=True)
libc.mmap.restype = ctypes.c_void_p
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                      ctypes.c_int, ctypes.c_int, ctypes.c_long]
libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
libc.pread.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_longlong]
libc.pread.restype = ctypes.c_longlong

def cached_bytes(path):
    real = os.path.realpath(path); size = os.stat(real).st_size
    fd = os.open(real, os.O_RDONLY)
    try:
        addr = libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, fd, 0)
        if addr in (None, ctypes.c_void_p(-1).value):
            return None
        npages = (size + PAGE - 1) // PAGE
        vec = ctypes.create_string_buffer(npages)
        rc = libc.mincore(ctypes.c_void_p(addr), size, vec)
        resident = sum(1 for b in vec.raw[:npages] if b & 1)
        libc.munmap(ctypes.c_void_p(addr), ctypes.c_size_t(size))
        return None if rc != 0 else resident * PAGE
    finally:
        os.close(fd)

def read_buffered(path, chunk):
    real = os.path.realpath(path); n = 0
    buf = bytearray(chunk); mv = memoryview(buf)
    with open(real, "rb", buffering=0) as f:
        while True:
            got = f.readinto(mv)
            if not got: break
            n += got
    return n

def read_odirect(path, chunk):
    O_DIRECT = 0o40000
    real = os.path.realpath(path)
    fd = os.open(real, os.O_RDONLY | O_DIRECT)
    raw = mmap.mmap(-1, chunk + ALIGN)
    base = ctypes.addressof(ctypes.c_char.from_buffer(raw))
    aligned = (base + ALIGN - 1) & ~(ALIGN - 1)
    n = 0
    try:
        while True:
            got = libc.pread(fd, ctypes.c_void_p(aligned), chunk, n)
            if got <= 0: break
            n += got
    finally:
        os.close(fd); raw.close()
    return n

def sweep(fn, threads, chunk, label):
    t = time.perf_counter()
    with ThreadPoolExecutor(max_workers=threads) as ex:
        got = sum(ex.map(lambda p: fn(p, chunk), FILES))
    dt = time.perf_counter() - t
    print(f"  {label:12s} threads={threads:<3d} chunk={chunk>>20:>3d}MiB "
          f"{got/GIB:7.2f} GiB / {dt:6.2f}s = {got/GIB/dt:6.2f} GiB/s", flush=True)

print(f"model: {len(FILES)} shards, {TOTAL/GIB:.2f} GiB")
res = 0; ok = True
for f in FILES:
    c = cached_bytes(f)
    if c is None: ok = False; break
    res += c
print(f"page cache resident: {res/GIB:.2f} / {TOTAL/GIB:.2f} GiB ({100*res/TOTAL:.1f}%)"
      if ok else "mincore failed")
print("\n=== O_DIRECT: true GPFS path, page cache bypassed ===")
for th in (4, 8, 16, 32): sweep(read_odirect, th, 32 << 20, "O_DIRECT")
print("\n=== buffered read(): what the nogds pread path sees ===")
for th in (4, 8, 16, 32): sweep(read_buffered, th, 32 << 20, "buffered")
