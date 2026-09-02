"""Deferred monkey-patching: attach spans to functions in modules that have
not been imported yet.

vLLM/torch internals cannot be patched from ``sitecustomize`` because none of
them are importable yet (and importing them there would both distort the
measurement and break vLLM's own import order). Instead we install one
``sys.meta_path`` finder that wraps each module's loader, and run registered
callbacks the instant a module finishes executing.

The same hook exposes ``add_exec_observer`` so :mod:`cs_imports` can time
module execution without a second layer of loader wrapping.
"""

import functools
import importlib
import inspect
import sys
import threading

import cs_trace as T

_lock = threading.RLock()
_pending = {}          # module name -> [callback(module), ...]
_observers = []        # (on_start(name), on_end(name, exc)) pairs
_installed = False
_results = []          # (target, status) for coverage reporting
_busy = threading.local()


# --------------------------------------------------------------------------
# import hook
# --------------------------------------------------------------------------

class _LoaderProxy(object):
    """Delegating proxy around a real loader, timing ``exec_module``."""

    def __init__(self, loader, fullname):
        self._cs_loader = loader
        self._cs_name = fullname

    def __getattr__(self, item):
        return getattr(self._cs_loader, item)

    def create_module(self, spec):
        return self._cs_loader.create_module(spec)

    def exec_module(self, module):
        name = self._cs_name
        for on_start, _ in _observers:
            try:
                on_start(name)
            except Exception:
                pass
        exc = None
        try:
            self._cs_loader.exec_module(module)
        except BaseException as e:      # noqa: BLE001 - re-raised below
            exc = e
            raise
        finally:
            for _, on_end in _observers:
                try:
                    on_end(name, exc)
                except Exception:
                    pass
            if exc is None:
                _run_callbacks(name, module)

    # Some loaders are used through the legacy load_module path.
    def load_module(self, fullname):
        return self._cs_loader.load_module(fullname)


class _Finder(object):
    """Finds nothing itself; delegates to the other finders and wraps loaders."""

    def find_spec(self, fullname, path=None, target=None):
        if getattr(_busy, "on", False):
            return None
        if fullname.startswith("cs_") or fullname == "sitecustomize":
            return None
        _busy.on = True
        try:
            for finder in list(sys.meta_path):
                if finder is self:
                    continue
                find_spec = getattr(finder, "find_spec", None)
                if find_spec is None:
                    continue
                try:
                    spec = find_spec(fullname, path, target)
                except Exception:
                    continue
                if spec is None:
                    continue
                loader = getattr(spec, "loader", None)
                if loader is not None and hasattr(loader, "exec_module"):
                    try:
                        spec.loader = _LoaderProxy(loader, fullname)
                    except Exception:
                        pass
                return spec
        finally:
            _busy.on = False
        return None


def install_hook():
    global _installed
    with _lock:
        if _installed:
            return
        sys.meta_path.insert(0, _Finder())
        _installed = True


def add_exec_observer(on_start, on_end):
    _observers.append((on_start, on_end))
    install_hook()


def after_import(module_name, callback):
    """Run ``callback(module)`` once ``module_name`` is imported (or now)."""
    install_hook()
    mod = sys.modules.get(module_name)
    if mod is not None and getattr(mod, "__spec__", None) is not None \
            and not getattr(mod.__spec__, "_initializing", False):
        _safe(callback, mod, module_name)
        return
    with _lock:
        _pending.setdefault(module_name, []).append(callback)


def _run_callbacks(name, module):
    with _lock:
        cbs = _pending.pop(name, None)
    if not cbs:
        return
    for cb in cbs:
        _safe(cb, module, name)


def _safe(cb, module, name):
    try:
        cb(module)
    except Exception as e:
        T.tracer.error("after_import:" + name, e)


# --------------------------------------------------------------------------
# wrappers
# --------------------------------------------------------------------------

def _mk_sync(fn, name, cat, argfn, min_dur):
    @functools.wraps(fn)
    def wrapper(*a, **kw):
        tok = T.tracer.begin(name, cat, **_args(argfn, a, kw))
        try:
            out = fn(*a, **kw)
        except BaseException as e:
            T.tracer.end(tok, min_dur=0.0, error=repr(e)[:200])
            raise
        T.tracer.end(tok, min_dur=min_dur, **_ret(argfn, out))
        return out
    return wrapper


def _mk_async(fn, name, cat, argfn, min_dur):
    @functools.wraps(fn)
    async def wrapper(*a, **kw):
        tok = T.tracer.begin(name, cat, **_args(argfn, a, kw))
        try:
            out = await fn(*a, **kw)
        except BaseException as e:
            T.tracer.end(tok, min_dur=0.0, error=repr(e)[:200])
            raise
        T.tracer.end(tok, min_dur=min_dur, **_ret(argfn, out))
        return out
    return wrapper


def _mk_asyncgen(fn, name, cat, argfn, min_dur):
    """For ``@asynccontextmanager`` targets: time setup and teardown halves."""

    @functools.wraps(fn)
    async def wrapper(*a, **kw):
        tok = T.tracer.begin(name + ".setup", cat, **_args(argfn, a, kw))
        agen = fn(*a, **kw)
        first = True
        try:
            async for item in agen:
                if first:
                    T.tracer.end(tok, min_dur=min_dur)
                    first = False
                    tok = None
                yield item
                tok = T.tracer.begin(name + ".teardown", cat)
        except BaseException as e:
            if tok is not None:
                T.tracer.end(tok, min_dur=0.0, error=repr(e)[:200])
                tok = None
            raise
        finally:
            if tok is not None:
                T.tracer.end(tok, min_dur=min_dur)
    return wrapper



class _TimedACM(object):
    """Times ``__aenter__`` of an async context manager (e.g. a function already
    decorated with ``@asynccontextmanager``, where the useful work happens on
    entry, not when the manager object is created)."""

    __slots__ = ("_cm", "_name", "_cat", "_args", "_min", "_tok")

    def __init__(self, cm, name, cat, args, min_dur):
        self._cm, self._name, self._cat = cm, name, cat
        self._args, self._min, self._tok = args, min_dur, None

    async def __aenter__(self):
        self._tok = T.tracer.begin(self._name + ".enter", self._cat,
                                   **self._args)
        try:
            out = await self._cm.__aenter__()
        except BaseException as e:
            T.tracer.end(self._tok, min_dur=0.0, error=repr(e)[:200])
            self._tok = None
            raise
        T.tracer.end(self._tok, min_dur=self._min)
        self._tok = None
        return out

    async def __aexit__(self, *exc):
        tok = T.tracer.begin(self._name + ".exit", self._cat)
        try:
            return await self._cm.__aexit__(*exc)
        finally:
            T.tracer.end(tok, min_dur=self._min)


def _mk_acm(fn, name, cat, argfn, min_dur):
    @functools.wraps(fn)
    def wrapper(*a, **kw):
        cm = fn(*a, **kw)
        if not hasattr(cm, "__aenter__"):
            return cm
        return _TimedACM(cm, name, cat, _args(argfn, a, kw), min_dur)
    return wrapper


def _args(argfn, a, kw):
    if argfn is None:
        return {}
    try:
        out = argfn(*a, **kw)
        return out if isinstance(out, dict) else {}
    except Exception:
        return {}


def _ret(argfn, out):
    return {}


def wrap_callable(fn, name, cat="misc", argfn=None, min_dur=None, kind=None):
    if getattr(fn, "_cs_wrapped", False):
        return fn
    md = T.tracer.min_dur if min_dur is None else min_dur
    if kind == "acm":
        w = _mk_acm(fn, name, cat, argfn, md)
    elif inspect.isasyncgenfunction(fn):
        w = _mk_asyncgen(fn, name, cat, argfn, md)
    elif inspect.iscoroutinefunction(fn):
        w = _mk_async(fn, name, cat, argfn, md)
    else:
        w = _mk_sync(fn, name, cat, argfn, md)
    w._cs_wrapped = True
    w._cs_orig = fn
    return w


def _resolve(root, path):
    """Walk ``a.b.c`` -> (owner_of_c, 'c')."""
    parts = path.split(".")
    obj = root
    for p in parts[:-1]:
        obj = getattr(obj, p)
    return obj, parts[-1]


def patch(module_name, path, name=None, cat="misc", argfn=None,
          min_dur=None, required=False, kind=None):
    """Register a span around ``module_name:path`` (e.g. ``Worker.load_model``).

    Silently records a ``missing`` result if the attribute does not exist in
    the installed version -- vLLM internals move between releases and a probe
    must never be the reason a run fails.
    """
    label = "%s:%s" % (module_name, path)
    span_name = name or (module_name.split(".")[-1] + "." + path)

    def _apply(module):
        try:
            owner, attr = _resolve(module, path)
        except Exception:
            _results.append((label, "missing"))
            return
        orig = getattr(owner, attr, None)
        if orig is None:
            _results.append((label, "missing"))
            return
        try:
            if isinstance(inspect.getattr_static(owner, attr, None),
                          staticmethod):
                setattr(owner, attr, staticmethod(
                    wrap_callable(orig, span_name, cat, argfn, min_dur, kind)))
            elif isinstance(inspect.getattr_static(owner, attr, None),
                            classmethod):
                inner = orig.__func__
                setattr(owner, attr, classmethod(
                    wrap_callable(inner, span_name, cat, argfn, min_dur, kind)))
            elif callable(orig):
                setattr(owner, attr,
                        wrap_callable(orig, span_name, cat, argfn, min_dur,
                                      kind))
            else:
                _results.append((label, "not-callable"))
                return
            _results.append((label, "ok"))
        except Exception as e:
            _results.append((label, "failed:%s" % type(e).__name__))
            if required:
                T.tracer.error("patch:" + label, e)

    after_import(module_name, _apply)


def patch_many(module_name, specs, cat="misc"):
    """specs: iterable of ``path`` or ``(path, span_name)`` or
    ``(path, span_name, cat)`` or ``(path, span_name, cat, argfn)``."""
    for s in specs:
        if isinstance(s, str):
            patch(module_name, s, cat=cat)
        else:
            s = list(s) + [None] * (5 - len(s))
            patch(module_name, s[0], name=s[1], cat=s[2] or cat, argfn=s[3],
                  kind=s[4])


def coverage():
    ok = sum(1 for _, s in _results if s == "ok")
    missing = [t for t, s in _results if s != "ok"]
    pending = sorted(_pending)
    return {"applied": ok, "not_applied": missing, "never_imported": pending}
