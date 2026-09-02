"""Network probe: DNS, TCP, TLS, HTTP and Hugging Face Hub traffic.

Cold start touches the network far more than people expect even when weights
are "already local": tokenizer/config metadata lookups, ``model_info`` HEAD
requests, usage telemetry, and (on a cache miss) the weight download itself.
Each of those is a serial, latency-bound step on the critical path, so they get
individual spans:

    dns.getaddrinfo   tcp.connect   tls.handshake   http.request
    hf.hub_download   hf.metadata   hf.snapshot

Bulk transfer volume is measured two ways: per-request content length where the
client exposes it, and netns-wide counters sampled by :mod:`cs_sampler`
(``/proc/self/net/dev``), which also catches traffic from native code (NCCL,
RDMA, object-store SDKs) that never passes through Python sockets.
"""

import sys

import cs_patch
import cs_trace as T

_installed = False


def _addr(a):
    try:
        if isinstance(a, (tuple, list)) and len(a) >= 2:
            return "%s:%s" % (a[0], a[1])
        return str(a)[:120]
    except Exception:
        return "?"


def install():
    global _installed
    if _installed or not T.enabled("net"):
        return
    _installed = True
    _patch_stdlib()
    _patch_clients()


def _patch_stdlib():
    """Patch the stdlib network primitives *lazily*.

    ``socket``, ``ssl`` and ``http.client`` are not imported here on purpose:
    importing ``http.client`` alone pulls in ``email`` (~50 modules), and doing
    that from ``sitecustomize`` would move real work onto the critical path
    before vLLM has asked for it -- inflating the very number we are measuring.
    ``cs_patch.after_import`` applies each patch the moment the application
    imports the module (or immediately, if it already has).
    """
    def _on_socket(socket):
        orig_gai = socket.getaddrinfo

        def getaddrinfo(host, port, *a, **kw):
            tok = T.tracer.begin("dns.getaddrinfo", "network",
                                 host=str(host)[:120], port=str(port))
            try:
                return orig_gai(host, port, *a, **kw)
            finally:
                T.tracer.end(tok, min_dur=0.002)
        getaddrinfo._cs_wrapped = True
        socket.getaddrinfo = getaddrinfo

        orig_connect = socket.socket.connect

        def connect(self, address, *a, **kw):
            tok = T.tracer.begin("tcp.connect", "network", peer=_addr(address))
            try:
                return orig_connect(self, address, *a, **kw)
            finally:
                T.tracer.end(tok, min_dur=0.002)
        socket.socket.connect = connect

        orig_cc = socket.create_connection

        def create_connection(address, *a, **kw):
            tok = T.tracer.begin("tcp.create_connection", "network",
                                 peer=_addr(address))
            try:
                return orig_cc(address, *a, **kw)
            finally:
                T.tracer.end(tok, min_dur=0.002)
        socket.create_connection = create_connection

    def _on_ssl(ssl):
        orig_hs = ssl.SSLSocket.do_handshake

        def do_handshake(self, *a, **kw):
            tok = T.tracer.begin("tls.handshake", "network")
            try:
                return orig_hs(self, *a, **kw)
            finally:
                T.tracer.end(tok, min_dur=0.002)
        ssl.SSLSocket.do_handshake = do_handshake

    def _on_httpclient(hc):
        def _req_args(self, method=None, url=None, *a, **kw):
            return {"method": str(method), "url": str(url)[:200],
                    "host": getattr(self, "host", "?")}

        hc.HTTPConnection.request = cs_patch.wrap_callable(
            hc.HTTPConnection.request, "http.request", "network",
            argfn=_req_args, min_dur=0.002)
        hc.HTTPConnection.getresponse = cs_patch.wrap_callable(
            hc.HTTPConnection.getresponse, "http.getresponse", "network",
            min_dur=0.002)

    cs_patch.after_import("socket", _on_socket)
    cs_patch.after_import("ssl", _on_ssl)
    cs_patch.after_import("http.client", _on_httpclient)


def _urlopen_args(self, method=None, url=None, *a, **kw):
    try:
        host = getattr(self, "host", "?")
        scheme = getattr(self, "scheme", "http")
        return {"method": str(method), "url": "%s://%s%s" % (
            scheme, host, str(url)[:200])}
    except Exception:
        return {}


def _requests_args(self, method=None, url=None, *a, **kw):
    return {"method": str(method), "url": str(url)[:250]}


def _patch_clients():
    # urllib3 is the transport under requests and huggingface_hub.
    cs_patch.patch("urllib3.connectionpool", "HTTPConnectionPool.urlopen",
                   name="http.urlopen", cat="network", argfn=_urlopen_args,
                   min_dur=0.002)
    cs_patch.patch("requests.sessions", "Session.request",
                   name="http.requests", cat="network", argfn=_requests_args,
                   min_dur=0.002)
    for mod, path in (("httpx._client", "Client.send"),
                      ("httpx._client", "AsyncClient.send")):
        cs_patch.patch(mod, path, name="http.httpx", cat="network",
                       min_dur=0.002)

    # Hugging Face Hub: metadata round-trips and weight transfer.
    hf = [
        ("hf_hub_download", "hf.hub_download", "network", _hf_dl_args),
        ("get_hf_file_metadata", "hf.file_metadata", "network", None),
        ("http_get", "hf.http_get", "network", None),
        ("_request_wrapper", "hf.request", "network", None),
        ("xet_get", "hf.xet_get", "network", None),
    ]
    for path, name, cat, argfn in hf:
        cs_patch.patch("huggingface_hub.file_download", path, name=name,
                       cat=cat, argfn=argfn, min_dur=0.002)
    cs_patch.patch("huggingface_hub._snapshot_download", "snapshot_download",
                   name="hf.snapshot_download", cat="network")
    cs_patch.patch("huggingface_hub.hf_api", "HfApi.model_info",
                   name="hf.model_info", cat="network")
    cs_patch.patch("huggingface_hub.hf_api", "HfApi.repo_info",
                   name="hf.repo_info", cat="network")
    # transformers' resolver: hits the Hub unless HF_HUB_OFFLINE is set.
    cs_patch.patch("transformers.utils.hub", "cached_file",
                   name="hf.cached_file", cat="network")
    cs_patch.patch("transformers.utils.hub", "has_file",
                   name="hf.has_file", cat="network")


def _hf_dl_args(*a, **kw):
    out = {}
    for k in ("repo_id", "filename", "revision", "local_dir", "cache_dir"):
        if k in kw:
            out[k] = str(kw[k])[:200]
    if a and not out:
        out["arg0"] = str(a[0])[:200]
    return out
