"""Serve TextScan's frontend and stream sentence analysis from a worker process.

Static files are an explicit allowlist, preloaded into memory, gzip-compressed once,
and served with ETags and long-lived caching (URLs are versioned by content hash).
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
import multiprocessing
import os
import queue
import re
import socket
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from backend.detector_worker import MAX_SENTENCES, worker_main

ROOT = Path(__file__).resolve().parent
MAX_TEXT_CHARACTERS = 12_000
MAX_BODY_BYTES = 100_000
MIN_WORDS = 20
MAX_IN_FLIGHT = 2
STREAM_TIMEOUT_SECONDS = 900
KEEPALIVE_SECONDS = 15
WATCHDOG_SECONDS = 5
_WORD_RE = re.compile(r"\b[\w’'-]+\b", re.UNICODE)
_TERMINAL = {"complete", "error"}

_CTX = multiprocessing.get_context("spawn")
_LOCK = threading.Lock()
_STOP = threading.Event()
_WAITERS: dict[str, queue.Queue] = {}
_ACTIVE: set[str] = set()
_WORKER: dict[str, Any] = {"jobs": None, "results": None, "cancel": None, "ready": None, "proc": None}
_STATIC: dict[str, dict[str, Any]] = {}

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
}

# Built at startup by _load_static(): inline <style> blocks in index.html are allowed
# by SHA-256 hash only, so 'unsafe-inline' is never needed.
CSP_TEMPLATE = (
    "default-src 'none'; script-src 'self'; style-src 'self'{hashes}; img-src 'self' data:; "
    "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
)
_CSP = CSP_TEMPLATE.format(hashes="")


# ---------------------------------------------------------------- static assets

def _asset(body: bytes, mime: str, cache: str) -> dict[str, Any]:
    tag = hashlib.sha1(body).hexdigest()[:16]
    packed = gzip.compress(body, 9, mtime=0) if len(body) > 512 else None
    return {"raw": body, "gzip": packed, "mime": mime, "cache": cache, "etag": f'"{tag}"'}


def _load_static() -> None:
    """Read the allowlisted files once and build versioned, compressed responses."""
    immutable = "public, max-age=31536000, immutable"
    versions = {"__CSS_V__": "0", "__JS_V__": "0"}

    for name, mime, key in (
        ("styles.css", "text/css; charset=utf-8", "__CSS_V__"),
        ("app.js", "text/javascript; charset=utf-8", "__JS_V__"),
    ):
        path = ROOT / name
        if not path.is_file():
            print(f"warning: {name} not found; it will return 404", file=sys.stderr)
            continue
        body = path.read_bytes()
        _STATIC[f"/{name}"] = _asset(body, mime, immutable)
        versions[key] = hashlib.sha1(body).hexdigest()[:10]

    html_path = next((p for p in (ROOT / "index.html", ROOT / "Index.html") if p.is_file()), None)
    if html_path is None:
        raise SystemExit("index.html not found next to server.py")
    html = html_path.read_text(encoding="utf-8")
    for placeholder, value in versions.items():
        html = html.replace(placeholder, value)

    global _CSP
    hashes = "".join(
        " 'sha256-%s'" % base64.b64encode(hashlib.sha256(block.encode("utf-8")).digest()).decode("ascii")
        for block in re.findall(r"<style[^>]*>(.*?)</style>", html, re.DOTALL)
    )
    _CSP = CSP_TEMPLATE.format(hashes=hashes)
    _STATIC["/"] = _asset(html.encode("utf-8"), "text/html; charset=utf-8", "no-cache")


# ---------------------------------------------------------------- worker lifecycle

def _route_worker_events(results: Any) -> None:
    while True:
        event = results.get()
        if event is None:
            return
        request_id = event.get("request_id")
        if not request_id:
            continue
        with _LOCK:
            waiter = _WAITERS.get(request_id)
            if event.get("type") in _TERMINAL:
                _ACTIVE.discard(request_id)
        if waiter is not None:
            try:
                waiter.put_nowait(event)
            except queue.Full:
                pass


def _start_worker() -> None:
    jobs = _CTX.Queue(maxsize=MAX_IN_FLIGHT)
    results = _CTX.Queue()
    cancel = _CTX.Queue()
    ready = _CTX.Event()
    proc = _CTX.Process(
        target=worker_main,
        args=(jobs, results, cancel, ready),
        name="textscan-model-worker",
        daemon=True,
    )
    proc.start()
    threading.Thread(target=_route_worker_events, args=(results,), name="textscan-router", daemon=True).start()
    with _LOCK:
        _WORKER.update(jobs=jobs, results=results, cancel=cancel, ready=ready, proc=proc)


def _restart_worker() -> None:
    """Fail in-flight requests, drop the dead worker, and start a fresh one."""
    with _LOCK:
        old = dict(_WORKER)
        waiters = list(_WAITERS.values())
        _ACTIVE.clear()
    failure = {"type": "error", "message": "The analysis worker restarted. Please retry."}
    for waiter in waiters:
        try:
            waiter.put_nowait(failure)
        except queue.Full:
            pass
    proc = old.get("proc")
    if proc is not None and proc.is_alive():
        proc.terminate()
    if old.get("results") is not None:
        try:
            old["results"].put(None)
        except Exception:
            pass
    _start_worker()


def _watchdog() -> None:
    while not _STOP.wait(WATCHDOG_SECONDS):
        proc = _WORKER["proc"]
        if proc is not None and not proc.is_alive():
            _restart_worker()


def _cancel(request_id: str) -> None:
    try:
        _WORKER["cancel"].put_nowait(request_id)
    except Exception:
        pass


# ---------------------------------------------------------------- HTTP handler

class TextScanHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "TextScan"
    sys_version = ""
    timeout = 30  # drops idle or stalled sockets (slow-client protection)

    def setup(self) -> None:
        super().setup()
        # Send small SSE chunks immediately instead of waiting for Nagle's algorithm.
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def log_message(self, format: str, *args) -> None:
        return  # never log request bodies or submitted text

    def end_headers(self) -> None:
        for name, value in SECURITY_HEADERS.items():
            self.send_header(name, value)
        self.send_header("Content-Security-Policy", _CSP)
        super().end_headers()

    # -- responses

    def _send_json(self, status: int, payload: dict, *, close: bool = False) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        if close:
            self.close_connection = True
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _serve_static(self, head_only: bool) -> None:
        path = urlsplit(self.path).path
        asset = _STATIC.get("/" if path == "/index.html" else path)
        if asset is None:
            self._send_json(404, {"error": "Not found."})
            return

        use_gzip = asset["gzip"] is not None and "gzip" in self.headers.get("Accept-Encoding", "")
        body = asset["gzip"] if use_gzip else asset["raw"]
        etag = asset["etag"][:-1] + '-gz"' if use_gzip else asset["etag"]

        if etag in (self.headers.get("If-None-Match") or ""):
            self.send_response(304)
            self.send_header("ETag", etag)
            self.send_header("Cache-Control", asset["cache"])
            self.send_header("Vary", "Accept-Encoding")
            self.end_headers()
            return

        self.send_response(200)
        self.send_header("Content-Type", asset["mime"])
        self.send_header("Content-Length", str(len(body)))
        self.send_header("ETag", etag)
        self.send_header("Cache-Control", asset["cache"])
        self.send_header("Vary", "Accept-Encoding")
        if use_gzip:
            self.send_header("Content-Encoding", "gzip")
        self.end_headers()
        if not head_only:
            self.wfile.write(body)

    def _begin_stream(self) -> None:
        self.close_connection = True
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Connection", "close")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        self.wfile.flush()

    def _write_chunk(self, content: str) -> None:
        encoded = content.encode("utf-8")
        self.wfile.write(b"%X\r\n%s\r\n" % (len(encoded), encoded))
        self.wfile.flush()

    def _write_event(self, event: dict) -> None:
        self._write_chunk(f"data: {json.dumps(event, ensure_ascii=False, separators=(',', ':'))}\n\n")

    # -- routes

    def do_HEAD(self) -> None:
        if urlsplit(self.path).path.startswith("/api/"):
            self._send_json(404, {"error": "API route not found."})
        else:
            self._serve_static(head_only=True)

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path == "/api/health":
            proc = _WORKER["proc"]
            alive = proc is not None and proc.is_alive()
            ready = _WORKER["ready"] is not None and _WORKER["ready"].is_set()
            self._send_json(200 if alive else 503, {
                "status": "ok" if alive else "worker_unavailable",
                "analysis_worker": "running" if alive else "stopped",
                "model_ready": ready,
                "streaming": True,
            })
        elif path.startswith("/api/"):
            self._send_json(404, {"error": "API route not found."})
        else:
            self._serve_static(head_only=False)

    def do_POST(self) -> None:
        # Request bodies are not always drained on early errors, so never reuse this connection.
        self.close_connection = True
        if urlsplit(self.path).path != "/api/scan":
            self._send_json(404, {"error": "API route not found."}, close=True)
            return
        if self.headers.get_content_type() != "application/json":
            self._send_json(415, {"error": "Send a JSON request body."}, close=True)
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(400, {"error": "Invalid request size."}, close=True)
            return
        if content_length <= 0:
            self._send_json(400, {"error": "Request body is empty."}, close=True)
            return
        if content_length > MAX_BODY_BYTES:
            self._send_json(413, {"error": "Request body is too large."}, close=True)
            return

        try:
            payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, TimeoutError):
            self._send_json(400, {"error": "Request body must be valid UTF-8 JSON."}, close=True)
            return
        if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
            self._send_json(400, {"error": "Include a text field containing a string."})
            return

        text = payload["text"].strip()
        if len(text) > MAX_TEXT_CHARACTERS:
            self._send_json(413, {"error": f"Text must be {MAX_TEXT_CHARACTERS:,} characters or fewer."})
            return
        if len(_WORD_RE.findall(text)) < MIN_WORDS:
            self._send_json(422, {"error": f"Please enter at least {MIN_WORDS} words for an analysis."})
            return

        proc = _WORKER["proc"]
        if proc is None or not proc.is_alive():
            self._send_json(503, {"error": "The analysis worker is not available. Please retry shortly."})
            return

        request_id = uuid.uuid4().hex
        waiter: queue.Queue = queue.Queue(maxsize=MAX_SENTENCES + 4)
        with _LOCK:
            if len(_ACTIVE) >= MAX_IN_FLIGHT:
                busy = True
            else:
                busy = False
                _ACTIVE.add(request_id)
                _WAITERS[request_id] = waiter
        if busy:
            self._send_json(429, {"error": "The analysis worker is busy. Please try again shortly."})
            return

        try:
            _WORKER["jobs"].put_nowait({"request_id": request_id, "text": text})
        except Exception as error:
            with _LOCK:
                _ACTIVE.discard(request_id)
                _WAITERS.pop(request_id, None)
            if isinstance(error, queue.Full):
                self._send_json(429, {"error": "The analysis queue is full. Please try again shortly."})
            else:
                self._send_json(503, {"error": "The analysis worker could not accept this request."})
            return

        self._stream_results(request_id, waiter)

    def _stream_results(self, request_id: str, waiter: queue.Queue) -> None:
        self._begin_stream()
        deadline = time.monotonic() + STREAM_TIMEOUT_SECONDS
        finished = False
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._write_event({"type": "error", "message": "This scan exceeded the time limit. Please retry."})
                    break
                try:
                    event = waiter.get(timeout=min(KEEPALIVE_SECONDS, remaining))
                except queue.Empty:
                    self._write_chunk(": keep-alive\n\n")
                    continue
                self._write_event(event)
                if event.get("type") in _TERMINAL:
                    finished = True
                    break
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            pass
        finally:
            with _LOCK:
                _WAITERS.pop(request_id, None)
            if not finished:
                # Client left or timed out: stop wasting CPU. The worker's terminal
                # event frees the slot.
                _cancel(request_id)


class TextScanHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 64

    def handle_error(self, request, client_address) -> None:
        if isinstance(sys.exc_info()[1], (BrokenPipeError, ConnectionResetError, TimeoutError)):
            return
        super().handle_error(request, client_address)


def main() -> None:
    port = int(os.environ.get("PORT", "3000"))
    host = os.environ.get("HOST", "0.0.0.0")
    _load_static()
    server = TextScanHTTPServer((host, port), TextScanHandler)
    _start_worker()
    threading.Thread(target=_watchdog, name="textscan-watchdog", daemon=True).start()
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        _STOP.set()
        server.server_close()
        try:
            _WORKER["jobs"].put(None, timeout=2)
        except Exception:
            pass
        proc = _WORKER["proc"]
        if proc is not None:
            proc.join(timeout=10)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=2)


if __name__ == "__main__":
    main()
