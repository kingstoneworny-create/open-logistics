"""Embedded, dependency-free HTTP/JSON microservice.

Run it next to an existing stack (WMS, ERP, Shopify app, ...) and talk to it
over plain HTTP::

    python -m open_logistics serve --port 8080

Endpoints (all JSON; ``/api/...`` and ``/api/v1/...`` are equivalent):

=======  ======================  ==============================================
Method   Path                    Purpose
=======  ======================  ==============================================
GET      /health                 Liveness probe
GET      /api/v1/info            Version, uptime, and active configuration
POST     /api/v1/dispatch        Orders + fleet + depot -> optimised routes
POST     /api/v1/inventory       Stock items -> ROP / safety stock / EOQ
POST     /api/v1/wave-pick       Orders or order IDs -> pick waves + tasks
=======  ======================  ==============================================

Errors come back as ``{"error": {"type": ..., "message": ...}}`` with 400
(invalid input), 404, 405, 413 (body too large), or 500.

Built on :class:`http.server.ThreadingHTTPServer`; intended for internal
networks or behind a reverse proxy that terminates TLS and handles auth.
"""

from __future__ import annotations

import json
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, Optional, Tuple

from open_logistics import __version__
from open_logistics.api.engine import LogisticsEngine
from open_logistics.core.exceptions import LogisticsError, ValidationError

__all__ = ["LogisticsHTTPHandler", "LogisticsHTTPServer", "create_server", "serve"]

MAX_BODY_BYTES = 5 * 1024 * 1024


class LogisticsHTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer carrying a shared :class:`LogisticsEngine`."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: Tuple[str, int],
        engine: LogisticsEngine,
        verbose: bool = False,
    ) -> None:
        self.engine = engine
        self.verbose = verbose
        self.started_at = time.time()
        super().__init__(server_address, LogisticsHTTPHandler)


class LogisticsHTTPHandler(BaseHTTPRequestHandler):
    """Routes requests to the server's :class:`LogisticsEngine`."""

    server: LogisticsHTTPServer  # narrowed type
    server_version = f"OpenLogistics/{__version__}"
    protocol_version = "HTTP/1.1"

    _GET_ROUTES = {"/health", "/api/health", "/api/v1/health", "/api/info", "/api/v1/info"}
    _POST_ROUTES = {"dispatch", "inventory", "wave-pick"}

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _normalise(path: str) -> str:
        path = path.split("?", 1)[0].rstrip("/") or "/"
        if path.startswith("/api/v1/"):
            return "/api/" + path[len("/api/v1/"):]
        return path

    def _post_action(self, path: str) -> Optional[str]:
        if path.startswith("/api/"):
            action = path[len("/api/"):]
            if action in self._POST_ROUTES:
                return action
        return None

    def _is_known(self, path: str) -> bool:
        return self._normalise(path) in {self._normalise(p) for p in self._GET_ROUTES} or (
            self._post_action(self._normalise(path)) is not None
        )

    def _send_json(self, status: int, body: Any) -> None:
        data = json.dumps(body, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def _send_error(self, status: int, err_type: str, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        payload: Dict[str, Any] = {"type": err_type, "message": message}
        if extra:
            payload["context"] = extra
        self._send_json(status, {"error": payload})

    def _read_json(self) -> Any:
        length_header = self.headers.get("Content-Length")
        try:
            length = int(length_header) if length_header is not None else 0
        except ValueError as exc:
            raise ValidationError("Invalid Content-Length header") from exc
        if length < 0:
            raise ValidationError("Invalid Content-Length header")
        if length > MAX_BODY_BYTES:
            raise OverflowError(f"Request body exceeds {MAX_BODY_BYTES} bytes")
        raw = self.rfile.read(length) if length else b""
        if not raw:
            raise ValidationError("Request body is empty; expected a JSON object")
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValidationError(f"Malformed JSON: {exc}") from exc

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 (stdlib signature)
        if getattr(self.server, "verbose", False):
            super().log_message(format, *args)

    # --------------------------------------------------------------- handlers
    def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
        path = self._normalise(self.path)
        if path in ("/health", "/api/health"):
            self._send_json(HTTPStatus.OK, {"status": "healthy", "version": __version__})
        elif path == "/api/info":
            self._send_json(
                HTTPStatus.OK,
                {
                    "name": "open-logistics",
                    "version": __version__,
                    "uptime_seconds": round(time.time() - self.server.started_at, 3),
                    "config": self.server.engine.config.to_dict(),
                    "endpoints": ["/health", "/api/v1/info", "/api/v1/dispatch", "/api/v1/inventory", "/api/v1/wave-pick"],
                },
            )
        elif self._post_action(path):
            self._method_not_allowed()
        else:
            self._send_error(HTTPStatus.NOT_FOUND, "NotFound", f"No route for GET {self.path}")

    do_HEAD = do_GET

    def do_POST(self) -> None:  # noqa: N802
        path = self._normalise(self.path)
        action = self._post_action(path)
        if action is None:
            if self._is_known(path):
                self._method_not_allowed()
            else:
                self._send_error(HTTPStatus.NOT_FOUND, "NotFound", f"No route for POST {self.path}")
            return

        engine = self.server.engine
        handlers: Dict[str, Callable[[Any], Dict[str, Any]]] = {
            "dispatch": engine.dispatch_from_payload,
            "inventory": engine.inventory_from_payload,
            "wave-pick": engine.wave_from_payload,
        }
        try:
            payload = self._read_json()
            result = handlers[action](payload)
        except OverflowError as exc:
            self._send_error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "PayloadTooLarge", str(exc))
            self.close_connection = True
        except LogisticsError as exc:
            # Domain/validation failures are the client's problem -> 400.
            self._send_error(HTTPStatus.BAD_REQUEST, type(exc).__name__, exc.message if hasattr(exc, "message") else str(exc), getattr(exc, "context", None) or None)
        except Exception as exc:  # pragma: no cover - defensive
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "InternalError", f"{type(exc).__name__}: {exc}")
        else:
            self._send_json(HTTPStatus.OK, result)

    def _method_not_allowed(self) -> None:
        self._send_error(HTTPStatus.METHOD_NOT_ALLOWED, "MethodNotAllowed", f"{self.command} not allowed on {self.path}")

    def _unsupported(self) -> None:
        # Drain any body so keep-alive connections stay in sync.
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if 0 < length <= MAX_BODY_BYTES:
                self.rfile.read(length)
        except ValueError:
            pass
        if self._is_known(self.path):
            self._method_not_allowed()
        else:
            self._send_error(HTTPStatus.NOT_FOUND, "NotFound", f"No route for {self.command} {self.path}")

    do_PUT = _unsupported
    do_DELETE = _unsupported
    do_PATCH = _unsupported


def create_server(
    host: str = "127.0.0.1",
    port: int = 8080,
    engine: Optional[LogisticsEngine] = None,
    verbose: bool = False,
) -> LogisticsHTTPServer:
    """Create (but do not start) the HTTP server. Call ``serve_forever()`` to run.

    Pass ``port=0`` to bind a random free port (see ``server.server_address``).
    """
    return LogisticsHTTPServer((host, int(port)), engine or LogisticsEngine(), verbose=verbose)


def serve(host: str = "127.0.0.1", port: int = 8080, engine: Optional[LogisticsEngine] = None, verbose: bool = True) -> None:
    """Run the server in the foreground until Ctrl+C."""
    server = create_server(host, port, engine=engine, verbose=verbose)
    bound_host, bound_port = server.server_address[:2]
    print(f"open-logistics {__version__} listening on http://{bound_host}:{bound_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
