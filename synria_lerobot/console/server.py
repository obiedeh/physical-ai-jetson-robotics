"""Serve a loopback operator surface with same-origin, token-bound mutations."""

from __future__ import annotations

import hmac
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import parse_qs, urlsplit

from synria_lerobot.console.playback import confined_file

STATIC_ROOT = Path(__file__).with_name("static")
MAX_BODY = 64 * 1024
CSP = (
    "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' blob:; "
    "connect-src 'self'; media-src 'self' blob:; base-uri 'none'; "
    "frame-ancestors 'none'; form-action 'self'"
)


class ConsoleAPI(Protocol):
    """Keep HTTP request parsing separate from recorder ownership and disk mutation."""

    def read(self, route: list[str], query: dict[str, list[str]]) -> Any:
        """Return read-only JSON for a validated relative API route."""

    def mutate(self, route: list[str], payload: dict[str, Any]) -> Any:
        """Apply one authenticated operation and return its restored server state."""

    def media(self, route: list[str], query: dict[str, list[str]]) -> bytes:
        """Read a session-scoped JPEG, never a caller-supplied filesystem path."""


def make_server(
    service: ConsoleAPI, *, token: str, host: str = "127.0.0.1", port: int = 0,
    allow_remote: bool = False,
) -> ThreadingHTTPServer:
    """Create an HTTP server whose browser origin cannot mutate it without its launch token."""
    if ":" in host:
        raise ValueError("this console supports IPv4 bind addresses only")
    if host not in {"127.0.0.1", "localhost"} and not allow_remote:
        raise ValueError("non-loopback bind requires --allow-remote and trusted network isolation")
    if len(token) < 32:
        raise ValueError("launch token must contain at least 32 characters")

    class Handler(BaseHTTPRequestHandler):
        """Constrain every request before routing it to the operator service."""

        server_version = "SynriaConsole/1"

        def setup(self) -> None:
            """Bound idle request sockets so abandoned pages cannot exhaust request threads."""
            super().setup()
            self.connection.settimeout(10)

        def log_message(self, format: str, *args: Any) -> None:
            """Avoid logging launch secrets, operator metadata or request bodies."""

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            """Apply the same restrictive response headers to assets, data and errors."""
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Content-Security-Policy", CSP)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, payload: Any) -> None:
            """Serialize finite JSON without executable response content."""
            self._send(
                code, json.dumps(payload, allow_nan=False).encode("utf-8"),
                "application/json; charset=utf-8",
            )

        def _guard(self, *, mutation: bool) -> None:
            """Reject cross-site requests and alternate Host values before any service call."""
            actual_port = server.server_address[1]
            aliases = {host}
            if host in {"127.0.0.1", "localhost"}:
                aliases.update({"127.0.0.1", "localhost"})
            allowed_hosts = {f"{name}:{actual_port}" for name in aliases}
            hosts = self.headers.get_all("Host", [])
            if len(hosts) != 1 or hosts[0] not in allowed_hosts:
                raise PermissionError("Host is not this console's bound address")
            origins = self.headers.get_all("Origin", [])
            if (mutation and len(origins) != 1) or len(origins) > 1:
                raise PermissionError("a single same-origin Origin is required")
            if origins and origins[0] != f"http://{hosts[0]}":
                raise PermissionError("Origin is not this console's origin")
            if mutation:
                tokens = self.headers.get_all("X-Console-Token", [])
                if len(tokens) != 1 or not hmac.compare_digest(tokens[0], token):
                    raise PermissionError("launch token is missing or invalid")

        def _route(self) -> tuple[list[str], dict[str, list[str]]]:
            """Parse a relative route without encoded path aliases or traversal segments."""
            parsed = urlsplit(self.path)
            if parsed.scheme or parsed.netloc or "%" in parsed.path or "\\" in parsed.path:
                raise ValueError("invalid request path")
            parts = parsed.path.strip("/").split("/")
            if ".." in parts or "." in parts:
                raise ValueError("path traversal is refused")
            return parts, parse_qs(parsed.query)

        def _error(self, error: Exception) -> None:
            """Map expected refusal types to stable HTTP errors without exposing tracebacks."""
            if isinstance(error, PermissionError):
                code = 403
            elif isinstance(error, (KeyError, FileNotFoundError, IndexError)):
                code = 404
            elif isinstance(error, (ValueError, TypeError, json.JSONDecodeError)):
                code = 400
            elif isinstance(error, (RuntimeError, OSError)):
                code = 409
            else:
                code = 500
            self._json(code, {"error": str(error) if code != 500 else "internal request failure"})

        def do_GET(self) -> None:
            """Serve read-only state and package-owned assets; GET never changes a session."""
            try:
                self._guard(mutation=False)
                route, query = self._route()
                if route[0] == "api":
                    self._json(200, service.read(route[1:], query))
                elif route[0] == "media":
                    self._send(200, service.media(route[1:], query), "image/jpeg")
                else:
                    name = "index.html" if route == [""] else "/".join(route)
                    path = confined_file(STATIC_ROOT, name)
                    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                    self._send(200, path.read_bytes(), content_type)
            except Exception as error:
                self._error(error)

        def do_POST(self) -> None:
            """Authenticate a bounded JSON mutation without accepting file paths or code."""
            try:
                self._guard(mutation=True)
                route, query = self._route()
                if route[0] != "api" or query:
                    raise ValueError("mutations require an API path without query parameters")
                lengths = self.headers.get_all("Content-Length", [])
                if len(lengths) != 1 or self.headers.get("Transfer-Encoding"):
                    raise ValueError("one explicit request length is required")
                size = int(lengths[0])
                if not 0 < size <= MAX_BODY:
                    raise ValueError("request body exceeds the console limit")
                if self.headers.get_content_type() != "application/json":
                    raise ValueError("mutations require application/json")
                body = self.rfile.read(size)
                if len(body) != size:
                    raise ValueError("incomplete request body")
                payload = json.loads(body)
                if not isinstance(payload, dict):
                    raise ValueError("request must be a JSON object")
                self._json(200, service.mutate(route[1:], payload))
            except Exception as error:
                self._error(error)

        def do_DELETE(self) -> None:
            """Reject alternative mutation methods so every write uses the same guarded path."""
            self._json(405, {"error": "state changes require POST"})

        do_PUT = do_DELETE
        do_PATCH = do_DELETE

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server
