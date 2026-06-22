"""Persistent TCP socket bridge to BRL-CAD's libmcpcad listener.

Speaks the libmcpcad length-prefixed frame protocol (see mcpcad.h):

    MC | length (4 bytes, big-endian) | payload

every message, in both directions.  A request payload is a raw MGED
command string; a reply payload begins with an ``OK\\n`` or
``ERR <code>\\n`` status line followed by the command output.

This replaces the previous newline + ``<<END_OF_RESPONSE>>`` sentinel
scheme that targeted a Tcl ``uplevel`` listener.  The command text on
the wire is unchanged - only the framing differs - so the MCP tools
above this layer are untouched.  The libmcpcad status line is mapped
back onto the ``SUCCESS:`` / ``ERROR:`` prefix convention the tools
already understand.

A single long-lived connection is reused so MGED state (edit modes,
selections) is preserved across consecutive ``send_command`` calls.
"""

from __future__ import annotations

import logging
import re
import socket
import struct
import threading

from brlcad_mcp.config import settings

logger = logging.getLogger(__name__)

# libmcpcad frame: 'M' 'C' + uint32 big-endian length + payload.
_FRAME_MAGIC = b"MC"
_FRAME_HDRLEN = 6

# Longest command the listener accepts (MCPCAD_MAXLINE - 1).
_MAX_PAYLOAD = 4095

# Regex to strip non-printable control characters (keep newlines and tabs).
_CONTROL_CHAR_RE = re.compile(r"[^\x20-\x7E\n\t]")


def _clean_response(raw: str) -> str:
    """Strip control characters, collapse excessive whitespace, and trim."""
    cleaned = _CONTROL_CHAR_RE.sub("", raw)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


class _MgedConnection:
    """Thread-safe persistent connection to the libmcpcad listener."""

    def __init__(self) -> None:
        self._sock: socket.socket | None = None
        self._lock = threading.Lock()

    # -- connection management -----------------------------------------------

    def _connect(self) -> socket.socket:
        """Open (or reopen) the TCP connection."""
        cfg = settings.brlcad
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(cfg.timeout)
        sock.connect((cfg.host, cfg.port))
        logger.info("Connected to libmcpcad listener at %s:%s", cfg.host, cfg.port)
        return sock

    def _ensure_connected(self) -> socket.socket:
        """Return the existing socket, reconnecting if necessary."""
        if self._sock is None:
            self._sock = self._connect()
        return self._sock

    def _disconnect(self) -> None:
        """Close the socket and reset state."""
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    # -- framed read/write ---------------------------------------------------

    def _recv_exact(self, sock: socket.socket, n: int) -> bytes:
        """Read exactly *n* bytes, or raise on short read / closed socket."""
        cfg = settings.brlcad
        buf = bytearray()
        while len(buf) < n:
            try:
                chunk = sock.recv(min(n - len(buf), cfg.buffer_size))
            except TimeoutError as exc:
                raise TimeoutError(
                    f"libmcpcad listener at {cfg.host}:{cfg.port} timed out"
                ) from exc
            if not chunk:
                self._disconnect()
                raise ConnectionError(
                    "libmcpcad listener closed the connection unexpectedly"
                )
            buf += chunk
        return bytes(buf)

    def _read_frame(self, sock: socket.socket) -> str:
        """Read one framed reply and return its decoded payload."""
        hdr = self._recv_exact(sock, _FRAME_HDRLEN)
        if hdr[:2] != _FRAME_MAGIC:
            self._disconnect()
            raise ConnectionError(f"bad reply frame magic: {hdr[:2]!r}")
        (length,) = struct.unpack(">I", hdr[2:6])
        body = self._recv_exact(sock, length)
        return body.decode("utf-8", errors="replace")

    # -- public API ----------------------------------------------------------

    def send_command(self, cmd: str) -> str:
        """Send *cmd* to the listener and return the cleaned response.

        The libmcpcad ``OK`` / ``ERR <code>`` status is translated to the
        ``SUCCESS:`` / ``ERROR:`` prefix the MCP tools expect.  The
        connection is reused across calls; one automatic reconnect is
        attempted if it drops.
        """
        cfg = settings.brlcad
        payload = cmd.encode("utf-8")
        if len(payload) > _MAX_PAYLOAD:
            return f"ERROR: command exceeds {_MAX_PAYLOAD}-byte listener limit"
        if b"\x00" in payload:
            return "ERROR: command contains NUL bytes"

        frame = _FRAME_MAGIC + struct.pack(">I", len(payload)) + payload

        with self._lock:
            for attempt in range(2):  # retry once on broken connection
                try:
                    sock = self._ensure_connected()
                    sock.sendall(frame)
                    body = self._read_frame(sock)

                    status, _, output = body.partition("\n")
                    output = _clean_response(output)
                    logger.debug("CMD  -> %s", cmd)
                    logger.debug("RESP <- [%s] %s", status, output)

                    if status.startswith("OK"):
                        return f"SUCCESS: {output}" if output else "SUCCESS:"
                    # status looks like "ERR <code>"; surface code + text
                    detail = output if output else status
                    return f"ERROR: {detail}"
                except (ConnectionError, TimeoutError, OSError) as exc:
                    self._disconnect()
                    if attempt == 0:
                        logger.warning("Connection lost, reconnecting: %s", exc)
                        continue
                    if isinstance(exc, TimeoutError):
                        raise
                    raise ConnectionError(
                        f"Could not reach libmcpcad listener at "
                        f"{cfg.host}:{cfg.port}: {exc}"
                    ) from exc

        # Unreachable, but keeps type checkers happy.
        raise ConnectionError("Failed to send command")  # pragma: no cover


# Module-level singleton.
_connection = _MgedConnection()


def send_command(cmd: str) -> str:
    """Send *cmd* to the listener via the persistent connection."""
    return _connection.send_command(cmd)
