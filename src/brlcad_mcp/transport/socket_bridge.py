"""Persistent TCP socket bridge for sending MGED commands to BRL-CAD's Tcl listener.

Maintains a single long-lived connection so that MGED state (edit modes,
selections) is preserved across consecutive ``send_command`` calls.
"""

from __future__ import annotations

import logging
import re
import socket
import threading

from brlcad_mcp.config import settings

logger = logging.getLogger(__name__)

# Must match the delimiter in listener.tcl
_END_OF_RESPONSE = "<<END_OF_RESPONSE>>"

# Regex to strip non-printable control characters (keep newlines and tabs).
_CONTROL_CHAR_RE = re.compile(r"[^\x20-\x7E\n\t]")


def _clean_response(raw: str) -> str:
    """Strip control characters, collapse excessive whitespace, and trim."""
    cleaned = _CONTROL_CHAR_RE.sub("", raw)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


class _MgedConnection:
    """Thread-safe persistent TCP connection to the MGED Tcl listener."""

    def __init__(self) -> None:
        self._sock: socket.socket | None = None
        self._buffer: str = ""
        self._lock = threading.Lock()

    # -- connection management -----------------------------------------------

    def _connect(self) -> socket.socket:
        """Open (or reopen) the TCP connection."""
        cfg = settings.brlcad
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(cfg.timeout)
        sock.connect((cfg.host, cfg.port))
        logger.info("Connected to MGED listener at %s:%s", cfg.host, cfg.port)
        self._buffer = ""
        return sock

    def _ensure_connected(self) -> socket.socket:
        """Return the existing socket, reconnecting if necessary."""
        if self._sock is None:
            logger.info("No existing connection (id=%d), opening new one", id(self))
            self._sock = self._connect()
        else:
            logger.info("Reusing existing connection (id=%d)", id(self))
        return self._sock

    def _disconnect(self) -> None:
        """Close the socket and reset state."""
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
            self._buffer = ""

    # -- reading with delimiter ----------------------------------------------

    def _read_until_delimiter(self, sock: socket.socket) -> str:
        """Read lines from the socket until the end-of-response delimiter."""
        cfg = settings.brlcad
        while _END_OF_RESPONSE not in self._buffer:
            try:
                chunk = sock.recv(cfg.buffer_size)
            except socket.timeout as exc:
                raise TimeoutError(
                    f"BRL-CAD listener at {cfg.host}:{cfg.port} timed out"
                ) from exc
            if not chunk:
                # Connection closed unexpectedly.
                self._disconnect()
                raise ConnectionError(
                    "MGED listener closed the connection unexpectedly"
                )
            self._buffer += chunk.decode("utf-8", errors="replace")

        # Split on the delimiter — keep anything after it for the next call.
        response, _, self._buffer = self._buffer.partition(_END_OF_RESPONSE)
        # Strip the trailing newline the delimiter's `puts` adds.
        self._buffer = self._buffer.lstrip("\n")
        return response

    # -- public API ----------------------------------------------------------

    def send_command(self, cmd: str) -> str:
        """Send *cmd* to MGED and return the cleaned response.

        The connection is reused across calls so MGED state is preserved.
        If the connection drops, one automatic reconnect is attempted.
        """
        cfg = settings.brlcad
        with self._lock:
            for attempt in range(2):  # retry once on broken connection
                try:
                    sock = self._ensure_connected()
                    sock.sendall(f"{cmd}\n".encode("utf-8"))
                    raw = self._read_until_delimiter(sock)
                    response = _clean_response(raw)
                    logger.debug("CMD  → %s", cmd)
                    logger.debug("RESP ← %s", response)
                    return response
                except (ConnectionError, TimeoutError, OSError) as exc:
                    self._disconnect()
                    if attempt == 0:
                        logger.warning("Connection lost, reconnecting: %s", exc)
                        continue
                    # Second attempt failed — surface the error.
                    if isinstance(exc, TimeoutError):
                        raise
                    raise ConnectionError(
                        f"Could not reach BRL-CAD listener at "
                        f"{cfg.host}:{cfg.port}: {exc}"
                    ) from exc

        # Unreachable, but keeps type checkers happy.
        raise ConnectionError("Failed to send command")  # pragma: no cover


# Module-level singleton.
_connection = _MgedConnection()


def send_command(cmd: str) -> str:
    """Send *cmd* to MGED via the persistent connection."""
    return _connection.send_command(cmd)
