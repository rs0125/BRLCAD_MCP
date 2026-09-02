"""Shared test fixtures.

The centerpiece is :class:`MockListener` - a protocol-accurate stand-in for
BRL-CAD's libmcpcad listener.  It speaks the exact length-prefixed frame
protocol (``MC`` + u32 big-endian length + payload) and runs a tiny stateful
"fake MGED" so the whole stack - real ``socket_bridge``, real tools, real
agent - can be exercised end to end with **no BRL-CAD build and no API key**.

Deterministic tests use it directly.  Natural-language tests (marked ``llm``)
reuse it but only run when ``--run-llm`` is passed and an API key is present.
"""

from __future__ import annotations

import socket
import struct
import threading
from types import SimpleNamespace

import dotenv
import pytest

# Isolate the suite from any .env on the machine.
#
# config.py reads the nearest .env at import, which is right for running the
# tool and wrong for testing it: whether the tests pass would otherwise depend
# on whether this checkout happens to be configured, and on what for.  A tree
# with BRLCAD_IPC_PATH set really did start failing test_no_ipc_path_means_tcp.
#
# Done by neutering dotenv here rather than by giving the config an "ignore
# .env" switch: that switch would be a production setting whose only purpose is
# to make tests pass, and it could not even be set in a .env file (the check
# would run before the file was read), which is a fair sign it belongs in the
# tests instead.  conftest is imported before any test module, so config's own
# ``from dotenv import ...`` picks these up.
#
# The subprocess tests in test_env_discovery.py are unaffected, which is the
# point: they get a real dotenv because they are separate processes.
dotenv.load_dotenv = lambda *a, **k: False
dotenv.find_dotenv = lambda *a, **k: ""

_MAGIC = b"MC"


# ---------------------------------------------------------------------------
# Fake MGED — minimal stateful command interpreter
# ---------------------------------------------------------------------------

class FakeMged:
    """A tiny in-memory stand-in for an MGED geometry database.

    Implements just enough command surface for the tools and the agent:
    object creation, listing, combinations, deletion, and display no-ops.
    Unknown commands return an error, mirroring real GED behaviour.
    """

    # display / view commands that succeed with no output
    _NOOP = {"draw", "erase", "autoview", "z", "d", "b", "blast"}
    # commands that create a single named object as argv[1]
    _CREATE = {"in", "make", "r", "c", "comb", "g"}

    def __init__(self) -> None:
        self.db: set[str] = set()
        self.title = "Mock BRL-CAD Database"

    def execute(self, line: str) -> tuple[int, str]:
        """Run *line*; return (status, output).  status 0 == success."""
        argv = line.split()
        if not argv:
            return 1, "no command"
        cmd, args = argv[0], argv[1:]

        if cmd == "ls":
            names = [n for n in args if not n.startswith("-")]
            if names:  # ls with explicit patterns -> only matching
                shown = sorted(n for n in self.db if n in names)
            else:
                shown = sorted(self.db)
            return 0, "  ".join(shown)

        if cmd in self._CREATE:
            if not args:
                return 1, f"{cmd}: missing object name"
            self.db.add(args[0])
            return 0, args[0] if cmd == "in" else ""

        if cmd in {"kill", "killall"}:
            for n in args:
                self.db.discard(n)
            return 0, ""

        if cmd in self._NOOP:
            return 0, ""

        if cmd == "l":
            if args and args[0] in self.db:
                return 0, f"{args[0]}:  (mock detail)"
            return 1, f"{cmd}: {args[0] if args else ''} does not exist"

        if cmd == "bb":
            if args and args[0] in self.db:
                return 0, f"Bounding Box Dimensions for {args[0]}: 10 10 10"
            return 1, f"bb: {args[0] if args else ''} not found"

        if cmd == "title":
            if args:
                self.title = " ".join(args)
                return 0, ""
            return 0, self.title

        if cmd in {"help", "man", "?"}:
            topic = args[0] if args else ""
            return 0, f"Usage for {topic}: <mock help text>"

        return 1, f"unknown command: {cmd}"


# ---------------------------------------------------------------------------
# Mock listener — frame protocol over a real loopback socket
# ---------------------------------------------------------------------------

class MockListener:
    """Threaded loopback server speaking the libmcpcad frame protocol."""

    def __init__(self) -> None:
        self.mged = FakeMged()
        self.received: list[str] = []  # every command line, in order
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(8)
        self.port = self._sock.getsockname()[1]
        self._stop = False
        self._thread = threading.Thread(target=self._serve, daemon=True)

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop = True
        try:
            self._sock.close()
        except OSError:
            pass

    # -- frame I/O -----------------------------------------------------------

    @staticmethod
    def _recv_exact(conn: socket.socket, n: int) -> bytes | None:
        buf = b""
        while len(buf) < n:
            chunk = conn.recv(n - len(buf))
            if not chunk:
                return None
            buf += chunk
        return buf

    def _serve(self) -> None:
        while not self._stop:
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            with conn:
                self._handle(conn)

    def _handle(self, conn: socket.socket) -> None:
        while not self._stop:
            hdr = self._recv_exact(conn, 6)
            if hdr is None:
                return
            if hdr[:2] != _MAGIC:
                # bad magic -> mimic listener: reply ERR -5 and hang up
                self._send(conn, "ERR -5\nnot an mcpcad frame stream; closing")
                return
            (length,) = struct.unpack(">I", hdr[2:6])
            payload = self._recv_exact(conn, length)
            if payload is None:
                return
            cmd = payload.decode("utf-8", errors="replace")
            self.received.append(cmd)
            status, output = self.mged.execute(cmd)
            body = f"OK\n{output}" if status == 0 else f"ERR {status}\n{output}"
            self._send(conn, body)

    @staticmethod
    def _send(conn: socket.socket, body: str) -> None:
        data = body.encode("utf-8")
        conn.sendall(_MAGIC + struct.pack(">I", len(data)) + data)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def listener(monkeypatch):
    """Start a MockListener and point the socket bridge at it."""
    from brlcad_mcp.transport import socket_bridge

    srv = MockListener()
    srv.start()

    fake_settings = SimpleNamespace(
        brlcad=SimpleNamespace(
            host="127.0.0.1", port=srv.port, timeout=5.0, buffer_size=4096,
            ipc_path="",   # these fakes exercise the TCP transport
        )
    )
    monkeypatch.setattr(socket_bridge, "settings", fake_settings)
    # fresh connection so no socket leaks across tests
    monkeypatch.setattr(socket_bridge, "_connection", socket_bridge._MgedConnection())

    try:
        yield srv
    finally:
        srv.stop()
