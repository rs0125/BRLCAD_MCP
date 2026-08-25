"""The client picks its transport from configuration, not a mode flag.

Presence of BRLCAD_IPC_PATH selects a Unix-domain socket; otherwise the
loopback port is used.  Same convention as LLM_BASE_URL selecting the model
endpoint, so there is one rule across the configuration rather than two.
"""

import importlib
import os
import socket
import tempfile
import threading

import pytest


def _reload(monkeypatch, **env):
    for k in ("BRLCAD_IPC_PATH", "BRLCAD_HOST", "BRLCAD_PORT",
              "BRLCAD_TIMEOUT"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    import brlcad_mcp.config as cfg
    import brlcad_mcp.transport.socket_bridge as sb
    importlib.reload(cfg)
    importlib.reload(sb)
    return cfg, sb


def test_no_ipc_path_means_tcp(monkeypatch):
    cfg, _ = _reload(monkeypatch)
    assert cfg.settings.brlcad.ipc_path == ""


def test_ipc_path_is_read_from_the_environment(monkeypatch):
    cfg, _ = _reload(monkeypatch, BRLCAD_IPC_PATH="/tmp/x.sock")
    assert cfg.settings.brlcad.ipc_path == "/tmp/x.sock"


def test_blank_ipc_path_is_unset_not_a_socket_named_empty(monkeypatch):
    # Copying .env.example verbatim leaves it empty.
    cfg, _ = _reload(monkeypatch, BRLCAD_IPC_PATH="")
    assert cfg.settings.brlcad.ipc_path == ""


def test_round_trip_over_a_unix_socket(monkeypatch):
    """The real path: a Unix-domain listener, driven by the real client."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "mcp.sock")
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(path)
        srv.listen(1)
        seen = []

        def serve():
            conn, _ = srv.accept()
            with conn:
                # one framed request in, one framed reply out
                hdr = conn.recv(6)
                n = int.from_bytes(hdr[2:6], "big")
                seen.append(conn.recv(n).decode())
                body = b"OK\nunix-ok"
                conn.sendall(b"MC" + len(body).to_bytes(4, "big") + body)

        t = threading.Thread(target=serve, daemon=True)
        t.start()

        _, sb = _reload(monkeypatch, BRLCAD_IPC_PATH=path)
        out = sb.send_command("ls")
        t.join(timeout=5)
        srv.close()

    assert seen == ["ls"]
    assert "unix-ok" in out


def test_a_missing_socket_reports_the_path_not_a_port(monkeypatch):
    _, sb = _reload(monkeypatch, BRLCAD_IPC_PATH="/tmp/definitely-not-here.sock")
    with pytest.raises(ConnectionError) as exc:
        sb.send_command("ls")
    assert "/tmp/definitely-not-here.sock" in str(exc.value)


def test_a_timeout_names_the_ipc_socket_not_a_tcp_address(monkeypatch):
    """A diagnostic that names the wrong transport sends you debugging the
    wrong thing.  Seen live: 'listener at 127.0.0.1:5555 timed out' while the
    client was in fact connected over a Unix socket."""
    import socket as _socket
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "slow.sock")
        srv = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
        srv.bind(path)
        srv.listen(1)
        accepted = []

        def serve():           # accept, then never reply
            conn, _ = srv.accept()
            accepted.append(conn)

        t = threading.Thread(target=serve, daemon=True)
        t.start()

        _, sb = _reload(monkeypatch, BRLCAD_IPC_PATH=path,
                        BRLCAD_TIMEOUT="0.3")
        with pytest.raises(TimeoutError) as exc:
            sb.send_command("ls")
        msg = str(exc.value)
        srv.close()
        for c in accepted:
            c.close()

    assert path in msg
    assert "127.0.0.1" not in msg
