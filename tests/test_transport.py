"""Tests for the transport layer against a real (mock) frame listener.

Unlike mocking the socket, these drive the actual ``socket_bridge`` framing
code over a loopback connection to ``MockListener``, so the length-prefix
encode/decode path is exercised for real.
"""

from types import SimpleNamespace

import pytest

from brlcad_mcp.transport import send_command


def test_success_maps_to_success_prefix(listener):
    resp = send_command("ls")
    assert resp.startswith("SUCCESS:")


def test_command_reaches_listener_verbatim(listener):
    send_command("in ball.s sph 0 0 0 10")
    assert listener.received[-1] == "in ball.s sph 0 0 0 10"


def test_create_then_list_roundtrip(listener):
    send_command("in ball.s sph 0 0 0 10")
    resp = send_command("ls")
    assert "ball.s" in resp


def test_error_maps_to_error_prefix(listener):
    resp = send_command("frobnicate now")
    assert resp.startswith("ERROR:")
    assert "unknown command" in resp


def test_persistent_connection_preserves_state(listener):
    # two calls on one persistent connection see the same database
    send_command("in a.s sph 0 0 0 1")
    send_command("in b.s sph 0 0 0 1")
    resp = send_command("ls")
    assert "a.s" in resp and "b.s" in resp


def test_oversize_command_rejected_locally(listener):
    # longer than MCPCAD_MAXLINE-1; bridge refuses before sending
    resp = send_command("x" * 5000)
    assert resp.startswith("ERROR:")
    assert "limit" in resp
    assert listener.received == []  # never hit the wire


def test_nul_byte_command_rejected_locally(listener):
    resp = send_command("ls\x00rm")
    assert resp.startswith("ERROR:")
    assert "NUL" in resp
    assert listener.received == []


def test_reconnect_after_listener_drop(listener):
    send_command("in a.s sph 0 0 0 1")
    # force the bridge's socket closed; next call must transparently reconnect
    from brlcad_mcp.transport import socket_bridge

    socket_bridge._connection._disconnect()
    resp = send_command("ls")
    assert "a.s" in resp


def test_connection_refused_raises(monkeypatch):
    # point at a port with nothing listening
    from brlcad_mcp.transport import socket_bridge

    fake = SimpleNamespace(
        brlcad=SimpleNamespace(host="127.0.0.1", port=1, timeout=1.0, buffer_size=4096)
    )
    monkeypatch.setattr(socket_bridge, "settings", fake)
    monkeypatch.setattr(socket_bridge, "_connection", socket_bridge._MgedConnection())
    with pytest.raises(ConnectionError):
        send_command("ls")
