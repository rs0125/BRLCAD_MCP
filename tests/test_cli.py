"""Which client `brlcad-mcp chat` actually launches.

Pinned because it silently regressed: v2 was built and used, while the installed
console script still started the deprecated v1 single-agent client, so anyone
running the documented command got the old agent.
"""

import pytest

import brlcad_mcp.client.agent as v1_agent
import client_v2.main as v2_main
from brlcad_mcp.cli import main


def test_chat_launches_client_v2(monkeypatch):
    launched = []
    monkeypatch.setattr(v2_main, "main", lambda: launched.append("v2"))
    monkeypatch.setattr(v1_agent, "main", lambda: launched.append("v1"))
    main(["chat"])
    assert launched == ["v2"]


def test_chat_v1_flag_launches_the_legacy_client(monkeypatch):
    launched = []
    monkeypatch.setattr(v2_main, "main", lambda: launched.append("v2"))
    monkeypatch.setattr(v1_agent, "main", lambda: launched.append("v1"))
    main(["chat", "--v1"])
    assert launched == ["v1"]


def test_serve_starts_the_tool_server(monkeypatch):
    # Imported here, not at module scope: importing the app registers every tool
    # module, which this file otherwise has no need to do.
    import brlcad_mcp.server.app as app
    started = []
    monkeypatch.setattr(app, "serve", lambda: started.append("serve"))
    main(["serve"])
    assert started == ["serve"]


def test_no_subcommand_exits_nonzero(capsys):
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 1
    assert "brlcad-mcp" in capsys.readouterr().out
