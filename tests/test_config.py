"""Tests for brlcad_mcp.config defaults and composition."""

from brlcad_mcp.config import BRLCADConfig, LLMConfig, ServerConfig, Settings


def test_default_brlcad_config():
    cfg = BRLCADConfig()
    assert cfg.host == "127.0.0.1"
    assert cfg.port == 5555
    assert cfg.timeout == 5.0
    assert cfg.buffer_size == 4096


def test_default_llm_config():
    cfg = LLMConfig()
    assert cfg.model == "gpt-4o"
    assert cfg.temperature == 0.0


def test_default_server_config():
    cfg = ServerConfig()
    assert cfg.name == "BRL-CAD-MCP"
    assert cfg.transport == "stdio"


def test_brlcad_config_reads_env(monkeypatch):
    monkeypatch.setenv("BRLCAD_PORT", "6000")
    monkeypatch.setenv("BRLCAD_HOST", "0.0.0.0")
    cfg = BRLCADConfig()
    assert cfg.port == 6000
    assert cfg.host == "0.0.0.0"


def test_settings_composition():
    s = Settings()
    assert isinstance(s.brlcad, BRLCADConfig)
    assert isinstance(s.llm, LLMConfig)
    assert isinstance(s.server, ServerConfig)
