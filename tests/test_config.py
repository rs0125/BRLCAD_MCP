"""Tests for brlcad_mcp.config defaults and composition."""

from brlcad_mcp.config import BRLCADConfig, LLMConfig, ServerConfig, Settings


def test_default_brlcad_config(monkeypatch):
    # Hermetic: an exported BRLCAD_* in the developer's shell (e.g. pointing at a
    # test listener on another port) must not make this fail.
    for var in ("BRLCAD_HOST", "BRLCAD_PORT", "BRLCAD_TIMEOUT",
                "BRLCAD_BUFFER_SIZE"):
        monkeypatch.delenv(var, raising=False)
    cfg = BRLCADConfig()
    assert cfg.host == "127.0.0.1"
    assert cfg.port == 5555
    assert cfg.timeout == 5.0
    assert cfg.buffer_size == 4096


def test_default_llm_config(monkeypatch):
    # Code default when no env override (the .env in the repo pins Sol, so drop
    # it here to test the fallback baked into the dataclass).
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_TEMPERATURE", raising=False)
    cfg = LLMConfig()
    assert cfg.model == "gpt-5.6-sol"
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
