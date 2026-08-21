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


# --- blank values in .env -------------------------------------------------
#
# .env.example ships several keys with no value, as placeholders showing what can
# be set. The setup instructions say to copy it to .env, which SETS those
# variables to the empty string, so os.getenv(name, default) returns "" and the
# default never applies. Found by installing a release archive and importing it:
# float("") raised on BRLCAD_RENDER_TIMEOUT before the config module finished
# loading, so nothing could start at all.

def test_a_blank_numeric_setting_falls_back_to_its_default(monkeypatch):
    from brlcad_mcp.config import _env_num
    for blank in ("", "   "):
        monkeypatch.setenv("SOME_TIMEOUT", blank)
        assert _env_num("SOME_TIMEOUT", "1800", float) == 1800.0
    monkeypatch.delenv("SOME_TIMEOUT")
    assert _env_num("SOME_TIMEOUT", "1800", float) == 1800.0


def test_a_malformed_numeric_setting_does_not_stop_startup(monkeypatch):
    """A bad tuning value should not prevent the program running."""
    from brlcad_mcp.config import _env_num
    monkeypatch.setenv("SOME_TIMEOUT", "not-a-number")
    assert _env_num("SOME_TIMEOUT", "1800", float) == 1800.0


def test_a_blank_string_setting_falls_back_to_its_default(monkeypatch):
    """A blank directory once meant writing renders to the current directory."""
    from brlcad_mcp.config import _env_str
    monkeypatch.setenv("SOME_DIR", "")
    assert _env_str("SOME_DIR", "/tmp/fallback") == "/tmp/fallback"
    monkeypatch.setenv("SOME_DIR", "/explicit")
    assert _env_str("SOME_DIR", "/tmp/fallback") == "/explicit"


def test_copying_env_example_verbatim_still_imports(tmp_path, monkeypatch):
    """The end-to-end version of the bug: every key in .env.example set blank."""
    import os
    import re
    example = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           ".env.example")
    names = [m.group(1) for m in
             re.finditer(r'^([A-Z_]+)=', open(example).read(), re.M)]
    assert names, "no keys parsed from .env.example"
    for name in names:
        monkeypatch.setenv(name, "")
    import importlib

    import brlcad_mcp.config as C
    importlib.reload(C)                     # must not raise
    assert C.settings.brlcad.port > 0
    assert C.settings.render.timeout > 0
    assert C.settings.render.output_dir      # not the current directory
    importlib.reload(C)
