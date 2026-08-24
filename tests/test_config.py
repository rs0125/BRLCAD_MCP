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


# --------------------------------------------------------------- BYOK / providers

_LLM_VARS = ("LLM_PROVIDER", "LLM_MODEL", "LLM_API_KEY", "LLM_BASE_URL",
             "LLM_API", "LLM_EFFORT", "LLM_TEMPERATURE", "LLM_EXTRA",
             "OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_BASE_URL",
             "OPENAI_API_BASE", "OPENAI_REASONING_EFFORT", "OPENAI_TEMPERATURE")


def _clean(monkeypatch):
    for v in _LLM_VARS:
        monkeypatch.delenv(v, raising=False)


def test_provider_defaults_to_openai(monkeypatch):
    _clean(monkeypatch)
    assert LLMConfig().provider == "openai"


def test_llm_vars_take_effect(monkeypatch):
    _clean(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_MODEL", "claude-sonnet-5")
    monkeypatch.setenv("LLM_API_KEY", "sk-ant")
    monkeypatch.setenv("LLM_BASE_URL", "http://gw/v1")
    cfg = LLMConfig()
    assert (cfg.provider, cfg.model) == ("anthropic", "claude-sonnet-5")
    assert (cfg.api_key, cfg.base_url) == ("sk-ant", "http://gw/v1")


def test_openai_vars_still_work_as_fallbacks(monkeypatch):
    # Existing .env files and the shipped release must keep working untouched.
    _clean(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-old")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
    monkeypatch.setenv("OPENAI_REASONING_EFFORT", "high")
    cfg = LLMConfig()
    assert cfg.api_key == "sk-old"
    assert cfg.model == "gpt-4o"
    assert cfg.reasoning_effort == "high"
    assert cfg.provider == "openai"


def test_llm_vars_win_over_openai_vars(monkeypatch):
    _clean(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-old")
    monkeypatch.setenv("LLM_API_KEY", "sk-new")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
    monkeypatch.setenv("LLM_MODEL", "gemini-3-pro")
    cfg = LLMConfig()
    assert cfg.api_key == "sk-new"
    assert cfg.model == "gemini-3-pro"


def test_base_url_accepts_either_openai_spelling(monkeypatch):
    # The OpenAI SDK uses OPENAI_BASE_URL; langchain reads OPENAI_API_BASE.
    _clean(monkeypatch)
    monkeypatch.setenv("OPENAI_API_BASE", "http://a/v1")
    assert LLMConfig().base_url == "http://a/v1"
    monkeypatch.setenv("OPENAI_BASE_URL", "http://b/v1")
    assert LLMConfig().base_url == "http://b/v1"


def test_extra_parses_json_and_survives_garbage(monkeypatch):
    _clean(monkeypatch)
    monkeypatch.setenv("LLM_EXTRA", '{"max_retries": 9}')
    assert LLMConfig().extra == {"max_retries": 9}
    # A typo in a config file must not crash the whole client on import.
    monkeypatch.setenv("LLM_EXTRA", "{not json")
    assert LLMConfig().extra == {}
    # A JSON scalar is not a kwargs mapping either.
    monkeypatch.setenv("LLM_EXTRA", '"just a string"')
    assert LLMConfig().extra == {}


def test_blank_llm_vars_are_treated_as_unset(monkeypatch):
    # Copying .env.example verbatim leaves these empty, which must not become
    # a provider named "" or a base_url of "".
    _clean(monkeypatch)
    for v in ("LLM_PROVIDER", "LLM_BASE_URL", "LLM_API", "LLM_MODEL"):
        monkeypatch.setenv(v, "")
    cfg = LLMConfig()
    assert cfg.provider == "openai"
    assert cfg.base_url == ""
    assert cfg.api_dialect == ""
    assert cfg.model == "gpt-5.6-sol"
