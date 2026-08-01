"""client-v2 debug flag: _env_bool parsing and settings.debug from env."""

from brlcad_mcp.config import Settings, _env_bool


def test_env_bool_truthy_and_falsy(monkeypatch):
    monkeypatch.delenv("X_FLAG", raising=False)
    assert _env_bool("X_FLAG") is False           # unset -> default
    for v in ("1", "true", "TRUE", "Yes", "on"):
        monkeypatch.setenv("X_FLAG", v)
        assert _env_bool("X_FLAG") is True, v
    for v in ("0", "false", "no", "off", ""):
        monkeypatch.setenv("X_FLAG", v)
        assert _env_bool("X_FLAG") is False, v


def test_settings_debug_reads_client_v2_debug(monkeypatch):
    monkeypatch.setenv("CLIENT_V2_DEBUG", "true")
    assert Settings().debug is True
    monkeypatch.setenv("CLIENT_V2_DEBUG", "false")
    assert Settings().debug is False
    monkeypatch.delenv("CLIENT_V2_DEBUG", raising=False)
    assert Settings().debug is False
