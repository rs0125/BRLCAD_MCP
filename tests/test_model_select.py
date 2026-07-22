"""Model-family detection and the ChatOpenAI kwargs chosen per family."""

from brlcad_mcp.client.agent import _is_reasoning_model, _model_kwargs


def test_reasoning_models_detected():
    for m in ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.5",
              "o3", "o4-mini"):
        assert _is_reasoning_model(m), m


def test_legacy_chat_models_not_flagged():
    for m in ("gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4-turbo"):
        assert not _is_reasoning_model(m), m


def test_reasoning_model_kwargs_force_none_and_drop_temperature():
    # With function tools on chat/completions, reasoning models require
    # reasoning_effort='none' and must NOT get a temperature arg.
    kw = _model_kwargs("gpt-5.6-sol", "", 0.0)
    assert kw == {"model": "gpt-5.6-sol", "reasoning_effort": "none"}
    # Even if a higher effort is configured, it is forced to 'none' here.
    kw = _model_kwargs("gpt-5.6-sol", "max", 0.0)
    assert kw["reasoning_effort"] == "none"
    assert "temperature" not in kw


def test_legacy_model_kwargs_use_temperature():
    kw = _model_kwargs("gpt-4o", "", 0.3)
    assert kw == {"model": "gpt-4o", "temperature": 0.3}
    assert "reasoning_effort" not in kw
