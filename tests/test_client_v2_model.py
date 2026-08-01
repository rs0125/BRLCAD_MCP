"""client-v2 model layer: family detection and Responses-API kwargs.

No network / no API key -- only the pure kwargs-selection logic.
"""

from client_v2.model import is_reasoning_model, model_config, normalize_effort


def test_reasoning_models_detected():
    for m in ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.5",
              "o3", "o4-mini"):
        assert is_reasoning_model(m), m


def test_legacy_chat_models_not_flagged():
    for m in ("gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4-turbo"):
        assert not is_reasoning_model(m), m


def test_effort_normalizes_and_defaults_to_medium():
    assert normalize_effort("high") == "high"
    assert normalize_effort("LOW") == "low"
    assert normalize_effort("") == "medium"       # unset -> default
    assert normalize_effort("bogus") == "medium"  # invalid -> default


def test_reasoning_model_gets_real_effort_over_responses_api():
    # The whole point of v2: a real effort (not 'none'), no temperature.
    cfg = model_config("gpt-5.6-sol", "high", 0.0)
    assert cfg["use_responses_api"] is True
    # responses/v1 keeps reasoning items in message content so they survive
    # across tool calls (see model_config docstring).
    assert cfg["output_version"] == "responses/v1"
    assert cfg["reasoning"] == {"effort": "high"}
    assert "temperature" not in cfg
    # Unset effort still reasons, at the medium default.
    assert model_config("gpt-5.6-sol", "", 0.0)["reasoning"] == {"effort": "medium"}


def test_legacy_model_uses_temperature_no_reasoning():
    cfg = model_config("gpt-4o", "", 0.3)
    assert cfg == {"model": "gpt-4o", "use_responses_api": True,
                   "output_version": "responses/v1", "temperature": 0.3}
    assert "reasoning" not in cfg
