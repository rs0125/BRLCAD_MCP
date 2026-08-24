"""Tests for the provider layer: which kwargs each backend gets, and why.

These are pure-function tests on purpose.  Building the kwargs is where every
provider difference lives, and it needs no integration package installed and no
network -- so the whole matrix (OpenAI, a gateway, Anthropic, Gemini, Ollama, an
unknown provider) is checked here, offline.
"""

import pytest

from client_v2.providers import (
    build_kwargs,
    normalize_provider,
    resolve_dialect,
    serializes_tool_calls,
)

# ---------------------------------------------------------------- provider ids

def test_provider_names_normalize():
    assert normalize_provider("OpenAI") == "openai"
    assert normalize_provider("google-genai") == "google_genai"
    assert normalize_provider("  anthropic  ") == "anthropic"


def test_common_aliases_map_to_langchain_names():
    # What a user would plausibly type vs what init_chat_model expects.
    assert normalize_provider("google") == "google_genai"
    assert normalize_provider("gemini") == "google_genai"
    assert normalize_provider("azure") == "azure_openai"
    assert normalize_provider("claude") == "anthropic"


def test_unknown_provider_passes_through_untouched():
    # We must not gatekeep: any LangChain integration should be reachable.
    assert normalize_provider("some_new_vendor") == "some_new_vendor"


def test_empty_provider_defaults_to_openai():
    assert normalize_provider("") == "openai"
    assert normalize_provider(None) == "openai"


# ------------------------------------------------------------------- dialect

def test_plain_openai_uses_the_responses_api():
    # No endpoint override means real OpenAI, which supports /v1/responses.
    assert resolve_dialect("openai", base_url="", requested="") == "responses"


def test_openai_dialect_with_a_gateway_falls_back_to_chat():
    # THE restricted-network case: a self-hosted or proxied endpoint almost
    # never implements /v1/responses, so defaulting to it 404s.  A base_url
    # override is the signal that we are not talking to OpenAI itself.
    assert resolve_dialect("openai", base_url="http://gw/v1", requested="") == "chat"


def test_dialect_can_be_forced_either_way():
    assert resolve_dialect("openai", "http://gw/v1", "responses") == "responses"
    assert resolve_dialect("openai", "", "chat") == "chat"


def test_non_openai_providers_have_no_dialect_choice():
    # 'responses' is an OpenAI concept; it is meaningless elsewhere.
    for prov in ("anthropic", "google_genai", "ollama", "some_new_vendor"):
        assert resolve_dialect(prov, "", "") == "chat"


# -------------------------------------------------------------- openai kwargs

def _kw(**over):
    base = dict(provider="openai", model="gpt-5.6-sol", api_key="k",
                base_url="", temperature=0.0, effort="", extra=None,
                dialect=None)
    base.update(over)
    return build_kwargs(**base)


def test_default_openai_path_is_unchanged_from_before_byok():
    # Regression guard: the measured 14/18 run used exactly these settings, so
    # adding provider support must not quietly alter the default backend.
    kw = _kw()
    assert kw["model"] == "gpt-5.6-sol"
    assert kw["use_responses_api"] is True
    assert kw["output_version"] == "responses/v1"
    assert kw["reasoning"] == {"effort": "medium"}
    assert "temperature" not in kw          # reasoning models reject it
    assert kw["api_key"] == "k"


def test_legacy_openai_chat_model_gets_temperature_not_reasoning():
    kw = _kw(model="gpt-4o", temperature=0.3)
    assert kw["temperature"] == 0.3
    assert "reasoning" not in kw


def test_openai_over_a_gateway_drops_responses_only_kwargs():
    # A gateway gets plain chat-completions: no responses api, and no
    # reasoning= (which is a Responses-API shape the gateway will reject).
    kw = _kw(model="qwen2.5-coder", base_url="http://gw/v1", effort="high")
    assert kw["base_url"] == "http://gw/v1"
    assert "use_responses_api" not in kw
    assert "output_version" not in kw
    assert "reasoning" not in kw
    # A model we do not recognise as an OpenAI reasoning model gets no effort
    # field at all -- a gateway serving llama would reject one.
    assert "reasoning_effort" not in kw


def test_openai_reasoning_model_on_chat_must_pin_effort_to_none():
    # Verified against the live API: omitting reasoning_effort is NOT enough.
    # The server applies the model's default effort and then refuses,
    #   "Function tools with reasoning_effort are not supported for gpt-5.6-sol
    #    in /v1/chat/completions ... or set reasoning_effort to 'none'."
    # and this agent always registers tools.
    kw = _kw(model="gpt-5.6-sol", base_url="http://proxy/v1", effort="high")
    assert kw["reasoning_effort"] == "none"
    assert "reasoning" not in kw
    assert "temperature" not in kw       # reasoning models reject it


# ----------------------------------------------------- other-provider kwargs

def test_anthropic_gets_the_standard_reasoning_effort_field():
    # langchain-anthropic exposes reasoning_effort as a standard param and
    # translates it to Anthropic's thinking budget itself.
    kw = build_kwargs(provider="anthropic", model="claude-sonnet-5", api_key="k",
                      base_url="", temperature=0.0, effort="high", extra=None,
                      dialect=None)
    assert kw["reasoning_effort"] == "high"
    assert "reasoning" not in kw            # that is the OpenAI shape
    assert "use_responses_api" not in kw


def test_gemini_drops_base_url_because_it_has_no_such_field():
    # ChatGoogleGenerativeAI has no endpoint kwarg; passing one raises.
    kw = build_kwargs(provider="google_genai", model="gemini-3-pro", api_key="k",
                      base_url="http://nope/v1", temperature=0.0, effort="",
                      extra=None, dialect=None)
    assert "base_url" not in kw


def test_ollama_needs_no_key_and_takes_an_endpoint():
    kw = build_kwargs(provider="ollama", model="qwen2.5-coder", api_key="",
                      base_url="http://localhost:11434", temperature=0.0,
                      effort="high", extra=None, dialect=None)
    assert kw["base_url"] == "http://localhost:11434"
    assert "api_key" not in kw              # empty key must not be forwarded
    # Ollama has no reasoning_effort field; sending one would be rejected.
    assert "reasoning_effort" not in kw


def test_unknown_provider_gets_only_portable_kwargs():
    # For a provider we know nothing about, send only what LangChain documents
    # as standard.  Anything exotic is the user's job via LLM_EXTRA.
    kw = build_kwargs(provider="some_new_vendor", model="m", api_key="k",
                      base_url="http://x/v1", temperature=0.2, effort="high",
                      extra=None, dialect=None)
    assert set(kw) <= {"model", "api_key", "base_url", "temperature"}


# ------------------------------------------------------------------- extras

def test_extra_kwargs_are_merged_and_win():
    kw = _kw(extra={"max_retries": 9, "temperature": 1.0})
    assert kw["max_retries"] == 9
    assert kw["temperature"] == 1.0          # explicit override beats our default


def test_extra_can_reach_a_provider_quirk_we_do_not_model():
    kw = build_kwargs(provider="ollama", model="m", api_key="", base_url="",
                      temperature=0.0, effort="", extra={"num_predict": 512},
                      dialect=None)
    assert kw["num_predict"] == 512


# --------------------------------------------------------- tool serialization

def test_parallel_tool_calls_is_only_disabled_where_it_exists():
    # parallel_tool_calls is an OpenAI kwarg.  Binding it on Gemini or Ollama
    # is an error, so it must be gated by provider rather than applied blindly.
    assert serializes_tool_calls("openai") is True
    assert serializes_tool_calls("azure_openai") is True
    assert serializes_tool_calls("google_genai") is False
    assert serializes_tool_calls("ollama") is False
    assert serializes_tool_calls("some_new_vendor") is False


# ------------------------------------------------------------------ hygiene

def test_no_empty_values_are_forwarded():
    # An unset env var must look unset to the provider, not like "".
    kw = build_kwargs(provider="openai", model="gpt-4o", api_key="", base_url="",
                      temperature=0.0, effort="", extra=None, dialect=None)
    assert "api_key" not in kw
    assert "base_url" not in kw


def test_model_is_always_present():
    for prov in ("openai", "anthropic", "google_genai", "ollama", "whatever"):
        kw = build_kwargs(provider=prov, model="m", api_key="k", base_url="",
                          temperature=0.0, effort="", extra=None, dialect=None)
        assert kw["model"] == "m"


@pytest.mark.parametrize("effort", ["", "  ", "nonsense"])
def test_bad_effort_falls_back_to_medium_not_an_error(effort):
    kw = _kw(effort=effort)
    assert kw["reasoning"] == {"effort": "medium"}


# ------------------------------------------------- construction (integration)
#
# These build real client objects (no network: construction does not call the
# API).  Providers whose package is absent are skipped rather than failed, since
# they are optional extras by design.

def _build(monkeypatch, **env):
    for k in ("LLM_PROVIDER", "LLM_MODEL", "LLM_API_KEY", "LLM_BASE_URL",
              "LLM_API", "LLM_EFFORT", "LLM_EXTRA", "OPENAI_API_KEY",
              "OPENAI_MODEL", "OPENAI_BASE_URL", "OPENAI_API_BASE"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    import importlib

    import brlcad_mcp.config as cfg
    import client_v2.model as m
    importlib.reload(cfg)
    importlib.reload(m)
    return m


def test_default_backend_builds_a_chatopenai(monkeypatch):
    m = _build(monkeypatch, LLM_API_KEY="k")
    model = m.build_model()
    # .bind() wraps it, so reach the underlying client.
    inner = getattr(model, "bound", model)
    assert type(inner).__name__ == "ChatOpenAI"
    assert m.describe_backend() == "openai:gpt-5.6-sol [responses]"


def test_openai_dialect_gateway_builds_and_reports_chat(monkeypatch):
    m = _build(monkeypatch, LLM_API_KEY="k", LLM_MODEL="qwen2.5-coder",
               LLM_BASE_URL="http://127.0.0.1:9/v1")
    model = m.build_model()
    inner = getattr(model, "bound", model)
    assert type(inner).__name__ == "ChatOpenAI"
    assert str(inner.root_client.base_url).rstrip("/") == "http://127.0.0.1:9/v1"
    assert "[chat]" in m.describe_backend()


def test_a_missing_provider_package_exits_with_the_pip_hint(monkeypatch):
    pytest.importorskip  # noqa: B018  (keep the import used)
    try:
        import langchain_anthropic  # noqa: F401
        pytest.skip("langchain-anthropic is installed, so this cannot fail")
    except ImportError:
        pass
    m = _build(monkeypatch, LLM_PROVIDER="anthropic", LLM_API_KEY="k",
               LLM_MODEL="claude-sonnet-5")
    with pytest.raises(SystemExit):
        m.build_model()


def test_an_unknown_provider_exits_rather_than_guessing(monkeypatch):
    m = _build(monkeypatch, LLM_PROVIDER="not-a-real-provider",
               LLM_API_KEY="k", LLM_MODEL="m")
    with pytest.raises(SystemExit):
        m.build_model()


def test_ollama_needs_no_api_key_to_get_past_the_gate(monkeypatch):
    pytest.importorskip("langchain_ollama")
    m = _build(monkeypatch, LLM_PROVIDER="ollama", LLM_MODEL="qwen2.5-coder",
               LLM_BASE_URL="http://127.0.0.1:11434")
    model = m.build_model()
    assert type(model).__name__ == "ChatOllama"   # not wrapped: no .bind()


# ------------------------------------------- parallel_tool_calls placement
#
# Regression: binding parallel_tool_calls on the *model* poisons every
# tool-less call.  The graph's planner, formatter and the worker's own
# summarisation middleware all invoke the model with no tools, and
# chat-completions rejects the kwarg there:
#   "Invalid value for 'parallel_tool_calls': 'parallel_tool_calls' is only
#    allowed when 'tools' are specified."
# The Responses API tolerates it, which is why this only showed up once a
# gateway/chat backend became reachable.

def test_build_model_does_not_carry_parallel_tool_calls(monkeypatch):
    m = _build(monkeypatch, LLM_API_KEY="k")
    model = m.build_model()
    assert getattr(model, "kwargs", {}).get("parallel_tool_calls") is None


def test_the_tool_loop_model_does_carry_it(monkeypatch):
    m = _build(monkeypatch, LLM_API_KEY="k")
    bound = m.for_tool_loop(m.build_model())
    assert bound.kwargs["parallel_tool_calls"] is False


def test_tool_loop_binding_is_skipped_where_unsupported(monkeypatch):
    pytest.importorskip("langchain_ollama")
    m = _build(monkeypatch, LLM_PROVIDER="ollama", LLM_MODEL="q",
               LLM_BASE_URL="http://127.0.0.1:11434")
    model = m.build_model()
    assert m.for_tool_loop(model) is model      # untouched, kwarg does not exist
