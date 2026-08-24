"""Which kwargs each LLM backend gets, and why.

The graph only ever holds a LangChain ``BaseChatModel``; it does not know or
care what is behind it.  So supporting a backend reduces to one question: what
kwargs does ``init_chat_model`` need for it?  That is all this module answers.

Two tiers are supported:

**An OpenAI-dialect endpoint** (``provider=openai`` plus ``base_url``).  The
provider name here means the *wire format*, not the vendor: llama.cpp's server,
vLLM, LM Studio, Ollama's ``/v1`` port, LocalAI, a LiteLLM proxy or a corporate
gateway all serve ``/v1/chat/completions``.  Config only, no extra package.

**A dedicated integration** (``provider=anthropic``, ``google_genai``,
``ollama``, ``bedrock``, ...).  Needs its ``langchain-*`` package installed,
and in exchange gets provider-native behaviour that a compatibility shim drops.

Most differences between backends are already absorbed by LangChain's standard
parameters -- ``model``, ``api_key``, ``temperature``, ``max_tokens``,
``max_retries`` and notably ``reasoning_effort``, which OpenAI, Anthropic,
Google, Bedrock and xAI all accept and translate themselves.  What is left is
the short list of genuine outliers below.  Anything we do not model is reachable
through ``LLM_EXTRA`` without a code change here.
"""

from __future__ import annotations

# What users plausibly type -> what init_chat_model expects.  Anything not
# listed passes through unchanged, so a LangChain integration we have never
# heard of is still reachable by name.
_ALIASES = {
    "google": "google_genai",
    "gemini": "google_genai",
    "googleai": "google_genai",
    "azure": "azure_openai",
    "claude": "anthropic",
    "vertex": "google_vertexai",
}

# Providers whose client class is OpenAI's, so OpenAI-only kwargs are valid.
_OPENAI_FAMILY = ("openai", "azure_openai")

# ChatGoogleGenerativeAI has no endpoint field at all -- passing base_url
# raises rather than being ignored, so it has to be dropped.
_NO_ENDPOINT = ("google_genai", "google_vertexai")

# Providers exposing LangChain's standard ``reasoning_effort`` field.  Ollama is
# deliberately absent: it uses a differently-shaped ``reasoning`` kwarg, and
# sending reasoning_effort would be rejected.
_STANDARD_EFFORT = ("anthropic", "google_genai", "bedrock_converse", "xai",
                    "fireworks", "deepseek")

# Effort levels accepted, weakest to strongest.
_VALID_EFFORTS = ("minimal", "low", "medium", "high")
_DEFAULT_EFFORT = "medium"

# Substrings marking an OpenAI reasoning model (vs a legacy chat model like
# gpt-4o).  Only consulted for the OpenAI family; meaningless for other vendors,
# whose integrations decide this themselves.
_REASONING_HINTS = ("gpt-5", "sol", "terra", "luna", "o1", "o3", "o4")


def normalize_provider(provider: str | None) -> str:
    """Canonical provider id, defaulting to openai.  Unknown names pass through."""
    p = (provider or "").strip().lower().replace("-", "_")
    if not p:
        return "openai"
    return _ALIASES.get(p, p)


def is_openai_reasoning_model(model_id: str) -> bool:
    """True for GPT-5.x / Sol / Terra / Luna / o-series."""
    m = (model_id or "").lower()
    return any(t in m for t in _REASONING_HINTS)


def normalize_effort(effort: str | None) -> str:
    """Clamp a configured effort to a valid level, defaulting to medium."""
    e = (effort or "").strip().lower()
    return e if e in _VALID_EFFORTS else _DEFAULT_EFFORT


def resolve_dialect(provider: str, base_url: str, requested: str) -> str:
    """Pick the OpenAI wire dialect: ``responses`` or ``chat``.

    An explicit request always wins.  Otherwise the rule is that a *base_url
    override means we are not talking to OpenAI itself*, and near-nothing else
    implements ``/v1/responses`` -- so a gateway gets plain chat-completions
    rather than a confident 404 on every call.  Non-OpenAI providers have no
    such concept and are always ``chat``.
    """
    want = (requested or "").strip().lower()
    if want in ("responses", "chat"):
        return want
    if normalize_provider(provider) not in _OPENAI_FAMILY:
        return "chat"
    return "chat" if (base_url or "").strip() else "responses"


def serializes_tool_calls(provider: str) -> bool:
    """Whether ``parallel_tool_calls=False`` may be bound for this provider.

    The graph wants one tool call at a time, but the kwarg is OpenAI's.  Binding
    it on Gemini or Ollama is an error, so it is gated rather than blanket.
    """
    return normalize_provider(provider) in _OPENAI_FAMILY


def build_kwargs(provider: str, model: str, api_key: str, base_url: str,
                 temperature: float, effort: str, extra: dict | None,
                 dialect: str | None = None) -> dict:
    """kwargs for ``init_chat_model`` for one backend (pure, hence testable).

    Empty values are omitted rather than forwarded: an unset environment
    variable must look unset to the provider, not like ``""``.  ``extra`` is
    merged last and wins, which is what makes a provider quirk we do not model
    a config change instead of a patch.
    """
    prov = normalize_provider(provider)
    kw: dict = {"model": model}

    if (api_key or "").strip():
        kw["api_key"] = api_key
    if (base_url or "").strip() and prov not in _NO_ENDPOINT:
        kw["base_url"] = base_url

    if prov in _OPENAI_FAMILY:
        d = resolve_dialect(prov, base_url, dialect or "")
        reasoning = is_openai_reasoning_model(model)
        if d == "responses":
            kw["use_responses_api"] = True
            # Puts reasoning + tool items in AIMessage.content, so the model's
            # reasoning carries across tool calls as the graph replays history.
            kw["output_version"] = "responses/v1"
            if reasoning:
                kw["reasoning"] = {"effort": normalize_effort(effort)}
            else:
                kw["temperature"] = temperature
        elif reasoning:
            # Plain chat-completions with an OpenAI reasoning model.  Effort has
            # to be pinned to 'none' rather than left off: the server otherwise
            # applies the model's default effort and rejects it because tools
            # are registered --
            #   "Function tools with reasoning_effort are not supported for
            #    gpt-5.6-sol in /v1/chat/completions ... or set
            #    reasoning_effort to 'none'."
            # This is why the Responses API is the default; it is the only way
            # such a model can reason *and* call tools.  No temperature either,
            # which reasoning models reject.
            kw["reasoning_effort"] = "none"
        else:
            # Anything else on the OpenAI dialect -- a legacy chat model, or a
            # gateway serving something we know nothing about.  Send no effort
            # field at all, since a non-OpenAI backend would reject one.
            kw["temperature"] = temperature
    elif prov in _STANDARD_EFFORT:
        if (effort or "").strip():
            kw["reasoning_effort"] = normalize_effort(effort)
        kw["temperature"] = temperature
    else:
        # A provider we know nothing about (including a local runtime): send
        # only what LangChain documents as standard for every integration.
        kw["temperature"] = temperature

    if extra:
        kw.update(extra)
    return kw
