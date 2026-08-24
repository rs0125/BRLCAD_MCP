"""Model layer for client-v2 — any LangChain chat backend.

The graph only ever holds a ``BaseChatModel``, so which vendor is behind it is a
configuration question.  ``init_chat_model`` does the construction (27 providers
built in, and any other LangChain integration by name), and
:mod:`client_v2.providers` decides the kwargs.  See ``docs/PROVIDERS.md``.

The default is unchanged: GPT-5.6 Sol over OpenAI's **Responses API**.  That
matters because reasoning models reject ``reasoning_effort`` alongside function
tools on ``/v1/chat/completions`` unless it is ``'none'`` -- and this agent
always registers tools -- so on chat-completions Sol never actually reasons.
The Responses API lifts that restriction.  An endpoint override switches to
chat-completions automatically, since almost nothing but OpenAI implements
``/v1/responses``.
"""

from __future__ import annotations

import sys

from langchain.chat_models import init_chat_model

from brlcad_mcp.config import settings
from client_v2.providers import (
    build_kwargs,
    is_openai_reasoning_model,
    normalize_effort,
    normalize_provider,
    resolve_dialect,
    serializes_tool_calls,
)

# Re-exported for callers and tests that predate the provider split.
__all__ = ["build_model", "for_tool_loop", "model_config", "is_reasoning_model",
           "normalize_effort", "describe_backend"]

is_reasoning_model = is_openai_reasoning_model


def model_config(model_id: str, effort: str, temperature: float) -> dict:
    """kwargs for the default OpenAI Responses path (kept for callers/tests)."""
    return build_kwargs(provider="openai", model=model_id, api_key="",
                        base_url="", temperature=temperature, effort=effort,
                        extra=None, dialect="responses")


def describe_backend() -> str:
    """One line naming the backend, for the banner and for run logs.

    Worth printing: once the endpoint is configurable, "which model am I
    actually talking to" stops being obvious, and a silent fallback to the
    default would be the confusing failure.
    """
    llm = settings.llm
    prov = normalize_provider(llm.provider)
    where = f" @ {llm.base_url}" if llm.base_url else ""
    dialect = ""
    if serializes_tool_calls(prov):
        dialect = f" [{resolve_dialect(prov, llm.base_url, llm.api_dialect)}]"
    return f"{prov}:{llm.model}{where}{dialect}"


def for_tool_loop(model):
    """The model as the worker's tool loop should see it.

    Tool calls are serialized (``parallel_tool_calls=False``) so the graph sees
    one tool at a time -- important once the worker executes ordered skill
    steps.  It is bound HERE rather than on the model itself because most nodes
    call the model with no tools at all (intake, planner, formatter, and the
    worker's own summarisation middleware), and chat-completions rejects the
    kwarg when no tools are present:

        Invalid value for 'parallel_tool_calls': 'parallel_tool_calls' is only
        allowed when 'tools' are specified.

    The Responses API tolerates it, so binding it globally went unnoticed until
    a chat-dialect backend became reachable.  It is also an OpenAI kwarg, so it
    is skipped entirely for providers that have no such option.
    """
    if not serializes_tool_calls(settings.llm.provider):
        return model
    if not hasattr(model, "bind"):
        return model
    return model.bind(parallel_tool_calls=False)


def build_model():
    """Instantiate the configured LLM backend.

    Returned unbound: see :func:`for_tool_loop` for why tool-call
    serialization is applied at the tool loop instead of here.
    """
    llm = settings.llm
    prov = normalize_provider(llm.provider)

    # Ollama and other local runtimes legitimately have no key; everything else
    # needs one, and failing here beats a confusing 401 mid-run.
    if not llm.api_key and prov not in ("ollama",) and not llm.base_url:
        print("ERROR: no API key set. Set LLM_API_KEY (or OPENAI_API_KEY).")
        sys.exit(1)

    kwargs = build_kwargs(
        provider=prov, model=llm.model, api_key=llm.api_key,
        base_url=llm.base_url, temperature=llm.temperature,
        effort=llm.reasoning_effort, extra=llm.extra,
        dialect=llm.api_dialect,
    )
    try:
        model = init_chat_model(model_provider=prov, **kwargs)
    except ImportError as exc:
        # init_chat_model already names the pip package to install.
        print(f"ERROR: {exc}")
        sys.exit(1)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    return model
