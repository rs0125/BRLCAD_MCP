"""Model layer for client-v2 — OpenAI **Responses API**.

Why the Responses API: reasoning models (GPT-5.6 Sol / Terra / Luna, o-series)
reject ``reasoning_effort`` alongside function tools on ``/v1/chat/completions``
unless it is ``'none'`` — and this agent always registers tools.  The shipping
client (``brlcad_mcp.client``) therefore forced ``'none'``, so Sol never
actually reasoned.  The Responses API lifts that restriction: a reasoning model
can reason *with* tools, controlled by a real ``reasoning={"effort": ...}``.

Legacy chat models (gpt-4o) still take a ``temperature`` and no reasoning.
"""

from __future__ import annotations

import sys

from langchain_openai import ChatOpenAI

from brlcad_mcp.config import settings

# Substrings that mark a reasoning model (vs a legacy chat model like gpt-4o).
_REASONING_HINTS = ("gpt-5", "sol", "terra", "luna", "o1", "o3", "o4")
# Effort levels the Responses API accepts, weakest to strongest.
_VALID_EFFORTS = ("minimal", "low", "medium", "high")
_DEFAULT_EFFORT = "medium"


def is_reasoning_model(model_id: str) -> bool:
    """True for GPT-5.x / Sol / Terra / Luna / o-series reasoning models."""
    m = model_id.lower()
    return any(t in m for t in _REASONING_HINTS)


def normalize_effort(effort: str) -> str:
    """Clamp a configured effort to a valid level, defaulting to medium."""
    e = (effort or "").strip().lower()
    return e if e in _VALID_EFFORTS else _DEFAULT_EFFORT


def model_config(model_id: str, effort: str, temperature: float) -> dict:
    """ChatOpenAI kwargs for the Responses API, chosen by model family (pure).

    Reasoning models get a real ``reasoning`` effort (never 'none') and NO
    temperature; legacy chat models get a temperature and no reasoning.  Both
    run over the Responses API for a single, consistent code path.

    ``output_version="responses/v1"`` (recommended by langchain-openai for new
    apps) puts reasoning + tool items into ``AIMessage.content`` instead of
    ``additional_kwargs``.  Because the graph forwards the full message history
    on each turn, the model's reasoning items are carried across tool calls --
    which OpenAI reports is worth ~3% on tool-heavy benchmarks and lifts prompt-
    cache hits from ~40% to ~80% (docs: Responses API reasoning items).
    """
    cfg: dict = {
        "model": model_id,
        "use_responses_api": True,
        "output_version": "responses/v1",
    }
    if is_reasoning_model(model_id):
        cfg["reasoning"] = {"effort": normalize_effort(effort)}
    else:
        cfg["temperature"] = temperature
    return cfg


def build_model():
    """Instantiate the LLM backend for the configured model, over Responses API.

    Tool calls are serialized (``parallel_tool_calls=False``) so the graph sees
    one tool at a time — important once the worker executes ordered skill steps.
    """
    if not settings.llm.api_key:
        print("ERROR: OPENAI_API_KEY environment variable is not set.")
        sys.exit(1)
    cfg = model_config(settings.llm.model, settings.llm.reasoning_effort,
                       settings.llm.temperature)
    return ChatOpenAI(**cfg).bind(parallel_tool_calls=False)
