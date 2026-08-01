"""Thin, role-specific prompts -- the replacement for v1's ~4k-token monolith.

Each agent gets only what its role needs.  Operational *recipes* live in skill
definitions (injected by the SkillsMiddleware), safety rules live in tool
preconditions, and loop control lives in the graph -- so none of that is
duplicated in a prompt.  That also removes v1's contradiction (a "stop after one
tool" rule sitting next to multi-tool recipes): stopping is now the graph's job.

The texts themselves are files under ``definitions/``, one per role, so they can
be edited without touching Python -- see :mod:`client_v2.prompts.library` for the
loading rules and the ``CLIENT_V2_PROMPTS_DIR`` override.

They are written to OpenAI's reasoning-model guidance: state the goal, the
constraints and an explicit output contract; do NOT prescribe intermediate steps
or ask for chain-of-thought; do not over-emphasise thoroughness (it causes tool
overuse).  Reasoning models also default to no markdown, which suits this
terminal, so the prompts deliberately never re-enable it.
"""

from client_v2.prompts.library import (
    BUILTIN_DIR,
    OVERRIDE_DIR_ENV,
    PROMPTS,
    REQUIRED,
    PromptLibrary,
    resolve,
)

__all__ = [
    "BUILTIN_DIR",
    "OVERRIDE_DIR_ENV",
    "PROMPTS",
    "REQUIRED",
    "PromptLibrary",
    "resolve",
]
