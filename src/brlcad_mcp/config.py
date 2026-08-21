"""Centralized configuration loaded from environment variables and .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Resolve project root (two levels up from this file → src/brlcad_mcp/config.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load .env from project root (if it exists)
load_dotenv(PROJECT_ROOT / ".env")


def _env_str(name: str, default: str) -> str:
    """A string setting, treating blank as unset.

    Same trap as :func:`_env_num`: a key present in ``.env`` with no value sets
    the variable to "", so ``os.getenv(name, default)`` skips the default.  For a
    directory that meant writing renders to the current working directory
    instead of the intended one.  Not used where blank carries meaning, such as
    an absent API key.
    """
    return (os.getenv(name) or "").strip() or default


def _env_num(name: str, default: str, cast):
    """Parse a numeric environment variable, treating blank as unset.

    ``.env.example`` ships several keys with no value, as placeholders showing
    what can be set.  Copying it to ``.env`` -- which the setup instructions tell
    you to do -- therefore SETS those variables to the empty string, so
    ``os.getenv(name, default)`` returns "" rather than the default and the cast
    raises on import.  Anything blank or unparseable falls back to the default
    instead, since a malformed tuning value should not stop the program starting.
    """
    raw = (os.getenv(name) or "").strip()
    try:
        return cast(raw or default)
    except ValueError:
        return cast(default)


def _env_bool(name: str, default: bool = False) -> bool:
    """Parse a truthy/falsy environment variable (1/true/yes/on)."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class BRLCADConfig:
    """Settings for the BRL-CAD TCP socket bridge."""

    host: str = field(default_factory=lambda: _env_str("BRLCAD_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _env_num("BRLCAD_PORT", "5555", int))
    timeout: float = field(
        default_factory=lambda: _env_num("BRLCAD_TIMEOUT", "5.0", float)
    )
    buffer_size: int = field(
        default_factory=lambda: _env_num("BRLCAD_BUFFER_SIZE", "4096", int)
    )


@dataclass(frozen=True)
class LLMConfig:
    """Settings for the OpenAI / LLM backend.

    ``model`` defaults to GPT-5.6 Sol.  Confirm the exact model id in your
    OpenAI dashboard and override with OPENAI_MODEL if it differs.  Sol is a
    reasoning-style model: it uses ``reasoning_effort`` and rejects a
    ``temperature`` argument, so the client picks the right one automatically
    (see agent._build_model).
    """

    api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )
    model: str = field(
        default_factory=lambda: _env_str("OPENAI_MODEL", "gpt-5.6-sol")
    )
    temperature: float = field(
        default_factory=lambda: _env_num("OPENAI_TEMPERATURE", "0", float)
    )
    # Reserved for reasoning models.  NOTE: on /v1/chat/completions (what the
    # client uses) reasoning models reject reasoning_effort together with
    # function tools unless it is 'none', so the client forces 'none' regardless
    # of this value.  Extended effort (high/max) would need the Responses API.
    reasoning_effort: str = field(
        default_factory=lambda: os.getenv("OPENAI_REASONING_EFFORT", "")
    )


@dataclass(frozen=True)
class ServerConfig:
    """Settings for the MCP tool server."""

    name: str = "BRL-CAD-MCP"
    transport: str = field(
        default_factory=lambda: _env_str("MCP_TRANSPORT", "stdio")
    )


@dataclass(frozen=True)
class RenderConfig:
    """Settings for the headless renderer.

    Rendering runs entirely over the socket via the ged ``rt`` command (which
    writes PNG directly), so no BRL-CAD binaries or PATH are needed.

    ``timeout`` bounds how long we wait for a single render to finish (ged_rt is
    async, so this is the poll deadline).  Raise it for slow renders -- large
    assemblies, high ambient-occlusion sample counts, or photon mapping.
    """

    output_dir: str = field(
        default_factory=lambda: _env_str(
            "BRLCAD_RENDER_DIR", os.path.expanduser("~/brlcad_renders")
        )
    )
    timeout: float = field(
        default_factory=lambda: _env_num("BRLCAD_RENDER_TIMEOUT", "1800", float)
    )
    # Restore points for destructive raw edits.  Deliberately NOT under
    # output_dir: they used to live in the render folder, so clearing a render
    # cache silently deleted the only way back from a bad `kill`.
    backup_dir: str = field(
        default_factory=lambda: _env_str(
            "BRLCAD_BACKUP_DIR", os.path.expanduser("~/brlcad_backups")
        )
    )
    # Saved build specs -- the source of truth `edit_build` applies ops to and
    # `verify_model_dimensions(name=...)` reads back.  Empty means "derive from
    # output_dir", which is the historical layout and is kept so an existing
    # store is not orphaned; set it to keep specs out of the render folder, which
    # is otherwise a cache someone may reasonably delete.
    spec_dir: str = field(
        default_factory=lambda: os.getenv("BRLCAD_SPEC_DIR", "")
    )


@dataclass(frozen=True)
class Settings:
    """Top-level settings container aggregating all sub-configs."""

    brlcad: BRLCADConfig = field(default_factory=BRLCADConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    # client-v2: when true, the REPL streams a per-node agent trace.
    debug: bool = field(default_factory=lambda: _env_bool("CLIENT_V2_DEBUG"))


# Module-level singleton — import and use directly
settings = Settings()
