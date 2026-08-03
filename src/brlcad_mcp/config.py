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


def _env_bool(name: str, default: bool = False) -> bool:
    """Parse a truthy/falsy environment variable (1/true/yes/on)."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class BRLCADConfig:
    """Settings for the BRL-CAD TCP socket bridge."""

    host: str = field(default_factory=lambda: os.getenv("BRLCAD_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(os.getenv("BRLCAD_PORT", "5555")))
    timeout: float = field(
        default_factory=lambda: float(os.getenv("BRLCAD_TIMEOUT", "5.0"))
    )
    buffer_size: int = field(
        default_factory=lambda: int(os.getenv("BRLCAD_BUFFER_SIZE", "4096"))
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
        default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-5.6-sol")
    )
    temperature: float = field(
        default_factory=lambda: float(os.getenv("OPENAI_TEMPERATURE", "0"))
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
        default_factory=lambda: os.getenv("MCP_TRANSPORT", "stdio")
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
        default_factory=lambda: os.getenv(
            "BRLCAD_RENDER_DIR", os.path.expanduser("~/brlcad_renders")
        )
    )
    timeout: float = field(
        default_factory=lambda: float(os.getenv("BRLCAD_RENDER_TIMEOUT", "1800"))
    )
    # Restore points for destructive raw edits.  Deliberately NOT under
    # output_dir: they used to live in the render folder, so clearing a render
    # cache silently deleted the only way back from a bad `kill`.
    backup_dir: str = field(
        default_factory=lambda: os.getenv(
            "BRLCAD_BACKUP_DIR", os.path.expanduser("~/brlcad_backups")
        )
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
