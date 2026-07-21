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
    """Settings for the OpenAI / LLM backend."""

    api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )
    model: str = field(
        default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o")
    )
    temperature: float = field(
        default_factory=lambda: float(os.getenv("OPENAI_TEMPERATURE", "0"))
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


@dataclass(frozen=True)
class Settings:
    """Top-level settings container aggregating all sub-configs."""

    brlcad: BRLCADConfig = field(default_factory=BRLCADConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    render: RenderConfig = field(default_factory=RenderConfig)


# Module-level singleton — import and use directly
settings = Settings()
