"""Centralized configuration loaded from environment variables and .env file."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

# Resolve project root (two levels up from this file → src/brlcad_mcp/config.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_env_files() -> list[Path]:
    """Load .env, nearest-first, and report which files were actually read.

    Two lookups, because one is not enough:

    * The .env nearest the WORKING DIRECTORY wins. ``PROJECT_ROOT`` follows
      where this file lives, not where you are, so a second checkout -- or an
      unpacked release run with a ``brlcad-mcp`` from elsewhere on PATH -- was
      silently reading the *other* tree's .env. That is very hard to see: the
      symptom is a transport you did not configure, because an unset
      ``BRLCAD_IPC_PATH`` falls back to host/port and 127.0.0.1:5555 is also
      the built-in default.
    * Then the one beside the package. Right for an editable checkout, and the
      only candidate a non-editable install has -- though there ``PROJECT_ROOT``
      lands inside site-packages and holds no .env, so such an install is
      configured by real environment variables alone.

    ``load_dotenv`` never overwrites a variable that is already set, so loading
    the nearest file first is what makes it take precedence, and a real exported
    variable still beats both.
    """
    loaded: list[Path] = []
    nearest = find_dotenv(usecwd=True)
    if nearest:
        load_dotenv(nearest)
        loaded.append(Path(nearest).resolve())
    beside = (PROJECT_ROOT / ".env").resolve()
    if beside.is_file() and beside not in loaded:
        load_dotenv(beside)
        loaded.append(beside)
    return loaded


# Recorded rather than discarded: the banner names these, so "which config am I
# actually running" is one line of output instead of an investigation.
ENV_FILES = _load_env_files()


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


def _env_first(names: tuple[str, ...], default: str = "") -> str:
    """First of several environment variables that is set and non-blank.

    Lets a general ``LLM_*`` name take precedence while the original
    ``OPENAI_*`` name keeps working, so existing ``.env`` files and the shipped
    release are unaffected by the move to configurable providers.
    """
    for name in names:
        value = (os.getenv(name) or "").strip()
        if value:
            return value
    return default


def _env_json_obj(name: str) -> dict:
    """Parse an environment variable holding a JSON object, or return {}.

    This is the escape hatch for provider kwargs we do not model, so a typo in
    it must not stop the client starting -- and a JSON scalar or list is not a
    kwargs mapping, so those are rejected too.
    """
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _env_bool(name: str, default: bool = False) -> bool:
    """Parse a truthy/falsy environment variable (1/true/yes/on)."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class BRLCADConfig:
    """Where the libmcpcad listener is, and how patiently to talk to it.

    Two transports carry the same protocol.  Setting ``ipc_path`` selects a
    Unix-domain socket (``mcp_listen ipc`` in MGED); leaving it unset uses the
    loopback port (``mcp_listen <port>``).  Presence selects the transport
    rather than a separate mode setting, so there is one rule to remember.
    """

    # Path to a Unix-domain socket.  Empty means use host/port instead.
    ipc_path: str = field(
        default_factory=lambda: _env_first(("BRLCAD_IPC_PATH",))
    )
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
    """Which model backend to talk to, and how.

    Any LangChain chat integration can be used, not just OpenAI.  Two ways in:

    * **An OpenAI-dialect endpoint** -- leave ``provider`` at ``openai`` and set
      ``base_url``.  The provider name there means the *wire format*, not the
      vendor, so llama.cpp's server, vLLM, LM Studio, Ollama's ``/v1`` port,
      LocalAI or a corporate gateway all work with no extra package.
    * **A dedicated integration** -- set ``provider`` to ``anthropic``,
      ``google_genai``, ``ollama``, ``bedrock`` and so on, and install that
      provider's ``langchain-*`` package.

    Every setting has an ``LLM_*`` name.  The original ``OPENAI_*`` names still
    work as fallbacks, so an existing ``.env`` needs no edit.
    """

    provider: str = field(
        default_factory=lambda: _env_first(("LLM_PROVIDER",), "openai")
    )
    api_key: str = field(
        default_factory=lambda: _env_first(("LLM_API_KEY", "OPENAI_API_KEY"))
    )
    model: str = field(
        default_factory=lambda: _env_first(("LLM_MODEL", "OPENAI_MODEL"),
                                           "gpt-5.6-sol")
    )
    # Endpoint override.  Both OPENAI_* spellings are accepted because the
    # OpenAI SDK reads OPENAI_BASE_URL while langchain-openai reads
    # OPENAI_API_BASE, and someone following either doc should just work.
    base_url: str = field(
        default_factory=lambda: _env_first(
            ("LLM_BASE_URL", "OPENAI_BASE_URL", "OPENAI_API_BASE")
        )
    )
    # OpenAI wire dialect: 'responses' or 'chat'.  Blank means decide from
    # whether base_url is set -- see providers.resolve_dialect.  Only consulted
    # for the OpenAI family; other providers have no such concept.
    api_dialect: str = field(
        default_factory=lambda: _env_first(("LLM_API",))
    )
    temperature: float = field(
        default_factory=lambda: _env_num("LLM_TEMPERATURE",
                                         str(_env_num("OPENAI_TEMPERATURE",
                                                      "0", float)), float)
    )
    # Reasoning/thinking effort: minimal|low|medium|high.  LangChain treats this
    # as a standard parameter and each provider translates it (OpenAI reasoning
    # effort, Anthropic thinking budget, ...), so one setting covers them all.
    reasoning_effort: str = field(
        default_factory=lambda: _env_first(("LLM_EFFORT",
                                            "OPENAI_REASONING_EFFORT"))
    )
    # Arbitrary extra kwargs as JSON, merged last and winning.  This is what
    # makes a provider quirk we do not model a config change rather than a
    # patch, e.g. {"num_predict": 512} for Ollama.
    extra: dict = field(default_factory=lambda: _env_json_obj("LLM_EXTRA"))


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
