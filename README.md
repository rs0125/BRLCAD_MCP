# BRL-CAD MCP Agent

A natural language interface for [BRL-CAD](https://brlcad.org/) solid modeling, powered by the **Model Context Protocol (MCP)**, **LangGraph**, and **OpenAI**. This project allows users to create and manipulate 3D geometry in BRL-CAD through conversational English instead of memorizing MGED command syntax.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Available Tools](#available-tools)
- [Adding New Tools](#adding-new-tools)
- [Project Structure](#project-structure)
- [License](#license)

---

## Overview

BRL-CAD is a powerful open-source Constructive Solid Geometry (CSG) modeling system, but its Tcl-based MGED interface has a steep learning curve. This project bridges that gap by placing a conversational AI agent in front of BRL-CAD. Users describe geometry in plain English, and a GPT-4o-backed agent translates those descriptions into precise MGED commands, executes them against a live BRL-CAD instance, and reports the results.

The system is built on three layers:

1. **libmcpcad Listener** -- the native BRL-CAD command listener (the `mcp_listen` command in MGED), which accepts commands over a length-prefixed local socket and executes them through the C `ged_exec()` pipeline, returning `OK` / `ERR <code>` framed responses. (See the [`libmcpcad` branch](https://github.com/rs0125/brlcad/tree/libmcpcad).)
2. **MCP Tool Server** -- a Python FastMCP server that exposes typed, validated tool functions (sphere creation, boolean operations, dynamic command discovery, generic command execution, overlap resolution, model health auditing, and rendering) to any MCP-compatible client.
3. **LangGraph Agent Client** -- a ReAct agent that reasons about user requests, selects the appropriate tools, and orchestrates multi-step modeling operations with real-time tool-call visibility.

---

## Architecture

The system is composed of three layers that communicate in a chain:

- **Client** (`client/agent.py`) — A LangGraph ReAct agent that takes natural-language input from the user, reasons about the request, and decides which MCP tool(s) to call. It maintains a **persistent MCP session** with the server subprocess over **stdio**, so all tool calls share the same server process and TCP connection to MGED.

- **Server** (`server/app.py` + `server/tools/*`) — A FastMCP tool server that exposes 12 typed, validated tools: dedicated geometry tools (sphere, cylinder, box, booleans), meta-tools for dynamic command discovery and execution, and higher-level tools for overlap resolution, model health auditing, and rendering. When a tool is invoked, it builds the corresponding MGED command(s) and sends them to BRL-CAD over a **persistent** connection via the transport layer (`transport/socket_bridge.py`). Responses are keyed off `SUCCESS:`/`ERROR:` prefixes that the transport derives from the listener's framed status.

- **Transport** (`transport/socket_bridge.py`) — Speaks the libmcpcad length-prefixed frame protocol (`MC` + u32 big-endian length + payload) in both directions, keeping a single long-lived connection so MGED state persists across calls. It translates the listener's `OK` / `ERR <code>` status into the `SUCCESS:` / `ERROR:` prefix the tools expect. The command text on the wire is plain MGED syntax — only the framing is structured.

**Request lifecycle:** User prompt → Agent selects tool → MCP server builds MGED command → framed socket → libmcpcad listener runs it via `ged_exec()` → result flows back through the same chain.

> The pre-libmcpcad Tcl `uplevel` listener is preserved under [`legacy/`](legacy/) for historical reference and older MGED builds.

---

## Prerequisites

- **Python 3.10+**
- **BRL-CAD** with the `libmcpcad` listener (the `mcp_listen` MGED command — see the [`libmcpcad` branch](https://github.com/rs0125/brlcad/tree/libmcpcad))
- **OpenAI API key** with access to the GPT-4o model (or another supported model)

---

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/brlcad-mcp.git
   cd brlcad-mcp
   ```

2. **Create and activate a virtual environment:**

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install the package (editable mode recommended for development):**

   ```bash
   pip install -e ".[dev]"
   ```

   This installs all runtime and development dependencies defined in `pyproject.toml`.

4. **Configure environment variables:**

   ```bash
   cp .env.example .env
   ```

   Edit `.env` and set your OpenAI API key (see [Configuration](#configuration)).

---

## Configuration

All configuration is managed through a `.env` file in the project root. A template is provided in `.env.example`.

| Variable              | Description                                  | Default       |
|-----------------------|----------------------------------------------|---------------|
| `OPENAI_API_KEY`      | Your OpenAI API key (required)               | --            |
| `OPENAI_MODEL`        | The OpenAI model to use (confirm exact id in your dashboard) | `gpt-5.6-sol` |
| `OPENAI_REASONING_EFFORT` | For reasoning models: `low`/`medium`/`high`/`max` (blank = `high`) | -- |
| `OPENAI_TEMPERATURE`  | Sampling temperature for non-reasoning models (e.g. `gpt-4o`) | `0`           |
| `BRLCAD_HOST`         | Host where the libmcpcad listener is running | `127.0.0.1`   |
| `BRLCAD_PORT`         | Port the listener (`mcp_listen`) is bound to | `5555`        |
| `BRLCAD_TIMEOUT`      | Socket timeout in seconds                    | `5.0`         |
| `BRLCAD_BUFFER_SIZE`  | Receive buffer size in bytes                 | `4096`        |
| `BRLCAD_RENDER_DIR`   | Directory where rendered images are written  | `~/brlcad_renders` |
| `BRLCAD_RENDER_TIMEOUT` | Seconds to wait for one render (raise for slow renders) | `1800` |
| `MCP_TRANSPORT`       | MCP transport type                           | `stdio`       |

**Important:** Never commit your `.env` file. It is already included in `.gitignore`.

---

## Usage

### Step 1: Start the BRL-CAD libmcpcad Listener

Open MGED on a database and start the listener with the native `mcp_listen`
command (loopback only, port matches `BRLCAD_PORT`):

```bash
mged my_model.g
```

Inside the MGED console:

```tcl
mcp_listen 5555
```

You should see:

```
mcp_listen: listening on 127.0.0.1:5555
```

> Building BRL-CAD with the `mcp_listen` command requires the
> [`libmcpcad` branch](https://github.com/rs0125/brlcad/tree/libmcpcad).
> For experimentation without an MGED GUI, that branch also builds a
> standalone `mcpcad_test_server <port> <db.g>` harness exposing the same
> protocol.

### Step 2: Run the Agent Client

In a separate terminal, activate the virtual environment and start the client:

```bash
source .venv/bin/activate
brlcad-mcp chat
```

Or equivalently:

```bash
python -m brlcad_mcp chat
```

You should see:

```
Starting local MCP Client...
Successfully loaded 12 tool(s) from BRL-CAD!

=================================================
 BRL-CAD Terminal Agent Active. Type 'exit' to quit.
=================================================
```

### Step 3: Describe Geometry in Plain English

```
You: Create a sphere named ball.s at the origin with radius 10

AI is calculating geometry...

AI: Created sphere ball.s at (0.0, 0.0, 0.0) with radius 10.0. The sphere is now visible in the MGED viewer.
```

Type `exit` or `quit` to terminate the session.

---

## Available Tools

The MCP server currently exposes **12 tools** organised into four groups: dedicated geometry tools, meta-tools for dynamic command discovery/execution, geometry-analysis tools, and rendering.

### Dedicated Geometry Tools

These handle draw/autoview automatically and have strict parameter validation.

#### `create_sphere`

Creates a solid sphere primitive in BRL-CAD.

| Parameter | Type    | Description                         |
|-----------|---------|-------------------------------------|
| `name`    | string  | Name of the sphere (e.g., `ball.s`) |
| `x`       | float   | X coordinate of the center          |
| `y`       | float   | Y coordinate of the center          |
| `z`       | float   | Z coordinate of the center          |
| `radius`  | float   | Radius of the sphere                |

#### `create_cylinder`

Creates a right circular cylinder (RCC) in BRL-CAD.

| Parameter   | Type   | Description                            |
|-------------|--------|----------------------------------------|
| `name`      | string | Name of the cylinder (e.g., `tube.s`)  |
| `base_x`    | float  | X coordinate of the base center        |
| `base_y`    | float  | Y coordinate of the base center        |
| `base_z`    | float  | Z coordinate of the base center        |
| `height_x`  | float  | X component of the height vector       |
| `height_y`  | float  | Y component of the height vector       |
| `height_z`  | float  | Z component of the height vector       |
| `radius`    | float  | Radius of the cylinder                 |

#### `create_box`

Creates an axis-aligned box (RPP) in BRL-CAD.

| Parameter | Type   | Description               |
|-----------|--------|---------------------------|
| `name`    | string | Name of the box            |
| `x_min`   | float  | Minimum X coordinate       |
| `y_min`   | float  | Minimum Y coordinate       |
| `z_min`   | float  | Minimum Z coordinate       |
| `x_max`   | float  | Maximum X coordinate       |
| `y_max`   | float  | Maximum Y coordinate       |
| `z_max`   | float  | Maximum Z coordinate       |

#### `boolean_combination`

Performs a CSG boolean operation to combine two existing objects.

| Parameter       | Type   | Description                                                        |
|-----------------|--------|--------------------------------------------------------------------|
| `output_name`   | string | Name of the resulting region (e.g., `result.r`)                    |
| `base_object`   | string | The primary object                                                 |
| `operator`      | string | Boolean operator: `u` (union), `-` (subtract), `+` (intersect)    |
| `target_object` | string | The secondary object                                               |

### Meta-Tools (Dynamic Command Discovery & Execution)

These tools allow the agent to discover, learn, and execute **any** MGED command — not just the ones with dedicated tool wrappers.

#### `list_commands`

Browse all available MGED commands with one-liner descriptions, optionally filtered by category (e.g., `creation`, `editing`, `display`).

#### `get_command_help`

Fetch the full help / man-page text for a specific MGED command to learn its exact argument syntax.

#### `execute_command`

Execute an arbitrary MGED command string. Supports `auto_draw` and `object_name` parameters for automatic view refresh after geometry changes.

#### `analyze_command_error`

Diagnose a failed MGED command: fetches help text, checks object existence, and executes a corrected command. Supports up to 5 retry attempts to prevent infinite loops.

### Geometry Analysis Tools

These run BRL-CAD's own analysers over the socket and summarise the results for the agent.

#### `model_health_report`

Audit an object with BRL-CAD's validators (lint checks plus `gqa` overlap detection on a fixed grid) and return a single grouped health report. Uses `gqa -g <grid>` so it never hangs on coincident faces.

#### `separate_overlap`

Resolve a single overlapping pair non-destructively by sliding the smaller part clear along the exit axis. Refuses bare solids and reports a part's parents so the caller can confirm it is moving the meaningful assembly, not a leaf.

#### `resolve_overlaps`

Sweep an assembly for overlaps and resolve each by minimal sliding, re-running `gqa` to verify. Greedy per-pair (nested cascades may need several passes); it moves parts, it does not subtract geometry.

### Rendering

#### `render_model`

Render the model the listener currently has open to a PNG, over the socket via the ged `rt` command (no `.g` path needed, no `opendb`). View presets (iso, front, side, top, back, and rear-quarter isometrics) or a custom azimuth/elevation, at three lighting levels: `studio` (default, camera-relative three-point rig — every angle lit the same), `model` (world-fixed rig — fixed-sun look), and `ambient` (evenly-lit, no rig). Renders entirely over the socket — `rt` writes the PNG directly, so no BRL-CAD binaries or `PATH` setup are required.

---

## Adding New Tools

The system is designed so that the agent client never needs modification when new tools are added. Tools are discovered dynamically at startup via the MCP protocol.

To add a new tool:

1. Choose the appropriate file under `src/brlcad_mcp/server/tools/` (or create a new module for a new category).
2. Define a new function decorated with `@mcp.tool()`.
3. Use Pydantic `Field` annotations for all parameters to provide type-safe descriptions for the LLM.
4. Construct the appropriate MGED command string and send it through `send_command()`.

**Example -- adding a cone tool to `primitives.py`:**

```python
from pydantic import Field

from brlcad_mcp.server.app import mcp
from brlcad_mcp.server.tools.helpers import check_mged_result, parse_response
from brlcad_mcp.transport import send_command

@mcp.tool()
def create_cone(
    name: str = Field(..., description="Name of the cone, e.g., 'cone.s'"),
    base_x: float = Field(..., description="X coordinate of the base center"),
    base_y: float = Field(..., description="Y coordinate of the base center"),
    base_z: float = Field(..., description="Z coordinate of the base center"),
    height_x: float = Field(..., description="X component of the height vector"),
    height_y: float = Field(..., description="Y component of the height vector"),
    height_z: float = Field(..., description="Z component of the height vector"),
    base_radius: float = Field(..., description="Radius at the base"),
    top_radius: float = Field(..., description="Radius at the top"),
) -> str:
    """Creates a truncated general cone (TGC) in BRL-CAD."""
    cmd = f"in {name} tgc {base_x} {base_y} {base_z} {height_x} {height_y} {height_z} {base_radius} {top_radius}"
    result = send_command(cmd)
    error = check_mged_result(result, command=cmd)
    if error:
        return error
    send_command(f"draw {name}")
    send_command("autoview")
    return f"Created cone '{name}'. Output: {parse_response(result)}"
```

If you create a new tool module (e.g., `transforms.py`), import it in `src/brlcad_mcp/server/tools/__init__.py` to ensure it gets registered.

---

## Testing

The suite needs **no BRL-CAD build and no API key** for the deterministic
tests: a `MockListener` (in `tests/conftest.py`) speaks the exact libmcpcad
frame protocol and runs a small stateful fake-MGED, so the real transport,
tools, and agent are exercised end to end over a loopback socket.

```bash
pip install -e '.[dev]'

# deterministic tests — fast, hermetic, CI-safe (no network, no LLM)
pytest

# also run the natural-language tests (real LLM; needs OPENAI_API_KEY, costs money)
pytest --run-llm
```

Natural-language tests are marked `llm` and skipped by default. They drive
the real LangGraph agent against the mock listener and assert on the
resulting fake-database state, so they validate the full
prompt → LLM → tool → transport chain without a live BRL-CAD instance.

---

## Project Structure

```
brlcad-mcp/
├── src/
│   └── brlcad_mcp/
│       ├── __init__.py              # Package metadata and version
│       ├── __main__.py              # python -m brlcad_mcp entry point
│       ├── cli.py                   # CLI argument parsing (serve / chat)
│       ├── config.py                # Centralised settings from env / .env
│       ├── client/
│       │   ├── __init__.py
│       │   └── agent.py             # LangGraph ReAct agent + chat loop
│       ├── server/
│       │   ├── __init__.py
│       │   ├── app.py               # FastMCP application instance
│       │   └── tools/
│       │       ├── __init__.py      # Auto-imports tool modules
│       │       ├── catalog.py       # MGED command catalog (categories + descriptions)
│       │       ├── helpers.py       # Shared error detection + response parsing
│       │       ├── primitives.py    # Sphere, cylinder, box creation tools
│       │       ├── boolean.py       # CSG boolean operations
│       │       ├── discovery.py     # list_commands + get_command_help tools
│       │       ├── execution.py     # execute_command + analyze_command_error tools
│       │       ├── geometry_ops.py  # separate_overlap + resolve_overlaps tools
│       │       ├── health.py        # model_health_report tool
│       │       └── rendering.py     # render_model tool (rt over the socket)
│       └── transport/
│           ├── __init__.py
│           └── socket_bridge.py     # Persistent libmcpcad frame connection
├── tests/
│   ├── __init__.py
│   ├── conftest.py                  # MockListener (fake MGED) + fixtures
│   ├── test_config.py
│   ├── test_helpers.py
│   ├── test_transport.py            # real bridge ↔ mock frame listener
│   ├── test_tools.py
│   ├── test_discovery.py
│   ├── test_rendering.py            # render tool logic (hermetic)
│   └── test_agent_nl.py             # natural-language tests (opt-in: --run-llm)
├── examples/
│   └── nl_demo.py                   # scripted natural-language walkthrough
├── legacy/
│   ├── README.md
│   └── listener.tcl                 # superseded Tcl uplevel listener
├── .env.example                     # Template for environment variables
├── .gitignore
├── pyproject.toml                   # PEP 621 packaging + tool config
├── LICENSE
└── README.md
```

---

## License

This project is licensed under the [MIT License](LICENSE).

BRL-CAD itself is licensed separately under the LGPL 2.1 — see the [BRL-CAD project](https://brlcad.org/) for details.
