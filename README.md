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

1. **Tcl Socket Listener** -- runs inside BRL-CAD's MGED and accepts commands over a persistent TCP socket, returning delimited responses with `SUCCESS:`/`ERROR:` prefixes.
2. **MCP Tool Server** -- a Python FastMCP server that exposes typed, validated tool functions (sphere creation, boolean operations, dynamic command discovery, and generic command execution) to any MCP-compatible client.
3. **LangGraph Agent Client** -- a ReAct agent that reasons about user requests, selects the appropriate tools, and orchestrates multi-step modeling operations with real-time tool-call visibility.

---

## Architecture

The system is composed of three layers that communicate in a chain:

- **Client** (`client/agent.py`) — A LangGraph ReAct agent that takes natural-language input from the user, reasons about the request, and decides which MCP tool(s) to call. It maintains a **persistent MCP session** with the server subprocess over **stdio**, so all tool calls share the same server process and TCP connection to MGED.

- **Server** (`server/app.py` + `server/tools/*`) — A FastMCP tool server that exposes 8 typed, validated tools: 4 dedicated geometry tools (sphere, cylinder, box, booleans) and 4 meta-tools for dynamic command discovery and execution. When a tool is invoked, it builds the corresponding MGED command and sends it to BRL-CAD over a **persistent TCP** connection (port 5555) via the transport layer (`transport/socket_bridge.py`). All responses are parsed using `SUCCESS:`/`ERROR:` prefixes from the listener.

- **Bridge** (`scripts/listener.tcl`) — A lightweight Tcl script sourced inside BRL-CAD's MGED session. It listens on a TCP socket, evaluates incoming commands in the global MGED scope using `uplevel #0`, wraps results with `SUCCESS:`/`ERROR:` prefixes, and terminates each response with a `<<END_OF_RESPONSE>>` delimiter so the Python client can reliably read multi-line output.

**Request lifecycle:** User prompt → Agent selects tool → MCP server builds MGED command → TCP socket → Tcl listener evaluates in MGED → result flows back through the same chain.

---

## Prerequisites

- **Python 3.10+**
- **BRL-CAD** (with MGED) installed and accessible on your system
- **OpenAI API key** with access to the GPT-4o model (or another supported model)
- **Tcl** (bundled with BRL-CAD's MGED)

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
| `OPENAI_MODEL`        | The OpenAI model to use                      | `gpt-4o`      |
| `OPENAI_TEMPERATURE`  | Sampling temperature (0 = deterministic)     | `0`           |
| `BRLCAD_HOST`         | Hostname where the Tcl listener is running   | `127.0.0.1`   |
| `BRLCAD_PORT`         | TCP port for the Tcl socket bridge           | `5555`        |
| `BRLCAD_TIMEOUT`      | Socket timeout in seconds                    | `5.0`         |
| `BRLCAD_BUFFER_SIZE`  | TCP receive buffer size in bytes             | `4096`        |
| `MCP_TRANSPORT`       | MCP transport type                           | `stdio`       |

**Important:** Never commit your `.env` file. It is already included in `.gitignore`.

---

## Usage

### Step 1: Start BRL-CAD with the Tcl Listener

Open MGED and create (or open) a database, then source the listener script:

```bash
mged my_model.g
```

Inside the MGED console:

```tcl
source /path/to/brlcad-mcp/scripts/listener.tcl
```

You should see:

```
=================================================
 BRL-CAD AI Bridge active.
 Listening for MCP commands on localhost:5555...
=================================================
```

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
Successfully loaded 8 tool(s) from BRL-CAD!

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

The MCP server currently exposes **8 tools** organised into two groups:

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
│       │       └── execution.py     # execute_command + analyze_command_error tools
│       └── transport/
│           ├── __init__.py
│           └── socket_bridge.py     # Persistent TCP connection to BRL-CAD
├── scripts/
│   └── listener.tcl                 # Tcl socket bridge for BRL-CAD MGED
├── tests/
│   ├── __init__.py
│   ├── test_config.py
│   ├── test_transport.py
│   └── test_tools.py
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
