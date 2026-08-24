# BRL-CAD MCP Agent

A natural language interface for [BRL-CAD](https://brlcad.org/) solid modeling, powered by the **Model Context Protocol (MCP)**, **LangGraph**, and **OpenAI**. This project allows users to create and manipulate 3D geometry in BRL-CAD through conversational English instead of memorizing MGED command syntax.

---

## Table of Contents

- [Overview](#overview)
- [Project Status](#project-status)
- [Architecture](#architecture)
- [The Agent (client-v2)](#the-agent-client-v2)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Available Tools](#available-tools)
- [Evaluation](#evaluation)
- [Sample Data](#sample-data)
- [Adding New Tools](#adding-new-tools)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [License](#license)

---

## Overview

BRL-CAD is a powerful open-source Constructive Solid Geometry (CSG) modeling system, but its Tcl-based MGED interface has a steep learning curve. This project bridges that gap by placing a conversational AI agent in front of BRL-CAD. Users describe geometry in plain English — or hand it a reference image — and the agent turns that into precise MGED commands, executes them against a live BRL-CAD instance, verifies the result against the model's own raytracer, and reports back.

The system is built on three layers:

1. **libmcpcad Listener** -- the native BRL-CAD command listener (the `mcp_listen` command in MGED), which accepts commands over a length-prefixed local socket and executes them through the C `ged_exec()` pipeline, returning `OK` / `ERR <code>` framed responses. (See the [`libmcpcad` branch](https://github.com/rs0125/brlcad/tree/libmcpcad).)
2. **MCP Tool Server** -- a Python FastMCP server that exposes typed, validated tool functions (primitive creation, boolean operations, dynamic command discovery, generic command execution, overlap resolution, model health auditing, spec-driven reconstruction, engine-truth verification, rendering, and rollback) to any MCP-compatible client.
3. **Agent Client (`client_v2`)** -- a LangGraph state machine of **thin, single-role agents** (intake → planner → authorize → worker/executor → verifier → formatter) driving **skill definitions loaded from YAML at runtime**. This replaced the original single ReAct agent behind one ~4k-token system prompt; see [The Agent](#the-agent-client-v2).

---

## Project Status

**What works today.** Natural-language and image-driven modelling against a live
BRL-CAD, end to end: a reference drawing goes in, a parametric CSG region comes
out, and the result is checked against BRL-CAD's own raytracer rather than
against the spec the agent invented.

At a glance:

| | |
|---|---|
| MCP tools | **21**, typed and validated |
| Skills (behaviour as data, editable at runtime) | **13** YAML definitions |
| Spec primitives | box, cylinder, sphere, wedge, cone |
| Evaluation corpus | **54 cases** over **20 real engineering drawings** |
| Test suite | 435 tests, no network or API key required |

**How correctness is established.** Reliability is a measured number, not an
impression — see [Evaluation](#evaluation). Every geometry verdict comes from
engine truth (`bb` plus `nirt` rays) compared against ground truth authored by
hand from the drawing. The vision check is advisory and cannot overturn it.

---

## Architecture

The system is composed of three layers that communicate in a chain:

- **Client** (`client_v2/`) — A LangGraph `StateGraph` of thin agents. Intake classifies the turn as chat or work; the planner turns work into an ordered, parameterized plan over the skill registry; an authorization gate can genuinely halt the graph for a user decision; the plan then runs either deterministically (executor, no model calls) or through the worker (the **only** agent with tool access); the verifier gates on engine truth and can kick work back to the planner; the formatter produces the final answer. It maintains a **persistent MCP session** with the server subprocess over **stdio**, so all tool calls share the same server process and TCP connection to MGED.

- **Server** (`server/app.py` + `server/tools/*`) — A FastMCP tool server that exposes 21 typed, validated tools: dedicated geometry tools (sphere, cylinder, box, booleans), meta-tools for dynamic command discovery and execution, and higher-level tools for overlap resolution, model health auditing, spec-driven building, verification, rendering, rollback, and recording the assumptions a build rests on. When a tool is invoked, it builds the corresponding MGED command(s) and sends them to BRL-CAD over a **persistent** connection via the transport layer (`transport/socket_bridge.py`). Responses are keyed off `SUCCESS:`/`ERROR:` prefixes that the transport derives from the listener's framed status.

- **Transport** (`transport/socket_bridge.py`) — Speaks the libmcpcad length-prefixed frame protocol (`MC` + u32 big-endian length + payload) in both directions, keeping a single long-lived connection so MGED state persists across calls. It translates the listener's `OK` / `ERR <code>` status into the `SUCCESS:` / `ERROR:` prefix the tools expect. The command text on the wire is plain MGED syntax — only the framing is structured.

**Request lifecycle:** User prompt → intake routes it → planner emits a plan → (authorize) → worker/executor calls a tool → MCP server builds MGED command → framed socket → libmcpcad listener runs it via `ged_exec()` → result flows back → verifier gates it → formatter answers.

> The pre-libmcpcad Tcl `uplevel` listener is preserved under [`legacy/`](legacy/) for historical reference and older MGED builds.

---

## The Agent (client-v2)

`client_v2/` is the shipping client (`brlcad-mcp chat`). It exists because a
single agent behind one large system prompt bundled agent identity, an MGED
command reference, guardrails, tool selection, several workflows, recovery
handling and rendering concepts into one blob — which works in a demo and fails
unpredictably in use. v2 splits that into roles, moves behavior out of prose into
data, and makes reliability measurable.

### Graph

```
intake --chat--> respond ------------------------------------------> END
       --work--> planner -> authorize (may HALT for a decision)
                              +--(deterministic plan)--> executor --+
                              |                                     |
                              +--(otherwise)---------> worker ------+
                                                                    |
                                         verifier <-----------------+
                                           |    |
                    revise (bounded) <-----+    +--> visual_check
                       back to planner               |     |     |
                             ^-----------------------+     |     +--> END
                              (visual mismatch, once)      +--> formatter --> END
```

Two boundaries are load-bearing:

- **`agents/` vs `pipeline/`** — anything under `pipeline/` is deterministic and
  reproducible (plan schema, executor). Anything under `agents/` asks a model.
  Where a behavior lives tells you whether it can be flaky.
- **`terminal/` vs everything else** — the graph has no opinion about how a human
  is talking to it, which is what lets the eval harness drive the same graph
  headlessly.

### Skills: behavior as data, editable at runtime

Capabilities are **YAML definitions**, not prompt prose, under
`client_v2/skills/definitions/`. Each spells out I/O params, preconditions,
dependencies, steps, examples, cautions, success/abort criteria, recovery actions
and effects. A *workflow* is a skill whose steps compose other skills — the
target demonstration workflow is `model_from_dimensioned_sketch` (reference image
or sketch → verified model + orthographic check renders).

The registry uses progressive disclosure: a lean catalog is always in context,
and a skill's full detail is injected only when it is in play. `/reload` re-reads
the definitions **and** the role prompts in a running session, so a skill can be
debugged and retried without a restart.

Role prompts are likewise files, not string literals:
`client_v2/prompts/definitions/*.md`, re-read per model call. Point
`CLIENT_V2_PROMPTS_DIR` at a directory to override individual prompts without
touching the repo.

### Verification is engine truth, not vibes

The gate is BRL-CAD's own raytracer, not a render and not a vision model.
`verify_model_dimensions` samples the built region and compares, per ray, the
material thickness the spec predicts (evaluated analytically) against the
thickness `nirt` reports, plus a bounding-box check. The sample is a 3×3 grid on
each axis — 27 rays — plus 15 more per subtracted cavity: one down its centre on
each axis and four offset toward its walls, which is what makes a cavity of the
wrong *diameter* detectable rather than merely present. A 30-part LEGO brick with
three bores is therefore checked with 72 rays, a single cone with 27. One shape-agnostic
comparison catches a subtraction that silently did not apply, a cavity in the
wrong place or of the wrong size or diameter, a blind pocket and its depth, and
wrong overall dimensions. A failure fails the turn and kicks it back to the
planner, bounded by a revision budget.

There is also an optional `visual_check` node that compares check renders to a
reference image. It is deliberately non-gating — an unclear verdict defaults to
MATCH — so correctness always rests on engine truth.

### Watching a turn

`CLIENT_V2_DEBUG=true` shows what the agent is doing as it does it:

```
--- trace ---
▸ intake
    route: work
    [planner] AI: {"steps": [{"skill": "build_model_spec", …}]}
▸ planner
    plan: {'steps': [{'skill': 'build_model_spec', …}]}
▸ authorize
    authorized: True
    [worker] ▸ build_from_spec({'spec': {'name': 'bearing', …}})
    [worker]   ✓ Built bearing.r with 4 part(s). Check renders: front, side, top…
    [worker] ▸ verify_model_dimensions({'region': 'bearing.r'})
    [worker]   ✓ Verification of 'bearing.r': PASS (40 rays, bbox within 1mm)
    [worker] AI: Built bearing.r and verified it.
▸ worker
    (+5 message(s))
▸ verifier
    verification: {'passed': True, 'checked': True, 'failures': []}
--- end trace ---
```

Two sources, because they answer different questions. Indented lines are printed
**live** by a callback handler as each model reply and tool call happens, each
labelled with the node it came from. Flush-left `▸` lines appear as each node
**finishes**, showing the state it wrote — the route, the plan, the verdict.

The live half has to be a callback rather than a stream reader: the worker runs
its tool loop in a nested invocation, so the graph's own node updates surface
nothing until that whole loop is done — a long build would print silence for
minutes and then dump everything at once. Callbacks propagate into the nested run
(the same reason the run log can see it), so tool calls appear as they are made.

This is not raw chain-of-thought: reasoning items come back encrypted, and only
the model's own summary is human-readable. The trace prints that summary (as
`· …` lines) when the model returns one — which requires asking for it, by adding
`"summary": "auto"` to the `reasoning` config in `client_v2/model.py`.

### Run logs

Every session appends newline-delimited JSON to
`~/brlcad_agent_logs/run_<stamp>.jsonl` (override with `CLIENT_V2_LOG_DIR`):
per-node state writes, every model call with its prompt/reply/usage (including
the ones inside the worker's tool loop), interrupts and the final answer. Base64
images are replaced by their byte count. A logging failure never costs a build.

### Evals

`evals/` scores the pipeline against ground truth read off the drawing, rather
than against the spec the agent invented. It has its own section below —
see [Evaluation](#evaluation).

---

## Prerequisites

- **Python 3.10+**
- **BRL-CAD** with the `libmcpcad` listener (the `mcp_listen` MGED command — see the [`libmcpcad` branch](https://github.com/rs0125/brlcad/tree/libmcpcad))
- **An API key for a model backend.** OpenAI by default. Any LangChain chat
  backend can be used instead, including Claude, Gemini, Ollama, Bedrock, or any
  OpenAI-compatible endpoint such as a local model or a company gateway. See
  [Model backends](docs/PROVIDERS.md). A reasoning-capable model is recommended,
  the model must support tool calling, and image input requires a vision-capable
  model.

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

The model backend is configurable: OpenAI by default, or Claude, Gemini, Ollama,
Bedrock, a local model, or a company gateway. [Model
backends](docs/PROVIDERS.md) covers the two ways in and the trade-off between
them. The older `OPENAI_*` names still work as fallbacks, so an existing `.env`
needs no edit.

| Variable              | Description                                  | Default       |
|-----------------------|----------------------------------------------|---------------|
| `LLM_PROVIDER`        | Backend, or the OpenAI wire format for a compatible endpoint | `openai` |
| `LLM_MODEL`           | Model id, passed through untouched           | `gpt-5.6-sol` |
| `LLM_API_KEY`         | API key for the backend                      | --            |
| `LLM_BASE_URL`        | Endpoint override — a gateway, or a local model | --         |
| `LLM_EFFORT`          | Reasoning effort: `minimal`/`low`/`medium`/`high` | provider default |
| `LLM_API`             | OpenAI wire dialect: `responses`/`chat`      | auto          |
| `LLM_TEMPERATURE`     | Sampling temperature for non-reasoning models (e.g. `gpt-4o`) | `0`           |
| `LLM_EXTRA`           | Extra backend kwargs as JSON, merged last    | `{}`          |
| `BRLCAD_HOST`         | Host where the libmcpcad listener is running | `127.0.0.1`   |
| `BRLCAD_PORT`         | Port the listener (`mcp_listen`) is bound to | `5555`        |
| `BRLCAD_TIMEOUT`      | Socket timeout in seconds                    | `5.0`         |
| `BRLCAD_BUFFER_SIZE`  | Receive buffer size in bytes                 | `4096`        |
| `BRLCAD_RENDER_DIR`   | Directory where rendered images are written  | `~/brlcad_renders` |
| `BRLCAD_RENDER_TIMEOUT` | Seconds to wait for one render (raise for slow renders) | `1800` |
| `MCP_TRANSPORT`       | MCP transport type                           | `stdio`       |
| `CLIENT_V2_DEBUG`     | Show the live agent trace instead of only the final answer (see [Watching a turn](#watching-a-turn)) | `false` |
| `CLIENT_V2_LOG_DIR`   | Where per-run JSONL logs are written         | `~/brlcad_agent_logs` |
| `CLIENT_V2_PROMPTS_DIR` | Directory of role-prompt overrides, overlaid on the built-ins | -- |

### Using a different model backend

Two ways in, covered fully in [Model backends](docs/PROVIDERS.md).

**Any OpenAI-compatible endpoint** needs configuration only, no extra package.
This covers local models and gateways — llama.cpp, vLLM, LM Studio, LocalAI,
Ollama's `/v1` port, a LiteLLM proxy, a company endpoint:

```bash
LLM_BASE_URL=http://localhost:8080/v1
LLM_MODEL=whatever-the-server-calls-it
```

`LLM_PROVIDER=openai` names the wire format rather than the vendor, and the model
string is passed through unchanged. Setting `LLM_BASE_URL` also switches the
client from OpenAI's Responses API to chat-completions, since few other endpoints
implement `/v1/responses`.

**A native integration** needs one install and gets provider-specific features a
compatibility shim drops:

```bash
pip install '.[anthropic]'      # or .[google] / .[ollama] / .[bedrock]
```

```bash
LLM_PROVIDER=anthropic
LLM_MODEL=claude-sonnet-4-5
LLM_API_KEY=sk-ant-...
```

In either case the model must support tool calling, since every build,
verification and render goes through a tool, and the image-to-CAD workflow
additionally requires vision. The worker's tool loop is bounded by call limits so
a model that does not terminate cannot run indefinitely; see
[Model backends](docs/PROVIDERS.md) for the values. The deterministic tool path
(`./evals/run.sh`) requires no model at all.

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
Starting client-v2 MCP client...
Loaded 20 tool(s) from BRL-CAD; 13 skill(s).
run log: /home/you/brlcad_agent_logs/run_20260803-101500.jsonl
=================================================
 client-v2 active. /help for commands, 'exit' to quit.
=================================================
```

`brlcad-mcp chat --v1` runs the deprecated single-agent client instead; it is kept
only so a v1/v2 comparison is still possible.

### Step 3: Describe Geometry in Plain English

```
You: Create a sphere named ball.s at the origin with radius 10

AI: Created sphere ball.s at (0.0, 0.0, 0.0) with radius 10.0. The sphere is now visible in the MGED viewer.
```

### Step 4: Or Hand It a Reference Image

```
You: /image ~/Downloads/bracket.png model this L-bracket in mm
  (attached 1 image(s))

AI needs a decision: I read 50 x 50 x 10 mm with a 6 mm bore, 12 mm from each
    outer edge. Confirm these dimensions before I build?
You: approved

AI: Built bracket.r ... Verification of 'bracket.r': PASS
```

REPL commands (`/help` lists them all):

| Command | Effect |
|---|---|
| `/image <path…> [prompt]` | attach image file(s) and send a message (aliases `/images`, `/img`) |
| `/paste [prompt]` | attach an image from the clipboard (alias `/clip`) |
| `/skills` | list the loaded skill definitions |
| `/prompts` | list the loaded role prompts |
| `/reload` | re-read skills and prompts from disk, mid-session |
| `exit` \| `quit` | leave |

Set `CLIENT_V2_DEBUG=true` to stream a node-by-node trace instead of just the
final answer.

---

## Available Tools

The MCP server currently exposes **20 tools**: dedicated geometry tools, meta-tools for dynamic command discovery/execution, geometry-analysis tools, rendering, spec-driven reconstruction, engine-truth verification, and rollback.

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

#### `verify_model_dimensions`

Check a spec-backed region against its spec objectively, using BRL-CAD's own
raytracer: a bounding-box comparison plus ~40 `nirt` rays whose measured material
thickness is compared to what the spec predicts analytically. Shape-agnostic, so
one check covers missing subtractions, mislocated or mis-sized cavities, wrong
hole diameters, blind pockets and their depth, and wrong overall dimensions. This
is the verifier's gate — no render, no vision model.

#### `model_health_report`

Audit an object with BRL-CAD's validators (lint checks plus `gqa` overlap detection on a fixed grid) and return a single grouped health report. Uses `gqa -g <grid>` so it never hangs on coincident faces.

#### `separate_overlap`

Resolve a single overlapping pair non-destructively by sliding the smaller part clear along the exit axis. Refuses bare solids and reports a part's parents so the caller can confirm it is moving the meaningful assembly, not a leaf.

#### `resolve_overlaps`

Sweep an assembly for overlaps and resolve each by minimal sliding, re-running `gqa` to verify. Greedy per-pair (nested cascades may need several passes); it moves parts, it does not subtract geometry.

### Rendering

#### `render_model`

Render the model the listener currently has open to a PNG, over the socket via the ged `rt` command (no `.g` path needed, no `opendb`). View presets (iso, front, side, top, back, and rear-quarter isometrics) or a custom azimuth/elevation, at three lighting levels: `studio` (default, camera-relative three-point rig — every angle lit the same), `model` (world-fixed rig — fixed-sun look), and `ambient` (evenly-lit, no rig). Renders entirely over the socket — `rt` writes the PNG directly, so no BRL-CAD binaries or `PATH` setup are required.

In client-v2 the rendered PNGs are fed back to a model in the `visual_check` node, which compares them against the reference image the user attached. That check is non-gating by design; correctness rests on `verify_model_dimensions`.

#### `render_previews`

Batch several `view:lighting` variants into one timestamped folder as small labelled stamps (A/B/C…), for the stamp-first workflow: preview cheaply, let the user pick a label, then render the final.

### Reconstruction (build from a reference image)

#### `build_from_spec`

Build a CSG region deterministically from a JSON spec and render the requested check views in one step. Saves a versioned spec so edits stay reversible.

Five primitives, unioned or subtracted:

| shape | parameters | what it is for |
|---|---|---|
| `box` | `center`, `size` | axis-aligned blocks, plates, walls |
| `cylinder` | `center` (base), `height` vector, `radius` | bores, bosses, studs, pins |
| `sphere` | `center`, `radius` | domes, ball ends |
| `wedge` | `center`, `size`, `top_size` | a box whose top face has its own footprint: tapered flanks, chamfers, and triangular gussets via `top_size: [x, 0]` |
| `cone` | `center` (base), `height` vector, `radius`, `top_radius` | truncated cones — tapered bosses, countersinks, turned profiles |

The vocabulary is deliberately far smaller than BRL-CAD's 41 primitives, and the
constraint is **not** what the engine can build — it builds all of them, and
`execute_command` reaches any of them. It is what the analytic CSG kernel can
intersect a ray with, because that prediction is what verification compares
against. A primitive without an exact ray-intersection function cannot be
verified, so adding one means adding its maths to `server/tools/csg.py` and
nothing else.

#### `declare_assumption`

Record a decision the model rests on — `topic`, `chose`, `over`, `reason` — as a
row beside the saved spec, so it survives the conversation.

Declared whenever the reference does not determine an answer: a dimension that is
missing, ambiguous, or contradicted elsewhere on the drawing. A model built from
a drawing is only as trustworthy as the readings behind it, and those readings
are otherwise invisible — a sentence in a reply that scrolls away, leaving the
geometry looking authoritative while the judgement that produced it is gone. An
undeclared assumption is indistinguishable from a misread, both to a reviewer and
to the record.

Because it is structured rather than prose, `promote_draft` can report a build's
assumptions from the record instead of from the agent's memory of what it said
several turns ago, and the evaluation harness can check that a contradiction was
actually acknowledged rather than silently resolved.

#### `edit_build` / `undo_build` / `list_builds`

Edit a spec-backed model with a small list of ops (move / update / add / remove a part) and rebuild — no need to re-send the whole model. `undo_build` reverts to the previous version; `list_builds` shows the saved versions.

### Recovery (raw destructive edits)

#### `restore_backup` / `list_backups`

Raw destructive commands run through `execute_command` (`kill`, `rm`, `mv`, `r` redefining an existing region…) are auto-snapshotted first: the objects they would delete or overwrite are `keep`-exported to a small `.g` under `<render_dir>/backups/`. `restore_backup` rolls back the last one (via `kill` + `dbconcat`); `list_backups` lists the restore points. This is the safety net for hand edits; spec-backed models should prefer `undo_build`.

---

## Evaluation

The harness answers the question the project could not previously answer with a
number: *how often does this actually produce correct geometry from a drawing?*

The distinction that makes it worth trusting: `verify_model_dimensions` proves a
build matches **the spec the agent wrote**, which can never tell you the spec
matched the drawing — there, the spec is both the input and the standard. The
harness instead compares the build against **ground truth read off the drawing by
hand**, so the agent is never grading its own homework.

### Running it

```bash
./evals/run.sh                                          # tool mode -- no API key
./evals/run.sh --mode agent --auto-approve              # unattended baseline
./evals/run.sh --mode agent --case '*_guided'           # one tier across the corpus
./evals/run.sh --mode agent --auto-approve --repeat 3   # pass^k
```

`evals/run.sh` owns the listener: it starts one on a throwaway `.g`, waits for
the socket, runs the harness, and kills it on the way out.

| mode | what it measures |
|---|---|
| `tool` | builds each case from its ground-truth spec. Deterministic and **needs no API key** — this is the one that catches pipeline regressions, and the quickest way to confirm an install works. |
| `agent` | gives the agent the prompt and scores whatever it built. This is the reliability metric. |

Agent mode has two shapes, and comparing them is the point. By default the case's
`approval` answers the workflow's confirmation halt, so the human-in-the-loop
path is exercised rather than bypassed. With `--auto-approve` there is no human
and no simulated user: the worker prompt gains a delta telling it to resolve
ambiguity itself and record each decision with `declare_assumption`. The agent is
then the only stochastic thing in the loop, so a failure is attributable to it
and nothing else.

`OPENAI_MODEL` selects the worker and `EVAL_JUDGE_MODEL` pins the vision judge
independently — without that, comparing two models would also change the judge,
and a score difference could be either model building better or judging more
leniently with no way to separate them.

### The corpus

54 cases over 20 drawings: sheet-metal brackets, LEGO bricks, plates and stepped
blocks from engineering-drawing textbooks, a dual-dimensioned bracket, hand
sketches, a photograph, and a turned chess piece.

Every drawing appears twice — once with the terse instruction a real user would
type, and once **guided**, supplying only what the image cannot carry: the unit,
the placement convention, which of two conflicting callouts controls. A guided
prompt never states what the drawing already shows, because every sentence added
removes a measurement. The gap between a pair is what disambiguation is worth.

Cases sit in four tiers, and a tier is a claim about the **drawing**, not about
difficulty:

| tier | scored on |
|---|---|
| **easy** | full envelope plus feature rays |
| **medium** | envelope plus one deliberate trap — decimal inches with no unit word, `22 [0.87]` dual dimensioning, inner-versus-outer flange readings |
| **ambiguous** | **no envelope at all.** These sheets genuinely under-determine the part, so asserting a bounding box would score *our* reading against the agent's. Judged on printed features and on whether the gaps were declared. |
| **hard** | expected to fail, and kept for that. A corpus where everything passes measures nothing. |

Cases live in `evals/cases/*.yaml` (hand-authored specs) and
`evals/image_cases.py` (a drawing plus the sentence a user would type).

### What a case can assert

| check | compares |
|---|---|
| `spec` | the full sweep: existence, derived bounding box, and a sampled ray per feature |
| `bbox` | outer lengths. Orientation-free by default — a drawing fixes a part's three lengths, rarely which way up you build it — with `oriented: true` where the prompt genuinely pins the axes |
| `bbox_ratio` | proportions only, normalised. The one geometric thing a reference with no printed unit can assert |
| `rays` | hand-authored probes: absolute, offset from the *measured* corner, or fractions of the measured bounding box |
| `dimensions` | a floor on how many of the printed values appear in the agent's own words |
| `conflicts` | that a single `declare_assumption` row names **both sides** of a contradiction. Each side is a group of equivalent framings, since a 6.3 mm cavity against a 1.0 mm roof on a 9.6 mm body is the same clash as 6.3 against 8.6 |
| `min_declarations` | a floor on assumptions for an under-dimensioned sheet. Declaring *nothing* there is the failure: the geometry can only have come from invented numbers, and inventing them silently is what makes such a drawing dangerous to work from |

A case passes only if every one of its checks passes.

### How ground truth earns trust

Ground truth is worth nothing unless it is right, so every check is validated in
**both directions** before being relied on:

- a **correct** model must pass — and for the scale-free checks, pass at two
  different scales;
- a **plausibly wrong** model must fail: continuous ridges instead of eight
  separate studs, a square cake instead of a round one, a 57 mm cube with the
  grooves never cut, a one-tier cake with exactly the right envelope.

That second half is the important one. Several of those wrong models satisfy the
bounding box **exactly** and are caught only by the rays — which is the whole
argument for the rays existing.

### pass^k

`--repeat K` runs the suite K times and reports `pass^k` (every run passed)
beside `pass@1` (the average single run). The gap between them *is* the
reliability signal: a single pass can report a figure that looks finished while
repeats reveal cases that only pass sometimes.

The suite repeats as a whole rather than each case K times in a row, so an
interrupted job still holds one sample of everything instead of ten samples of
the first case. Results are appended and flushed after each pass, and the runner
restarts the listener between passes, so a long job survives interruption.

### What a run produces

One self-contained directory per run under `evals/runs/<stamp>_<shape>/`, with
`latest` symlinked to the newest:

```
report.txt       the verdict per case, with failing checks named
results.jsonl    one row per case per pass -- every check and its outcome
log.jsonl        every model call, node write and interrupt
renders/         the check views the agent produced
models/          per-case standalone .g   (open with: mged <file>)
specs/           saved build specs, and assumptions.jsonl
backups/         restore points from destructive raw commands
```

Everything a verdict depends on is in that directory rather than in live process
state, so a finished run can be re-scored under revised checks without
rebuilding any geometry.

---

## Sample Data

`evals/images/` holds the 20 reference drawings the corpus is built from, in the
repository so a case is runnable from a fresh clone rather than depending on one
machine's downloads folder. They span the range deliberately:

| kind | examples | what it exercises |
|---|---|---|
| Clean dimensioned drawings | `roundbracket.jpg`, `engineeringdrawing.jpg` | multi-view reading, section views, an explicit "ALL DIMENSIONS IN mm" |
| Stepped blocks | `textbook3.jpg`, `textbook3.jpeg` | pure orthogonal CSG, every dimension printed |
| LEGO bricks and plates | `lego.jpg`, `lego2.jpg`, `lego3.jpg` | repeated features, stud grids, and one genuinely self-contradictory sheet |
| Unit traps | `2headedpart.jpg`, `sheetmetalbracket.png`, `metalbracketcomplicated2.jpg` | decimal inches with no unit word, and mm with bracketed inches |
| Under-dimensioned sheets | `triangularpartdrawing.jpg`, `textbookbracket.png` | whether the agent declares what the drawing never states |
| Hand-drawn and photographed | `handdrawnpart.jpg`, `birthdaycake.jpg`, `rubic.jpg` | hostile input: a sketch photographed mid-draw, a sketch with no unit, a photo with no dimensions at all |
| Turned profiles | `chesspieces.png` | curved profiles against a straight-sided primitive set |

To point a case at a drawing of your own, drop it in `evals/images/` and add an
entry to `evals/image_cases.py` — a bare filename resolves against that
directory, and absolute or `~` paths work too for a one-off.

One drawing in the folder, `deltabracket.jpg`, is deliberately **not** in the
corpus: several of its rotated dimensions are not legible enough at source
resolution to hand-author ground truth worth trusting, and wrong ground truth is
worse than none. It is kept as a candidate for a better scan.

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
pytest          # 435 tests, hermetic and CI-safe: no network, no LLM, no API key
```

Every part of the v2 graph is tested with injected fakes (`tests/v2_fakes.py`) —
models, tools, classifier and registry are all constructor arguments — so routing,
planning, execution, verification, kick-backs and the authorization halt are all
exercised offline. The analytic CSG kernel (`server/tools/csg.py`) is pure maths
and tested directly, primitive by primitive.

Every test runs on every invocation; nothing is skipped or gated behind a flag,
because a test that never executes is not coverage.

Reliability of the *whole pipeline against real geometry* is a separate question,
measured by the [eval harness](#evaluation) rather than by pytest — pytest tells
you the plumbing is sound, the harness tells you how often the geometry is right.

---

## Project Structure

The tree mirrors the architecture rather than being a flat pile of modules, so the
design is legible from `ls`.

```
brlcad-mcp/
├── client_v2/                       # THE AGENT (brlcad-mcp chat)
│   ├── main.py                      # REPL entry point (python -m client_v2.main)
│   ├── graph.py                     # wires the agents into a LangGraph state machine
│   ├── state.py                     # the shared state every node reads and updates
│   ├── model.py                     # LLM factory (Responses API, reasoning effort)
│   ├── runlog.py                    # per-run JSONL logging (nodes, model calls)
│   ├── agents/                      # one module per ROLE, each with a single job
│   │   ├── conversational.py        #   intake: chat-vs-work routing + turn reset
│   │   ├── planner.py               #   work request -> ordered, parameterized plan
│   │   ├── authorize.py             #   real graph halt for a user decision
│   │   ├── worker.py                #   the only agent with tool access
│   │   ├── verifier.py              #   engine-truth gate + bounded kick-back
│   │   ├── visual.py                #   non-gating render-vs-reference comparison
│   │   └── formatter.py             #   final answer in the required form
│   ├── pipeline/                    # DETERMINISTIC machinery, no model calls
│   │   ├── plan.py                  #   plan schema + parsing/validation
│   │   └── executor.py              #   runs a deterministic plan, binds ${refs}
│   ├── skills/
│   │   ├── registry.py              #   schema + loader + hot reload
│   │   ├── middleware.py            #   prompt injection (catalog / active detail)
│   │   └── definitions/*.yaml       #   THE SKILLS AND WORKFLOWS
│   ├── prompts/
│   │   ├── library.py               #   file-backed prompts, re-read per call
│   │   └── definitions/*.md          #   the thin per-role prompts
│   └── terminal/                    # REPL-facing only
│       ├── attachments.py           #   /image, /paste, command parsing
│       └── trace.py                 #   debug trace formatting
├── evals/
│   ├── run.sh                       # starts a throwaway listener, runs the harness
│   ├── harness.py                   # scores tool mode and agent mode vs ground truth
│   ├── image_cases.py               # drawing + the sentence a user would type
│   ├── cases/*.yaml                 # golden cases (prompt, image, spec, assertions)
│   ├── images/                      # the 20 reference drawings the corpus is built on
│   └── runs/<stamp>/                # one self-contained directory per run (gitignored)
├── src/
│   └── brlcad_mcp/
│       ├── __init__.py              # Package metadata and version
│       ├── __main__.py              # python -m brlcad_mcp entry point
│       ├── cli.py                   # CLI argument parsing (serve / chat [--v1])
│       ├── config.py                # Centralised settings from env / .env
│       ├── client/
│       │   ├── __init__.py
│       │   └── agent.py             # DEPRECATED v1 single-agent client
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
│       │       ├── reconstruct.py   # build_from_spec / edit_build / undo_build
│       │       ├── csg.py           # analytic ray-vs-spec kernel (pure)
│       │       ├── verify.py        # verify_model_dimensions (bb + nirt)
│       │       ├── snapshots.py     # auto-snapshot + restore_backup
│       │       └── rendering.py     # render_model / render_previews (rt over the socket)
│       └── transport/
│           ├── __init__.py
│           └── socket_bridge.py     # Persistent libmcpcad frame connection
├── tests/
│   ├── conftest.py                  # MockListener (fake MGED) + fixtures
│   ├── test_transport.py            # real bridge ↔ mock frame listener
│   ├── test_tools.py                # server tools over the mock listener
│   ├── test_csg.py / test_verify.py # verification kernel + verdicts
│   ├── test_evals_harness.py        # the harness: scoring, corpus invariants
│   └── test_client_v2_*.py          # one file per v2 component
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
