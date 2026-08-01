"""client-v2: a thin-agent redesign of the BRL-CAD MCP client.

Layout mirrors the architecture (see DESIGN.md):

    main.py       the REPL entry point (python -m client_v2.main)
    graph.py      wires the agents into a LangGraph state machine
    state.py      the shared state every node reads and updates
    model.py      the LLM factory (OpenAI Responses API, real reasoning)
    prompts.py    thin per-role prompts -- the monolith's replacement
    agents/       one module per role: intake, planner, worker, verifier,
                  visual, formatter
    pipeline/     deterministic plan schema + executor (no model calls)
    skills/       the structured YAML definitions, their registry, and the
                  middleware that injects them
    terminal/     REPL input parsing and the debug trace

The MCP server and its tools are shared with v1 and unchanged by this work;
they are the verbs the worker calls.
"""
