"""CLI entry point for brlcad-mcp.

Usage
-----
    brlcad-mcp serve        # start the MCP tool server
    brlcad-mcp chat         # start the interactive agent (client-v2)
    brlcad-mcp chat --v1    # the legacy single-agent client (deprecated)

``chat`` runs client-v2: thin single-role agents (intake -> planner -> authorize
-> worker/executor -> verifier -> formatter) over dynamically-loaded skill
definitions.  The v1 client is one ReAct agent behind a single ~4k-token system
prompt; it is kept only so a v1/v2 comparison is still possible and will be
removed.
"""

from __future__ import annotations

import argparse
import sys


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="brlcad-mcp",
        description="BRL-CAD Model Context Protocol agent and server.",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("serve", help="Start the MCP tool server (stdio transport)")
    chat = sub.add_parser("chat", help="Start the interactive agent CLI (client-v2)")
    chat.add_argument(
        "--v1", action="store_true",
        help="Run the deprecated single-agent client instead of client-v2",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments and dispatch to the appropriate sub-command."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "serve":
        from brlcad_mcp.server.app import serve

        serve()
    elif args.command == "chat":
        if args.v1:
            print("brlcad-mcp chat --v1: the legacy single-agent client "
                  "(deprecated; omit --v1 for client-v2).", file=sys.stderr)
            from brlcad_mcp.client.agent import main as agent_main
        else:
            from client_v2.main import main as agent_main

        agent_main()
    else:
        parser.print_help()
        sys.exit(1)
