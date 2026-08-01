"""Turning a line of REPL input into an agent message, including images.

A reference image is the input for the ``model_from_dimensioned_sketch``
workflow, so the client has to be able to attach one: ``/image path...`` reads
files, ``/paste`` grabs one from the clipboard.  Images travel to the model as
OpenAI multimodal ``image_url`` parts (base64 data URIs).

Command matching is on the first whitespace-delimited word, so ``/images`` is
not mistaken for ``/image`` -- a prefix match silently ate the plural's ``s`` and
made every ``/images`` call fail with a confusing "no image found".
"""

from __future__ import annotations

import base64
import mimetypes
import subprocess
from dataclasses import dataclass
from pathlib import Path

from langchain_core.messages import HumanMessage

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}

IMAGE_CMDS = ("/image", "/images", "/img")
PASTE_CMDS = ("/paste", "/clip")
HELP_CMDS = ("/help", "/?")
SKILLS_CMDS = ("/skills",)
RELOAD_CMDS = ("/reload",)
QUIT_WORDS = ("exit", "quit")

DEFAULT_IMAGE_PROMPT = "Here is a reference image."

HELP = """Commands:
  /image <path> [more paths] [prompt]  attach image file(s) and send a message
                                       (aliases: /images, /img)
  /paste [prompt]                      attach an image from the clipboard
                                       (alias: /clip)
  /skills                              list the loaded skill definitions
  /reload                              re-read the skill definitions from disk
  /help                                show this help
  exit | quit                          leave
Drag-and-drop a file into the terminal to paste its path after /image.
Anything else is sent to the agent as a normal message."""


@dataclass
class ReplCommand:
    """A local command handled by the REPL, not sent to the agent."""

    name: str            # "help" | "skills" | "reload" | "quit"


def data_uri(data: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def image_part_from_file(path: Path) -> dict:
    """An OpenAI multimodal image_url part built from an image file."""
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    return {"type": "image_url",
            "image_url": {"url": data_uri(path.read_bytes(), mime)}}


def clipboard_image_part() -> dict | None:
    """Best-effort clipboard image (Wayland then X11); None if unavailable."""
    for cmd in (["wl-paste", "--type", "image/png"],
                ["xclip", "-selection", "clipboard", "-t", "image/png", "-o"]):
        try:
            out = subprocess.run(cmd, capture_output=True, timeout=5)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
        if out.returncode == 0 and out.stdout:
            return {"type": "image_url",
                    "image_url": {"url": data_uri(out.stdout, "image/png")}}
    return None


def image_message(rest: str) -> HumanMessage:
    """Build a multimodal message from ``/image`` arguments.

    ``rest`` is everything after the command word: leading image paths, then an
    optional free-text prompt.  Raises ``ValueError`` with a precise message
    when nothing usable is found -- naming the path we actually tried, since a
    typo or wrong working directory is the usual cause.
    """
    tokens = rest.split()
    images: list[Path] = []
    prompt_tokens: list[str] = []
    for i, token in enumerate(tokens):
        path = Path(token.strip("\"'")).expanduser()
        looks_like_image = path.suffix.lower() in IMAGE_EXTS
        if looks_like_image and path.is_file():
            images.append(path)
        elif looks_like_image:
            raise ValueError(f"Could not find that image file: {path}")
        else:
            prompt_tokens = tokens[i:]
            break
    if not images:
        raise ValueError(
            "Usage: /image <path> [more paths] [prompt]. No image path found "
            "(paths must come first; supported extensions: "
            f"{', '.join(sorted(IMAGE_EXTS))}).")
    prompt = " ".join(prompt_tokens).strip() or DEFAULT_IMAGE_PROMPT
    parts: list[dict] = [{"type": "text", "text": prompt}]
    parts += [image_part_from_file(p) for p in images]
    return HumanMessage(content=parts)


def paste_message(rest: str) -> HumanMessage:
    """Build a multimodal message from a clipboard image."""
    part = clipboard_image_part()
    if part is None:
        raise ValueError(
            "No image on the clipboard (needs wl-paste or xclip, and an image "
            "copied). Use /image <path> instead.")
    prompt = rest.strip() or DEFAULT_IMAGE_PROMPT
    return HumanMessage(content=[{"type": "text", "text": prompt}, part])


def attached_image_count(message) -> int:
    """How many images a message carries (0 for plain text)."""
    content = getattr(message, "content", None)
    if not isinstance(content, list):
        return 0
    return sum(1 for part in content
               if isinstance(part, dict) and part.get("type") == "image_url")


def parse_input(text: str):
    """Turn a REPL line into a ReplCommand, a HumanMessage, or plain text.

    Returns a :class:`ReplCommand` for local commands, a ``HumanMessage`` for
    attachments, or a ``("user", text)`` tuple for ordinary input.  Raises
    ``ValueError`` (with a user-facing message) for a malformed or unknown
    command, so the REPL can print it and carry on.
    """
    stripped = text.strip()
    command, _, rest = stripped.partition(" ")
    key = command.lower()

    if stripped.lower() in QUIT_WORDS:
        return ReplCommand("quit")
    if key in HELP_CMDS or stripped.lower() == "help":
        return ReplCommand("help")
    if key in SKILLS_CMDS:
        return ReplCommand("skills")
    if key in RELOAD_CMDS:
        return ReplCommand("reload")
    if key in IMAGE_CMDS:
        return image_message(rest)
    if key in PASTE_CMDS:
        return paste_message(rest)
    if command.startswith("/"):
        # Better than quietly sending "/imag foo.png" to the model as prose.
        raise ValueError(
            f"Unknown command: {command}. Type /help for the command list.")
    return ("user", stripped)
