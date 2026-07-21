"""Tests for the CLI client's input handling (/image, /paste, plain text).

These exercise _build_message without a live agent or API key -- they only
check how a line of REPL input becomes an agent message.
"""

import base64

import pytest
from langchain_core.messages import HumanMessage

from brlcad_mcp.client import agent as A

# A minimal 1x1 PNG.
_PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
)


def test_plain_text_is_a_user_tuple():
    assert A._build_message("make a sphere") == ("user", "make a sphere")


def test_image_command_builds_multimodal_message(tmp_path):
    p = tmp_path / "ref.png"
    p.write_bytes(_PNG_1X1)
    msg = A._build_message(f"/image {p} design this")
    assert isinstance(msg, HumanMessage)
    text_part, image_part = msg.content
    assert text_part == {"type": "text", "text": "design this"}
    assert image_part["type"] == "image_url"
    assert image_part["image_url"]["url"].startswith("data:image/png;base64,")


def test_image_command_defaults_prompt_when_none_given(tmp_path):
    p = tmp_path / "ref.png"
    p.write_bytes(_PNG_1X1)
    msg = A._build_message(f"/image {p}")
    assert msg.content[0]["text"]  # non-empty default prompt
    assert len(msg.content) == 2  # text + one image


def test_image_command_accepts_multiple_files(tmp_path):
    a, b = tmp_path / "front.png", tmp_path / "side.png"
    a.write_bytes(_PNG_1X1)
    b.write_bytes(_PNG_1X1)
    msg = A._build_message(f"/image {a} {b} build it")
    assert msg.content[0] == {"type": "text", "text": "build it"}
    assert sum(part["type"] == "image_url" for part in msg.content) == 2


def test_image_command_without_valid_file_raises():
    with pytest.raises(ValueError):
        A._build_message("/image /no/such/file.png")
