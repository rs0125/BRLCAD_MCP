"""client-v2 REPL input: commands, image attachment, plain text.

Exercised against the real reference images in ~/Downloads/test_brlcad_images
when present (so the data-URI path is proven on actual JPEG/PNG files), with a
generated PNG as the fallback so the suite still runs anywhere.
"""

import base64
import os

import pytest
from langchain_core.messages import HumanMessage

from client_v2.terminal.attachments import (
    DEFAULT_IMAGE_PROMPT,
    ReplCommand,
    attached_image_count,
    image_message,
    parse_input,
)

# A minimal 1x1 PNG, for when the sample images are not on this machine.
_PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
)
_SAMPLES = os.path.expanduser("~/Downloads/test_brlcad_images")


@pytest.fixture
def image_file(tmp_path):
    """Path to a real reference image, or a generated PNG as a fallback."""
    sample = os.path.join(_SAMPLES, "Lbracket.png")
    if os.path.isfile(sample):
        return sample
    path = tmp_path / "ref.png"
    path.write_bytes(_PNG_1X1)
    return str(path)


# --- local commands -------------------------------------------------------

def test_quit_help_skills_reload_are_local_commands():
    for text, name in (("exit", "quit"), ("quit", "quit"), ("/help", "help"),
                       ("help", "help"), ("/?", "help"), ("/skills", "skills"),
                       ("/reload", "reload")):
        parsed = parse_input(text)
        assert isinstance(parsed, ReplCommand) and parsed.name == name, text


def test_plain_text_becomes_a_user_turn():
    assert parse_input("build a washer") == ("user", "build a washer")


def test_unknown_slash_command_is_reported_not_sent_to_the_model():
    with pytest.raises(ValueError, match="Unknown command"):
        parse_input("/imag foo.png")


def test_word_boundary_match_so_images_is_not_read_as_image():
    # Regression: prefix matching ate the plural's "s" and every /images failed.
    with pytest.raises(ValueError, match="Unknown command"):
        parse_input("/imageify something")


# --- image attachment -----------------------------------------------------

def test_image_command_attaches_a_real_file(image_file):
    msg = parse_input(f"/image {image_file} model this bracket")
    assert isinstance(msg, HumanMessage)
    text_part, image_part = msg.content
    assert text_part == {"type": "text", "text": "model this bracket"}
    assert image_part["image_url"]["url"].startswith("data:image/")
    assert attached_image_count(msg) == 1


def test_all_image_aliases_work(image_file):
    for cmd in ("/image", "/images", "/img", "/IMAGES"):
        assert attached_image_count(parse_input(f"{cmd} {image_file}")) == 1


def test_prompt_defaults_when_only_a_path_is_given(image_file):
    msg = parse_input(f"/image {image_file}")
    assert msg.content[0] == {"type": "text", "text": DEFAULT_IMAGE_PROMPT}


def test_several_images_attach_together(tmp_path, image_file):
    second = tmp_path / "side.png"
    second.write_bytes(_PNG_1X1)
    msg = parse_input(f"/image {image_file} {second} front and side")
    assert attached_image_count(msg) == 2
    assert msg.content[0] == {"type": "text", "text": "front and side"}


def test_missing_file_names_the_path_it_tried():
    with pytest.raises(ValueError, match="Could not find that image file"):
        parse_input("/image Downloads/nope.jpg draw this")


def test_no_path_at_all_shows_usage():
    with pytest.raises(ValueError, match="No image path found"):
        parse_input("/image just some words")


def test_attached_image_count_is_zero_for_plain_messages():
    assert attached_image_count(HumanMessage(content="hello")) == 0
    assert attached_image_count(None) == 0


@pytest.mark.skipif(not os.path.isdir(_SAMPLES),
                    reason="sample reference images not on this machine")
def test_every_sample_reference_image_can_be_attached():
    # The JPEGs matter: mime detection must not hard-code image/png.
    names = [n for n in sorted(os.listdir(_SAMPLES))
             if n.lower().endswith((".png", ".jpg", ".jpeg"))]
    assert names, "expected sample images"
    for name in names:
        msg = image_message(os.path.join(_SAMPLES, name))
        url = msg.content[1]["image_url"]["url"]
        expected = "data:image/jpeg" if name.lower().endswith(
            (".jpg", ".jpeg")) else "data:image/png"
        assert url.startswith(expected), name
