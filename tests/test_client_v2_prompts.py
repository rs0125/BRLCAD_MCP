"""client-v2 prompt library: file-backed prompts, hot reload, and overrides."""

from pathlib import Path

import pytest

from client_v2.prompts import PROMPTS, REQUIRED, PromptLibrary, resolve
from client_v2.prompts import library as L
from client_v2.skills import SkillDef, SkillRegistry
from client_v2.skills.middleware import compose_worker_prompt


def _write_all(directory, **overrides):
    """A complete set of prompt files in *directory*."""
    directory.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED:
        text = overrides.get(name, f"{name} prompt body")
        (directory / f"{name}.md").write_text(text)
    return directory


# --- the shipped prompts ---------------------------------------------------

def test_every_required_prompt_ships_and_is_non_empty():
    # The point of the guard: an agent must never run on an empty system prompt.
    for name in REQUIRED:
        assert PROMPTS.text(name).strip(), name


def test_names_lists_the_loaded_prompts():
    assert set(REQUIRED) <= set(PROMPTS.names())


def test_unknown_prompt_names_what_is_loaded():
    with pytest.raises(KeyError) as exc:
        PROMPTS.text("nope")
    assert "worker" in str(exc.value)      # tells you the real options


def test_the_prompts_are_files_on_disk_not_literals():
    # Editing the file is the supported way to change a prompt, so the shipped
    # text must actually come from definitions/.
    for name in REQUIRED:
        on_disk = Path(L.BUILTIN_DIR, f"{name}.md").read_text().strip()
        assert on_disk == PROMPTS.text(name), name


# --- loading rules ---------------------------------------------------------

def test_a_missing_required_prompt_fails_loudly_with_its_path(tmp_path, monkeypatch):
    _write_all(tmp_path)
    (tmp_path / "worker.md").unlink()
    monkeypatch.setattr(L, "BUILTIN_DIR", str(tmp_path))
    with pytest.raises(ValueError) as exc:
        PromptLibrary.load()
    assert "worker" in str(exc.value) and "worker.md" in str(exc.value)


def test_a_blank_prompt_file_counts_as_missing(tmp_path, monkeypatch):
    _write_all(tmp_path, formatter="   \n  ")
    monkeypatch.setattr(L, "BUILTIN_DIR", str(tmp_path))
    with pytest.raises(ValueError) as exc:
        PromptLibrary.load()
    assert "formatter" in str(exc.value) and "empty" in str(exc.value)


def test_txt_files_load_too(tmp_path, monkeypatch):
    _write_all(tmp_path)
    (tmp_path / "worker.md").unlink()
    (tmp_path / "worker.txt").write_text("plain text worker")
    monkeypatch.setattr(L, "BUILTIN_DIR", str(tmp_path))
    assert PromptLibrary.load().text("worker") == "plain text worker"


def test_whitespace_inside_a_prompt_is_preserved(tmp_path, monkeypatch):
    # Line breaks and blank lines are part of the prompt; nothing is reflowed.
    body = "line one\n\n  indented two\nline three"
    _write_all(tmp_path, worker=f"\n{body}\n")
    monkeypatch.setattr(L, "BUILTIN_DIR", str(tmp_path))
    assert PromptLibrary.load().text("worker") == body


# --- hot reload ------------------------------------------------------------

def test_reload_picks_up_an_edit_in_place(tmp_path, monkeypatch):
    _write_all(tmp_path)
    monkeypatch.setattr(L, "BUILTIN_DIR", str(tmp_path))
    lib = PromptLibrary.load()
    assert lib.text("worker") == "worker prompt body"

    (tmp_path / "worker.md").write_text("EDITED worker")
    status = lib.reload()

    assert lib.text("worker") == "EDITED worker"   # same object, new text
    assert "changed: worker" in status


def test_reload_reports_no_changes_when_nothing_moved(tmp_path, monkeypatch):
    _write_all(tmp_path)
    monkeypatch.setattr(L, "BUILTIN_DIR", str(tmp_path))
    lib = PromptLibrary.load()
    assert "no changes" in lib.reload()


def test_a_bad_edit_cannot_take_down_a_running_agent(tmp_path, monkeypatch):
    # Emptying a prompt file mid-session must keep the last good text, the same
    # way a malformed skill YAML does.
    _write_all(tmp_path)
    monkeypatch.setattr(L, "BUILTIN_DIR", str(tmp_path))
    lib = PromptLibrary.load()

    (tmp_path / "planner.md").write_text("")
    status = lib.reload()

    assert "failed" in status and "planner" in status
    assert lib.text("planner") == "planner prompt body"    # kept


# --- the override directory -----------------------------------------------

def test_override_dir_replaces_one_prompt_and_leaves_the_rest(tmp_path, monkeypatch):
    builtin = _write_all(tmp_path / "builtin", worker="builtin worker")
    override = tmp_path / "override"
    override.mkdir()
    (override / "worker.md").write_text("my worker")

    monkeypatch.setattr(L, "BUILTIN_DIR", str(builtin))
    monkeypatch.setenv(L.OVERRIDE_DIR_ENV, str(override))

    lib = PromptLibrary.load()
    assert lib.text("worker") == "my worker"                 # overridden
    assert lib.text("formatter") == "formatter prompt body"  # still built-in
    assert str(override) in lib.catalog()


def test_a_missing_override_dir_is_simply_ignored(tmp_path, monkeypatch):
    _write_all(tmp_path)
    monkeypatch.setattr(L, "BUILTIN_DIR", str(tmp_path))
    monkeypatch.setenv(L.OVERRIDE_DIR_ENV, str(tmp_path / "nope"))
    assert PromptLibrary.load().text("worker") == "worker prompt body"


# --- resolve() and the worker path ----------------------------------------

def test_resolve_handles_text_callable_and_default():
    assert resolve("fixed", "worker") == "fixed"
    assert resolve(lambda: "late", "worker") == "late"
    assert resolve(None, "worker") == PROMPTS.text("worker")


def _registry():
    return SkillRegistry({"s": SkillDef(id="s", description="a skill")})


def test_the_worker_prompt_defaults_to_the_library():
    prompt = compose_worker_prompt(None, _registry())
    assert PROMPTS.text("worker").splitlines()[0] in prompt


def test_an_explicit_worker_prompt_still_wins():
    # Tests pin an exact string; that must not start reading files.
    assert compose_worker_prompt("BASE", _registry()).startswith("BASE")


def test_an_edited_worker_prompt_reaches_the_worker_without_a_restart(
        tmp_path, monkeypatch):
    # THE POINT of resolving per call: the middleware composes the prompt on
    # every model call, so a /reload mid-session changes what the worker is told.
    _write_all(tmp_path, worker="ORIGINAL worker")
    monkeypatch.setattr(L, "BUILTIN_DIR", str(tmp_path))
    lib = PromptLibrary.load()
    monkeypatch.setattr(L, "PROMPTS", lib)

    assert "ORIGINAL worker" in compose_worker_prompt(None, _registry())
    (tmp_path / "worker.md").write_text("REVISED worker")
    lib.reload()
    assert "REVISED worker" in compose_worker_prompt(None, _registry())
