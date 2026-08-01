"""Prompt texts as editable files, not string literals in Python.

Every role prompt lives in ``definitions/<name>.md`` and is read at runtime, so
tuning what an agent is told no longer means editing code -- the same reason the
skill definitions are YAML rather than prose in a prompt.  ``/reload`` picks up an
edit in a running session, because the agents ask the library for their text at
call time instead of capturing it at import.

Files are read VERBATIM: whitespace, line breaks and blank lines are part of the
prompt.  Nothing is reflowed, unwrapped or markdown-rendered, so what the file
looks like is exactly what the model receives.  The ``.md`` extension is for the
benefit of editors; the content is plain text (``.txt`` is accepted too).

Two safety properties:

1. **A missing prompt is a startup error, not an empty prompt.** The names the
   code asks for are declared in :data:`REQUIRED`, and loading fails with the
   offending path -- an agent running on an empty system prompt would otherwise
   degrade quietly and look like a model problem.
2. **A bad edit cannot take down a running agent.** :meth:`PromptLibrary.reload`
   keeps the current texts and reports the problem, matching how the skill
   registry behaves.

Set ``CLIENT_V2_PROMPTS_DIR`` to a directory to override prompts per-file
without touching the repo: it is overlaid on the built-ins, so dropping a single
``worker.md`` in there replaces that one prompt and leaves the rest alone.
"""

from __future__ import annotations

import os

# The prompt names the code actually asks for.  A name here with no file is a
# hard error; a file with no name here is simply available via text().
REQUIRED = ("worker", "formatter", "planner", "chat", "visual")

BUILTIN_DIR = os.path.join(os.path.dirname(__file__), "definitions")
OVERRIDE_DIR_ENV = "CLIENT_V2_PROMPTS_DIR"
_SUFFIXES = (".md", ".txt")


def override_dir() -> str | None:
    """The configured override directory, if any."""
    raw = os.environ.get(OVERRIDE_DIR_ENV, "").strip()
    return os.path.expanduser(raw) if raw else None


def _read_dir(path: str | None) -> dict[str, str]:
    """``{name: text}`` for every prompt file in *path* (empty if absent)."""
    texts: dict[str, str] = {}
    if not path or not os.path.isdir(path):
        return texts
    for name in sorted(os.listdir(path)):
        stem, ext = os.path.splitext(name)
        if ext.lower() not in _SUFFIXES:
            continue
        with open(os.path.join(path, name)) as fh:
            texts[stem] = fh.read().strip()
    return texts


def _problems(texts: dict[str, str], where: str) -> list[str]:
    """Why *texts* is unusable: a required prompt missing or blank."""
    out = []
    for name in REQUIRED:
        if name not in texts:
            out.append(f"missing prompt '{name}' (expected {where}/{name}.md)")
        elif not texts[name]:
            out.append(f"prompt '{name}' is empty ({where}/{name}.md)")
    return out


class PromptLibrary:
    """Named prompt texts, reloadable in place."""

    def __init__(self, texts: dict[str, str]):
        self._texts = texts

    @classmethod
    def load(cls) -> PromptLibrary:
        """Read the built-ins, overlay any override dir, and validate."""
        texts = {**_read_dir(BUILTIN_DIR), **_read_dir(override_dir())}
        problems = _problems(texts, BUILTIN_DIR)
        if problems:
            raise ValueError("cannot load prompts: " + "; ".join(problems))
        return cls(texts)

    def text(self, name: str) -> str:
        """The prompt named *name*; raises if it was never loaded."""
        try:
            return self._texts[name]
        except KeyError:
            raise KeyError(
                f"no prompt '{name}'; loaded: {', '.join(self.names())}") from None

    def names(self) -> list[str]:
        return sorted(self._texts)

    def reload(self) -> str:
        """Re-read from disk IN PLACE; keeps current texts on a bad edit.

        Mutating rather than rebuilding is what lets a running graph pick up an
        edit: the agents hold this object, not the strings it returned.
        """
        try:
            texts = {**_read_dir(BUILTIN_DIR), **_read_dir(override_dir())}
        except OSError as exc:
            return f"prompt reload failed (keeping current): {exc}"
        problems = _problems(texts, BUILTIN_DIR)
        if problems:
            return ("prompt reload failed (keeping current): "
                    + "; ".join(problems))
        changed = [n for n, t in texts.items() if self._texts.get(n) != t]
        self._texts = texts
        status = f"reloaded {len(texts)} prompt(s)"
        return f"{status}; changed: {', '.join(sorted(changed))}" if changed \
            else f"{status}; no changes"

    def catalog(self) -> str:
        """Listing of the loaded prompts with their sizes and source."""
        source = override_dir()
        lines = [f"prompts from {BUILTIN_DIR}"]
        if source:
            lines.append(f"overridden from {source} ({OVERRIDE_DIR_ENV})")
        lines += [f"- {n} ({len(self._texts[n])} chars)" for n in self.names()]
        return "\n".join(lines)


# The singleton the agents read.  Loaded eagerly so a missing prompt fails at
# import with a clear message rather than mid-turn.
PROMPTS = PromptLibrary.load()


def resolve(prompt, name: str) -> str:
    """Prompt text from a string, a callable, or None (-> the library's *name*).

    Lets a caller pin an exact string (tests do) while production passes nothing
    and gets the live file contents on every call.
    """
    if prompt is None:
        return PROMPTS.text(name)
    if callable(prompt):
        return str(prompt())
    return str(prompt)
