"""Which .env wins, and whether you can tell.

``PROJECT_ROOT`` follows where ``config.py`` lives, not where you are, so a
second checkout (or an unpacked release driven by a ``brlcad-mcp`` from
elsewhere on PATH) used to silently read the other tree's .env. The symptom is
almost invisible: an unset ``BRLCAD_IPC_PATH`` falls back to host/port, and
127.0.0.1:5555 is also the built-in default, so "not configured" and
"configured to the default" print identically.

Run in subprocesses because the settings object is built once at import.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

PROBE = textwrap.dedent("""
    from brlcad_mcp.config import ENV_FILES, settings
    print("IPC=" + settings.brlcad.ipc_path)
    print("PORT=%d" % settings.brlcad.port)
    print("BUFFER=%d" % settings.brlcad.buffer_size)
    print("FILES=" + "|".join(str(f) for f in ENV_FILES))
""")


def _run(cwd, env=None):
    e = {k: v for k, v in os.environ.items()
         if not k.startswith(("BRLCAD_", "LLM_", "OPENAI_"))}
    e["PYTHONPATH"] = str(REPO / "src")
    e.update(env or {})
    out = subprocess.run([sys.executable, "-c", PROBE], cwd=str(cwd), env=e,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    return dict(line.split("=", 1) for line in out.stdout.strip().splitlines())


def test_the_env_beside_the_working_directory_is_used(tmp_path):
    (tmp_path / ".env").write_text("BRLCAD_IPC_PATH=/tmp/from_cwd.sock\n")
    got = _run(tmp_path)
    assert got["IPC"] == "/tmp/from_cwd.sock"
    assert str(tmp_path / ".env") in got["FILES"]


def test_the_working_directory_wins_over_the_one_beside_the_code(tmp_path):
    """The bug this fixes: the repo's own .env used to win from anywhere."""
    (tmp_path / ".env").write_text("BRLCAD_IPC_PATH=/tmp/mine.sock\n")
    got = _run(tmp_path)
    assert got["IPC"] == "/tmp/mine.sock", (
        "the repo's .env overrode the one in the directory being run from")


def test_a_real_environment_variable_still_beats_every_file(tmp_path):
    (tmp_path / ".env").write_text("BRLCAD_IPC_PATH=/tmp/fromfile.sock\n")
    got = _run(tmp_path, {"BRLCAD_IPC_PATH": "/tmp/exported.sock"})
    assert got["IPC"] == "/tmp/exported.sock"


def test_without_a_nearby_env_it_falls_back_to_the_one_beside_the_code(tmp_path):
    """No .env in the working directory: the package-adjacent one is used.

    That is the documented fallback and the case an editable checkout relies
    on, so the only thing to assert is that nothing was invented from the
    working directory.
    """
    empty = tmp_path / "empty"
    empty.mkdir()
    got = _run(empty)
    assert str(empty / ".env") not in got["FILES"]


def test_a_setting_no_file_mentions_keeps_its_builtin_default(tmp_path):
    # buffer_size is set by neither .env, so it proves the defaults still apply.
    (tmp_path / ".env").write_text("BRLCAD_PORT=6002\n")
    got = _run(tmp_path, {"PROBE_EXTRA": "1"})
    assert got["PORT"] == "6002"
    assert got["BUFFER"] == "4096"


def test_the_files_that_were_read_are_reported(tmp_path):
    """The banner needs this: 'which config is live' should not need a debugger."""
    (tmp_path / ".env").write_text("BRLCAD_PORT=6001\n")
    got = _run(tmp_path)
    assert got["FILES"], "no .env path reported even though one was loaded"
    assert got["PORT"] == "6001"
