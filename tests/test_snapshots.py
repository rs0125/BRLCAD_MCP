"""Tests for the destructive-command snapshot/restore logic.

These cover the pure pieces -- which commands are destructive, which tokens
are candidate object names, ls parsing, and manifest listing order -- without
a live socket or a running listener.
"""

from brlcad_mcp.server.tools import snapshots as S
from brlcad_mcp.server.tools.helpers import destructive_targets


def test_non_destructive_commands_have_no_targets():
    for cmd in ("ls", "l bracket.r", "draw thing.s", "in a.s sph 0 0 0 5",
                "bb widget.r", "search . -type sph"):
        assert destructive_targets(cmd) == [], cmd


def test_kill_targets_are_the_named_objects():
    assert destructive_targets("kill hole_x.s hole_y.s") == ["hole_x.s",
                                                             "hole_y.s"]
    assert destructive_targets("killall stud.s") == ["stud.s"]


def test_region_redefine_skips_operators_and_numbers():
    # 'r' redefines a region; u/-/+ are set operators, not object names.
    targets = destructive_targets("r bracket.r u plate.s - hole.s")
    assert targets == ["bracket.r", "plate.s", "hole.s"]


def test_flags_and_numbers_are_dropped():
    # mv/rm with a flag and a rcc-like numeric tail: only real names survive.
    assert destructive_targets("mv -f old.s new.s") == ["old.s", "new.s"]
    assert destructive_targets("kill a.s a.s") == ["a.s"]  # de-duplicated


def test_ls_name_parsing_strips_decorations():
    out = "bracket.r/ plate.s hole.s _GLOBAL@ sub*"
    assert S._parse_ls_names(out) == {"bracket.r", "plate.s", "hole.s",
                                      "_GLOBAL", "sub"}


def test_manifest_listing_is_newest_first(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "_backups_root", lambda: str(tmp_path))
    # Backup filenames embed a sortable timestamp; newer name sorts later, so
    # listing reverses to put newest first.
    for stamp, obj in (("20260101_000000_000001", "old.s"),
                       ("20260101_000000_000002", "new.s")):
        g = str(tmp_path / f"snap_{stamp}.g")
        (tmp_path / f"snap_{stamp}.g").write_text("")  # dummy .g
        S._write_manifest(g, f"kill {obj}", [obj], stamp)
    manifests = S._list_manifests()
    assert [m["objects"][0] for m in manifests] == ["new.s", "old.s"]


def test_sidecar_path_pairs_with_g_file():
    assert S._sidecar_path("/x/snap_1.g") == "/x/snap_1.json"
