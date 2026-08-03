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
    # Both roots must be isolated: listing also reads the legacy location, and
    # patching only one let this test pick up the developer's REAL backups.
    monkeypatch.setattr(S, "_legacy_backups_root", lambda: str(tmp_path / "none"))
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


def test_ls_parsing_strips_region_marker():
    # Regression: 'plate.r/R' must reduce to 'plate.r', or a destructive command
    # naming a REGION never matches the live listing and the snapshot is skipped.
    names = S._parse_ls_names("plate.r/R  body.s  sub.c/  _GLOBAL@")
    assert names == {"plate.r", "body.s", "sub.c", "_GLOBAL"}


def test_region_destructive_target_is_seen_as_live():
    live = S._parse_ls_names("plate.r/R  body.s")
    assert "plate.r" in live          # so `kill plate.r` gets snapshotted


# --- globs: the command that destroys most was protected least -------------

def test_a_glob_is_expanded_against_the_database_not_compared_to_it():
    # THE BUG: `kill *` matched no live name literally, so the snapshot logic
    # concluded "nothing to lose" and ran the single most destructive command
    # form with NO restore point behind it.
    from brlcad_mcp.server.tools.helpers import expand_targets
    live = {"plate.r", "hole.s", "body.s", "query_rayffff"}
    assert set(expand_targets(["*"], live)) == live
    assert expand_targets(["hole*"], live) == ["hole.s"]
    assert expand_targets(["*.s"], live) == ["body.s", "hole.s"]
    assert expand_targets(["nope*"], live) == []


def test_plain_names_still_have_to_exist():
    from brlcad_mcp.server.tools.helpers import expand_targets
    live = {"plate.r"}
    assert expand_targets(["plate.r", "ghost.s"], live) == ["plate.r"]


def test_expanded_targets_are_deduplicated_in_order():
    from brlcad_mcp.server.tools.helpers import expand_targets
    live = {"a.s", "b.s"}
    assert expand_targets(["a.s", "*", "a.s"], live) == ["a.s", "b.s"]


# --- a no-op reported as success -------------------------------------------

def test_a_kill_that_removed_nothing_is_flagged():
    # MGED returned SUCCESS for `kill *` while deleting nothing; the agent only
    # found out by re-running ls itself.
    note = S.describe_effect(["a.s", "b.s"], ["a.s", "b.s"])
    assert "removed NOTHING" in note and "wildcard" in note


def test_a_partial_kill_says_what_survived():
    note = S.describe_effect(["a.s", "b.s"], ["b.s"])
    assert "1 of 2" in note and "b.s" in note


def test_a_clean_kill_adds_no_noise():
    assert S.describe_effect(["a.s"], []) == ""
    assert S.describe_effect([], []) == ""


def test_only_removing_verbs_get_the_effect_check():
    # `r`/`c`/`comb`/`g` REDEFINE and `mv` renames, so the named object is still
    # expected afterwards -- checking for absence would warn on every success.
    from brlcad_mcp.server.tools.helpers import removes_objects
    assert removes_objects("kill a.s") and removes_objects("killtree a.r")
    for cmd in ("r bracket.r u a.s - h.s", "comb x u a", "mv old new",
                "rm comb member", "g grp a b"):
        assert not removes_objects(cmd), cmd
    assert S.destructive_effect_note("r x u a.s", ["x", "a.s"]) == ""


# --- restore points are not in the render cache ---------------------------

def test_backups_live_outside_the_render_directory():
    # They used to sit in render_dir/backups, so clearing a render cache deleted
    # the only way back from a bad raw edit.
    from brlcad_mcp.config import settings
    assert S._backups_root() == settings.render.backup_dir
    assert not S._backups_root().startswith(settings.render.output_dir + "/")


def test_old_restore_points_are_still_listed(tmp_path, monkeypatch):
    import json as _json
    new, old = tmp_path / "new", tmp_path / "old"
    new.mkdir(), old.mkdir()
    (new / "snap_2.json").write_text(_json.dumps(
        {"backup": "/x/snap_2.g", "objects": ["b.s"], "created": "20260803_2"}))
    (old / "snap_1.json").write_text(_json.dumps(
        {"backup": "/x/snap_1.g", "objects": ["a.s"], "created": "20260803_1"}))
    monkeypatch.setattr(S, "_backups_root", lambda: str(new))
    monkeypatch.setattr(S, "_legacy_backups_root", lambda: str(old))
    manifests = S._list_manifests()
    assert [m["objects"][0] for m in manifests] == ["b.s", "a.s"]   # newest first
