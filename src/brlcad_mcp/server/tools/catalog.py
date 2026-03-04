"""Static MGED command catalog — categorised commands with one-liner descriptions.

This serves as a *fast, always-available* reference for the agent.  The
``list_commands`` tool can also query MGED's ``?`` command at runtime and
merge the results, but the static catalog gives better descriptions and
category grouping than MGED's raw output.
"""

from __future__ import annotations

COMMAND_CATALOG: dict[str, dict[str, str]] = {
    "primitives": {
        "in": "Create a primitive shape interactively (e.g. sph, rcc, rpp, tgc, tor, ell, arb8 …)",
        "make": "Create a primitive with default parameters, specifying only type and name",
        "bb": "Create a bounding box (RPP) around existing objects",
    },
    "booleans_and_combinations": {
        "r": "Create or extend a region (boolean combination that is renderable)",
        "c": "Create or extend a combination (group of objects with boolean ops)",
        "comb": "Low-level combination create / modify",
        "g": "Create a group (union-only combination) of objects",
        "i": "Add an instance (member) of an object to a combination",
        "rm": "Remove a member from a combination",
    },
    "object_management": {
        "ls": "List objects in the database (supports glob patterns and flags)",
        "l": "Display detailed information about an object (type, params, tree)",
        "cp": "Copy an object to a new name",
        "mv": "Rename / move an object",
        "mvall": "Rename an object and update all references to it",
        "kill": "Delete object(s) from the database",
        "killall": "Delete an object and all combinations referencing it",
        "killtree": "Delete an object and its entire subtree",
        "tops": "List top-level (un-referenced) objects",
        "find": "Find all references to an object in the database",
        "which": "Show which regions/combinations reference a given object",
        "paths": "Show all paths from top-level to a given object",
        "search": "Search the database with filters (type, name, attributes …)",
        "dbfind": "Find all paths containing a given object",
    },
    "display": {
        "draw": "Draw / display an object in the graphics window (alias: e)",
        "e": "Draw / display an object (alias for draw)",
        "erase": "Remove an object from the display (alias: d)",
        "d": "Remove an object from the display (alias for erase)",
        "B": "Blast — clear the display then draw the specified objects",
        "Z": "Zap — clear everything from the display",
        "who": "List all currently displayed objects",
        "autoview": "Automatically fit the view to show all displayed objects",
        "refresh": "Force a display refresh",
    },
    "view_controls": {
        "ae": "Set the azimuth and elevation of the view",
        "zoom": "Zoom the view in or out by a factor",
        "center": "Set the center point of the view",
        "size": "Set the view size (diameter of the visible area)",
        "eye_pt": "Set the eye point for perspective view",
        "lookat": "Point the view at specific coordinates",
        "viewsize": "Get or set the view size",
        "perspective": "Set perspective angle (0 = orthographic)",
        "knob": "Set view rotation/translation knobs",
        "sv": "Set view parameters from a predefined orientation (front, top, right …)",
    },
    "editing_and_transforms": {
        "oed": "Enter object-edit mode for a combination member",
        "sed": "Enter solid-edit mode for a primitive",
        "tra": "Translate (move) the currently edited object",
        "rot": "Rotate the currently edited object",
        "sca": "Scale the currently edited object",
        "oscale": "Scale an object non-interactively",
        "orotate": "Rotate an object non-interactively",
        "otranslate": "Translate an object non-interactively",
        "accept": "Accept current edits and exit edit mode",
        "reject": "Reject (discard) current edits and exit edit mode",
        "keypoint": "Set the keypoint for editing transformations",
        "mirface": "Mirror a face of an ARB",
        "permute": "Reorder (permute) the vertices of an ARB",
        "extrude": "Extrude a face of an ARB",
        "arced": "Edit combination arc (transformation matrix) directly",
        "push": "Push transformation matrices down to leaves",
        "xpush": "Push all matrices to leaves, resolving shared instances",
        "pull": "Pull transformations up from leaves to arcs",
    },
    "raytracing_and_analysis": {
        "rt": "Raytrace the current view to produce a rendered image",
        "rtcheck": "Check for overlapping regions via raytracing",
        "rtarea": "Compute presented and exposed surface areas via raytracing",
        "rtweight": "Compute weight/volume of objects via raytracing",
        "rtedge": "Raytrace producing an edge-only (line drawing) image",
        "nirt": "Interactive ray query — fire a ray and report intersections",
        "gqa": "Geometry quality analysis (weight, volume, overlaps, gaps)",
    },
    "database_and_file": {
        "opendb": "Open a .g database file",
        "title": "Get or set the database title",
        "units": "Get or set the working units (mm, cm, m, in, ft …)",
        "summary": "Show a summary of the database (counts by type)",
        "keep": "Copy objects into a separate .g file",
        "dbconcat": "Concatenate another .g database into the current one",
        "dup": "Check for duplicate names between current db and another .g",
        "attr": "Get, set, or list attributes on an object",
        "put": "Low-level: write a serialised object definition into the db",
        "get": "Low-level: retrieve the serialised definition of an object",
    },
    "appearance_and_materials": {
        "shader": "Set the shader (material appearance) on a region",
        "mater": "Set material properties (shader, color, inherit) on a region",
        "color": "Set the display color of an object",
        "edcolor": "Edit the color table interactively",
        "edcodes": "Edit region ID codes (region_id, aircode, material, LOS)",
        "rcodes": "Set region codes for multiple regions at once",
        "regdef": "Set default region ID for the next region created",
    },
    "utility": {
        "?": "List all available MGED commands",
        "help": "Show usage / man-page text for a specific command",
        "apropos": "Search command descriptions by keyword",
        "history": "Show recent command history",
        "status": "Show MGED status information",
        "dump": "Build a text dump (Tcl script) that recreates all geometry",
        "preview": "Preview an animation script",
        "rrt": "Run an external rt-family program with custom arguments",
        "saveview": "Save current view settings to a file for later rt usage",
        "loadview": "Load a saved view file",
    },
}

# Flattened set of all known command names (for quick membership checks).
ALL_KNOWN: set[str] = {
    cmd for cat in COMMAND_CATALOG.values() for cmd in cat
}


def categories_text(category: str | None = None) -> str:
    """Return a formatted text block of the command catalog."""
    lines: list[str] = []
    for cat_name, commands in COMMAND_CATALOG.items():
        if category and category.lower() != cat_name.lower():
            continue
        lines.append(f"\n=== {cat_name.upper().replace('_', ' ')} ===")
        for cmd, desc in commands.items():
            lines.append(f"  {cmd:16s} — {desc}")
    if not lines:
        return f"No category matching '{category}'."
    return "\n".join(lines)
