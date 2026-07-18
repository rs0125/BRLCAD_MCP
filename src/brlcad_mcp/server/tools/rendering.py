"""MCP tool — headless model rendering with rt.

Three lighting modes:

* ``ambient`` (default): a normal opaque render.  One rt pass, auto-framed on
  the object, with high ambient light and ambient occlusion.  Makes no changes
  to the database.
* ``studio``: a three-point, camera-RELATIVE light rig (key / fill / rim).  The
  rig rotates with the camera, so every angle is lit the same way -- ideal for
  consistent multi-angle product shots.
* ``model``: a three-point, WORLD-fixed rig.  The lights stay put relative to
  the model, so different camera angles show it lit from different sides -- a
  scene / fixed-sun look.

Both rig modes create temporary light regions in the database, render with a
locked view so the far-off lights do not shrink the model in frame, then remove
the temp lights afterwards.

``rt`` and ``pix-png`` run as subprocesses, so the server must run where
BRL-CAD's tools are available (or set ``BRLCAD_BIN`` to their directory).  The
``.g`` path is discovered from the live listener via ``opendb``, or passed
explicitly as ``db_path``.

Transparent / x-ray rendering is a separate case, layered on top of the
``studio`` lighting, and is handled by its own tool.
"""

from __future__ import annotations

import math
import os
import re
import subprocess
from pathlib import Path

from pydantic import Field

from brlcad_mcp.config import settings
from brlcad_mcp.server.app import mcp
from brlcad_mcp.server.tools.helpers import is_error_response, parse_response
from brlcad_mcp.transport import send_command

# View presets -> (azimuth, elevation) in degrees.
_VIEWS = {
    "iso": (35, 25),
    "iso2": (325, 25),
    "front": (0, 0),
    "side": (90, 0),
    "top": (0, 90),
    "rear": (180, 0),
}

# Camera-relative 3-point rig: name, (right, up, forward) offset factors,
# brightness, and shadow-ray count.  The shadow value is a RAY COUNT, not a
# flag: >1 gives soft shadows (the sphere lights have finite size), 0 = none.
# Brightness/ambient tuned via an A/B sweep ("punchy": low ambient fill so the
# directional lights dominate, keeping true material colour and clear shaping).
#
# MODEL SCALING (per mentor feedback on dark renders): the rig scales with the
# model through GEOMETRY, not brightness.  The offset factors are multiplied by
# the view size (see _rig_positions), so the lights always sit a proportional
# distance outside the model, and the light radius is a fixed fraction of the
# view size (_LIGHT_RADIUS_FRAC) so shadow softness stays consistent at any
# size.  Brightness is deliberately NOT scaled: BRL-CAD's light shader applies
# no distance attenuation (sh_light.c: sw_intensity is path attenuation only,
# no 1/r^2 term), so fixed values light a small and a large model identically
# -- verified across a ~13x size range (jeep, m35, havoc).  Scaling brightness
# by size would actually make large models over-bright relative to small ones.
_RIG = [
    ("key", (-1.0, 0.9, -0.7), 3.5, 12),
    ("fill", (1.0, 0.2, -0.5), 1.5, 0),
    ("rim", (0.1, 1.0, 0.9), 2.0, 8),
]
# Model-relative 3-point rig: same idea, but offsets are along the WORLD axes
# (X, Y, Z with Z up) instead of the camera basis.  The lights are fixed to the
# model, so rotating the camera reveals it lit from different sides (a scene /
# fixed-sun look) rather than the consistent-per-view look of the camera rig.
# Factors are (X, Y, Z), multiplied by the view size.  Front is taken as -Y.
_MODEL_RIG = [
    ("key", (-0.9, -1.0, 1.3), 3.5, 12),
    ("fill", (1.1, -0.2, 0.4), 1.5, 0),
    ("rim", (0.1, 1.2, 1.2), 2.0, 8),
]
_LIGHT_RADIUS_FRAC = 0.03  # light sphere radius as a fraction of view size
_LIGHT_PREFIX = "_rndrlight_"


def _bin(name: str) -> str:
    """Resolve a BRL-CAD tool, honoring BRLCAD_BIN when it is set."""
    d = settings.render.bin_dir
    return str(Path(d) / name) if d else name


def _subprocess_env() -> dict:
    """Environment for the rt / pix-png subprocesses.

    On a normal BRL-CAD install the binaries find their shared libraries via
    RPATH or the system linker, so nothing is needed here.  When running against
    a *build tree* (BRLCAD_BIN pointed at ``build/bin``), the libraries live in a
    sibling ``lib/`` that is not on the system linker path, so we prepend it to
    the loader search path.

    This stays release-safe: it derives the lib dir from BRLCAD_BIN rather than
    hardcoding one, only adds it when the directory actually exists, and
    *prepends* (never replaces) so any existing value is preserved.  With no
    BRLCAD_BIN set -- the usual case for an installed release where rt is on
    PATH -- it adds nothing and rt uses its own RPATH.
    """
    env = dict(os.environ)
    bin_dir = settings.render.bin_dir
    if bin_dir:
        lib = os.path.abspath(os.path.join(bin_dir, os.pardir, "lib"))
        if os.path.isdir(lib):
            # LD_LIBRARY_PATH on Linux, DYLD_LIBRARY_PATH on macOS
            for var in ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"):
                cur = env.get(var, "")
                env[var] = lib + (os.pathsep + cur if cur else "")
    return env


def _discover_db() -> str:
    """Ask the live listener which .g it currently has open."""
    try:
        resp = send_command("opendb")
    except (ConnectionError, TimeoutError):
        return ""
    if is_error_response(resp):
        return ""
    return parse_response(resp).strip()


def _frame(db: str, obj: str, az: float, el: float):
    """Pass 1: frame rt on *obj* alone and scrape the view it chose.

    Returns (viewsize, eye_pt, orientation, center) with the first three as the
    raw strings rt printed (fed straight back to rt in pass 2) and center as a
    3-tuple derived from the reported model bounds.
    """
    r = subprocess.run(
        [_bin("rt"), "-a", str(az), "-e", str(el), "-s", "16", "-o",
         os.devnull, db, *obj.split()],
        capture_output=True, text=True, timeout=120, env=_subprocess_env(),
    )
    size = eye = ori = None
    center = (0.0, 0.0, 0.0)
    for ln in r.stderr.splitlines():
        s = ln.strip()
        if s.startswith("Size:"):
            size = s.split(":", 1)[1].strip().rstrip("m")
        elif s.startswith("Eye_pos:"):
            eye = s.split(":", 1)[1].replace(",", " ").strip()
        elif s.startswith("Orientation:"):
            ori = s.split(":", 1)[1].replace(",", " ").strip()
        elif s.startswith("Model:"):
            nums = [float(n) for n in re.findall(r"-?\d+\.?\d*", s)]
            if len(nums) >= 6:
                center = ((nums[0] + nums[1]) / 2, (nums[2] + nums[3]) / 2,
                          (nums[4] + nums[5]) / 2)
    return size, eye, ori, center


def _rig_positions(eye: str, center, size: str):
    """World positions for the 3-point rig, placed relative to the camera."""
    ex, ey, ez = (float(v) for v in eye.split())
    cx, cy, cz = center
    d = float(size)  # view size: the offset factors below scale by this
    # forward: from eye toward the model center
    fx, fy, fz = cx - ex, cy - ey, cz - ez
    fl = math.hypot(fx, fy, fz) or 1.0
    fx, fy, fz = fx / fl, fy / fl, fz / fl
    # right = forward x world-up(0,0,1)
    rx, ry, rz = fy, -fx, 0.0
    rl = math.hypot(rx, ry, rz) or 1.0
    rx, ry, rz = rx / rl, ry / rl, rz / rl
    # up = right x forward
    ux = ry * fz - rz * fy
    uy = rz * fx - rx * fz
    uz = rx * fy - ry * fx
    out = []
    for name, (a, b, c), bright, sh in _RIG:
        px = cx + (a * rx + b * ux + c * fx) * d
        py = cy + (a * ry + b * uy + c * fy) * d
        pz = cz + (a * rz + b * uz + c * fz) * d
        out.append((name, px, py, pz, bright, sh))
    return out


def _rig_positions_model(center, size: str):
    """World positions for the model-relative rig (offsets along world axes).

    The lights are fixed to the model rather than the camera, so different
    camera angles show it lit from different directions.
    """
    cx, cy, cz = center
    d = float(size)  # view size: the world-axis offsets below scale by this
    out = []
    for name, (ax, ay, az), bright, sh in _MODEL_RIG:
        out.append((name, cx + ax * d, cy + ay * d, cz + az * d, bright, sh))
    return out


def _make_lights(positions, radius: float):
    """Create the temporary light regions; return their base names."""
    names = []
    send_command("units mm")
    for name, x, y, z, bright, sh in positions:
        n = _LIGHT_PREFIX + name
        send_command(f"in {n}.s sph {x:.3f} {y:.3f} {z:.3f} {radius:.3f}")
        send_command(f"r {n}.r u {n}.s")
        send_command(
            f'mater {n}.r "light bright {bright} shadows {sh} invisible 1" '
            f"255 250 240 0"
        )
        names.append(n)
    return names


def _remove_lights(names):
    """Delete the temporary light regions and their solids."""
    for n in names:
        try:
            send_command(f"kill {n}.r {n}.s")
        except (ConnectionError, TimeoutError):
            pass


@mcp.tool()
def render_model(
    obj: str = Field(
        ...,
        description="Object/assembly to render (e.g. 'all.g', 'havoc'). Space-"
        "separate to render several top objects together.",
    ),
    view: str = Field(
        default="iso",
        description="View preset: iso, iso2, front, side, top, rear. Overridden "
        "by azimuth/elevation when those are given.",
    ),
    azimuth: float | None = Field(
        default=None, description="Custom azimuth in degrees (overrides preset)."
    ),
    elevation: float | None = Field(
        default=None, description="Custom elevation in degrees (overrides preset)."
    ),
    lighting: str = Field(
        default="ambient",
        description="'ambient' = normal render (rt lighting + high ambient + "
        "ambient occlusion, no db changes). 'studio' = three-point camera-"
        "relative rig (each angle lit consistently). 'model' = three-point "
        "world-fixed rig (lights stay put, so angles are lit from different "
        "sides -- a scene / fixed-sun look).",
    ),
    size: int = Field(default=800, description="Square image resolution in pixels."),
    ambient: float | None = Field(
        default=None,
        description="Ambient light fraction (rt -A). Auto-picks 1.2 for ambient "
        "mode and 0.3 for the studio/model rigs; 1.0-1.5 gives bright, evenly-"
        "lit shots.",
    ),
    ambient_samples: int = Field(
        default=64,
        description="Ambient-occlusion samples. 0 disables; 32 is gritty, 200 "
        "smooth. Higher looks better on opaque models but is slower.",
    ),
    quality: str = Field(
        default="clean",
        description="'draft' (no anti-aliasing, fast) or 'clean' (hypersampled).",
    ),
    db_path: str = Field(
        default="",
        description="Path to the .g file to render (e.g. '~/models/ktank.g'). "
        "Required -- if you don't know it, ask the user which database/file "
        "they have open and pass it here.",
    ),
) -> str:
    """Render *obj* to a PNG with rt and return the image path.

    Does a normal opaque render.  ``lighting='ambient'`` (default) is a quick
    evenly-lit shot; ``'studio'`` is a camera-relative three-point rig (angles
    lit consistently); ``'model'`` is a world-fixed three-point rig (angles lit
    from different sides).
    """
    az, el = _VIEWS.get(view, _VIEWS["iso"])
    if azimuth is not None:
        az = azimuth
    if elevation is not None:
        el = elevation

    # NOTE: auto-discovery via `opendb` is disabled for now -- it segfaults a
    # live MGED listener (see docs/opendb-mged-crash.md).  Until that is fixed
    # (or a libmcpcad-level path query is added), the caller must pass db_path;
    # if it's missing, ask the user for it.  _discover_db() is kept for later.
    db = db_path
    if not db:
        return ("I need the path to the .g file to render. Which database are "
                "you working with? Provide it as db_path, e.g. "
                "'~/models/ktank.g'.")
    db = os.path.expanduser(db)
    if not os.path.exists(db):
        return f"Error: database file not found: {db}"

    amb = ambient if ambient is not None else (
        0.3 if lighting in ("studio", "model") else 1.2)
    out_dir = settings.render.output_dir
    os.makedirs(out_dir, exist_ok=True)
    stem = (f"{obj.split()[0].strip('/').replace('/', '_')}_"
            f"{view}_{int(az)}_{int(el)}_{lighting}")
    pix = os.path.join(out_dir, stem + ".pix")
    png = os.path.join(out_dir, stem + ".png")

    common = ["-s", str(size), "-A", str(amb)]
    if quality == "clean":
        common += ["-H", "8", "-J", "1"]
    if ambient_samples > 0:
        common += ["-c", f"set ambSamples={ambient_samples}"]

    lights: list[str] = []
    detail = ""
    try:
        if lighting in ("studio", "model"):
            vsize, eye, ori, center = _frame(db, obj, az, el)
            if not (vsize and eye and ori):
                return (f"Error: could not frame '{obj}' -- it may not be a valid "
                        f"object name. Run 'tops' to list the objects, then render "
                        f"one of those.")
            if lighting == "studio":
                positions = _rig_positions(eye, center, vsize)  # camera-relative
            else:
                positions = _rig_positions_model(center, vsize)  # world-fixed
            lights = _make_lights(
                positions,
                float(vsize) * _LIGHT_RADIUS_FRAC,  # radius scales with model
            )
            tops = [*obj.split(), *(f"{n}.r" for n in lights)]
            script = (f"viewsize {vsize};\norientation {ori};\neye_pt {eye};\n"
                      f"start 0;\nend;\n")
            r = subprocess.run(
                [_bin("rt"), "-M", *common, "-o", pix, db, *tops],
                input=script, capture_output=True, text=True, timeout=600,
                env=_subprocess_env(),
            )
            kind = "camera-relative" if lighting == "studio" else "world-fixed"
            detail = f"three-point rig ({kind}), viewsize={vsize}"
        else:
            r = subprocess.run(
                [_bin("rt"), "-a", str(az), "-e", str(el), *common, "-o", pix,
                 db, *obj.split()],
                capture_output=True, text=True, timeout=600,
                env=_subprocess_env(),
            )
            detail = "ambient + occlusion"
    except FileNotFoundError:
        return ("Error: 'rt' not found. Install BRL-CAD, or set BRLCAD_BIN to "
                "its bin directory.")
    except subprocess.TimeoutExpired:
        return "Error: render timed out. Try quality='draft' or a smaller size."
    finally:
        _remove_lights(lights)

    if not os.path.exists(pix) or os.path.getsize(pix) == 0:
        err = r.stderr or ""
        # rt found no geometry -- almost always a bad/nonexistent object name.
        if ("No primitives remaining" in err
                or "0 solids in 0 regions" in err):
            return (f"Error: '{obj}' has no drawable geometry -- it is probably "
                    f"not a valid object name. Run 'tops' to list the objects in "
                    f"this database, then render one of those.")
        return f"Error: rt produced no output.\n{err[-500:]}"

    try:
        with open(png, "wb") as fh:
            conv = subprocess.run(
                [_bin("pix-png"), "-s", str(size), pix],
                stdout=fh, stderr=subprocess.PIPE, timeout=180,
                env=_subprocess_env(),
            )
    except FileNotFoundError:
        return f"Error: 'pix-png' not found. Raw render is at {pix}."
    try:
        os.remove(pix)
    except OSError:
        pass
    if not os.path.exists(png) or os.path.getsize(png) == 0:
        return (f"Error: pix-png conversion failed.\n"
                f"{conv.stderr.decode(errors='ignore')[-300:]}")

    return (f"Rendered '{obj}' -> {png}\n"
            f"  view={view} (az={az:g}, el={el:g}), {size}px, quality={quality}\n"
            f"  lighting={lighting} ({detail}), ambient={amb:g}, "
            f"ambSamples={ambient_samples}")
