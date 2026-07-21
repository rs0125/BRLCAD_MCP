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
import subprocess
import time
from pathlib import Path

from pydantic import Field

from brlcad_mcp.config import settings
from brlcad_mcp.server.app import mcp
from brlcad_mcp.server.tools.helpers import is_error_response, parse_response
from brlcad_mcp.transport import send_command

# View presets -> (azimuth, elevation) in degrees.  iso/iso2 are the two
# front-quarter isometrics; iso_back/iso_back2 are their rear-quarter mirrors
# (azimuth 180 +/- 35) for back three-quarter shots.
_VIEWS = {
    "iso": (35, 25),
    "iso2": (325, 25),
    "iso_back": (215, 25),
    "iso_back2": (145, 25),
    "front": (0, 0),
    "back": (180, 0),
    "side": (90, 0),
    "side2": (270, 0),
    "top": (0, 90),
    "bottom": (0, -90),
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
    ("key", (-1.0, 0.9, -0.7), 4.5, 12),
    ("fill", (1.0, 0.2, -0.5), 2.2, 0),
    ("rim", (0.1, 1.0, 0.9), 2.8, 8),
]
# Model-relative 3-point rig: same idea, but offsets are along the WORLD axes
# (X, Y, Z with Z up) instead of the camera basis.  The lights are fixed to the
# model, so rotating the camera reveals it lit from different sides (a scene /
# fixed-sun look) rather than the consistent-per-view look of the camera rig.
# Factors are (X, Y, Z), multiplied by the view size.  Front is taken as -Y.
# The default views (iso/iso2, ae 35/325 el 25) put the camera in the +X/+Z half
# looking at the +Y and -Y faces, so the key sits front-right-high (+X,+Y,+Z) to
# actually light what those views see, the fill covers the -Y (iso2) side, and
# the rim comes over the top-back for separation.  Brightnesses are a touch hot
# so the fixed-sun look reads without washing the material colour.
_MODEL_RIG = [
    ("key", (0.9, 0.8, 1.2), 4.0, 12),
    ("fill", (0.6, -1.0, 0.4), 2.2, 0),
    ("rim", (-0.9, 0.4, 1.5), 2.5, 8),
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


def _rt_opts(size, amb, ambient_samples, quality):
    """Build the rt option string shared by both ged_rt render paths."""
    opts = f"-s {size} -A {amb:g}"
    if quality == "clean":
        opts += " -H 8 -J 1"
    if ambient_samples > 0:
        opts += f' -c "set ambSamples={ambient_samples}"'
    return opts


def _ged_rt_and_wait(opts, pix, size):
    """Fire `rt` over the socket and wait for the .pix to finish.

    ged_rt is async -- it launches rt and returns immediately -- so we poll for
    the file to reach its full size (W*H*3 bytes).  Returns None on success or an
    error string.
    """
    resp = send_command(f"rt {opts} -o {pix}")
    if is_error_response(resp):
        return f"Error: ged_rt failed: {parse_response(resp)}"
    expected = size * size * 3
    deadline = time.time() + 600
    while time.time() < deadline:
        if os.path.exists(pix) and os.path.getsize(pix) >= expected:
            return None
        time.sleep(0.1)
    return "Error: ged_rt render timed out (no complete .pix produced)."


def _draw_objects(obj):
    """Draw each top object; return an error string on the first bad name."""
    for o in obj.split():
        resp = send_command(f"draw {o}")
        if is_error_response(resp):
            return (f"Error drawing '{o}': {parse_response(resp)} -- is it a "
                    f"valid object? Run 'tops' to list the objects.")
    return None


def _view_params():
    """Read the listener's current view over the socket.

    Returns (viewsize_str, eye_str, center_tuple) in mm, or None if any getter
    returns something unparseable.  `size` is the view diagonal, `eye_pt` the
    camera position, `center` the view center -- exactly the inputs the light rig
    needs, so we no longer scrape them from a subprocess rt.
    """
    vsize = parse_response(send_command("size")).strip()
    eye = parse_response(send_command("eye_pt")).strip()
    cen = parse_response(send_command("center")).strip().split()
    try:
        center = tuple(float(v) for v in cen[:3])
        float(vsize)
        [float(v) for v in eye.split()]
    except (ValueError, IndexError):
        return None
    if len(center) != 3 or len(eye.split()) != 3:
        return None
    return vsize, eye, center


def _render_via_ged_rt(obj, az, el, size, amb, ambient_samples, quality, pix):
    """Ambient render over the socket with the ged `rt` command.

    ged_rt renders the listener's OWN open database (it uses gedp->dbip), so we
    need NO db path and never send `opendb` -- which sidesteps the MGED opendb
    crash entirely.  It renders the *displayed* objects, so we draw first, set
    the view, then rt.  Returns None on success or an error string.

    Side effect: this draws `obj` and changes the listener's current view
    (ae/autoview) -- visible in a live MGED.
    """
    try:
        os.remove(pix)  # clear any stale file so the poll is meaningful
    except OSError:
        pass
    err = _draw_objects(obj)
    if err:
        return err
    send_command(f"ae {az:g} {el:g}")
    send_command("autoview")
    return _ged_rt_and_wait(
        _rt_opts(size, amb, ambient_samples, quality), pix, size)


def _render_studio_via_ged_rt(obj, az, el, size, amb, ambient_samples,
                              quality, lighting, pix):
    """Studio/model render over the socket -- no db path.

    Same ged_rt idea as ambient, but we first build the three-point light rig on
    the live database: draw the object, set the view, read the view params over
    the socket (size/eye_pt/center), place the rig relative to that view, draw
    the (invisible) lights, then rt.  The lights are killed afterwards so the db
    is left as we found it apart from the object staying drawn.
    """
    try:
        os.remove(pix)
    except OSError:
        pass
    err = _draw_objects(obj)
    if err:
        return err
    send_command("units mm")  # rig math + light geometry are all in mm
    send_command(f"ae {az:g} {el:g}")
    send_command("autoview")
    view = _view_params()
    if view is None:
        return ("Error: could not read the view parameters over the socket "
                "(size/eye_pt/center). Is a model open in the listener?")
    vsize, eye, center = view
    positions = (_rig_positions(eye, center, vsize) if lighting == "studio"
                 else _rig_positions_model(center, vsize))
    lights = _make_lights(positions, float(vsize) * _LIGHT_RADIUS_FRAC)
    try:
        for n in lights:
            send_command(f"draw {n}.r")
        return _ged_rt_and_wait(
            _rt_opts(size, amb, ambient_samples, quality), pix, size)
    finally:
        _remove_lights(lights)


@mcp.tool()
def render_model(
    obj: str = Field(
        ...,
        description="Object/assembly to render (e.g. 'all.g', 'havoc'). Space-"
        "separate to render several top objects together.",
    ),
    view: str = Field(
        default="iso",
        description="View preset: iso, iso2 (front-quarter isometrics), iso_back, "
        "iso_back2 (rear-quarter isometrics), front, back, side, side2, top, "
        "bottom. For angles with no preset, pass azimuth/elevation instead "
        "(they override this). To render several distinct views, give each a "
        "different preset or azimuth/elevation -- repeating the same view "
        "overwrites the same file.",
    ),
    azimuth: float | None = Field(
        default=None, description="Custom azimuth in degrees (overrides preset)."
    ),
    elevation: float | None = Field(
        default=None, description="Custom elevation in degrees (overrides preset)."
    ),
    lighting: str = Field(
        default="studio",
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
        "mode and 1.0 for the studio/model rigs; 1.0-1.5 gives bright, evenly-"
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
) -> str:
    """Render *obj* to a PNG with rt and return the image path.

    Renders the model the listener currently has open, over the socket via the
    ged ``rt`` command -- no file path is needed, so do NOT ask the user for one.
    ``lighting='studio'`` (default) is a camera-relative three-point rig, so every
    angle (iso, iso2, side...) is lit consistently; ``'model'`` is a world-fixed
    three-point rig (lights stay put, so angles are lit from different sides -- a
    fixed-sun look); ``'ambient'`` is a quick evenly-lit shot.
    """
    az, el = _VIEWS.get(view, _VIEWS["iso"])
    if azimuth is not None:
        az = azimuth
    if elevation is not None:
        el = elevation

    amb = ambient if ambient is not None else (
        1.0 if lighting in ("studio", "model") else 1.2)
    out_dir = settings.render.output_dir
    os.makedirs(out_dir, exist_ok=True)
    stem = (f"{obj.split()[0].strip('/').replace('/', '_')}_"
            f"{view}_{int(az)}_{int(el)}_{lighting}")
    pix = os.path.join(out_dir, stem + ".pix")
    png = os.path.join(out_dir, stem + ".png")

    # Everything renders over the socket via ged_rt on the listener's own open
    # db -- no db path, no `opendb` (which sidesteps the MGED opendb crash).
    if lighting == "ambient":
        err = _render_via_ged_rt(
            obj, az, el, size, amb, ambient_samples, quality, pix)
        detail = "ambient + occlusion (via ged_rt over the socket)"
    elif lighting in ("studio", "model"):
        err = _render_studio_via_ged_rt(
            obj, az, el, size, amb, ambient_samples, quality, lighting, pix)
        kind = "camera-relative" if lighting == "studio" else "world-fixed"
        detail = f"three-point rig ({kind}), via ged_rt over the socket"
    else:
        return (f"Error: unknown lighting mode '{lighting}'. "
                f"Use 'ambient', 'studio', or 'model'.")
    if err:
        return err

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
