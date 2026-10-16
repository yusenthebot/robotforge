"""observe(find=...): the colored things a camera sees, located in the world, so an agent can say "pick the red
one" without knowing names.

One render gives the picture, its depth and which geom each pixel shows (render.Layers). The color finder
(vision) makes the blobs and locates each one from the picture and depth alone. What a blob is a picture of
is then read from the simulator, so an agent gets a name to pass to pick_and_place: a loose object, a robot's
part, or something fixed in the world. A loose object seen as two blobs (a highlight across it, an arm in
front) is reported once, by its biggest blob.
"""

from __future__ import annotations

import io
from typing import Any

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from robotforge.sim import vision
from robotforge.sim.render import Layers

EVERY = ("all", "any", "every", "colors")  # find= words that mean every named color
MAX_FOUND = 12  # the biggest finds reported; the rest are counted
# Measured (scripts/vision_bench.py, tests/test_find.py): a 4 cm cube half a metre away and seen from above
# came out 0.1 to 0.4 cm off; a 10 cm ball seen from a rover's low camera 2 cm short, its far half unseen.
ACCURACY = (
    "within about 0.5 cm for a small object seen from above; a camera looking from the side sees only the near "
    "half, so the middle can come out a few cm short"
)
MARK = (255, 190, 0)  # the boxes drawn round the finds


def colors(color: str) -> list[str]:
    """The colors `find=` asks for; an unknown or grey one fails here, before anything is rendered."""
    if color.strip().lower() in EVERY:
        return list(vision.NAMED)
    vision.color_range(color)  # raises ValueError naming the colors it knows
    return [color]


def find(layers: Layers, color: str, model: mujoco.MjModel) -> list[dict[str, Any]]:
    """Every blob of `color` (or of every named color) in the picture, biggest first, located and named."""
    found = [f for c in colors(color) for f in vision.find(layers.rgb, layers.depth, layers.camera, c)]
    found.sort(key=lambda f: -f.blob.area)
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for f in found:
        what = _what(f.blob, layers.geoms, model)
        loose = what.get("object")
        if loose is not None:
            if loose in seen:  # a smaller piece of an object already reported
                continue
            seen.add(loose)
        out.append(
            {
                "n": len(out) + 1,
                "color": f.color,
                **what,
                "position": [round(float(v), 3) for v in f.position],
                "size": [round(float(v), 3) for v in f.size],
                "pixels": f.blob.area,
                "at_pixel": [round(f.blob.u), round(f.blob.v)],
            }
        )
    return out


def summary(view: str, color: str, found: list[dict[str, Any]]) -> dict[str, Any]:
    """The `found` field of observe's result: the biggest MAX_FOUND finds, and how many more there were."""
    out: dict[str, Any] = {"view": view, "color": color, "objects": found[:MAX_FOUND]}
    if len(found) > MAX_FOUND:
        out["more"] = len(found) - MAX_FOUND
    out["how"] = (
        f"position: the estimated middle (m, world frame) from the picture's color and depth, {ACCURACY}. "
        "object (pick_and_place takes it), robot or fixed: what those pixels show"
    )
    if not found:
        out["hint"] = (
            "nothing of that color in view: try another view (overview, or a robot's camera from describe_model) "
            "or move the robot's camera"
        )
    return out


def marked_png(rgb: np.ndarray, found: list[dict[str, Any]]) -> bytes:
    """The picture with a numbered box round each reported find, as observe returns it."""
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    line = max(2, round(rgb.shape[1] / 320))
    for item in found[:MAX_FOUND]:
        u, v = item["at_pixel"]
        half = max(6.0, (item["pixels"] ** 0.5) * 0.6)
        draw.rectangle((u - half, v - half, u + half, v + half), outline=MARK, width=line)
        draw.text((u - half + 2, max(0.0, v - half - 12)), str(item["n"]), fill=MARK)
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _what(blob: vision.Blob, geoms: np.ndarray | None, model: mujoco.MjModel) -> dict[str, str]:
    """What most of a blob's pixels show: {"object": name} for a loose object, {"robot": name} for a part of a
    robot, {"fixed": name} for something built into the world; nothing when no one thing covers most of it."""
    if geoms is None:
        return {}
    ys, xs = blob.pixels()
    ids, counts = np.unique(geoms[ys, xs], return_counts=True)
    things: dict[tuple[str, int], int] = {}  # ("body", root body) or ("geom", geom on the world body): pixels
    for g, n in zip(ids.tolist(), counts.tolist(), strict=True):
        if g < 0:
            continue
        root = int(model.body_rootid[model.geom_bodyid[g]])  # a robot's or an object's top body
        key = ("body", root) if root else ("geom", g)
        things[key] = things.get(key, 0) + n
    if not things:
        return {}
    (kind, index), n = max(things.items(), key=lambda kv: kv[1])
    if n * 2 <= blob.area:
        return {}
    if kind == "geom":  # built into the world: a fixed prop, a wall, the floor
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index) or "floor"
        return {"fixed": name.split("/")[0]}
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, index) or f"body{index}"
    if "/" in name:
        return {"robot": name.split("/")[0]}
    joint = model.body_jntadr[index]
    free = model.body_jntnum[index] > 0 and int(model.jnt_type[joint]) == int(mujoco.mjtJoint.mjJNT_FREE)
    return {"object": name} if free else {"fixed": name}
