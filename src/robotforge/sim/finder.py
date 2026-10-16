"""observe(find=...): the colored things a camera sees, located in the world, so an agent can say "pick the red
one" without knowing names.

One render gives the picture, its depth and which geom each pixel shows (render.Layers). The color finder
(vision) makes the blobs and locates each one from the picture and depth alone. What a blob is a picture of
is then read from the simulator, so an agent gets a name to pass to pick_and_place: a loose object, a robot's
part, or something fixed in the world. Each of them is reported once, by its biggest blob (a highlight across
an object or an arm in front of it can split it in two), and loose objects come first, so a floor or a shelf
found as orange never pushes the apple off the list.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from robotforge.sim import vision
from robotforge.sim.render import Layers

EVERY = ("all", "any", "every", "colors")  # find= words that mean every named color
MAX_FOUND = 12  # finds reported, loose objects first; the rest are counted
KINDS = ("object", "robot", "fixed")  # the order finds are listed in: what can be picked comes first
# Measured with scripts/vision_bench.py --scenes (every ready-made scene, from every preset view and robot camera:
# 115 finds of loose objects). The 56 with no flag were 0.1 to 3.9 cm off, 91 % of them within about_cm and 96 %
# within twice it; flagged ones were up to 66 cm off (a pallet seen through the gaps between the crates on it).
# A red cube on a blue one at 12 spots (--stacks): 62 of its 64 finds with no flag were within about_cm, at most
# 1.1 cm off; from straight above it is flagged. Carried in the arm's gripper at 12 spots, each of its 72 finds
# said held_by (1.4 to 4.5 cm off: the gripper covers part of it).
HOW = (
    "position: the estimated middle (m, world frame) from the picture's color and depth. about_cm: how far off it "
    "may be; 9 in 10 finds with no flag were within it (from low down a ball comes out a little short). object "
    "(pick_and_place takes it), robot or fixed: what those pixels show"
)
# What each flag on a find means; observe's "how" explains only the ones its finds carry, to keep it short. Any of
# them: it may be much further off than about_cm.
DOUBTS = {
    "cut_off": "the picture's edge cuts it",
    "hidden": "something nearer covers part of it",
    "side_on": "only a face is seen; its middle is about half its depth further away",
    "raised": "it may stand on something under it that this view cannot show; then it is higher",
    "held_by": "that robot holds it, and its hand may cover part of it",
    "merged": "loose objects of its color touching it, found as one; the position is the middle of them all",
}
MARK = (255, 190, 0)  # the boxes drawn round the finds
FLAGS = ("cut_off", "hidden", "side_on", "raised")  # vision.Found's doubts, passed on when they hold
MERGED_SHARE = 0.03  # of a blob's pixels (and MIN_AREA at least): another loose object this big in it is part of it


@dataclass(frozen=True)
class Seen:
    """One find as observe reports it, and where its blob is in the picture (for the boxes drawn on it)."""

    entry: dict[str, Any]
    box: tuple[int, int, int, int]  # x0, y0, x1, y1, px


def colors(color: str) -> list[str]:
    """The colors `find=` asks for; an unknown or grey one fails here, before anything is rendered."""
    if color.strip().lower() in EVERY:
        return list(vision.NAMED)
    vision.color_range(color)  # raises ValueError naming the colors it knows
    return [color]


def find(layers: Layers, color: str) -> list[Seen]:
    """Every blob of `color` (or of every named color) in the picture, located and named: loose objects first,
    then each robot once, then each fixed thing once, then blobs no one thing covers; biggest first in each."""
    found = [f for c in colors(color) for f in vision.find(layers.rgb, layers.depth, layers.camera, c)]
    driven = _driven(layers.model)
    held = _held(layers.model, layers.touching, driven)
    named = [(f, *_what(f.blob, layers.geoms, layers.model, driven)) for f in found]
    named.sort(key=lambda fwm: (_rank(fwm[1], fwm[2]), -fwm[0].blob.area))
    out: list[Seen] = []
    seen: set[tuple[str, str]] = set()
    for f, what, merged in named:
        key = next(iter(what.items()), None)
        if key is not None:
            if key in seen:  # a smaller piece of something already reported (a highlight split it, an arm in front)
                continue
            seen.add(key)
        entry: dict[str, Any] = {
            "n": len(out) + 1,
            "color": f.color,
            **what,
            "position": [round(float(v), 3) for v in f.position],
            "size": [round(float(v), 3) for v in f.size],
            "pixels": f.blob.area,
            "about_cm": max(round(f.reach * 100, 1), 0.1),
            "at_pixel": [round(f.blob.u), round(f.blob.v)],
        }
        if merged:
            entry["merged"] = merged
        entry.update({flag: True for flag in FLAGS if getattr(f, flag)})
        if what.get("object") in held:
            entry["held_by"] = held[what["object"]]
        out.append(Seen(entry, f.blob.box))
    return out


def summary(view: str, color: str, found: list[Seen]) -> dict[str, Any]:
    """The `found` field of observe's result: the first MAX_FOUND finds, and how many more there were."""
    out: dict[str, Any] = {"view": view, "color": color, "objects": [s.entry for s in found[:MAX_FOUND]]}
    if len(found) > MAX_FOUND:
        out["more"] = len(found) - MAX_FOUND
    marks = [mark for mark in DOUBTS if any(mark in o for o in out["objects"])]
    out["how"] = HOW + (". May be much further off: " + "; ".join(f"{m}: {DOUBTS[m]}" for m in marks) if marks else "")
    if not found:
        out["hint"] = (
            "nothing of that color in view: try another view (overview, or a robot's camera from describe_model) "
            "or move the robot's camera"
        )
    return out


def marked_png(rgb: np.ndarray, found: list[Seen]) -> bytes:
    """The picture with a numbered box round each reported find's blob, as observe returns it."""
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    line = max(2, round(rgb.shape[1] / 320))
    pad = line + 1  # px between a blob and its box, so the box does not cover the blob's edge
    h, w = rgb.shape[:2]
    for seen in found[:MAX_FOUND]:
        x0, y0, x1, y1 = seen.box
        box = (max(x0 - pad, 0), max(y0 - pad, 0), min(x1 + pad, w - 1), min(y1 + pad, h - 1))
        draw.rectangle(box, outline=MARK, width=line)
        label = str(seen.entry["n"])
        above = box[1] - 12 >= 0  # the number sits on the box's top edge, or inside it at the picture's top
        draw.text((box[0] + 2, box[1] - 12 if above else box[1] + line + 1), label, fill=MARK)
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _rank(what: dict[str, str], merged: list[str]) -> int:
    """Where a find's kind comes in the list (KINDS): with loose objects in it, among the objects; blobs no one
    thing covers last."""
    kind = next(iter(what), None)
    if merged:
        return 0
    return KINDS.index(kind) if kind in KINDS else len(KINDS)


def _driven(model: mujoco.MjModel) -> set[int]:
    """The top bodies with a motor somewhere below them: robots, named with a prefix in a world ("arm/link1")
    or not in a plain model file (the demo's arm is "base", "turntable", ...)."""
    roots: set[int] = set()
    for a in range(model.nu):
        target, kind = int(model.actuator_trnid[a, 0]), int(model.actuator_trntype[a])
        if kind in (int(mujoco.mjtTrn.mjTRN_JOINT), int(mujoco.mjtTrn.mjTRN_JOINTINPARENT)):
            roots.add(int(model.body_rootid[model.jnt_bodyid[target]]))
        elif kind == int(mujoco.mjtTrn.mjTRN_SITE):
            roots.add(int(model.body_rootid[model.site_bodyid[target]]))
    return roots


def _held(model: mujoco.MjModel, touching: np.ndarray | None, driven: set[int]) -> dict[str, str]:
    """The loose objects a robot holds, and which robot: everything such an object touches is part of that robot
    (a gripper's fingers), and nothing else carries it (not the floor, a table or another object). From the
    simulator's contacts at the moment of the picture, as the names are."""
    if touching is None or not len(touching):
        return {}
    roots = model.body_rootid[model.geom_bodyid[touching]]
    others: dict[int, set[int]] = {}
    for a, b in roots.tolist():
        if a != b:
            others.setdefault(a, set()).add(b)
            others.setdefault(b, set()).add(a)
    held = {}
    for body, by in others.items():
        thing = _name(model, "body", body, driven) if body else {}
        if "object" not in thing or 0 in by:  # body 0 is the world's: the floor and everything fixed
            continue
        holders = [_name(model, "body", other, driven) for other in sorted(by)]
        if all("robot" in h for h in holders):
            held[thing["object"]] = holders[0]["robot"]
    return held


def _what(
    blob: vision.Blob, geoms: np.ndarray | None, model: mujoco.MjModel, driven: set[int]
) -> tuple[dict[str, str], list[str]]:
    """What most of a blob's pixels show: {"object": name} for a loose object, {"robot": name} for a part of a
    robot (anything with a motor in it), {"fixed": name} for something built into the world; nothing when no
    one thing covers most of it. And the other loose objects in it: touching things of one color are one blob
    (crates on a pallet, a blue cup against a blue arm), so its position is the middle of them all."""
    if geoms is None:
        return {}, []
    ys, xs = blob.pixels()
    ids, counts = np.unique(geoms[ys, xs], return_counts=True)
    things: dict[tuple[str, int], int] = {}  # ("body", root body) or ("geom", geom on the world body): pixels
    for g, n in zip(ids.tolist(), counts.tolist(), strict=True):
        if g < 0:
            continue
        root = int(model.body_rootid[model.geom_bodyid[g]])  # a robot's or an object's top body
        key = ("body", root) if root else ("geom", g)
        things[key] = things.get(key, 0) + n
    ranked = sorted(things.items(), key=lambda kv: -kv[1])
    what = _name(model, *ranked[0][0], driven) if ranked and ranked[0][1] * 2 > blob.area else {}
    least = max(vision.MIN_AREA, MERGED_SHARE * blob.area)
    names = [_name(model, *key, driven) for key, n in ranked if n >= least]
    return what, [n["object"] for n in names if "object" in n and n != what]


def _name(model: mujoco.MjModel, kind: str, index: int, driven: set[int]) -> dict[str, str]:
    """{"object": name}, {"robot": name} or {"fixed": name} for a top body or a geom on the world body."""
    if kind == "geom":  # built into the world: a fixed prop, a wall, the floor
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index) or "floor"
        return {"fixed": name.split("/")[0]}
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, index) or f"body{index}"
    if "/" in name or index in driven:
        return {"robot": name.split("/")[0]}
    joint = model.body_jntadr[index]
    free = model.body_jntnum[index] > 0 and int(model.jnt_type[joint]) == int(mujoco.mjtJoint.mjJNT_FREE)
    return {"object": name} if free else {"fixed": name}
