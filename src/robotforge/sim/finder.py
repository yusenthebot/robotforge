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

import functools
import io
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

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
# said held_by (by view, 1.4 to 4.3 cm off in the median and 4.5 at most: the gripper covers part of it).
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
MARK = (255, 190, 0)  # the boxes drawn round the finds, and their numbers
TAG = (20, 20, 20)  # behind a number, so it reads on a light floor as on a dark one
MIN_LABEL = 12  # px: the numbers' height, at least; the default 10 px font was hard to read on a 640 px picture
LABEL_SHARE = 40  # the numbers are this share of the picture's width tall (16 px on a 640 px picture)
BIG_BOX = 4  # tags wide and high: a box this big (round a shelf, a counter) may have other finds' tags inside it
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
    try:
        vision.color_range(color)
    except ValueError as exc:
        raise ValueError(f'{exc}; "all" looks for every named one') from None
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
    """The picture with a numbered box round each reported find's blob, as observe returns it.

    A number sits on a dark tag next to its box, wherever it covers no other number and no other box (a box far
    bigger than the tag, round a shelf or a floor, may have tags inside it): on a cube standing on another, the
    lower one's number on its box's top edge was hidden under the upper one's box. In a crowd a tag goes further
    out, with a line to its box; only when nothing is free does it go inside its own box.
    """
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    h, w = rgb.shape[:2]
    line = max(2, round(w / 320))
    pad = line + 1  # px between a blob and its box, so the box does not cover the blob's edge
    font = _font(max(MIN_LABEL, round(w / LABEL_SHARE)))
    shown = found[:MAX_FOUND]
    boxes = [
        (max(x0 - pad, 0), max(y0 - pad, 0), min(x1 + pad, w - 1), min(y1 + pad, h - 1))
        for x0, y0, x1, y1 in (seen.box for seen in shown)
    ]
    for box in boxes:
        draw.rectangle(box, outline=MARK, width=line)
    tags: dict[int, tuple[int, int, int, int]] = {}
    texts: dict[int, tuple[str, int, int]] = {}
    # Small boxes choose first: a big one has room all round it.
    for i in sorted(range(len(shown)), key=lambda k: _area(boxes[k])):
        label = str(shown[i].entry["n"])
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        tw, th = right - left + 2 * line, bottom - top + 2 * line  # the tag: the number and a margin round it
        others = [b for k, b in enumerate(boxes) if k != i]
        places = [(x, y, x + tw - 1, y + th - 1) for x, y in _places(boxes[i], tw, th, line)]
        fits = [t for t in places if t[0] >= 0 and t[1] >= 0 and t[2] < w and t[3] < h]
        free = [
            t
            for t in fits
            if not any(_overlap(t, o) for o in tags.values()) and not any(_covers(t, o, tw, th) for o in others)
        ]
        tags[i] = (free or fits or places[-1:])[0]
        texts[i] = (label, line - left, line - top)
    for i, tag in tags.items():  # a line from a tag that does not touch its box
        x0, y0, x1, y1 = boxes[i]
        cx, cy = (tag[0] + tag[2]) / 2, (tag[1] + tag[3]) / 2
        bx, by = min(max(cx, x0), x1), min(max(cy, y0), y1)
        tx, ty = min(max(bx, tag[0]), tag[2]), min(max(by, tag[1]), tag[3])
        if abs(tx - bx) > 1 or abs(ty - by) > 1:
            draw.line((tx, ty, bx, by), fill=MARK, width=max(1, line // 2))
    for i, tag in tags.items():
        label, dx, dy = texts[i]
        draw.rectangle(tag, fill=TAG)
        draw.text((tag[0] + dx, tag[1] + dy), label, fill=MARK, font=font)
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _places(box: tuple[int, int, int, int], tw: int, th: int, line: int) -> list[tuple[int, int]]:
    """Where a tag tw by th may go round a box, nearest first: on its edges, then up to three tags further out,
    and last inside the box's corner."""
    x0, y0, x1, y1 = box
    near = [(x0, y0 - th), (x0, y1 + 1), (x1 + 1, y0), (x0 - tw, y0)]
    far = [
        place
        for k in (1, 2, 3)
        for place in (
            (x0 + k * tw, y0 - th),
            (x0 - k * tw, y0 - th),
            (x0, y0 - (k + 1) * th),
            (x0 + k * tw, y1 + 1),
            (x0 - k * tw, y1 + 1),
            (x0, y1 + 1 + k * th),
            (x1 + 1 + k * tw, y0),
            (x0 - (k + 1) * tw, y0),
        )
    ]
    return [*near, *far, (x0 + line, y0 + line)]


def _covers(tag: tuple[int, int, int, int], box: tuple[int, int, int, int], tw: int, th: int) -> bool:
    """The tag hides part of the box: it overlaps it, and the box is not so big (BIG_BOX tags each way) that a
    tag inside it is just a label on the scenery."""
    if not _overlap(tag, box):
        return False
    big = box[2] - box[0] >= BIG_BOX * tw and box[3] - box[1] >= BIG_BOX * th
    inside = box[0] < tag[0] and tag[2] < box[2] and box[1] < tag[1] and tag[3] < box[3]
    return not (big and inside)


def _overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def _area(box: tuple[int, int, int, int]) -> int:
    return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)


@functools.cache
def _font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    """Pillow's own font at `size` px (it ships one since 10.1), or its small bitmap font where FreeType is missing."""
    try:
        return ImageFont.load_default(size=size)
    except (ImportError, OSError, TypeError):
        return ImageFont.load_default()


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
    """The loose objects a robot holds, and which robot, from the simulator's contacts at the moment of the
    picture, as the names are. Loose objects touching each other go together (a cube resting on the one the
    gripper closed on is carried too): a group is held when what it touches outside itself is only robots' parts,
    nothing fixed and not the floor."""
    if touching is None or not len(touching):
        return {}
    kinds: dict[int, str] = {}  # top body: "object", "robot" or "fixed"

    def kind(body: int) -> str:
        if body not in kinds:
            kinds[body] = next(iter(_name(model, "body", body, driven))) if body else "fixed"
        return kinds[body]

    links: dict[int, set[int]] = {}
    for a, b in model.body_rootid[model.geom_bodyid[touching]].tolist():
        if a != b:
            links.setdefault(a, set()).add(b)
            links.setdefault(b, set()).add(a)
    held: dict[str, str] = {}
    done: set[int] = set()
    for start in links:
        if start in done or kind(start) != "object":
            continue
        group: set[int] = set()
        outside: set[int] = set()
        todo = [start]
        while todo:  # the loose objects touching this one, and what they touch that is not loose
            body = todo.pop()
            if body in group:
                continue
            group.add(body)
            for other in links[body]:
                if kind(other) == "object":
                    todo.append(other)
                else:
                    outside.add(other)
        done |= group
        if outside and all(kind(o) == "robot" for o in outside):
            robot = _name(model, "body", min(outside), driven)["robot"]
            held.update({_name(model, "body", body, driven)["object"]: robot for body in group})
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
