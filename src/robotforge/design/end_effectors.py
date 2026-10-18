"""End effectors that hold without fingers: a suction cup and an electromagnet, for any arm.

Both hold with MuJoCo's adhesion actuator, which pulls the pad and whatever touches it together with a force
injected at their contacts. A suction cup holds anything its flat pad seals on, flat things a parallel gripper
cannot get its fingers around included (a tile, a plate). An electromagnet holds only steel: its core collides
with steel objects alone (a contact bit only they answer to), so only their contacts carry its pull.

A tool is one body on the arm's hand with the arm's tool point ("tcp") on its face, and an actuator called
"gripper" like the fingers' own: move_arm, gripper and pick_and_place work it unchanged, with "close" turning it
on (it holds) and "open" off. Our arm template builds a tool in place of its fingers; any other arm (a library
arm, an arm built from parts) gets one in front of its fingers, which stay open and answer to "fingers".
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import mujoco
import numpy as np

from robotforge.design.visuals import Visuals
from robotforge.design.visuals import mesh as M
from robotforge.design.visuals import parts as P

GEOM = mujoco.mjtGeom
Kind = Literal["suction", "magnet"]
KINDS: tuple[str, ...] = ("suction", "magnet")
FINGERS = "gripper"  # the end effector an arm is built with, when none is chosen
NAMES = (FINGERS, *KINDS)
MAGNET_BIT = 2  # contact bit of a magnet's core; steel objects answer to it (see steel_affinity)
DARK = [0.16, 0.17, 0.2, 1.0]
CUP_RUBBER = [0.12, 0.13, 0.15, 1.0]
MAGNET_RED = [0.78, 0.16, 0.12, 1.0]
STEEL = [0.62, 0.64, 0.67, 1.0]
# The pad finds what it touches this far before touching it, and pulls on it from there; a contact inside the
# margin pushes back only once the two really meet (the gap), so the reach does not hold things off the pad.
# Kept short: at 3 mm, the floor under a pressed 8 mm tile was "touching" the pad too, and the pull on the floor
# held the arm down.
PAD_MARGIN = 0.001  # m
# A soft contact pushes back in proportion to the mass it moves, so a 60 N pull sank a 17 g tile 27 mm into the pad
# (through it, to the floor). The pad's contacts are nearly hard instead, and its settings win the mix with
# whatever it touches: the same tile now gives way by less than a tenth of a millimetre.
PAD_CONTACT = {"solref": [0.004, 1.0], "solimp": [0.99, 0.999, 0.001, 0.5, 2.0], "solmix": 1000.0}
# A cup sealed on a face resists being peeled off it all round its rim; a handful of point contacts do not, and a
# 14 cm plate held at its middle tipped over and hung 7 cm below the cup. Rolling friction (condim 6) resists the
# turn, sliding friction the slip, as a seal does.
PAD_FRICTION = {"friction": [1.2, 0.02, 0.01], "condim": 6}
# How far a tool in front of fingers sticks out past the old tool point: past the fingertips of our gripper (12 mm),
# a Panda's (about 10 mm) and a PiPER's (15 mm), with room to spare so they never touch what the pad holds.
AHEAD_OF_FINGERS = 0.04  # m


@dataclass(frozen=True)
class ToolSpec:
    """How a tool holds: its pull and the face it holds with."""

    title: str
    pull: float  # N at full power, shared among the pad's contacts
    pad_half: float  # m, half the side of the square pad (collision), or the core's radius
    length: float  # m from the mount to the pad's face
    holds: str  # what it holds, for agents


TOOLS: dict[str, ToolSpec] = {
    # A 40 mm cup on a vacuum generator at -60 kPa: 0.6 bar on 12.6 cm2 is 75 N; less for leaks and wear.
    "suction": ToolSpec("suction cup", 60.0, 0.016, 0.045, "anything with a flat face the cup can seal on"),
    # A 45 mm holding magnet of the kind on small cranes and sorting cells pulls a few hundred N on thick steel.
    "magnet": ToolSpec("electromagnet", 150.0, 0.02, 0.04, "steel things only"),
}


def canonical(name: str | None) -> str | None:
    """The tool `name` means ("Suction Cup", "electromagnet", "fingers"), or `name` itself when there is none."""
    if name is None:
        return None
    word = name.strip().lower().replace("-", " ").replace("_", " ")
    aliases = {
        "suction": "suction",
        "suction cup": "suction",
        "vacuum": "suction",
        "cup": "suction",
        "magnet": "magnet",
        "electromagnet": "magnet",
        "electro magnet": "magnet",
        "gripper": FINGERS,
        "fingers": FINGERS,
        "parallel gripper": FINGERS,
        "jaws": FINGERS,
    }
    return aliases.get(word, name)


def check(name: str) -> str:
    """`name` as a tool's name, or a ValueError that lists them."""
    found = canonical(name)
    if found not in NAMES:
        raise ValueError(
            f"unknown end_effector {name!r}; choose suction (a suction cup: flat things), magnet (an electromagnet: "
            f"steel things) or gripper (the arm's own fingers)"
        )
    return found


# ---- building a tool ---------------------------------------------------------------------------------


def add_tool(
    spec: mujoco.MjSpec,
    body: mujoco.MjsBody,
    kind: str,
    vis: Visuals | None,
    name: str,
    pos: Sequence[float] = (0.0, 0.0, 0.0),
    quat: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
) -> float:
    """Build tool `kind` on `body` as a child body called `name` at `pos` and `quat` (the tool along that frame's
    +z), with its adhesion actuator `<prefix>gripper` (the prefix is `name` up to its last "/"). Returns the
    distance from that frame's origin to the pad's face, where the tool point goes."""
    tool = TOOLS[kind]
    own = vis is None
    vis = vis or Visuals(spec)
    holder = body.add_body(name=name, pos=list(pos), quat=list(quat))
    if kind == "suction":
        face, grip_body = _suction(holder, tool, vis), holder
    else:
        face, grip_body = _magnet(holder, tool, vis, name)
    if own:
        vis.finish()
    prefix = name.rsplit("/", 1)[0] + "/" if "/" in name else ""
    act = spec.add_actuator(name=f"{prefix}gripper", target=grip_body.name, trntype=mujoco.mjtTrn.mjTRN_BODY)
    act.set_to_adhesion(gain=tool.pull)
    act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
    act.ctrlrange = [0.0, 1.0]
    return face


def _suction(holder: mujoco.MjsBody, tool: ToolSpec, vis: Visuals) -> float:
    """A vacuum generator on a flange, a two-fold bellows and a flat rubber cup; a square pad does the holding (a
    box face meets a flat face in four contacts, so a held tile cannot pivot on one point)."""
    face = tool.length
    pad_h = 0.003
    # Collision: the generator (so the tool cannot pass through a table) and the pad. Masses as built: about 60 g.
    body_geom = holder.add_geom(
        type=GEOM.mjGEOM_CYLINDER, size=[0.011, 0.012, 0], pos=[0, 0, 0.012], rgba=DARK, mass=0.04
    )
    pad = holder.add_geom(
        type=GEOM.mjGEOM_BOX,
        size=[tool.pad_half, tool.pad_half, pad_h / 2],
        pos=[0, 0, face - pad_h / 2],
        rgba=CUP_RUBBER,
        mass=0.01,
        **PAD_FRICTION,
        margin=PAD_MARGIN,
        gap=PAD_MARGIN,
        **PAD_CONTACT,
    )
    vis.hide(body_geom)
    vis.hide(pad)
    # Looks (visual only): flange, generator with its port, bellows, cup lip.
    vis.add(holder, "metal", P.METAL, M.cylinder(0.016, 0.004, segments=36, bevel=0.0008, z0=0.0))
    vis.add(holder, "dark", DARK, M.cylinder(0.011, 0.02, segments=32, bevel=0.0015, z0=0.004))
    port = M.cylinder(0.003, 0.008, segments=12).moved(rot=M.rot_y(math.pi / 2))
    vis.add(holder, "metal", P.METAL, port, (0.014, 0, 0.013))
    vis.add(holder, "metal", P.METAL, M.cylinder(0.006, 0.004, segments=24, z0=0.024))
    bellows = [(0.006, 0.027), (0.011, 0.029), (0.0125, 0.031), (0.009, 0.0335), (0.0125, 0.036), (0.011, 0.038)]
    lip = [(0.0085, 0.0395), (0.0135, 0.041), (0.0175, face - 0.0012), (0.0172, face), (0.0, face)]
    vis.add(holder, "rubber", CUP_RUBBER, M.lathe([(0.0, 0.027), *bellows, *lip], segments=36))
    return face


def _magnet(holder: mujoco.MjsBody, tool: ToolSpec, vis: Visuals, name: str) -> tuple[float, mujoco.MjsBody]:
    """A round red housing with a steel core at its face. The core is a body of its own, the one the actuator
    pulls with, so that only the core's contacts (steel ones) carry the pull."""
    face = tool.length
    r = tool.pad_half + 0.004
    housing = holder.add_geom(
        type=GEOM.mjGEOM_CYLINDER,
        size=[r, face / 2 - 0.0005, 0],
        pos=[0, 0, face / 2 - 0.0005],
        rgba=MAGNET_RED,
        mass=0.25,
    )
    core_body = holder.add_body(name=f"{name}_core", pos=[0, 0, face])
    # Square, under its round look: a box face meets a flat face in contacts at its corners, which share the pull
    # evenly. As a disc it held 5.25 kg of a 150 N pull; as a square it holds as the cup does for its pull.
    side = tool.pad_half * 0.88
    core = core_body.add_geom(
        type=GEOM.mjGEOM_BOX,
        size=[side, side, 0.002],
        pos=[0, 0, -0.002],
        rgba=STEEL,
        mass=0.02,
        contype=MAGNET_BIT,
        conaffinity=0,
        **PAD_FRICTION,
        margin=PAD_MARGIN,
        gap=PAD_MARGIN,
        **PAD_CONTACT,
    )
    vis.hide(housing)
    vis.hide(core)
    vis.add(holder, "metal", P.METAL, M.cylinder(0.016, 0.004, segments=36, bevel=0.0008, z0=0.0))
    vis.add(holder, "paint", MAGNET_RED, M.cylinder(r, face - 0.009, segments=48, bevel=0.002, z0=0.004))
    vis.add(holder, "dark", DARK, M.cylinder(r * 1.02, 0.003, segments=48, bevel=0.0008, z0=face - 0.006))
    vis.add(holder, "dark", DARK, P.bolt_circle(r * 0.62, 4, 0.0016, 0.001, math.pi / 4), (0, 0, 0.0045))
    cable = M.cylinder(0.0025, 0.012, segments=10).moved(rot=M.rot_y(math.pi / 2))
    vis.add(holder, "rubber", P.RUBBER, cable, (r + 0.003, 0, 0.012))
    vis.add(core_body, "metal", STEEL, M.cylinder(tool.pad_half, 0.003, segments=40, bevel=0.0006, z0=-0.003))
    return face, core_body


# ---- fitting a tool onto any arm -----------------------------------------------------------------------


def tool_points(spec: mujoco.MjSpec) -> list[mujoco.MjsSite]:
    """The tool points of a robot's spec: "tcp" for an arm, "<part>/tcp" for each arm a robot carries."""
    return [s for s in spec.sites if s.name == "tcp" or s.name.endswith("/tcp")]


def fit(spec: mujoco.MjSpec, kind: str, home_ctrl: dict[str, float]) -> list[str]:
    """Put tool `kind` on every arm of a robot's spec, in front of its tool point, and move the tool point onto the
    pad's face. Fingers already there stay open, out of the way, under the name "fingers". Returns notes; raises
    ValueError for a robot with no tool point."""
    sites = tool_points(spec)
    if not sites:
        raise ValueError(
            f"it has no tool point to put a {TOOLS[kind].title} on: end_effector is for arms (and robots carrying "
            "one); build the arm with a tool point, e.g. a gripper template on its last part"
        )
    notes = []
    for site in sites:
        prefix = site.name[: -len("tcp")]
        fingers = next((a for a in spec.actuators if a.name == f"{prefix}gripper"), None)
        if fingers is not None and int(fingers.trntype) == mujoco.mjtTrn.mjTRN_BODY:
            # Built with a cup or a magnet already (a suction_cup part on a built arm): it keeps the one it has.
            notes.append(f"{prefix[:-1] or 'the arm'} keeps the tool it was built with")
            continue
        ahead = 0.0
        if fingers is not None:  # open, out of the way, and no longer what "gripper" works
            fingers.name = f"{prefix}fingers"
            opened = float(fingers.ctrlrange[1]) if fingers.ctrllimited == mujoco.mjtLimited.mjLIMITED_TRUE else 0.0
            home_ctrl.pop(f"{prefix}gripper", None)
            home_ctrl[f"{prefix}fingers"] = opened
            ahead = AHEAD_OF_FINGERS
        axis = _forward(site.quat)
        at = np.asarray(site.pos, float) + axis * ahead
        face = add_tool(spec, site.parent, kind, None, f"{prefix}{kind}", at, site.quat)
        site.pos = list(at + axis * face)
        home_ctrl[f"{prefix}gripper"] = 0.0
        where = f" on {prefix[:-1]}" if prefix else ""
        kept = "; its fingers stay open (actuator 'fingers')" if fingers is not None else ""
        notes.append(f"{TOOLS[kind].title}{where}: holds {TOOLS[kind].holds}{kept}")
    return notes


def _forward(quat: Sequence[float]) -> np.ndarray:
    """The +z axis of a frame turned by `quat`, in its parent's frame: where a tool point faces."""
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, np.asarray(quat, float))
    return rot.reshape(3, 3)[:, 2].copy()


def steel_affinity() -> int:
    """The conaffinity a steel object's geoms take, so a magnet's core touches them (and only them)."""
    return 1 | MAGNET_BIT


def magnetic(model: mujoco.MjModel, geoms: Sequence[int]) -> bool:
    """Whether any of `geoms` is steel to a magnet: it answers to the magnet core's contact bit."""
    return any(int(model.geom_conaffinity[g]) & MAGNET_BIT for g in geoms)


def tool_of(model: mujoco.MjModel, actuator: int) -> str | None:
    """ "suction" or "magnet" for a tool's adhesion actuator, None for fingers (or anything else)."""
    if actuator < 0 or int(model.actuator_trntype[actuator]) != mujoco.mjtTrn.mjTRN_BODY:
        return None
    body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.actuator_trnid[actuator, 0])) or ""
    return "magnet" if body.endswith("_core") else "suction"
