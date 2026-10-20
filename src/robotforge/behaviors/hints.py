"""Taking hold of a generated object where it is meant to be held, and putting it where it is meant to go.

`choices(...)` turns an object's grasp hints into the tool points to try, in the order a person would: the hint that
suits the object best first (a mug by its handle, a bottle round its body, a tool across its shaft), each placed where
the object is now and turned with it, left out when the gripper does not open wide enough for it or the object is not
the way up the hint was made for. The manager tries them in turn with the arm's own planning and keeps the first the
arm can reach and carry; `grasp="handle"` asks for one kind.

`place_for(...)` is the other end: a fixture's slot (a rack's second socket, a tray's inside, a shelf board) or the flat
top of another object, as the point the load's base goes to, how high it must be carried and where it ends up.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from robotforge import objects
from robotforge.behaviors.arm import ArmMotion
from robotforge.objects.kit import Built, Grasp, Slot, center_of_mass, top_under
from robotforge.sim.geometry import collides

UPRIGHT = 0.9  # cosine: an object tipped further than about 26 degrees is not the way up its hints were made for
OPEN_MARGIN = 0.004  # m a gripper opens beyond the width a hint needs
SOCKET_REACH = 0.8  # of a socket's radius: where the base of what is put in it must be to count as in it
FORCE_MIN = 20.0  # N: what a firm hold squeezes with at least
FORCE_MAX = 90.0  # N: and at most (a Panda's hand holds 70 N for good, 140 N for a moment)
PAD_TORSION = 0.005  # m: torsional friction of the pads of the library's grippers
PAD_TORSION_FIRM = 0.012  # m: what a firm hold gives them (a soft pad pressed on a smooth thing, about 5 mm of patch)
TURN_SAFETY = 6.0  # how much more than the turning of a load a squeeze must resist
ROLL_SAFETY = 3.0  # the same for a load rolling about the pinch
ROLL_ARM = 0.01  # m: the lever the pressure over a pad has against a load rolling about the pinch
CAPACITY_ROOM = 0.09  # m a typical thing wants round it, to count how many an area holds
SPREAD_FROM = 4  # things an area must hold, by CAPACITY_ROOM, to be filled from its middle outward and spread apart
FINGER_GAP = 0.025  # m left between two things put in one tray, beyond their own widths
ROOM = 0.055  # m between the middles of two things put in one tray
EDGE = 0.03  # m from a tray's wall that a thing is put at least
STEP = 0.015  # m between the spots tried in a tray
NEAR_Z = 0.12  # m: a thing this far above or below a tray's floor is not in it
BOOST_MAX = 4.0  # times a gripper's force limit a firm hold may raise it
PALM_MARGIN = 0.004  # m the underside of the hand keeps clear of the object's top
TIP_MARGIN = 0.001  # m the fingertips keep over the floor
JAW_ANGLES = 12  # directions tried for the jaws of a grasp that has no preferred one
PAD_ROOM = 0.025  # m beyond the open jaws that must be free of walls and neighbours
HOVER = 0.012  # m above its resting place an object is let go (the planner adds 1 cm; this is for a socket's chamfer)


@dataclass
class Choice:
    """One way to take hold: the tool point in the world, how far the load reaches below it, the jaws' normals, and
    how hard the fingers have to squeeze (N) with how far short of closed they stand then (m)."""

    point: np.ndarray
    half_height: float
    faces: np.ndarray | None
    hint: Grasp | None
    why: str = ""
    force: float = 10.0
    short: float = 0.01

    def describe(self) -> dict[str, Any]:
        if self.hint is None:
            return {"kind": "body"}
        return {"kind": self.hint.kind, "at": [round(float(v), 3) for v in self.point], "note": self.hint.note}


def built_for(entry: Any) -> Built | None:
    """The generated object an entry is (None for the shapes that are not generated)."""
    return None if entry is None else objects.built_of(entry)


@dataclass(frozen=True)
class Palm:
    """The hand as a top-down grasp meets an object: what hangs over the fingers."""

    gap: float  # m from the tool point up to the underside of the hand: nothing of the object may reach above it
    tip: float  # m the fingertips reach below the tool point (the floor must be this far under it)
    half_jaw: float  # half the width of the hand along the way the jaws close
    half_across: float  # half its width across that
    links: float = 0.0  # m the parts that move with the jaws (a Robotiq's links and springs) reach above the tool point


DEFAULT_PALM = Palm(0.03, 0.01, 0.1, 0.03)  # a Panda's hand, when the model cannot be measured


def palm_of(model: mujoco.MjModel, arm: ArmMotion) -> Palm:
    """Measure the arm's hand: the shapes on its tool body (and what is fixed to it) are the palm, the shapes of the
    parts that move with the jaws are the fingers. Worked out once per arm, from the bounding boxes of the shapes."""
    cached = getattr(arm, "_palm", None)
    if cached is not None:
        return cached
    palm = _measure_palm(model, arm) or DEFAULT_PALM
    arm._palm = palm  # type: ignore[attr-defined]
    return palm


def _measure_palm(model: mujoco.MjModel, arm: ArmMotion) -> Palm | None:
    if arm.gripper < 0 or arm.closing is None:
        return None
    d = arm.ik.data
    arm.ik.point(arm.chain, arm.q)
    tool = arm.chain.body
    rot = d.xmat[tool].reshape(3, 3)
    axis = rot @ arm.tool_axis
    jaw = rot @ arm.closing
    across = np.cross(axis, jaw)
    across /= np.linalg.norm(across) or 1.0
    tcp = d.site_xpos[arm.site]
    palm_front, palm_jaw, palm_across, tip, links = -math.inf, 0.0, 0.0, -math.inf, 0.0
    found = False
    for g in range(model.ngeom):
        if not collides(model, g):
            continue
        body = int(model.geom_bodyid[g])
        if not _below(model, body, tool):
            continue
        r = d.geom_xmat[g].reshape(3, 3)
        c = d.geom_xpos[g] + r @ model.geom_aabb[g][:3]
        h = model.geom_aabb[g][3:]
        front = float((c - tcp) @ axis) + _reach(r, h, axis)
        if _fixed_to(model, body, tool):
            found = True
            palm_front = max(palm_front, front)
            palm_jaw = max(palm_jaw, abs(float((c - tcp) @ jaw)) + _reach(r, h, jaw))
            palm_across = max(palm_across, abs(float((c - tcp) @ across)) + _reach(r, h, across))
        else:
            tip = max(tip, front)
            links = max(links, _reach(r, h, axis) - float((c - tcp) @ axis))
    if not found:
        return None
    return Palm(
        gap=max(-palm_front, 0.0) if palm_front < 0 else 0.0,
        tip=max(tip, 0.0),
        half_jaw=palm_jaw,
        half_across=palm_across,
        links=links,
    )


def _reach(rot: np.ndarray, half: np.ndarray, direction: np.ndarray) -> float:
    """How far a box with half extents `half` in the frame `rot` reaches along a direction from its middle."""
    return float(np.abs(rot.T @ direction) @ half)


def finger_geoms(model: mujoco.MjModel, arm: ArmMotion) -> np.ndarray:
    """The solid shapes of everything that moves on the hand: its fingers and their pads and linkage (the shapes on
    bodies the tool body carries by a joint, the pads of a Robotiq among them, which ride on a follower link)."""
    tool = arm.chain.body
    found = [
        g
        for g in range(model.ngeom)
        if collides(model, g)
        and _below(model, int(model.geom_bodyid[g]), tool)
        and not _fixed_to(model, int(model.geom_bodyid[g]), tool)
    ]
    return np.array(found, int)


def _below(model: mujoco.MjModel, body: int, ancestor: int) -> bool:
    while body > 0:
        if body == ancestor:
            return True
        body = int(model.body_parentid[body])
    return False


def _fixed_to(model: mujoco.MjModel, body: int, tool: int) -> bool:
    """Whether `body` is the tool body or hangs from it by no joint at all (a palm, not a finger)."""
    while body != tool:
        if model.body_jntnum[body] > 0 or body <= 0:
            return False
        body = int(model.body_parentid[body])
    return True


def choices(
    data: mujoco.MjData,
    body: int,
    built: Built,
    opening: float | None,
    palm: Palm,
    want: str | None = None,
    model: mujoco.MjModel | None = None,
    high: bool = False,
) -> tuple[list[Choice], list[str]]:
    """The tool points to try for `built` as it lies now, best first, and why the others were left out. With `high`
    each is taken as near the top of what its hint allows as the hint allows (a cup to pour from is held by its rim, so
    that the fingers' upper links are not round its mouth). With the
    `model`, a grasp that may close its jaws in any direction (a round thing) closes them where there is room for the
    open fingers: not through the wall of the bin it lies in or the neighbour beside it."""
    rot = data.xmat[body].reshape(3, 3)
    base = data.xpos[body]
    tipped = float(rot[2, 2]) < UPRIGHT
    out: list[Choice] = []
    left: list[str] = []
    for hint in built.grasps:
        if want is not None and hint.kind != want:
            continue
        if hint.side:
            left.append(f"{hint.kind}: taken from the side, which the top-down planner cannot do")
            continue
        if tipped:
            left.append(f"{hint.kind}: the {built.shape} lies tipped over, its hints are for it as it stands")
            continue
        if opening is not None and hint.opening > opening - OPEN_MARGIN:
            left.append(f"{hint.kind}: needs {hint.opening * 100:.1f} cm, the gripper opens {opening * 100:.1f} cm")
            continue
        z, why = _height(hint, built, palm, high)
        if z is None:
            left.append(f"{hint.kind}: {why}")
            continue
        local = np.array([hint.pos[0], hint.pos[1], z])
        point = base + rot @ local
        jaw = rot @ np.array([hint.jaw[0], hint.jaw[1], 0.0])
        flat = np.array([jaw[0], jaw[1], 0.0])
        faces = None
        if np.linalg.norm(flat) > 0.5:
            unit = flat / np.linalg.norm(flat)
            faces = np.array([unit, unit])  # the jaws close along it, either way round
        elif model is not None and opening is not None:
            free = free_jaw(model, data, body, point, opening, palm)
            if free is not None:
                faces = np.array([free, free])
        force = _squeeze_for(built, hint, point, base, rot, faces)
        if force > FORCE_MAX:
            left.append(
                f"{hint.kind}: pinched there the {built.shape} ({built.mass * 1000:.0f} g) would turn in the fingers "
                f"unless they squeezed with {force:.0f} N, more than the {FORCE_MAX:.0f} N they have"
            )
            continue
        out.append(Choice(point, float(point[2] - base[2]), faces, hint, why, force, hint.opening / 2))
    return out, left


def free_jaw(
    model: mujoco.MjModel, data: mujoco.MjData, body: int, point: np.ndarray, opening: float, palm: Palm | None = None
) -> np.ndarray | None:
    """The horizontal direction (unit) the jaws should close along at `point` so that the open fingers, `opening` apart,
    have room, and the hand's palm (as wide along the jaws as `palm` says, at its height over the grasp) clears the
    neighbours too: the one whose two sides are clear the farthest of the JAW_ANGLES tried, found by rays along it at
    the height of the fingertips and of the palm. None when every direction is clear (nothing to avoid: the arm turns
    its hand as it likes)."""
    reach = opening / 2 + PAD_ROOM
    ids = np.zeros(1, dtype=np.int32)
    point = np.asarray(point, float)
    clear: list[float] = []
    for k in range(JAW_ANGLES):
        a = math.pi * k / JAW_ANGLES
        direction = np.array([math.cos(a), math.sin(a), 0.0])
        across = np.array([-direction[1], direction[0], 0.0])
        room = (
            1.0 + 0.05 / reach
        )  # as a share of what the open fingers need, and a little more to tell clear ways apart
        for sign in (1.0, -1.0):
            for dz in (-0.015, 0.005):  # the pads' lower and upper edge
                hit = mujoco.mj_ray(model, data, point + [0.0, 0.0, dz], sign * direction, None, 1, body, ids)
                if hit >= 0:
                    room = min(room, float(hit) / reach)
            if palm is None or palm.half_jaw <= reach:
                continue
            for side in (-palm.half_across, 0.0, palm.half_across):  # the hand's underside, edge to edge
                origin = point + across * side + np.array([0.0, 0.0, palm.gap + 0.01])
                hit = mujoco.mj_ray(model, data, origin, sign * direction, None, 1, body, ids)
                if hit >= 0 and hit < palm.half_jaw:
                    room = min(room, float(hit) / palm.half_jaw)
        clear.append(room)
    if min(clear) >= 1.0:
        return None
    best = int(np.argmax(clear))
    a = math.pi * best / JAW_ANGLES
    return np.array([math.cos(a), math.sin(a), 0.0])


def _squeeze_for(
    built: Built, hint: Grasp, point: np.ndarray, base: np.ndarray, rot: np.ndarray, faces: np.ndarray | None
) -> float:
    """The squeeze (N) that keeps the object from turning in the fingers, with a safety factor: its weight times how
    far its middle is from the line the pads press along (turning about that line, held by the pads' torsional
    friction: twice the larger of theirs and the object's, times the normal force on each), and its weight times how far
    its middle is along that line (rolling about the pinch, held by the pressure over the pad: the normal force times
    ROLL_ARM). At least FORCE_MIN; above FORCE_MAX it cannot be done."""
    mass = built.mass
    if mass <= 0:
        return FORCE_MIN
    com = base + rot @ center_of_mass(built)
    jaw = faces[0] if faces is not None else np.array([1.0, 0.0, 0.0])
    across = np.array([-jaw[1], jaw[0], 0.0])
    weight = mass * 9.81
    torsion = max(built.friction[1], PAD_TORSION)
    turn = abs(float((com - point) @ across)) * weight / (2.0 * torsion)
    roll = abs(float((com - point) @ jaw)) * weight / ROLL_ARM if faces is not None else 0.0
    return float(max(TURN_SAFETY * turn, ROLL_SAFETY * roll, FORCE_MIN))


def _height(hint: Grasp, built: Built, palm: Palm, high: bool = False) -> tuple[float | None, str]:
    """The height (object frame) the tool point goes to for a hint: where it asks, raised until the hand's underside
    clears what is under it, and no lower than the fingertips allow. None, and why, when that is not within the hint."""
    lo = hint.lo if hint.lo is not None else hint.pos[2]
    hi = hint.hi if hint.hi is not None else hint.pos[2]
    jaw = hint.jaw if hint.jaw != (0.0, 0.0) else (1.0, 0.0)
    top = top_under(built, (hint.pos[0], hint.pos[1]), jaw, palm.half_jaw, palm.half_across)
    need = max(top - palm.gap + PALM_MARGIN, palm.tip + TIP_MARGIN, lo)
    if need > hi + 1e-9:
        if top - palm.gap + PALM_MARGIN > hi:
            return None, (
                f"the hand's palm would meet the top of the {built.shape} ({top * 100:.1f} cm): it would have to hold "
                f"it at {need * 100:.1f} cm, above the {hi * 100:.1f} cm that grasp reaches"
            )
        return None, f"the fingertips would reach below the floor at {hi * 100:.1f} cm"
    z = hi if high else min(max(hint.pos[2], need), hi)
    return z, ("" if abs(z - hint.pos[2]) < 1e-9 else f"held at {z * 100:.1f} cm so the palm clears the top")


class FirmHold:
    """What a heavy thing held in the fingers needs, for as long as the arm carries it: MuJoCo's no-slip pass, a
    squeeze that is firm a centimetre short of closed, firm finger pads and the arm holding its own weight up. The
    settings a mobile errand makes (fetch.Errand._hold_fast), without its rolling friction: with it a can stayed wedged
    between the open fingers;
    put back by `stop`, which the arm calls when it is idle, when its controller is replaced or the simulation is
    reset."""

    def __init__(self, session: Any, arm: ArmMotion, body: int, force: float = 10.0, short: float = 0.01) -> None:
        self.session, self.arm, self.body = session, arm, body
        self.force, self.short = force, short
        self._saved: dict[str, Any] = {}

    def start(self, keep: bool = False) -> None:
        """Take hold firmly. With `keep` the settings stay when the arm has no more to do (it holds the thing up for
        as long as it is told to) until the jaws are opened, the controller is replaced or the simulation is reset."""
        from robotforge.behaviors import fetch

        model = self.session.model
        if model.opt.noslip_iterations < fetch.NOSLIP:
            self._saved["noslip"] = int(model.opt.noslip_iterations)
            model.opt.noslip_iterations = fetch.NOSLIP
        self._saved["weight"] = self.arm.carry_weight
        self.arm.carry_weight = True
        if self.arm.gripper >= 0:
            pads = finger_geoms(model, self.arm)
            self._saved["pad_friction"] = (pads, model.geom_condim[pads].copy(), model.geom_friction[pads].copy())
            model.geom_condim[pads] = np.maximum(model.geom_condim[pads], 4)  # the contact can resist turning
            model.geom_friction[pads, 1] = np.maximum(model.geom_friction[pads, 1], PAD_TORSION_FIRM)
            self._saved["pads"] = (
                pads,
                model.geom_solmix[pads].copy(),
                model.geom_solimp[pads].copy(),
                model.geom_solref[pads].copy(),
            )
            model.geom_solmix[pads] = fetch.FIRM_SOLMIX
            model.geom_solimp[pads] = fetch.FIRM_SOLIMP
            model.geom_solref[pads] = fetch.FIRM_SOLREF
            before = fetch.squeeze(model, self.arm.gripper, self.force, self.short)
            if before is not None:
                self._saved["squeeze"] = before
        self.arm.on_lift.append(self.boost)
        self.arm.on_open.append(self.stop)
        if not keep:
            self.arm.on_idle.append(self.stop)
        self.session.reset_hooks.append(self.stop)

    @property
    def live(self) -> bool:
        """Whether the settings are changed now (start has been called and stop has not)."""
        return "weight" in self._saved

    def boost(self) -> None:
        """The jaws have closed: if they squeeze less than the hold needs (a Robotiq's servo is as stiff as it gets and
        its force limit is what limits it), raise that limit so that they squeeze as hard as it takes, up to
        BOOST_MAX times what it was."""
        model, data = self.session.model, self.session.data
        g = self.arm.gripper
        if not self.live or g < 0 or not model.actuator_forcelimited[g]:
            return
        fingers = set(int(i) for i in finger_geoms(model, self.arm))
        mine = {i for i in range(model.ngeom) if int(model.geom_bodyid[i]) == self.body}
        normal = np.zeros(6)
        total, pairs = 0.0, 0
        for c in range(data.ncon):
            con = data.contact[c]
            if (con.geom1 in fingers and con.geom2 in mine) or (con.geom2 in fingers and con.geom1 in mine):
                mujoco.mj_contactForce(model, data, c, normal)
                total += abs(float(normal[0]))
                pairs += 1
        per_side = total / 2.0
        if pairs == 0 or per_side <= 0 or per_side >= self.force:
            return
        factor = min(self.force / per_side, BOOST_MAX)
        if "force_range" not in self._saved:
            self._saved["force_range"] = (g, model.actuator_forcerange[g].copy())
        model.actuator_forcerange[g] = self._saved["force_range"][1] * factor

    def relax(self) -> None:
        """Give the gripper's own servo back (the jaws are about to open: a stiffened one did not let go of a can)."""
        saved = self._saved.pop("squeeze", None)
        if saved is not None:
            g = self.arm.gripper
            self.session.model.actuator_gainprm[g], self.session.model.actuator_biasprm[g] = saved

    def stop(self) -> None:
        if not self.live:
            return
        model = self.session.model
        saved, self._saved = self._saved, {}
        if "noslip" in saved:
            model.opt.noslip_iterations = saved["noslip"]
        self.arm.carry_weight = saved["weight"]
        if "pad_friction" in saved:
            pads, condim, friction = saved["pad_friction"]
            model.geom_condim[pads], model.geom_friction[pads] = condim, friction
        if "force_range" in saved:
            g, limits = saved["force_range"]
            model.actuator_forcerange[g] = limits
        if "pads" in saved:
            pads, solmix, solimp, solref = saved["pads"]
            model.geom_solmix[pads], model.geom_solimp[pads], model.geom_solref[pads] = solmix, solimp, solref
        if "squeeze" in saved:
            g = self.arm.gripper
            model.actuator_gainprm[g], model.actuator_biasprm[g] = saved["squeeze"]
        if self.stop in self.session.reset_hooks:
            self.session.reset_hooks.remove(self.stop)


def kinds_of(built: Built) -> list[str]:
    return sorted({g.kind for g in built.grasps})


# ---- where it goes ---------------------------------------------------------------------------------------------


@dataclass
class Place:
    """Where a load's base goes: the point, the highest thing round it (the carried load must clear it) and the
    region `placed` judges it by."""

    point: np.ndarray
    top: float
    region: dict[str, Any]
    slot: str
    note: str = ""


def parse_target(text: str, names: list[str]) -> tuple[str, str | None]:
    """ "rack/2" is the object rack, slot 2; a name with no slash is the object."""
    if "/" in text:
        head, tail = text.rsplit("/", 1)
        if head in names:
            return head, tail
    return text, None


def place_for(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    name: str,
    built: Built,
    slot_name: str | None,
    taken: list[np.ndarray],
    radius: float = 0.0,
) -> Place | None:
    """The place on a generated object: the slot asked for, else the first free one, else its flat top. `taken` are
    the positions of other loose things (a socket with one in it is not free). None when it has no place to put things
    (a mug is not a shelf). `radius` is the footprint radius of what is put there: things in a tray are spread
    at least a finger's width apart."""
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body < 0:
        return None
    rot = data.xmat[body].reshape(3, 3)
    base = data.xpos[body]
    slots = list(built.slots)
    if not slots and built.facts.get("top_z") is not None:  # the flat top of a box or a book: things stand on it
        hx, hy = built.half
        slots = [Slot("top", (0.0, 0.0, float(built.facts["top_z"])), "surface", half=(hx * 0.8, hy * 0.8),
                      rim=float(built.facts["top_z"]))]  # fmt: skip
    if not slots:
        return None
    if slot_name is not None:
        found = [s for s in slots if s.name == slot_name]
        if not found:
            names = ", ".join(s.name for s in slots)
            raise KeyError(f"{name} has no place called {slot_name!r}; its places are {names}")
        slots = found
    chosen, note = None, ""
    for s in slots:
        spot = base + rot @ np.asarray(s.pos)
        if s.radius is None and s.half is not None:  # a tray or a top: room for several, put each where it is freest
            free = _free_spot(spot, rot, s, taken, max(ROOM, 2 * radius + FINGER_GAP), max(EDGE, radius + 0.005))
            if free is not None:
                chosen = (s, free)
                break
            continue
        busy = [p for p in taken if np.hypot(*(p[:2] - spot[:2])) < (s.radius or 0.05) and p[2] > spot[2] - 0.02]
        if not busy:
            chosen = (s, spot)
            break
    if chosen is None:
        raise LookupError(f"every place on {name} is taken ({', '.join(x.name for x in slots)})")
    s, spot = chosen
    if s.radius is not None:
        half = [s.radius * SOCKET_REACH, s.radius * SOCKET_REACH]
    elif s.half is not None:
        a = np.abs(rot[:2, :2]) @ np.asarray(s.half)
        half = [float(a[0]), float(a[1])]
    else:
        half = [0.04, 0.04]
    rim = float(base[2] + s.rim)
    region = {"center": [float(v) for v in spot], "half_xy": half, "min_z": float(spot[2] - 0.012)}
    if s.kind == "socket":
        spot = spot + np.array([0.0, 0.0, 0.0])
    return Place(spot, rim, region, s.name, note)


def _free_spot(
    centre: np.ndarray, rot: np.ndarray, slot: Slot, taken: list[np.ndarray], room: float = ROOM, edge: float = EDGE
) -> np.ndarray | None:
    """The spot in a slot's area farthest from what already stands in it (at least `room` from each), nearest the
    middle among equals, `edge` clear of the walls; None when the area is full."""
    hx, hy = slot.half
    hx, hy = max(hx - edge, 0.0), max(hy - edge, 0.0)
    near = [
        p[:2] for p in taken if -0.01 <= p[2] - centre[2] < NEAR_Z and np.hypot(*(p[:2] - centre[:2])) < hx + hy + room
    ]
    xs = np.linspace(-hx, hx, max(2, int(2 * hx / STEP) + 1)) if hx > 0 else [0.0]
    ys = np.linspace(-hy, hy, max(2, int(2 * hy / STEP) + 1)) if hy > 0 else [0.0]
    capacity = (int(2 * hx / CAPACITY_ROOM) + 1) * (int(2 * hy / CAPACITY_ROOM) + 1)
    # A roomy area is filled from its middle outward, spread apart; a small one (a tray for two things) from a corner,
    # packed, so that what it holds is not left a middle that nothing else fits beside.
    anchor = np.zeros(2) if capacity >= SPREAD_FROM else np.array([-hx, -hy])
    best, score = None, -np.inf
    for x in xs:
        for y in ys:
            p = centre + rot @ np.array([x, y, 0.0])
            gap = min((float(np.hypot(*(p[:2] - q))) for q in near), default=np.inf)
            if gap < room:
                continue
            value = (min(gap, 3 * room) if capacity >= SPREAD_FROM else 0.0) - float(
                np.hypot(x - anchor[0], y - anchor[1])
            )
            if value > score:
                best, score = p, value
    return best


def tilt_deg(data: mujoco.MjData, body: int) -> float:
    """How far a body's z axis is from the world's up, in degrees."""
    return float(math.degrees(math.acos(max(-1.0, min(1.0, float(data.xmat[body][8]))))))
