"""Turn a plan into a robot: a build of parts (the builder's input), or a template with params for a humanoid.

Each body is a proven layout from the gallery and the builder guide (the dog's legs, the scorpion's tripod, the
rover, the balancer, the snake, the arm template) stretched by the plan's factors, with the servos chosen for the
weight they carry, and dressed by the seed: a color scheme, materials and a few details (a head with camera eyes, a
lidar puck, a bumper, an antenna), so the robot looks made, and every body keeps the shape the built-in behaviors
recognize, so it moves as soon as it is added.
"""

from __future__ import annotations

import random
from typing import Any

from robotforge.design.gallery.base import GOLD, GRAPHITE, IVORY, STEEL, hanging_arm
from robotforge.design.generator.fit import fit_legs
from robotforge.design.generator.plan import HUMANOID_HEIGHT, Plan

# Color schemes: (main, accent). The main color is the body shell, the accent the joints' links and trim.
SCHEMES: tuple[tuple[str, str], ...] = (
    ("#e8702a", GRAPHITE),  # safety orange
    ("#1f8a8a", IVORY),  # teal
    ("#f2b134", "#1d2026"),  # signal yellow
    ("#2f6fd6", "#e9ecef"),  # cobalt
    ("#c8323c", STEEL),  # crimson
    ("#5cc8a8", GRAPHITE),  # mint
    ("#6b4fbf", GOLD),  # violet and gold
    (IVORY, GOLD),  # space white
    ("#6b7a3a", "#c9b38a"),  # olive
    ("#ef6f5e", "#23395d"),  # coral and navy
)
DARK = "#1d2026"
HIP_SPACING = 2.0  # a quadruped's front and back hips at least this many leg (thigh) lengths apart
HIP_WIDTH = 0.58  # and its left and right hips this many
TORSO_WIDTH = (0.14, 0.16)  # m times the size: how wide a two-armed quadruped is (its torso is 0.13)
ARM_BACK = 0.12  # an arm on a walker's back stands this share of its length behind the middle
# Its gripper hangs forward over the head at home: on a robot 1.6 times the size a head that big was in its way.
HEAD_UNDER_ARM = 0.08  # m, the largest head under an arm
CARGO_TALL = 1.5  # a rover with an arm: its cargo box at most this many times as tall as the rover is long


def r3(v: float) -> float:
    return round(v, 3)


class Look:
    """The seed's choices: colors, materials, details. Same plan, same look."""

    def __init__(self, plan: Plan):
        rng = random.Random(plan.seed * 7919 + 17)
        main, accent = SCHEMES[rng.randrange(len(SCHEMES))]
        self.main = plan.color or main
        self.accent = plan.accent or (accent if accent.lower() != self.main.lower() else GRAPHITE)
        self.links = rng.choice((GRAPHITE, self.accent, STEEL))
        self.material = plan.material or rng.choice(("pla", "pla", "aluminum", "carbon"))
        self.limb_material = plan.material or rng.choice(("pla", "carbon", "aluminum"))
        self.round_head = rng.random() < 0.5
        self.antenna = rng.random() < 0.5 and plan.arms == 0  # an arm swings where an antenna would stand
        self.bumper = rng.random() < 0.6
        self.jitter = rng.uniform(0.94, 1.06)  # proportions a little different from robot to robot


def make(plan: Plan) -> tuple[dict[str, Any], list[str]]:
    """The robot entry's source (`{"build": {...}}` or `{"template": ..., "params": {...}}`) and notes."""
    look = Look(plan)
    maker = {
        "legged": legged,
        "wheeled": wheeled,
        "balancer": balancer,
        "snake": snake,
        "arm": standing_arm,
        "humanoid": humanoid,
    }[plan.body]
    made, notes = maker(plan, look)
    if plan.style:
        _in_style(made, plan.style)
    return made, notes


def _in_style(made: dict[str, Any], style: str) -> None:
    """Draw the robot in a style: a builder spec's `style`, or the `style` of a template (and of the arms and hands a
    spec mounts)."""
    if "template" in made:
        made.setdefault("params", {})["style"] = style
        return
    made["style"] = style
    for part in made.get("parts", ()):
        mounted = part.get("robot")
        if mounted and mounted.get("template") in ("arm", "gripper", "quadruped", "humanoid", "rover"):
            mounted.setdefault("params", {})["style"] = style


# ---- details shared by the bodies --------------------------------------------------------------------------------


def head(parent: str, at: Any, size: float, look: Look, plan: Plan, pan: bool = True) -> list[dict[str, Any]]:
    """A head on a neck servo with two camera eyes (or a dark visor without cameras)."""
    d = r3(size)
    part: dict[str, Any] = {
        "name": "head",
        "parent": parent,
        "at": at,
        "joint": {"axis": "z", "range": [-90, 90]} if pan else "fixed",
        "shape": "sphere" if look.round_head else "box",
        "size": [d] if look.round_head else [d, r3(d * 1.15), r3(d * 0.8)],
        "direction": "up",
        "color": IVORY if look.main != IVORY else look.accent,
    }
    if pan:
        part["actuator"] = "sg90"
    out = [part]
    front = d / 2 * (0.9 if look.round_head else 1.0)
    z = d * (0.55 if look.round_head else 0.42)
    if plan.camera:
        part["sensors"] = [
            {"type": "camera", "name": "left_eye", "at": [r3(front), r3(d * 0.22), r3(z)]},
            {"type": "camera", "name": "right_eye", "at": [r3(front), r3(-d * 0.22), r3(z)]},
        ]
    else:
        out.append(
            {
                "name": "visor",
                "parent": "head",
                "at": [r3(front - 0.002), 0, r3(z)],
                "joint": "fixed",
                "shape": "box",
                "size": [0.006, r3(d * 0.7), r3(d * 0.25)],
                "direction": "forward",
                "color": DARK,
            }
        )
    if look.antenna:
        out.append(
            {
                "name": "antenna",
                "parent": "head",
                "at": [r3(-d * 0.2), r3(d * 0.25), r3(d * (1.0 if look.round_head else 0.8))],
                "joint": "fixed",
                "shape": "cylinder",
                "size": [0.006, r3(0.6 * d)],
                "direction": "up",
                "color": STEEL,
            }
        )
    return out


def base_sensors(plan: Plan, lidar_at: Any = "top", camera: bool = False) -> list[Any]:
    sensors: list[Any] = ["imu"]
    if plan.lidar:
        sensors.append({"type": "lidar", "at": lidar_at, "rays": 24})
    if camera and plan.camera:
        sensors.append({"type": "camera", "at": "front"})
    return sensors


TOOLS = {"suction": "suction_cup", "magnet": "magnet"}  # the templates that hold without fingers


def gripper(hand: str, color: str, opening: float = 0.06) -> dict[str, Any]:
    """The hand on the end of a chain built from parts: the gripper template, or a suction cup or magnet."""
    if hand in TOOLS:
        return {"template": TOOLS[hand]}
    return {"template": "gripper", "params": {"color": color, "opening": opening}}


def arm_on_top(plan: Plan, look: Look, reach: float, shift: float, home: dict[str, float] | None = None):
    """One arm template on the robot's back, its gripper (or cup or magnet) on the end."""
    params: dict[str, Any] = {"reach": r3(reach), "color": look.accent if look.accent != DARK else look.main}
    if plan.hand != "gripper":
        params["end_effector"] = plan.hand
    if reach > 0.42:
        params["servo"] = "bldc_joint"  # a long arm on hobby servos sags under its own weight
    robot: dict[str, Any] = {"template": "arm", "params": params}
    if home:
        robot["home"] = home
    return {"name": "arm", "at": "top", "shift": [r3(shift), 0, 0], "robot": robot}


# ---- legged ------------------------------------------------------------------------------------------------------


def legged(plan: Plan, look: Look) -> tuple[dict[str, Any], list[str]]:
    """Four legs (a dog) or six (an insect, or a spider when round), with the leg joints the builder's own check
    says hold it up with room to walk (fit.py)."""
    build, notes = (quadruped if plan.legs == 4 else spider if plan.round else insect)(plan, look)
    build, more = fit_legs(build)
    return build, notes + more


def quadruped(plan: Plan, look: Look) -> tuple[dict[str, Any], list[str]]:
    s, j = plan.size, look.jitter
    length = r3(0.3 * plan.length * s * j)
    width = r3(0.14 * plan.width * s)
    thick = r3(0.06 * s)
    leg = r3(0.12 * plan.height * s)
    notes: list[str] = []
    spacing = r3(length - 0.06 * s)
    if spacing < HIP_SPACING * leg:
        # Hips closer than two leg lengths, the front and back feet step into each other: 2026-10-19 sweep
        # (scripts/gen_bench.py sweep, 4 seeds, 0.25 m/s): body x0.6 and x0.7 walked 1 in 8, x0.8 4 in 8, x0.9 7 in
        # 8, while every one with its hips two legs apart walked; a dog 1.35 times as big with legs 1.4 times as
        # long pitched over in 4 of 5 variations until its body was lengthened.
        asked = length
        spacing = r3(HIP_SPACING * leg)
        length = r3(spacing + 0.06 * s)
        if length > asked * 1.03 and (plan.length < 1 or plan.height > 1):  # not for a seed's few percent
            why = "its front and back feet would step into each other"
            notes.append(
                f"its body is {length:.2f} m long, not shorter: with its hips closer than two leg lengths {why}; "
                "shorter legs make a shorter dog"
                if plan.length < 1
                else f"its body is {length:.2f} m long for its long legs: shorter, {why}"
            )
    narrowest, widest = TORSO_WIDTH
    if plan.arms >= 2 and not narrowest * s <= width <= widest * s:
        # Two arms hang from a torso as wide as a default dog: wider, its front thighs swung into them; narrower (x0.6
        # or x0.8), it tipped over or barely walked (2026-10-19 sweep).
        width = r3(min(max(width, narrowest * s), widest * s))
        notes.append(
            f"its body is {width:.2f} m wide: a two-armed dog is as wide as its torso, or it tips or kicks its arms"
        )
    if width < HIP_WIDTH * leg:
        # Narrow and long-legged, it tips over sideways: legs x1.8 on a body x0.6 or x0.8 wide walked 1 in 6, x0.9
        # wide 3 in 3; x1.6 legs walked from x0.7 wide (2026-10-19 sweep).
        asked = width
        width = r3(HIP_WIDTH * leg)
        if width > asked * 1.03 and (plan.width < 1 or plan.height > 1):
            notes.append(f"its body is {width:.2f} m wide, not narrower: on its long legs a narrower one tips over")
    hip_y = r3(width / 2)
    parts: list[dict[str, Any]] = [
        {
            "name": "left_hip",
            "at": [r3(spacing / 2), hip_y, 0],
            "joint": {"axis": "x", "range": [-35, 35]},
            "shape": "none",
            "mirror": True,
            "repeat": {"count": 2, "along": "back", "spacing": spacing},
        },
        {
            "name": "left_thigh",
            "parent": "left_hip",
            "at": [0, r3(0.035 * s), 0],
            "joint": {"axis": "y", "range": [-70, 140], "home": 46},
            "size": [r3(0.024 * s), leg],
            "direction": "down",
            "color": look.links,
            "material": look.limb_material,
        },
        {
            "name": "left_shin",
            "parent": "left_thigh",
            "at": "end",
            "joint": {"axis": "y", "range": [-150, -15], "home": -92},
            "size": [r3(0.019 * s), leg],
            "direction": "down",
            "foot": True,
            "sensors": ["touch"],
            "color": GRAPHITE if look.links != GRAPHITE else STEEL,
            "material": look.limb_material,
        },
    ]
    parts += _legged_dress(plan, look, length, width, thick)
    base = {
        "shape": "box",
        "size": [length, width, thick],
        # a shell over the battery, not a solid print, as heavy as it is long and wide
        "mass": r3(1.0 * s * length / 0.3 * width / 0.14),
        "sensors": base_sensors(plan, camera=_headless(plan)),
        "material": look.material,
    }
    return {"color": look.main, "base": base, "parts": parts}, notes


def _hex_leg(prefix: str, at: list[float], yaw: float, s: float, plan: Plan, look: Look) -> list[dict[str, Any]]:
    reach = plan.height
    parts = [
        {
            "name": f"{prefix}_coxa",
            "at": at,
            "rotate": [0, 0, yaw],
            "joint": {"axis": "z", "range": [-35, 35]},
            "shape": "box",
            "size": [r3(0.05 * s), r3(0.03 * s), r3(0.025 * s)],
            "direction": "forward",
            "color": GRAPHITE,
            "mirror": True,
        },
        {
            "name": f"{prefix}_femur",
            "parent": f"{prefix}_coxa",
            "at": "end",
            "joint": {"range": [-80, 80], "home": -20},
            "size": [r3(0.022 * s), r3(0.09 * s * reach)],
            "direction": "forward",
            "color": "main",
            "material": look.limb_material,
        },
        {
            "name": f"{prefix}_tibia",
            "parent": f"{prefix}_femur",
            "at": "end",
            "joint": {"range": [0, 150], "home": 110},
            "size": [r3(0.018 * s), r3(0.13 * s * reach)],
            "direction": "forward",
            "color": look.links if look.links != look.main else GRAPHITE,
            "foot": True,
            "material": look.limb_material,
        },
    ]
    return parts


def insect(plan: Plan, look: Look) -> tuple[dict[str, Any], list[str]]:
    """Three legs a side along a long body, the front pair angled forward and the rear pair back, like an ant."""
    s, j = plan.size, look.jitter
    length = r3(0.3 * plan.length * s * j)
    width = r3(0.13 * plan.width * s)
    thick = r3(0.05 * s)
    row = r3(length / 2 - 0.04 * s)
    y = r3(width / 2)
    parts: list[dict[str, Any]] = []
    for prefix, x, yaw in (("left_front", row, 45), ("left_middle", 0.0, 90), ("left_rear", -row, 135)):
        parts += _hex_leg(prefix, [x, y, 0], yaw, s, plan, look)
    parts += _legged_dress(plan, look, length, width, thick)
    base = {
        "shape": "capsule" if look.round_head else "box",
        "size": [r3(width), r3(length - width)] if look.round_head else [length, width, thick],
        "mass": r3(0.9 * s**3 * plan.length * plan.width),
        "sensors": base_sensors(plan, camera=_headless(plan)),
        "material": look.material,
    }
    if look.round_head:
        base["axis"] = "x"
    return {"color": look.main, "base": base, "parts": parts}, []


def spider(plan: Plan, look: Look) -> tuple[dict[str, Any], list[str]]:
    """Six legs in a ring round a round body, the builder guide's hexapod."""
    s = plan.size
    diameter = r3(0.26 * s * plan.width * look.jitter)
    leg = _hex_leg("leg", [r3(diameter * 0.433), r3(diameter * 0.25), 0], 30, s, plan, look)
    leg[0]["name"] = "coxa"
    leg[0].pop("mirror")
    leg[0]["repeat"] = {"count": 6, "around": "z"}
    leg[1].update(name="femur", parent="coxa")
    leg[2].update(name="tibia", parent="femur")
    parts = leg + _legged_dress(plan, look, diameter, diameter, r3(0.055 * s))
    base = {
        "shape": "cylinder",
        "size": [diameter, r3(0.055 * s)],
        "mass": r3(0.9 * s**3 * plan.width**2),
        "sensors": base_sensors(plan, camera=_headless(plan)),
        "material": look.material,
    }
    return {"color": look.main, "base": base, "parts": parts}, []


def _headless(plan: Plan) -> bool:
    """A camera on the base: there is no head (or torso with one) to carry the eyes."""
    return not plan.head


def _legged_dress(plan: Plan, look: Look, length: float, width: float, thick: float) -> list[dict[str, Any]]:
    """What sits on a walker's back: arms (one on the back, two on a torso in front), a head, a lidar puck."""
    s = plan.size
    out: list[dict[str, Any]] = []
    if plan.arms >= 2:
        if plan.legs == 6:
            # The front legs reach forward under the arms: the arms are held up like a mantis's, on a riser that
            # grows with the legs (their thighs rise toward the shoulders).
            raised, riser = True, (0.04 + 0.04 * plan.height) * s
        else:
            # Short legs bring the hands down to the floor: a riser keeps them clear while it walks.
            raised, riser = False, max(0.0, 1.0 - plan.height) * 0.25 * s
        shift = [r3(length * 0.3), 0, 0]
        out += torso_with_arms("top", shift, plan, look, riser=riser, raised=raised)
    else:
        if plan.arms == 1:
            out.append(arm_on_top(plan, look, min(0.38 * s, 0.42), -length * ARM_BACK))
        if plan.head:
            size = 0.07 * s if plan.arms == 0 else min(0.07 * s, HEAD_UNDER_ARM)
            out += head("base", [r3(length / 2 - 0.03 * s), 0, r3(thick / 2)], size, look, plan)
    return out


def torso_with_arms(
    at: Any,
    shift: list[float],
    plan: Plan,
    look: Look,
    parent: str = "base",
    riser: float = 0.0,
    raised: bool = False,
) -> list[dict[str, Any]]:
    """An upright torso with a head and two hanging arms with grippers (the centaur's), for two-armed robots; a riser
    column under it lifts the shoulders clear of what is below them (a rover's wheels)."""
    s = plan.size
    tall = r3(0.14 * s)
    parts: list[dict[str, Any]] = []
    if riser > 0:
        parts.append(
            {
                "name": "riser",
                "parent": parent,
                "at": at,
                "shift": shift,
                "joint": "fixed",
                "shape": "cylinder",
                "size": [r3(0.07 * s), r3(riser)],
                "mass": r3(0.2 * s**2),
                "direction": "up",
                "color": STEEL,
            }
        )
        parent, at, shift = "riser", "end", [0, 0, 0]
    parts += [
        {
            "name": "torso",
            "parent": parent,
            "at": at,
            "shift": shift,
            "joint": "fixed",
            "shape": "box",
            "size": [r3(0.09 * s), r3(0.13 * s), tall],
            "mass": r3(0.35 * s**3),
            "direction": "up",
            "color": IVORY if look.main != IVORY else look.accent,
        },
        {
            "name": "chest",
            "parent": "torso",
            "at": [r3(0.045 * s), 0, r3(0.085 * s)],
            "joint": "fixed",
            "shape": "box",
            "size": [0.008, r3(0.09 * s), r3(0.07 * s)],
            "direction": "forward",
            "color": "main",
        },
        {
            "name": "left_shoulder",
            "parent": "torso",
            "at": [0, r3(0.065 * s), r3(0.118 * s)],
            "joint": "fixed",
            "shape": "cylinder",
            "size": [r3(0.042 * s), r3(0.1 * s)],
            "mass": 0.03,
            "direction": "left",
            "color": "main",
            "mirror": True,
        },
    ]
    if plan.head:
        dressed = head("torso", "end", 0.075 * s, look, plan)
        dressed[0]["shift"] = [0, 0, 0.012]
        parts += dressed
    arms = hanging_arm(
        "left_shoulder",
        [0, r3(0.078 * s), r3(-0.021 * s)],
        upper=r3(0.18 * s),
        fore=r3(0.18 * s),
        color=IVORY if look.main != IVORY else STEEL,
        grip_color=look.main,
        home=(0, -60, -50, 110) if raised else (0, -20, -110, 130),
        mirror=False,
    )
    if plan.hand != "gripper":  # a suction cup or magnet on each wrist in place of the gripper
        arms[-1]["robot"] = gripper(plan.hand, look.main)
    return parts + arms


# ---- wheeled -----------------------------------------------------------------------------------------------------


def wheeled(plan: Plan, look: Look) -> tuple[dict[str, Any], list[str]]:
    s, j = plan.size, look.jitter
    length = r3(0.36 * plan.length * s * j)
    width = r3(0.24 * plan.width * s)
    thick = r3(0.08 * s)
    wheel_d = r3(0.11 * s * (1.15 if plan.wheels == 2 else 1.0))
    wheel_w = r3(0.035 * s)
    gear = 3 if s * plan.length * plan.width > 1.1 or plan.arms or plan.cargo else 2
    wheel = {"shape": "wheel", "size": [wheel_d, wheel_w], "actuator": {"model": "jga25", "gear": gear}}
    axle_z = r3(-thick / 2 + 0.01 * s)
    y = r3(width / 2 + wheel_w / 2 + 0.006)
    parts: list[dict[str, Any]] = []
    if plan.wheels == 2:
        parts.append({"name": "left_wheel", "at": [0, y, axle_z], "mirror": True, **wheel})
        caster = r3(wheel_d / 2 - 0.01 * s - 0.002)  # its top just under the base, its bottom on the floor
        cz = r3(axle_z - wheel_d / 2 + caster / 2)
        for name, x in (("front_caster", length / 2 - caster), ("rear_caster", -(length / 2 - caster))):
            parts.append({"name": name, "at": [r3(x), 0, cz], "shape": "caster", "size": [r3(caster)]})
    else:
        count = plan.wheels // 2
        spacing = r3((length - wheel_d * 0.9) / (count - 1))
        first = r3((length - wheel_d * 0.9) / 2)
        parts.append(
            {
                "name": "left_wheel",
                "at": [first, y, axle_z],
                "mirror": True,
                "repeat": {"count": count, "along": "back", "spacing": spacing},
                **wheel,
            }
        )
    deck = "base"
    if look.bumper:
        parts.append(
            {
                "name": "bumper",
                "at": "front",
                "joint": "fixed",
                "shape": "box",
                "size": [r3(0.025 * s), r3(width * 0.8), r3(thick * 0.6)],
                "direction": "forward",
                "offset": [0, 0, r3(-thick * 0.15)],
                "color": GRAPHITE,
            }
        )
    top_parts: list[dict[str, Any]] = []
    tall = plan.cargo or plan.height > 1.25
    notes: list[str] = []
    if tall:  # a courier's cargo box, as tall as the plan asks
        box_h = r3(0.3 * plan.height * s)
        if plan.arms and box_h > CARGO_TALL * length:
            # Short rovers with an arm on a box twice as tall as they are long tipped over driving off (2026-10-19).
            box_h = r3(CARGO_TALL * length)
            notes.append(
                f"its cargo box is {box_h:.2f} m tall, not taller: with an arm on it, a shorter rover tips over"
            )
        top_parts.append(
            {
                "name": "cargo",
                "at": "top",
                "shift": [r3(-length * 0.05), 0, 0],
                "joint": "fixed",
                "shape": "box",
                "size": [r3(length * 0.78), r3(width * 0.9), box_h],
                "mass": r3(0.6 * s**2 * plan.height),  # a thin-walled bin
                "direction": "up",
                "color": "main",
            }
        )
        top_parts.append(
            {
                "name": "lid",
                "parent": "cargo",
                "at": "end",
                "joint": "fixed",
                "shape": "box",
                "size": [r3(length * 0.8), r3(width * 0.92), 0.012],
                "direction": "up",
                "color": look.accent,
            }
        )
        top_parts.append(
            {
                "name": "light_bar",
                "parent": "cargo",
                "at": "front",
                "shift": [0, 0, r3(box_h * 0.3)],
                "joint": "fixed",
                "shape": "box",
                "size": [0.008, r3(width * 0.6), r3(0.025 * s)],
                "direction": "forward",
                "color": DARK,
                "sensors": [{"type": "camera", "at": "front"}] if plan.camera else [],
            }
        )
        deck = "lid"
    if plan.arms >= 2:
        top_parts += torso_with_arms("top", [r3(length * 0.12), 0, 0], plan, look, parent=deck, riser=0.16 * s)
    elif plan.arms == 1:
        arm = arm_on_top(plan, look, 0.42 * s, length * 0.22, home={"yaw": 180} if not tall else None)
        if deck != "base":
            arm["parent"] = deck
            arm["shift"] = [r3(length * 0.2), 0, 0]
        top_parts.append(arm)
    if plan.head and plan.arms == 0:  # an arm has the deck to itself: it folds back over it to drive
        mast_h = r3(0.12 * s * (1.0 if not tall else 0.5))
        top_parts += [
            {
                "name": "mast",
                "parent": deck,
                "at": "top",
                "shift": [r3(-length * 0.3), r3(width * 0.25), 0],
                "joint": "fixed",
                "shape": "cylinder",
                "size": [r3(0.022 * s), mast_h],
                "direction": "up",
                "color": STEEL,
            },
            *head("mast", "end", 0.06 * s, look, plan),
        ]
    parts += top_parts
    lidar_at = [r3(length * 0.32), 0, r3(thick / 2)] if not tall else "top"
    sensors = base_sensors(plan, lidar_at=lidar_at, camera=not plan.head or plan.arms > 0)
    if tall and plan.lidar:  # high up on the lid, where it sees over the cargo box
        sensors = ["imu"]
        for p in parts:
            if p["name"] == "lid":
                p["sensors"] = [{"type": "lidar", "at": "top", "rays": 24}]
    base = {
        "shape": "box",
        "size": [length, width, thick],
        "mass": r3(3.0 * s**3 * plan.length * plan.width * (2.0 if plan.arms or tall else 1.0)),
        "sensors": sensors,
        "material": look.material,
    }
    return {"color": look.main, "motor": "jga25", "base": base, "parts": parts}, notes


# ---- balancer, snake, arm, humanoid ------------------------------------------------------------------------------


def balancer(plan: Plan, look: Look) -> tuple[dict[str, Any], list[str]]:
    """Two wheels on one axle under a tall body: it balances by itself (LQR) and drives."""
    s = plan.size
    tall = r3(0.22 * plan.height * s)
    width = r3(0.14 * plan.width * s)
    depth = r3(0.07 * plan.length * s)
    wheel_d = r3(0.12 * s)
    parts: list[dict[str, Any]] = [
        {
            "name": "left_wheel",
            "at": [0, r3(width / 2 + 0.025), r3(-tall / 2)],
            "shape": "wheel",
            "size": [wheel_d, r3(0.025 * s)],
            "mirror": True,
        },
    ]
    if plan.head:
        parts += head("base", "top", 0.07 * s, look, plan, pan=False)
    notes: list[str] = []  # arms come off the plan in clamp()
    base = {
        "shape": "box",
        "size": [depth, width, tall],
        "sensors": base_sensors(plan, camera=not plan.head),
        "material": look.material,
    }
    return {"color": look.main, "motor": "jga25", "base": base, "parts": parts}, notes


def snake(plan: Plan, look: Look) -> tuple[dict[str, Any], list[str]]:
    """A chain of side-to-side segments on passive wheels (the builder guide's snake), a head with eyes in front and,
    when asked, a gripper for jaws on a neck that lifts it."""
    s = plan.size * plan.width
    # Carbon unless the sentence said a material: a stiff chain slithers further than a printed one (a x1.35 snake with
    # jaws and 12 segments: printed 35 %, carbon 70 % of the way; sweep --body snake --set material=pla,carbon).
    material = plan.material or "carbon"
    seg = r3(0.08 * s * plan.length)
    parts: list[dict[str, Any]] = [
        {
            "name": "segment",
            "at": "back",
            "joint": {"axis": "z", "range": [-70, 70]},
            "shape": "box",
            "size": [seg, r3(0.05 * s), r3(0.04 * s)],
            "direction": "back",
            "color": "main",
            "material": material,
            "repeat": {"count": plan.segments, "chain": True},
        },
        {
            "name": "left_wheel",
            "parent": "segment",
            "at": [r3(-seg / 2), r3(0.031 * s), r3(-0.004 * s)],
            "shape": "wheel",
            "size": [r3(0.036 * s), r3(0.008 * s)],
            "actuator": "none",
            "color": GRAPHITE,
            "mirror": True,
        },
        {
            "name": "stripe",
            "parent": "segment",
            "at": "top",
            "joint": "fixed",
            "shape": "box",
            "size": [r3(seg * 0.5), r3(0.03 * s), 0.004],
            "direction": "up",
            "color": look.accent,
        },
    ]
    sensors: list[Any] = ["imu"]
    if plan.camera and plan.arms == 0:
        sensors.append({"type": "camera", "at": "front"})
    if plan.lidar:
        sensors.append({"type": "lidar", "at": "top", "rays": 16})
    if plan.arms:  # jaws: a gripper on a neck that lifts the head off the ground
        parts += [
            {
                "name": "neck",
                "at": "front",
                "joint": {"axis": "y", "range": [-60, 30], "home": -30},  # jaws up: lower, they drag
                "shape": "box",
                "size": [r3(0.05 * s), r3(0.045 * s), r3(0.035 * s)],
                "direction": "forward",
                "color": look.accent,
                "sensors": [{"type": "camera", "at": "top", "direction": "forward"}] if plan.camera else [],
            },
            {
                "name": "jaws",
                "parent": "neck",
                "at": "end",
                "rotate": [0, 90, 0],
                "robot": gripper(plan.hand, look.accent, 0.05),
            },
        ]
    notes: list[str] = []  # a second arm comes off the plan in clamp()
    if plan.arms:
        notes.append("the jaws weigh its head down: it slithers slowly, about 60 % of the speed asked for")
    base = {"shape": "box", "size": [r3(0.09 * s), r3(0.06 * s), r3(0.045 * s)], "sensors": sensors}
    # Measured: on hobby servos a snake 1.6 times the size did not move at all (1.3 times: 0.43 m in 6 s on sts3215);
    # on joint modules it slithered 0.85 m in 6 s. On mg996r servos twelve segments went 15 to 44 % of the way when
    # carbon or aluminum, and 0.6-size snakes with jaws 30 %; on sts3215 all but a long aluminum one (53 %) went the
    # whole way (2026-10-19, scripts/gen_bench.py sweep --body snake).
    k = plan.size * plan.width
    chain = plan.segments * plan.length / 8  # times the 8-segment snake's length
    heavy = chain > 1.5
    servo = "bldc_joint" if k > 1.2 or heavy else "sts3215"
    return {"color": look.main, "servo": servo, "base": base, "parts": parts}, notes


def standing_arm(plan: Plan, look: Look) -> tuple[dict[str, Any], list[str]]:
    """An arm bolted to a pedestal: the arm template (IK, pick_and_place), a tentacle, or a torso with two."""
    s = plan.size
    two = plan.arms >= 2
    # Two arms hang from shoulders, so they stand on a pedestal the height of a workbench (the twin_arm design's),
    # held up over the table in front; a short one put the hands into the floor.
    pedestal_h = r3((0.36 if two else 0.12) * plan.height * s)
    base = {
        "shape": "cylinder",
        "size": [r3((0.22 if two else 0.16) * s * plan.width), pedestal_h],
        "fixed": True,
        "color": GRAPHITE,
        "material": look.material,
        "sensors": base_sensors(plan, lidar_at="front", camera=True)[1:],  # no imu: it is bolted down
    }
    if two:
        parts = torso_with_arms("top", [0, 0, 0], plan, look, raised=True)
        return {"color": look.main, "base": base, "parts": parts}, []
    if plan.tentacle:
        return {"color": look.main, "base": base, "parts": tentacle_parts(plan, look)}, []
    reach = r3(min(0.45 * plan.length * s, 1.0))
    params: dict[str, Any] = {"reach": reach, "dof": plan.joints, "color": look.main, "material": look.limb_material}
    if reach > 0.5:
        params["servo"] = "bldc_joint"
    if plan.hand != "gripper":
        params["end_effector"] = plan.hand
    parts = [{"name": "arm", "at": "top", "robot": {"template": "arm", "params": params}}]
    return {"color": look.main, "base": base, "parts": parts}, []


def tentacle_parts(plan: Plan, look: Look) -> list[dict[str, Any]]:
    """The gallery's tentacle: short segments bending up-down and side to side in turn, tapering to a gripper."""
    n = max(4, min(plan.segments, 10))
    s = plan.size
    parts: list[dict[str, Any]] = [
        {
            "name": "turret",
            "at": "top",
            "joint": {"axis": "z", "range": [-180, 180]},
            "shape": "cylinder",
            "size": [r3(0.09 * s), 0.03],
            "direction": "up",
            "color": GRAPHITE,
        }
    ]
    parent = "turret"
    for i in range(n):
        share = i / (n - 1)
        bend = (30 + 20 * share) if i % 2 == 0 and i >= 2 else 0
        segment: dict[str, Any] = {
            "name": f"segment_{i + 1}",
            "parent": parent,
            "at": "end",
            "joint": {"axis": "y" if i % 2 == 0 else "x", "range": [-75, 75], "home": round(bend)},
            "size": [r3((0.054 - 0.028 * share) * s), r3((0.078 - 0.026 * share) * s * plan.length * 8 / n)],
            "mass": r3(0.05 - 0.025 * share),
            "direction": "up",
            "color": "main" if i % 2 == 0 else look.accent,
            "actuator": "bldc_joint" if i < 3 else {"model": "sts3215", "gear": 2},
        }
        parts.append(segment)
        parent = segment["name"]
    parts.append({"name": "tip", "parent": parent, "at": "end", "robot": gripper(plan.hand, look.main)})
    return parts


def humanoid(plan: Plan, look: Look) -> tuple[dict[str, Any], list[str]]:
    """The humanoid template: it walks with its own controller, sized by the plan."""
    height = r3(HUMANOID_HEIGHT * plan.height * plan.size)  # clamp() keeps it where it walks
    params: dict[str, Any] = {"height": height, "color": look.main, "head": plan.head}
    # Only a material the sentence asked for: aluminum limbs a seed picked made 0.54 m humanoids fall.
    if plan.material in ("aluminum", "carbon"):
        params["material"] = plan.material
    if height > 0.5:  # measured: 0.54 m fell on hobby servos in some variations; on joint modules 0.65 m walks
        params["servo"] = "bldc_joint"
    return {"template": "humanoid", "params": params}, []  # what it cannot carry came off the plan in clamp()
