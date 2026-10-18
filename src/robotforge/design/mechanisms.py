"""Things in a scene that move on a joint of their own: drawers, cabinet doors, buttons, switches, levers, valve
wheels and conveyor belts, and soft things: a rope and a cloth.

Each mechanism is a fixed carcass (a static body named after the object) and one moving part on a joint, named
"<object>/<part>" for both the body and its joint, so "drawer" is the chest and "drawer/drawer" the drawer in it.
The part a hand takes hold of carries a site "<object>/handle", whose x axis runs along the bar or rim. Sizes follow
props.py: base on the ground at the middle of the footprint, length along x, depth along y, the front toward -y.

What each one is doing (how far open, pressed, which way up, how fast) is read by sim/mechanisms.py, which also
runs what needs running while time passes (a belt's wrap, counting presses). Arms work them with operate
(behaviors/operate.py).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import mujoco
import numpy as np

G = mujoco.mjtGeom
J = mujoco.mjtJoint
PANEL = 0.012  # m, carcass board thickness
CLEAR = 0.003  # m between a moving part and the carcass around it
METAL = [0.72, 0.74, 0.77, 1.0]
DARK = [0.2, 0.21, 0.23, 1.0]
RUBBER = [0.1, 0.1, 0.11, 1.0]
BAR = 0.016  # m, a handle bar's square section: a finger fits round it, a suction cup sits on its flat top
STANDOFF = 0.03  # m between a handle bar and the face it is on: room for a finger behind it
# m from the top of a drawer front or a door down to the middle of its bar. A gripper's palm sits 4 cm above the
# point between its fingertips (ours, and a Panda's); held from above lower down, it pressed on the front.
HANDLE_BELOW_TOP = 0.017
MATTE = "prop_matte"
# A conveyor's belt is one long plate that slides with the things on it and is put back by exactly this much
# whenever it has gone that far (see sim/mechanisms.py), so it can run for ever. The plate is longer than the
# frame by the same amount, never more past either end than this.
BELT_WRAP = 0.015  # m
BELT_THICK = 0.01  # m
CONVEYOR_KV = 200.0  # N per m/s: the drive holds its speed under a few kilos of boxes within a centimetre a second
BUTTON_TRAVEL = 0.008  # m a button goes down
PRESSED_AT = 0.6  # fraction of its travel past which a button reads as pressed


@dataclass(frozen=True)
class Mechanism:
    """How a mechanism moves and is worked: its moving part and joint, what operate can do with it and in which
    units, and how the joint's value reads."""

    part: str  # the moving part's name after "<object>/" (body and joint)
    actions: tuple[str, ...]  # what operate does with it
    unit: str  # "m" (slides) or "deg" (hinges)
    grip: str  # how a hand takes hold: "bar" (fingers round a bar, a cup on top), "knob" (a ball), "top" (pushed)
    summary: str


MECHANISMS: dict[str, Mechanism] = {
    "drawer": Mechanism("drawer", ("open", "close"), "m", "bar", "a chest with a drawer that pulls out toward -y"),
    "cabinet": Mechanism(
        "door", ("open", "close"), "deg", "bar", "a cabinet with a door hinged on its left, opening toward -y"
    ),
    "button": Mechanism("cap", ("press",), "m", "top", "a push button on a box; it springs back up"),
    "switch": Mechanism("toggle", ("press", "turn"), "deg", "knob", "a toggle switch: tipped toward -y is on"),
    "lever": Mechanism("lever", ("press", "turn"), "deg", "knob", "a lever on a box, tipping toward -y or +y"),
    "valve": Mechanism(
        "wheel", ("open", "close", "turn"), "deg", "bar", "a valve with a handwheel on top: counter-clockwise opens"
    ),
    "conveyor": Mechanism("belt", (), "m", "none", "a conveyor belt along x; things on it ride at its speed"),
}
SOFT = ("rope", "cloth")
SHAPES = (*MECHANISMS, *SOFT)
CONTAINERS = ("drawer",)  # pick_and_place can put things in these
DEFAULT_SPEED = 0.1  # m/s a conveyor runs at unless told otherwise
VALVE_TURNS = 2.0  # turns from closed to fully open
SWITCH_RANGE = math.radians(25)
LEVER_RANGE = math.radians(40)


def home(shape: str, name: str, speed: float | None) -> tuple[dict[str, float], dict[str, float]]:
    """A mechanism's starting joint values and controls, by name after "<object>/": a switch starts off, a lever
    back, a conveyor at its speed."""
    if shape == "switch":
        return {"toggle": -SWITCH_RANGE * 0.9}, {}
    if shape == "lever":
        return {"lever": -LEVER_RANGE * 0.9}, {}
    if shape == "conveyor":
        return {}, {"speed": DEFAULT_SPEED if speed is None else speed}
    return {}, {}


def build(
    spec: mujoco.MjSpec,
    name: str,
    shape: str,
    size: list[float],
    rgba: list[float],
    base: list[float],
    quat: list[float],
    fixed: bool,
) -> None:
    """Add mechanism or soft thing `shape` called `name` at `base` (on the ground there), turned by `quat`."""
    if shape in SOFT:
        _SOFT[shape](spec, name, size, rgba, base, quat, fixed)
        return
    if not any(m.name == MATTE for m in spec.materials):
        spec.add_material(name=MATTE, specular=0.08, shininess=0.1, reflectance=0.0)
    carcass = spec.worldbody.add_body(name=name, pos=base, quat=quat)
    _BUILDERS[shape](spec, carcass, name, size, rgba)
    # MuJoCo leaves out contacts between a body and its parent, but not when the parent is welded to the world, as
    # a carcass is: a button's cap sat on its own collar and a 5 N push moved it 1 mm.
    spec.add_exclude(bodyname1=name, bodyname2=f"{name}/{MECHANISMS[shape].part}")


# ---- helpers ------------------------------------------------------------------------------------------


def _box(body: mujoco.MjsBody, name: str | None, half, pos, rgba, collide: bool = True, **kw) -> mujoco.MjsGeom:
    extra = {} if collide else {"contype": 0, "conaffinity": 0}
    return body.add_geom(
        **({"name": name} if name else {}),
        type=G.mjGEOM_BOX,
        size=list(half),
        pos=list(pos),
        rgba=list(rgba),
        material=MATTE,
        **extra,
        **kw,
    )


def _shade(rgba: list[float], k: float) -> list[float]:
    return [min(1.0, c * k) for c in rgba[:3]] + [rgba[3]]


def _carcass(body: mujoco.MjsBody, name: str, length: float, depth: float, height: float, rgba) -> None:
    """An open-fronted box of boards: bottom, top, sides and back."""
    t = PANEL
    _box(body, name, [length / 2, depth / 2, t / 2], [0, 0, t / 2], rgba)
    _box(body, f"{name}/top", [length / 2, depth / 2, t / 2], [0, 0, height - t / 2], rgba)
    for k, sx in enumerate((1, -1)):
        side = [t / 2, depth / 2, (height - 2 * t) / 2]
        _box(body, f"{name}/side{k}", side, [sx * (length / 2 - t / 2), 0, height / 2], rgba)
    back = [length / 2 - t, t / 2, (height - 2 * t) / 2]
    _box(body, f"{name}/back", back, [0, depth / 2 - t / 2, height / 2], _shade(rgba, 0.9))


def _handle(body: mujoco.MjsBody, name: str, center: list[float], length: float, face_y: float) -> None:
    """A bar handle along x at `center` on a face at `face_y` (the bar stands STANDOFF in front of it, toward -y),
    with two posts, and the handle site on the bar."""
    _box(body, f"{name}/handle", [length / 2, BAR / 2, BAR / 2], center, METAL, friction=[1.2, 0.02, 0.002])
    for sx in (1, -1):
        post_y = (face_y + center[1] + BAR / 2) / 2
        half = [0.004, abs(face_y - (center[1] + BAR / 2)) / 2 + 0.001, 0.004]
        _box(body, None, half, [center[0] + sx * (length / 2 - 0.006), post_y, center[2]], METAL)
    body.add_site(name=f"{name}/handle", pos=center, size=[0.005, 0, 0], rgba=[1, 0.6, 0.1, 0])


# ---- drawer -------------------------------------------------------------------------------------------


def _drawer(spec: mujoco.MjSpec, carcass: mujoco.MjsBody, name: str, size: list[float], rgba) -> None:
    length, depth, height = size
    t = PANEL
    _carcass(carcass, name, length, depth, height, rgba)
    front_w, front_h = length - 2 * t - 2 * CLEAR, height - 2 * t - 2 * CLEAR
    inner_d = depth - t - 2 * t  # behind the front panel, short of the back
    drawer = carcass.add_body(name=f"{name}/drawer", pos=[0, -depth / 2 + t / 2, height / 2])
    drawer.add_joint(
        name=f"{name}/drawer",
        type=J.mjJNT_SLIDE,
        axis=[0, -1, 0],
        range=[0, 0.9 * inner_d],
        limited=mujoco.mjtLimited.mjLIMITED_TRUE,
        damping=3.0,
        frictionloss=0.6,
        armature=0.02,
    )
    face = _shade(rgba, 1.08)
    mass = {"mass": 0.12}
    _box(drawer, f"{name}/drawer_front", [front_w / 2, t / 2, front_h / 2], [0, 0, 0], face, **mass)
    tb = 0.006  # the drawer box's own thin boards
    box_w, wall_h = front_w - 0.004, front_h * 0.75
    floor_z = -front_h / 2 + tb / 2
    y_mid = t / 2 + inner_d / 2
    _box(drawer, f"{name}/drawer_bottom", [box_w / 2, inner_d / 2, tb / 2], [0, y_mid, floor_z], face, **mass)
    for k, sx in enumerate((1, -1)):
        _box(
            drawer,
            f"{name}/drawer_side{k}",
            [tb / 2, inner_d / 2, wall_h / 2],
            [sx * (box_w / 2 - tb / 2), y_mid, -front_h / 2 + wall_h / 2],
            face,
            **mass,
        )
    _box(
        drawer,
        f"{name}/drawer_back",
        [box_w / 2 - tb, tb / 2, wall_h / 2],
        [0, t / 2 + inner_d - tb / 2, -front_h / 2 + wall_h / 2],
        face,
        **mass,
    )
    bar = min(0.12, 0.55 * front_w)
    # Near the top edge: a hand coming down on it from above has its palm over the chest, not against its front.
    _handle(drawer, name, [0, -t / 2 - STANDOFF - BAR / 2, front_h / 2 - HANDLE_BELOW_TOP], bar, -t / 2)
    # Where things put in it rest: the middle of its floor, and how far its walls go each way (for pick_and_place).
    drawer.add_site(
        name=f"{name}/inside",
        pos=[0, y_mid, floor_z + tb / 2],
        size=[box_w / 2 - tb, inner_d / 2 - tb, wall_h - tb],
        type=G.mjGEOM_BOX,
        rgba=[0, 0, 0, 0],
    )


# ---- cabinet ------------------------------------------------------------------------------------------


def _cabinet(spec: mujoco.MjSpec, carcass: mujoco.MjsBody, name: str, size: list[float], rgba) -> None:
    length, depth, height = size
    t = PANEL
    _carcass(carcass, name, length, depth, height, rgba)
    _box(carcass, f"{name}/shelf", [length / 2 - t, depth / 2 - t, t / 2], [0, -t / 2, height / 2], rgba)
    # An overlay door in front of the carcass, hinged on its -x edge; opening turns its free edge toward -y.
    door = carcass.add_body(name=f"{name}/door", pos=[-length / 2, -depth / 2 - t / 2 - CLEAR, 0])
    door.add_joint(
        name=f"{name}/door",
        type=J.mjJNT_HINGE,
        axis=[0, 0, -1],
        range=[0, math.radians(110)],
        limited=mujoco.mjtLimited.mjLIMITED_TRUE,
        damping=0.4,
        frictionloss=0.15,
        armature=0.01,
    )
    _box(
        door,
        f"{name}/door_panel",
        [length / 2, t / 2, height / 2 - 0.002],
        [length / 2, 0, height / 2],
        _shade(rgba, 1.08),
        mass=0.5,
    )
    bar = min(0.1, 0.4 * length)
    center = [length - 0.025 - bar / 2, -t / 2 - STANDOFF - BAR / 2, height - 0.002 - HANDLE_BELOW_TOP]
    _handle(door, name, center, bar, -t / 2)


# ---- button, switch, lever ----------------------------------------------------------------------------


def _button(spec: mujoco.MjSpec, carcass: mujoco.MjsBody, name: str, size: list[float], rgba) -> None:
    diameter, height = size
    housing = max(diameter + 0.03, 0.06)
    cap_h = 0.012
    body_h = height - cap_h
    _box(carcass, name, [housing / 2, housing / 2, body_h / 2], [0, 0, body_h / 2], DARK)
    carcass.add_geom(  # the collar round the cap
        type=G.mjGEOM_CYLINDER,
        size=[diameter / 2 + 0.005, 0.002, 0],
        pos=[0, 0, body_h + 0.002],
        rgba=METAL,
        material=MATTE,
    )
    cap = carcass.add_body(name=f"{name}/cap", pos=[0, 0, body_h + cap_h / 2 + 0.004])
    # A spring with its rest point above the top stop: the cap sits up against it with a little preload, and a
    # press of about 1.5 N takes it all the way down, like a panel push button.
    cap.add_joint(
        name=f"{name}/cap",
        type=J.mjJNT_SLIDE,
        axis=[0, 0, 1],
        range=[-BUTTON_TRAVEL, 0],
        limited=mujoco.mjtLimited.mjLIMITED_TRUE,
        stiffness=150.0,
        springref=0.002,
        damping=1.5,
        armature=0.005,
    )
    cap.add_geom(
        name=f"{name}/cap",
        type=G.mjGEOM_CYLINDER,
        size=[diameter / 2, cap_h / 2, 0],
        rgba=rgba,
        mass=0.02,
        material=MATTE,
    )
    cap.add_site(name=f"{name}/handle", pos=[0, 0, cap_h / 2], size=[0.005, 0, 0], rgba=[1, 0.6, 0.1, 0])


def _tipping(
    carcass: mujoco.MjsBody,
    name: str,
    part: str,
    base: list[float],
    arm: float,
    knob: float,
    swing: float,
    rgba,
    hold: float,
) -> None:
    """A base box with a stick on top that tips about x (toward -y is positive), held where it is put by friction,
    and a ball on its end to take hold of."""
    bx, by, bz = base
    _box(carcass, name, [bx / 2, by / 2, bz / 2], [0, 0, bz / 2], DARK)
    stick = carcass.add_body(name=f"{name}/{part}", pos=[0, 0, bz])
    # Friction holds it where it is put. MuJoCo's friction gives way slowly under a steady load (it is a soft
    # constraint): a lever crept 4 degrees in 10 s under its own weight, onto its stop, and a hand reaching for its
    # knob closed on air. Some inertia in the joint and a stiffer friction keep that under 0.2 degrees.
    stick.add_joint(
        name=f"{name}/{part}",
        type=J.mjJNT_HINGE,
        axis=[1, 0, 0],
        range=[-swing, swing],
        limited=mujoco.mjtLimited.mjLIMITED_TRUE,
        frictionloss=hold,
        damping=hold * 0.2,
        armature=0.08 * hold + 0.001,
        solref_friction=[0.004, 1.0],
    )
    radius = max(0.003, knob * 0.45)
    stick.add_geom(
        type=G.mjGEOM_CAPSULE, fromto=[0, 0, 0, 0, 0, arm], size=[radius, 0, 0], rgba=METAL, mass=0.02, material=MATTE
    )
    stick.add_geom(
        name=f"{name}/knob",
        type=G.mjGEOM_SPHERE,
        size=[knob, 0, 0],
        pos=[0, 0, arm],
        rgba=rgba,
        mass=0.01,
        material=MATTE,
        friction=[1.6, 0.02, 0.002],
    )
    stick.add_site(name=f"{name}/handle", pos=[0, 0, arm], size=[0.005, 0, 0], rgba=[1, 0.6, 0.1, 0])


def _switch(spec: mujoco.MjSpec, carcass: mujoco.MjsBody, name: str, size: list[float], rgba) -> None:
    length, width, height = size
    _tipping(carcass, name, "toggle", [length, width, height], 0.035, 0.009, SWITCH_RANGE, rgba, hold=0.012)


def _lever(spec: mujoco.MjSpec, carcass: mujoco.MjsBody, name: str, size: list[float], rgba) -> None:
    arm = size[0]
    _tipping(carcass, name, "lever", [0.08, 0.06, 0.04], arm, 0.016, LEVER_RANGE, rgba, hold=0.25)


# ---- valve --------------------------------------------------------------------------------------------


def _valve(spec: mujoco.MjSpec, carcass: mujoco.MjsBody, name: str, size: list[float], rgba) -> None:
    diameter, height = size
    stem_top = height - 0.02
    carcass.add_geom(
        name=name, type=G.mjGEOM_CYLINDER, size=[0.05, 0.006, 0], pos=[0, 0, 0.006], rgba=DARK, material=MATTE
    )
    carcass.add_geom(
        type=G.mjGEOM_CYLINDER, size=[0.022, stem_top / 2, 0], pos=[0, 0, stem_top / 2], rgba=METAL, material=MATTE
    )
    carcass.add_geom(
        type=G.mjGEOM_CYLINDER, size=[0.035, 0.012, 0], pos=[0, 0, stem_top * 0.45], rgba=DARK, material=MATTE
    )  # the valve body round the pipe
    wheel = carcass.add_body(name=f"{name}/wheel", pos=[0, 0, height - 0.008])
    wheel.add_joint(
        name=f"{name}/wheel",
        type=J.mjJNT_HINGE,
        axis=[0, 0, 1],
        range=[0, VALVE_TURNS * 2 * math.pi],
        limited=mujoco.mjtLimited.mjLIMITED_TRUE,
        damping=0.02,
        frictionloss=0.04,
        armature=0.002,
    )
    rim_r = diameter / 2 - BAR / 2
    segments = 16
    seg_len = 2 * rim_r * math.tan(math.pi / segments) + 0.002
    for k in range(segments):
        a = 2 * math.pi * k / segments
        wheel.add_geom(
            name=f"{name}/rim{k}",
            type=G.mjGEOM_BOX,
            size=[seg_len / 2, BAR / 2, BAR * 0.4],
            pos=[rim_r * math.cos(a), rim_r * math.sin(a), 0],
            quat=[math.cos((a + math.pi / 2) / 2), 0, 0, math.sin((a + math.pi / 2) / 2)],
            rgba=rgba,
            mass=0.02,
            material=MATTE,
            friction=[1.6, 0.02, 0.002],  # a rubber-coated rim
        )
    # Dished, as handwheels are: the spokes run down to a hub below the rim, out of the way of a finger inside it.
    dish = 0.026
    for k in range(3):
        a = 2 * math.pi * k / 3 + math.pi / 6
        r = rim_r - BAR / 2
        wheel.add_geom(
            type=G.mjGEOM_CAPSULE,
            fromto=[0.012 * math.cos(a), 0.012 * math.sin(a), -dish, r * math.cos(a), r * math.sin(a), -0.003],
            size=[0.004, 0, 0],
            rgba=METAL,
            mass=0.01,
            material=MATTE,
        )
    wheel.add_geom(
        type=G.mjGEOM_CYLINDER, size=[0.016, 0.01, 0], pos=[0, 0, -dish], rgba=METAL, mass=0.02, material=MATTE
    )
    # Taken hold of at the front of its rim (toward -y), the bar running along the rim.
    wheel.add_site(name=f"{name}/handle", pos=[0, -rim_r, 0], size=[0.005, 0, 0], rgba=[1, 0.6, 0.1, 0])


# ---- conveyor -----------------------------------------------------------------------------------------


def _conveyor(spec: mujoco.MjSpec, carcass: mujoco.MjsBody, name: str, size: list[float], rgba) -> None:
    length, width, height = size
    top = height
    rail_h = 0.03
    # Frame: side rails that stand a little above the belt (they keep things on it), legs and end rollers.
    for k, sy in enumerate((1, -1)):
        _box(
            carcass,
            name if k == 0 else f"{name}/rail{k}",
            [length / 2 + 0.03, 0.012, (rail_h + 0.05) / 2],
            [0, sy * (width / 2 + 0.014), top + rail_h - (rail_h + 0.05) / 2],
            rgba,
        )
        for sx in (1, -1):
            leg_h = top - 0.05
            _box(
                carcass,
                None,
                [0.02, 0.02, leg_h / 2],
                [sx * (length / 2 - 0.04), sy * (width / 2 + 0.014), leg_h / 2],
                DARK,
            )
    for sx in (1, -1):
        carcass.add_geom(
            type=G.mjGEOM_CYLINDER,
            size=[BELT_THICK * 2, width / 2, 0],
            pos=[sx * length / 2, 0, top - BELT_THICK * 2],
            quat=[math.cos(math.pi / 4), math.sin(math.pi / 4), 0, 0],
            rgba=METAL,
            contype=0,
            conaffinity=0,
            material=MATTE,
        )
    _box(carcass, None, [length / 2, width / 2 - 0.002, 0.004], [0, 0, top - BELT_THICK - 0.006], DARK, collide=False)
    belt = carcass.add_body(name=f"{name}/belt", pos=[0, 0, 0])
    belt.add_joint(
        name=f"{name}/belt",
        type=J.mjJNT_SLIDE,
        axis=[1, 0, 0],
        range=[-1.0, 1.0],
        limited=mujoco.mjtLimited.mjLIMITED_FALSE,
        armature=0.5,
    )
    # One plate: as long as the frame plus one wrap, placed so that at any slide from 0 to a wrap it covers the
    # frame from end to end (it reaches past the frame's ends by no more than a wrap).
    belt.add_geom(
        name=f"{name}/belt",
        type=G.mjGEOM_BOX,
        size=[(length + BELT_WRAP) / 2, width / 2, BELT_THICK / 2],
        pos=[-BELT_WRAP / 2, 0, top - BELT_THICK / 2],
        rgba=RUBBER,
        mass=2.0,
        friction=[1.0, 0.02, 0.002],
        material=MATTE,
    )
    # Cleats every wrap: the belt is seen to run, and the put-back by one wrap leaves them where they were.
    count = int((length + BELT_WRAP) / BELT_WRAP)
    for k in range(count):
        x = -length / 2 - BELT_WRAP + (k + 0.5) * BELT_WRAP
        if abs(x) > length / 2 + BELT_WRAP / 2:
            continue
        belt.add_geom(
            type=G.mjGEOM_BOX,
            size=[0.0015, width / 2 - 0.004, 0.0006],
            pos=[x, 0, top + 0.0002],
            rgba=[0.2, 0.2, 0.22, 1],
            contype=0,
            conaffinity=0,
            group=2,
            material=MATTE,
        )
    act = spec.add_actuator(name=f"{name}/speed", target=f"{name}/belt", trntype=mujoco.mjtTrn.mjTRN_JOINT)
    act.set_to_velocity(kv=CONVEYOR_KV)
    act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
    act.ctrlrange = [-0.5, 0.5]
    act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
    act.forcerange = [-150.0, 150.0]


# ---- rope and cloth (MuJoCo composites and flex) ---------------------------------------------------------


def _rope(spec, name, size, rgba, base, quat, fixed) -> None:
    """A cable composite of capsules on ball joints. Fixed, it hangs from a hook at `base` (start: straight out
    along its own x, so it swings down); loose, it lies on the ground along x."""
    length, diameter = size
    count = max(8, min(60, round(length / (2.5 * diameter))))
    color = " ".join(f"{c:.3f}" for c in rgba)
    initial = "none" if fixed else "free"
    xml = f"""
    <mujoco>
      <worldbody>
        <body name="anchor">
          <composite prefix="seg" type="cable" curve="s" count="{count + 1} 1 1" size="{length}" initial="{initial}">
            <joint kind="main" damping="{2e-4 * length:.6f}" armature="1e-6"/>
            <geom type="capsule" size="{diameter / 2}" rgba="{color}" density="900" friction="1 0.01 0.001"
                  condim="3"/>
          </composite>
        </body>
      </worldbody>
    </mujoco>"""
    child = mujoco.MjSpec.from_string(xml)
    child.compiler.degree = False
    lift = 0.0 if fixed else diameter / 2
    frame = spec.worldbody.add_frame(pos=[base[0], base[1], base[2] + lift], quat=quat)
    spec.attach(child, prefix=f"{name}/", frame=frame)
    if fixed:  # the hook it hangs from
        spec.worldbody.add_geom(
            name=f"{name}/hook",
            type=G.mjGEOM_SPHERE,
            size=[diameter, 0, 0],
            pos=base,
            rgba=METAL,
            contype=0,
            conaffinity=0,
        )


def _cloth(spec, name, size, rgba, base, quat, fixed) -> None:
    """A square-meshed cloth (a flex of triangles whose edges keep their length): it lies flat at `base` and drapes
    over what is under it. Fixed, its back edge (+y) is pinned where it starts, like a curtain."""
    length, width = size
    spacing = 0.035
    nx, ny = max(3, round(length / spacing) + 1), max(3, round(width / spacing) + 1)
    color = " ".join(f"{c:.3f}" for c in rgba)
    pin = ""
    if fixed:
        ids = " ".join(str(i * ny + ny - 1) for i in range(nx))
        pin = f'<pin id="{ids}"/>'
    spacing = f"{length / (nx - 1):.5f} {width / (ny - 1):.5f} 1"
    mass = 0.15 * length * width / 0.09  # kg: 150 g a 30 cm square, a tea towel
    xml = f"""
    <mujoco>
      <worldbody>
        <flexcomp name="cloth" type="grid" count="{nx} {ny} 1" spacing="{spacing}"
                  radius="0.003" dim="2" mass="{mass:.4f}" rgba="{color}">
          <edge equality="true" damping="0.002"/>
          <contact condim="3" selfcollide="none" solref="0.01 1"/>
          {pin}
        </flexcomp>
      </worldbody>
    </mujoco>"""
    child = mujoco.MjSpec.from_string(xml)
    frame = spec.worldbody.add_frame(pos=[base[0], base[1], base[2] + 0.004], quat=quat)
    spec.attach(child, prefix=f"{name}/", frame=frame)


_BUILDERS: dict[str, Callable] = {
    "drawer": _drawer,
    "cabinet": _cabinet,
    "button": _button,
    "switch": _switch,
    "lever": _lever,
    "valve": _valve,
    "conveyor": _conveyor,
}
_SOFT: dict[str, Callable] = {"rope": _rope, "cloth": _cloth}


# ---- where things go, for pick_and_place ---------------------------------------------------------------


def inside(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Where things put into container `name` (a drawer) rest: the middle of its floor in the world, its half size
    across the floor in the world's x and y, and the height of its walls above the floor; None when it has none."""
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{name}/inside")
    if site < 0:
        return None
    rot = data.site_xmat[site].reshape(3, 3)
    half = model.site_size[site]
    across = np.abs(rot[:2, :2]) @ half[:2]
    return data.site_xpos[site].copy(), across, float(half[2])
