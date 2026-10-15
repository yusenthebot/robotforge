"""Cart-pole: a cart on a rail with a free-swinging pole, the classic balance-control lesson.

The pole has no motor. The only actuator pushes the cart along the rail, so the
controller has to keep the pole upright by moving the cart under it.
"""

from __future__ import annotations

import math

import mujoco
from pydantic import Field

from robotforge.design import components
from robotforge.design.templates.base import DARK, GEOM, SHELL, BuiltRobot, Template, TemplateParams, new_spec
from robotforge.design.visuals import Visuals
from robotforge.design.visuals import mesh as M
from robotforge.design.visuals import parts as P


class CartpoleParams(TemplateParams):
    rail_length: float = Field(1.6, ge=0.4, le=4.0, description="rail length in meters")
    rail_height: float = Field(0.3, ge=0.1, le=1.0, description="height of the rail above the floor in meters")
    pole_length: float = Field(0.6, ge=0.1, le=2.0, description="pole length in meters")
    pole_mass: float = Field(0.1, gt=0, le=5.0, description="pole mass in kg")
    cart_mass: float = Field(0.5, gt=0, le=20.0, description="cart mass in kg")
    max_force: float = Field(10.0, gt=0, le=200.0, description="strongest push on the cart in newtons")


class Cartpole(Template):
    name = "cartpole"
    summary = (
        "Cart-pole: a cart on a rail pushed by one motor, with an unpowered pole on top. "
        "The classic balance task for control and reinforcement learning."
    )
    Params = CartpoleParams

    def build(self, p: CartpoleParams) -> BuiltRobot:
        accent = components.color(p.color)
        spec = new_spec()
        half_rail = p.rail_length / 2
        cart_half = [0.06, 0.05, 0.03]

        vis = Visuals(spec)
        base = spec.worldbody.add_body(name="base")
        # The rail is only for looks: the slide joint keeps the cart on it, so it must not collide.
        rail = base.add_geom(
            type=GEOM.mjGEOM_BOX,
            size=[half_rail, 0.01, 0.01],
            rgba=DARK,
            contype=0,
            conaffinity=0,
            mass=0.1,
        )
        vis.hide(rail)
        for end in (1, -1):
            post = base.add_geom(
                type=GEOM.mjGEOM_BOX,
                size=[0.015, 0.015, p.rail_height / 2],
                pos=[end * (half_rail + 0.015), 0, -p.rail_height / 2],
                rgba=DARK,
                mass=0.1,
            )
            vis.hide(post)
        _dress_rail(vis, base, half_rail, p.rail_height)

        cart = base.add_body(name="cart")
        cart.add_joint(
            name="slider",
            type=mujoco.mjtJoint.mjJNT_SLIDE,
            axis=[1, 0, 0],
            range=[-(half_rail - cart_half[0]), half_rail - cart_half[0]],
            limited=mujoco.mjtLimited.mjLIMITED_TRUE,
            damping=0.05,
        )
        vis.hide(
            cart.add_geom(type=GEOM.mjGEOM_BOX, size=cart_half, rgba=accent, mass=p.cart_mass, contype=0, conaffinity=0)
        )
        _dress_cart(vis, cart, cart_half, accent)

        pole = cart.add_body(name="pole", pos=[0, -cart_half[1] - 0.012, 0])
        pole.add_joint(name="hinge", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0], damping=0.001)
        radius = max(0.008, 0.02 * p.pole_length)
        rod = pole.add_geom(
            type=GEOM.mjGEOM_CAPSULE,
            fromto=[0, 0, 0, 0, 0, p.pole_length],
            size=[radius, 0, 0],
            rgba=SHELL,
            mass=p.pole_mass * 0.8,
        )
        # A small weight at the tip makes the pole feel like the textbook "mass on a stick".
        weight = pole.add_geom(
            type=GEOM.mjGEOM_SPHERE,
            size=[2.2 * radius],
            pos=[0, 0, p.pole_length],
            rgba=accent,
            mass=p.pole_mass * 0.2,
        )
        vis.hide(rod)
        vis.hide(weight)
        _dress_pole(vis, pole, radius, p.pole_length, accent)
        vis.finish()
        pole.add_site(name="tip", pos=[0, 0, p.pole_length], size=[0.005, 0, 0], rgba=[0, 0, 0, 0])

        act = spec.add_actuator(name="cart", target="slider", trntype=mujoco.mjtTrn.mjTRN_JOINT)
        act.set_to_motor()
        act.gear[0] = p.max_force
        act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
        act.ctrlrange = [-1.0, 1.0]

        spec.add_sensor(
            name="tip_pos", type=mujoco.mjtSensor.mjSENS_FRAMEPOS, objtype=mujoco.mjtObj.mjOBJ_SITE, objname="tip"
        )
        return BuiltRobot(
            spec=spec,
            home_qpos={"slider": 0.0, "hinge": 0.0},
            home_ctrl={"cart": 0.0},
            spawn_height=p.rail_height,
            floating=False,
            notes=[
                f"cart-pole, pole {p.pole_length:.2f} m, push up to {p.max_force:.0f} N; "
                "the pole starts upright and falls unless a controller balances it"
            ],
        )


# ---- looks: visual-only meshes (no mass, no contacts) over the collision primitives above ----------


def _dress_rail(vis: Visuals, base, half_rail: float, height: float) -> None:
    """An aluminium extrusion with dark slots, on two dark stands with foot plates."""
    vis.add(base, "metal", P.METAL, M.rounded_box(half_rail, 0.01, 0.01, 0.0025, segments=2))
    for y in (0.0101, -0.0101):
        vis.add(base, "dark", DARK, M.box(half_rail * 0.995, 0.0004, 0.0022), (0, y, 0))
    for end in (1, -1):
        x = end * (half_rail + 0.015)
        post = M.rounded_box(0.015, 0.015, height / 2 + 0.012, 0.004, segments=2)
        vis.add(base, "dark", DARK, post, (x, 0, -height / 2 + 0.012))
        foot = M.rounded_box(0.045, 0.05, 0.004, 0.003, segments=2, corner=0.01)
        vis.add(base, "dark", DARK, foot, (x, 0, -height + 0.004))
        vis.add(base, "metal", P.METAL, P.bolt_circle(0.035, 4, 0.004, 0.003, math.pi / 4), (x, 0, -height + 0.008))
        stop = M.rounded_box(0.006, 0.02, 0.016, 0.003, segments=2)
        vis.add(base, "rubber", P.RUBBER, stop, (end * (half_rail - 0.006), 0, 0))


def _dress_cart(vis: Visuals, cart, half, accent) -> None:
    hx, hy, hz = half
    body = M.rounded_box(hx, hy, hz * 0.62, hz * 0.4, segments=3, corner=0.012)
    vis.add(cart, "shell", accent, body, (0, 0, hz * 0.38))
    block = M.rounded_box(hx * 0.9, hy * 0.82, hz * 0.45, hz * 0.2, segments=2, corner=0.006)
    vis.add(cart, "dark", DARK, block, (0, 0, -hz * 0.5))
    vis.add(cart, "led", P.LED_GREEN, P.led(0.003)["led"], (hx * 0.6, hy * 0.5, hz))
    # The pivot shaft out of the cart's front face, carrying the pole.
    shaft = M.cylinder(0.006, 0.014, segments=20).moved(rot=M.Z_TO_Y)
    vis.add(cart, "metal", P.METAL, shaft, (0, -hy - 0.005, 0))


def _dress_pole(vis: Visuals, pole, radius: float, length: float, accent) -> None:
    for preset, mesh in P.drum(radius * 1.7, 0.009, radius * 1.1).items():
        vis.add(pole, preset, DARK if preset == "dark" else P.METAL, mesh, (0, 0, 0), M.Z_TO_Y)
    vis.add(pole, "metal", P.METAL, M.cylinder(radius, length, segments=24, bevel=radius * 0.3, z0=0.0))
    vis.add(pole, "shell", accent, M.sphere(2.2 * radius, rings=14, segments=28), (0, 0, length))
