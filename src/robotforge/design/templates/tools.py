"""A suction cup and an electromagnet on their own, to mount on the end of an arm built from parts like the gripper
template: the tool point is on the pad's face and the "gripper" actuator turns it on and off."""

from __future__ import annotations

import mujoco

from robotforge.design import end_effectors
from robotforge.design.templates.base import DARK, GEOM, BuiltRobot, Template, TemplateParams, new_spec
from robotforge.design.visuals import Visuals
from robotforge.design.visuals import mesh as M
from robotforge.design.visuals import parts as P


def _tool_robot(kind: str) -> BuiltRobot:
    spec = new_spec()
    vis = Visuals(spec)
    hand = spec.worldbody.add_body(name="hand")
    # The same wrist plate as the gripper template's, so the tool can go wherever a gripper would.
    flange = hand.add_geom(type=GEOM.mjGEOM_CYLINDER, size=[0.02, 0.005, 0], pos=[0, 0, 0.005], rgba=DARK, mass=0.02)
    vis.hide(flange)
    vis.add(hand, "metal", P.METAL, M.cylinder(0.02, 0.01, segments=40, bevel=0.0012, z0=0.0))
    vis.add(hand, "dark", DARK, P.bolt_circle(0.0145, 4, 0.0018, 0.0012, 0.785), (0, 0, 0.0098))
    face = end_effectors.add_tool(spec, hand, kind, vis, kind, pos=[0, 0, 0.01])
    vis.finish()
    hand.add_site(name="tcp", pos=[0, 0, 0.01 + face], size=[0.006, 0, 0], rgba=[1, 0.6, 0.1, 0.6])
    spec.add_sensor(
        name="tcp_pos", type=mujoco.mjtSensor.mjSENS_FRAMEPOS, objtype=mujoco.mjtObj.mjOBJ_SITE, objname="tcp"
    )
    info = end_effectors.TOOLS[kind]
    return BuiltRobot(
        spec=spec,
        home_ctrl={"gripper": 0.0},
        floating=False,
        notes=[f"{info.title}: holds {info.holds}; gripper close turns it on, open lets go"],
    )


class SuctionCupParams(TemplateParams):
    color: str | list[float] = "black"


class SuctionCup(Template):
    name = "suction_cup"
    summary = (
        "A vacuum suction cup with a tool point on its face; mount it on the end of an arm built from parts (it points "
        "along the mount's +z). Holds anything with a flat face, flat things a gripper cannot grasp included."
    )
    Params = SuctionCupParams

    def build(self, p: SuctionCupParams) -> BuiltRobot:
        return _tool_robot("suction")


class MagnetParams(TemplateParams):
    color: str | list[float] = "red"


class Magnet(Template):
    name = "magnet"
    summary = (
        "An electromagnet with a tool point on its face; mount it on the end of an arm built from parts (it points "
        "along the mount's +z). Holds steel things only (objects with material steel)."
    )
    Params = MagnetParams

    def build(self, p: MagnetParams) -> BuiltRobot:
        return _tool_robot("magnet")
