"""A parallel two-finger gripper on its own, to mount on a robot built from parts (or anywhere else)."""

from __future__ import annotations

import mujoco
from pydantic import Field

from robotforge.design import components
from robotforge.design.templates.arm import add_gripper
from robotforge.design.templates.base import DARK, GEOM, BuiltRobot, Template, TemplateParams, new_spec


class GripperParams(TemplateParams):
    opening: float = Field(0.07, ge=0.02, le=0.2, description="max finger gap in meters")


class Gripper(Template):
    name = "gripper"
    summary = (
        "Parallel two-finger gripper with a tool point; mount it on the end of an arm built from parts "
        "(fingers point along the mount's +z) to use move_arm and pick_and_place."
    )
    Params = GripperParams

    def build(self, p: GripperParams) -> BuiltRobot:
        spec = new_spec()
        accent = components.color(p.color)
        hand = spec.worldbody.add_body(name="hand")
        # A flange like the wrist plate of a real gripper; it also keeps the palm off the arm's last joint.
        hand.add_geom(type=GEOM.mjGEOM_CYLINDER, size=[0.02, 0.005, 0], pos=[0, 0, 0.005], rgba=DARK, mass=0.02)
        palm = hand.add_body(name="palm", pos=[0, 0, 0.01])
        tcp_z = add_gripper(spec, palm, p.opening, accent)
        palm.add_site(name="tcp", pos=[0, 0, tcp_z], size=[0.006, 0, 0], rgba=[1, 0.6, 0.1, 0.6])
        palm.add_camera(name="wrist_cam", pos=[0.035, 0, 0.0], xyaxes=[0, -1, 0, -1, 0, 0], fovy=70)
        spec.add_sensor(
            name="tcp_pos", type=mujoco.mjtSensor.mjSENS_FRAMEPOS, objtype=mujoco.mjtObj.mjOBJ_SITE, objname="tcp"
        )
        return BuiltRobot(
            spec=spec,
            home_ctrl={"gripper": p.opening / 2},
            floating=False,
            notes=[f"gripper, opening {p.opening:.2f} m"],
        )
