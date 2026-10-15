"""Export a robotforge quadruped template as a UniLab task, so UniLab can teach it to walk.

Experimental (feasibility of training robotforge-designed robots with UniLab). Writes:

    <out>/scene.xml                       the robot alone, plus a floor, the sensors UniLab's
                                          rewards read and a "home" keyframe
    <out>/conf/task/<task>/mujoco.yaml    a UniLab task config: Go2's walking task
                                          (go2_joystick_flat/base.yaml from your UniLab checkout)
                                          with this robot's names, gains and sizes

then train, export and install it through robotforge:

    uv run python scripts/unilab_export_quadruped.py --out ~/robotforge/unilab/tasks/my_dog
    robotforge unilab train --task my_dog --conf-dir ~/robotforge/unilab/tasks/my_dog/conf \\
        --iterations 400 --name my_dog_walk --wait
    # in Python or an agent: move(robot="dog", speed=0.3, walker="my_dog_walk")

Needs $ROBOTFORGE_UNILAB_DIR (for the reward and observation template). Sizes that matter for the
rewards (standing height, swing height, speed range) are scaled from Go2 by standing height.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import mujoco
import yaml

from robotforge.behaviors.quadruped import TEMPLATE_LEGS
from robotforge.design import templates

SHORT = {"front_left": "FL", "front_right": "FR", "rear_left": "RL", "rear_right": "RR"}
GO2_HEIGHT = 0.30  # m, the standing height Go2's task rewards


def build_scene(params: dict) -> tuple[mujoco.MjSpec, dict]:
    built = templates.build("quadruped", params)
    spec = built.spec
    for leg in TEMPLATE_LEGS:  # name the feet so contact and position sensors can refer to them
        spheres = [g for g in spec.body(f"{leg}_shin").geoms if g.type == mujoco.mjtGeom.mjGEOM_SPHERE]
        if not spheres:
            raise SystemExit(f"cannot find the foot of {leg}")
        spheres[-1].name = SHORT[leg]
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05])
    root = spec.worldbody.first_body()
    root.pos = [0, 0, built.spawn_height]
    sensor, site, geom = mujoco.mjtSensor, mujoco.mjtObj.mjOBJ_SITE, mujoco.mjtObj.mjOBJ_GEOM
    spec.add_sensor(name="upvector", type=sensor.mjSENS_FRAMEZAXIS, objtype=site, objname="imu")
    spec.add_sensor(name="local_linvel", type=sensor.mjSENS_VELOCIMETER, objtype=site, objname="imu")
    for short in SHORT.values():
        spec.add_sensor(name=f"{short}_pos", type=sensor.mjSENS_FRAMEPOS, objtype=geom, objname=short)
        contact = spec.add_sensor(name=f"{short}_foot_contact", type=sensor.mjSENS_CONTACT)
        contact.objtype, contact.objname, contact.reftype, contact.refname = geom, "floor", geom, short
        contact.intprm[0] = 1  # data="found"
        contact.intprm[1] = 1  # reduce="mindist"
        contact.intprm[2] = 1  # num=1
    model = spec.compile()
    qpos = model.qpos0.copy()
    qpos[2] = built.spawn_height
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if name in built.home_qpos:
            qpos[model.jnt_qposadr[j]] = built.home_qpos[name]
    actuators = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in range(model.nu)]
    spec.add_key(name="home", qpos=qpos.tolist(), ctrl=[built.home_ctrl.get(a, 0.0) for a in actuators])
    spec.option.timestep = 0.01  # UniLab's sim_dt for this task
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    trunk = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, 1)
    joints = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(1, model.njnt)]
    info = {"trunk": trunk, "joints": joints, "actuators": actuators, "height": float(data.xpos[1][2])}
    return spec, info


def owner_config(base: dict, task_name: str, scene: Path, info: dict) -> dict:
    scale = info["height"] / GO2_HEIGHT
    env, reward = base["env"], base["reward"]
    env["scene"]["model_file"] = str(scene.resolve())
    env["scene"]["entities"]["robot"] = {
        "root_body_name": info["trunk"],
        "body_names": [info["trunk"]],
        "joint_names": info["joints"],
        "actuator_names": info["actuators"],
    }
    # Gains come from the robot's own servos; only jitter them, as Go2's task does with its absolute range.
    env["events"]["pd_gains"]["params"] = {"kp_range": [0.9, 1.1], "kd_range": [0.9, 1.1], "operation": "scale"}
    velocity = {k: [-0.2, 0.2] for k in ("x", "y", "z", "roll", "pitch", "yaw")}  # gentler starts for a small robot
    env["events"]["reset_root_state_uniform"]["params"]["velocity_range"] = velocity
    env["commands"]["twist"]["ranges"] = {
        "lin_vel_x": [round(-0.6 * scale, 2), round(1.0 * scale, 2)],
        "lin_vel_y": [round(-0.4 * scale, 2), round(0.4 * scale, 2)],
        "ang_vel_z": [-0.8, 0.8],
    }
    reward["base_height"]["params"]["target_height"] = round(info["height"], 3)
    reward["swing_feet_z"]["params"]["target_height"] = round(0.1 * scale, 3)
    reward["swing_feet_z"]["params"]["kernel"] = round(0.01 * scale, 4)
    reward["swing_feet_z"]["params"]["sensor_names"] = [f"{s}_pos" for s in SHORT.values()]
    reward["contact"]["params"]["sensor_names"] = [f"{s}_foot_contact" for s in SHORT.values()]
    return {
        "training": {"task_name": task_name, "sim_backend": "mujoco"},
        "algo": {  # as go2_joystick_flat/mujoco.yaml
            "num_envs": 1024,
            "max_iterations": 400,
            "obs_groups": {"actor": ["actor"], "critic": ["critic"]},
            "actor": {"obs_normalization": True, "distribution_cfg": {"init_std": 0.5}},
            "critic": {"obs_normalization": True},
            "algorithm": {"learning_rate": 3.0e-4, "entropy_coef": 1.0e-3},
        },
        "env": env,
        "reward": reward,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--task", default=None, help="task folder name (default: the --out folder's name)")
    parser.add_argument("--params", default="{}", help="quadruped template parameters as JSON")
    parser.add_argument("--unilab", type=Path, default=os.environ.get("ROBOTFORGE_UNILAB_DIR"))
    args = parser.parse_args()
    if args.unilab is None:
        parser.error("set ROBOTFORGE_UNILAB_DIR or pass --unilab")
    task = args.task or args.out.name
    base_yaml = args.unilab / "src/unilab/conf/ppo/task/go2_joystick_flat/base.yaml"
    base = yaml.safe_load(base_yaml.read_text(encoding="utf-8"))
    spec, info = build_scene(json.loads(args.params))
    args.out.mkdir(parents=True, exist_ok=True)
    scene = args.out / "scene.xml"
    scene.write_text(spec.to_xml(), encoding="utf-8")
    task_name = "Robotforge" + "".join(part.title() for part in task.split("_"))
    owner = owner_config(base, task_name, scene, info)
    folder = args.out / "conf" / "task" / task
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "mujoco.yaml").write_text(
        "# @package _global_\n" + yaml.safe_dump(owner, sort_keys=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "scene": str(scene),
                "conf_dir": str(args.out / "conf"),
                "task": task,
                "task_name": task_name,
                "standing_height": round(info["height"], 3),
                "commands": owner["env"]["commands"]["twist"]["ranges"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
