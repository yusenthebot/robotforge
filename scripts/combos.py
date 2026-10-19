"""Ten robots made with mount, each in one sentence, each moving and doing a small job (milestone G10).

    uv run python scripts/combos.py [--only NAME ...]

Each combination is built in a fresh world from the real Menagerie models the way an agent builds it (mount calls),
then does its job headless with the real behaviors (move, move_arm, pick_and_place, read_sensors) while this prints
what it measured: the mount checks, how far it went, whether the job worked. scripts/screenshots.py pictures them
(g10_<name>) and scripts/demos.py films a hand swap. Maintainer tool, not part of robotforge.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

BASE = {"length": 0.7, "width": 0.5, "wheel_radius": 0.1, "motor": "jga25"}  # a wheeled base an arm stands on
SMALL_BASE = {"length": 0.55, "width": 0.4, "wheel_radius": 0.07, "motor": "jga25"}
HEAVY_BASE = {**BASE, "material": "aluminum"}  # a 47 kg deck: a Panda swinging out does not tip it


@dataclass(frozen=True)
class Combo:
    name: str
    sentence: str
    build: Callable[[Any], None]  # on a Playground with a world open
    job: Callable[[Any], dict[str, Any]]  # runs headless; "ok" says whether it worked
    robot: str  # the carrier, for the camera
    view: tuple[float, float, float] = (1.3, -1.5, 0.75)
    lift: float = 0.35


# ---- jobs -------------------------------------------------------------------------------------------------------


def _run(pg: Any, seconds: float) -> None:
    pg.run_fast(lambda: False, seconds)


def _go(pg: Any, robot: str, speed: float, seconds: float) -> dict[str, Any]:
    before = pg.behaviors.pose(robot)
    pg.behaviors.move(robot, speed, 0.0)
    _run(pg, seconds)
    pg.behaviors.stop(robot)
    _run(pg, 1.5)
    after = pg.behaviors.pose(robot)
    went = float(np.linalg.norm(np.subtract(after["position"][:2], before["position"][:2])))
    return {"moved_m": round(went, 2), "upright": bool(after["upright"])}


def _pick(pg: Any, arm: str, obj: str, target: Any, limit: float = 120.0, go: bool = True) -> dict[str, Any]:
    motion, info = pg.behaviors.pick_and_place(arm, obj, target, go)
    start = float(pg.session.data.time)
    pg.run_fast(lambda: motion.done, limit)
    _run(pg, 1.0)
    placed = pg.behaviors.placed(info["object"], info["region"])
    took = float(pg.session.data.time) - start
    out = {"placed": bool(placed["success"]), "pick_s": round(took, 1)}
    if not placed["success"]:
        out["object_at"] = placed["object_position"]
    if hasattr(motion, "result") and not placed["success"]:
        out["errand"] = motion.result()
    return out


def _reach(pg: Any, arm: str, point: list[float]) -> dict[str, Any]:
    from robotforge.behaviors.manager import BehaviorError

    for _ in range(10):  # a carrier that walked comes to a stand first
        try:
            motion = pg.behaviors.move_arm(arm, point)
            break
        except BehaviorError as exc:
            if "coming to a stand" not in str(exc):
                raise
            _run(pg, 1.0)
    pg.run_fast(lambda: motion.done, 20.0)
    _run(pg, 0.5)
    tcp = pg.session.data.site_xpos[pg.session.model.site(f"{arm}/tcp").id]
    return {"tool_off_m": round(float(np.linalg.norm(tcp - np.asarray(point))), 3)}


def _sensor(pg: Any, robot: str, name: str) -> dict[str, Any]:
    reading, _ = pg.sensors.read(robot, [name])
    return reading["sensors"][name]


def _ok(out: dict[str, Any], *keys: str) -> dict[str, Any]:
    out["ok"] = all(bool(out.get(k)) for k in keys)
    return out


# ---- the ten ---------------------------------------------------------------------------------------------------


def _leap_cart(pg: Any) -> None:
    pg.add_robot("ur5e", library="ur5e")
    pg.mount("leap_hand", "ur5e", save_as="ur5e_leap")
    pg.remove("ur5e")
    pg.add_robot("cart", template="rover", params=BASE)
    pg.mount("ur5e_leap", "cart", name="arm")
    pg.add_object("cube", shape="box", size=[0.05] * 3, pos=[1.05, 0.3], color="orange")
    pg.add_object("bin", shape="bin", size=[0.2, 0.2, 0.08], pos=[0.95, -0.35], color="blue", fixed=True)


def _leap_cart_job(pg: Any) -> dict[str, Any]:
    out = _go(pg, "cart", 0.2, 2.0)
    out |= _pick(pg, "cart/arm", "cube", "bin")
    return _ok(out, "upright", "placed")


def _spot_two_arms(pg: Any) -> None:
    pg.add_robot("spot", library="spot")
    pg.mount("z1", "spot")
    pg.mount("piper", "spot", name="arm2")
    pg.add_object("stand", shape="table", size=[0.6, 0.4, 0.3], pos=[1.6, -0.6])
    pg.add_object("cube", shape="box", size=[0.04] * 3, pos=[1.6, -0.48, 0.32], color="orange")
    pg.add_object("shelf", shape="table", size=[0.6, 0.4, 0.3], pos=[1.6, 0.6])


def _spot_two_arms_job(pg: Any) -> dict[str, Any]:
    out = _go(pg, "spot", 0.3, 3.0)
    out |= _pick(pg, "spot/arm2", "cube", "shelf")
    return _ok(out, "upright", "placed")


def _go2_wrist_lidar(pg: Any) -> None:
    pg.add_robot("go2", library="go2")
    pg.mount("piper", "go2")
    pg.mount("lidar", "go2/arm", name="wrist_lidar")
    pg.add_object("wall", shape="wall", size=[2.0, 0.1, 0.6], pos=[1.2, 1.0])


def _go2_wrist_lidar_job(pg: Any) -> dict[str, Any]:
    out = _go(pg, "go2", 0.3, 2.0)
    out |= _reach(pg, "go2/arm", list(np.add(pg.behaviors.pose("go2")["position"], [0.25, 0.2, 0.15])))
    seen = _sensor(pg, "go2", "wrist_lidar")
    near = seen.get("nearest") or {}
    out["lidar_nearest_m"] = near.get("distance_m")
    return _ok(out, "upright", "lidar_nearest_m")


def _anymal_hand(pg: Any) -> None:
    pg.add_robot("anymal", library="anymal_c")
    pg.mount("ur5e", "anymal", adapter=False)  # 21 kg of arm: a plate would take ANYmal past the 22.1 kg it carries
    pg.mount("panda_hand", "anymal/arm", adapter=False)
    pg.add_object("ball", shape="sphere", size=[0.06], pos=[1.7, 0.45], color="orange")
    pg.add_object("bin", shape="bin", size=[0.22, 0.22, 0.08], pos=[1.7, -0.5], color="blue", fixed=True)


def _anymal_hand_job(pg: Any) -> dict[str, Any]:
    out = _go(pg, "anymal", 0.3, 3.0)
    out |= _pick(pg, "anymal/arm", "ball", "bin")
    return _ok(out, "upright", "placed")


def _allegro_ur5e(pg: Any) -> None:
    pg.add_robot("ur5e", library="ur5e")
    pg.mount("allegro_hand", "ur5e")
    pg.add_object("ball", shape="sphere", size=[0.06], pos=[0.5, 0.2], color="orange")
    pg.add_object("bin", shape="bin", size=[0.18, 0.18, 0.08], pos=[0.35, -0.35], color="blue", fixed=True)


def _allegro_ur5e_job(pg: Any) -> dict[str, Any]:
    return _ok(_pick(pg, "ur5e", "ball", "bin"), "placed")


def _suction_rover(pg: Any) -> None:
    pg.add_robot("rover", template="rover", params=SMALL_BASE)
    pg.mount("piper", "rover", shift=[-0.1, 0.0, 0.0])
    pg.mount("suction", "rover/arm")
    pg.add_object("tile", shape="box", size=[0.1, 0.1, 0.01], pos=[0.22, 0.3], color="orange")
    pg.add_object("tray", shape="bin", size=[0.2, 0.2, 0.04], pos=[0.12, -0.36], color="blue", fixed=True)


def _suction_rover_job(pg: Any) -> dict[str, Any]:
    out = _pick(pg, "rover/arm", "tile", "tray")
    out |= _go(pg, "rover", 0.2, 2.0)
    return _ok(out, "upright", "placed")


def _magnet_cart(pg: Any) -> None:
    pg.add_robot("cart", template="rover", params=BASE)
    pg.mount("ur5e", "cart")
    pg.mount("magnet", "cart/arm")
    pg.add_object("block", shape="box", size=[0.04] * 3, pos=[1.05, 0.25], material="steel", color="gray")
    pg.add_object("bin", shape="bin", size=[0.18, 0.18, 0.06], pos=[0.95, -0.35], color="blue", fixed=True)


def _magnet_cart_job(pg: Any) -> dict[str, Any]:
    out = _go(pg, "cart", 0.2, 2.0)
    out |= _pick(pg, "cart/arm", "block", "bin")
    return _ok(out, "upright", "placed")


def _t1_sensors(pg: Any) -> None:
    pg.add_robot("t1", library="t1")
    pg.mount("lidar", "t1", at="H2", name="head_lidar")
    pg.mount("camera", "t1", at="Trunk", params={"mode": "depth", "at": "front"}, name="chest_cam")
    pg.add_object("crate", shape="box", size=[0.4, 0.6, 1.5], pos=[2.0, 0.0], fixed=True, color="orange")


def _t1_sensors_job(pg: Any) -> dict[str, Any]:
    out = _go(pg, "t1", 0.12, 5.0)
    x = pg.behaviors.pose("t1")["position"][0]
    lidar = _sensor(pg, "t1", "head_lidar").get("ahead") or {}
    camera = _sensor(pg, "t1", "chest_cam").get("nearest") or {}
    out["crate_face_m"] = round(1.8 - x, 2)  # from the robot's middle
    out["lidar_ahead_m"] = lidar.get("distance_m")
    out["camera_nearest_m"] = camera.get("distance_m")
    return _ok(out, "upright", "lidar_ahead_m", "camera_nearest_m")


def _go1_piper_eye(pg: Any) -> None:
    pg.add_robot("arm", library="piper")
    pg.mount("camera", "arm", params={"mode": "depth"}, name="wrist_cam")
    pg.mount(None, "arm", save_as="piper_eye")
    pg.remove("arm")
    pg.add_robot("go1", library="go1")
    pg.mount("piper_eye", "go1", name="arm")
    pg.add_object("crate", shape="box", size=[0.3, 0.3, 0.3], pos=[1.6, 0.0], fixed=True, color="orange")


def _go1_piper_eye_job(pg: Any) -> dict[str, Any]:
    out = _go(pg, "go1", 0.3, 2.5)
    out["camera_center_m"] = _sensor(pg, "go1", "wrist_cam").get("center_m")
    return _ok(out, "upright", "camera_center_m")


def _panda_cart(pg: Any) -> None:
    pg.add_robot("cart", template="rover", params=HEAVY_BASE)
    pg.mount("panda", "cart")
    pg.mount("lidar3d", "cart", at="chassis", params={"at": "front"}, name="lidar3d")
    pg.add_object("stand", shape="table", size=[0.5, 0.4, 0.3], pos=[1.15, 0.45])
    pg.add_object("cube", shape="box", size=[0.05] * 3, pos=[1.05, 0.35, 0.33], color="orange")
    pg.add_object("bin", shape="bin", size=[0.2, 0.2, 0.08], pos=[1.0, -0.4], color="blue", fixed=True)
    pg.add_object("wall", shape="wall", size=[1.5, 0.1, 0.8], pos=[2.6, 0.0], yaw=90)


def _panda_cart_job(pg: Any) -> dict[str, Any]:
    out = _go(pg, "cart", 0.2, 2.0)
    seen = _sensor(pg, "cart", "lidar3d")
    out["lidar_points"] = seen.get("points")
    out["lidar_nearest_m"] = (seen.get("nearest") or {}).get("distance_m")
    out |= _pick(pg, "cart/arm", "cube", "bin")
    return _ok(out, "upright", "placed", "lidar_nearest_m")


# name: (the sentence, the carrier's name); each builds with _<name> and works with _<name>_job.
TEN = {
    "leap_cart": ("A UR5e with a LEAP hand, saved as a design, on a wheeled base.", "cart"),
    "spot_two_arms": ("Two arms on a Spot: a Unitree Z1 and an AgileX PiPER.", "spot"),
    "go2_wrist_lidar": ("A Go2 carrying a PiPER with a lidar on its wrist.", "go2"),
    "anymal_hand": ("An ANYmal C carrying a UR5e with a Franka Hand.", "anymal"),
    "allegro_ur5e": ("An Allegro hand on a UR5e.", "ur5e"),
    "suction_rover": ("A suction cup on a PiPER on a small rover.", "rover"),
    "magnet_cart": ("An electromagnet on a UR5e on a wheeled base.", "cart"),
    "t1_sensors": ("A Booster T1 humanoid with a lidar on its head and a depth camera on its chest.", "t1"),
    "go1_piper_eye": ("A PiPER with a depth camera on its wrist, saved as a design, on a Go1.", "go1"),
    "panda_cart": ("A Franka Panda on a wheeled base with a 3D lidar on its front.", "cart"),
}
# Cameras for the pictures where the default does not frame it: (offset from the point looked at, its height over the
# carrier's main body), m.
VIEWS = {
    "t1_sensors": ((2.3, -2.5, 0.35), -0.1),  # all of a 1.2 m humanoid, and the crate it measures
    "allegro_ur5e": ((1.0, -1.25, 0.65), 0.45),
}
COMBOS: dict[str, Combo] = {
    name: Combo(name, sentence, globals()[f"_{name}"], globals()[f"_{name}_job"], robot, *VIEWS.get(name, ()))
    for name, (sentence, robot) in TEN.items()
}


def run(combo: Combo) -> dict[str, Any]:
    from robotforge.playground import Playground, PlaygroundError

    os.environ["ROBOTFORGE_HOME"] = tempfile.mkdtemp(prefix="rf-combo-")
    pg = Playground.with_demo()
    pg.new_world(combo.name)
    t0 = time.monotonic()
    try:
        combo.build(pg)
    except PlaygroundError as exc:
        return {"ok": False, "refused": str(exc)}
    out: dict[str, Any] = {"build_s": round(time.monotonic() - t0, 1)}
    if os.environ.get("SHEET"):
        _filmed(pg, combo, os.environ["SHEET"])
    _run(pg, 1.0)
    try:
        out |= combo.job(pg)
    except Exception as exc:  # report and go on to the next one
        out |= {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return out


def _filmed(pg: Any, combo: Combo, out: str) -> None:
    """Debugging: every run_fast is run in 0.5 s pieces and a picture of the carrier taken after each; the contact
    sheet is written when the process ends."""
    import atexit

    import mujoco
    from PIL import Image

    m, d = pg.session.model, pg.session.data
    renderer = mujoco.Renderer(m, 240, 320)
    frames: list[np.ndarray] = []
    plain = pg.run_fast
    root = next(b for b in range(1, m.nbody) if m.body(b).name.startswith(f"{combo.robot}/"))

    def chunked(until: Callable[[], bool], seconds: float) -> float:
        left, ran = seconds, 0.0
        while left > 1e-9:
            piece = min(0.5, left)
            ran += plain(until, piece)
            left -= piece
            cam = mujoco.MjvCamera()
            cam.lookat[:] = d.xpos[root] + [0.3, 0.0, 0.0]
            cam.distance, cam.azimuth, cam.elevation = float(os.environ.get("FILM_DISTANCE", "2.0")), 230, -25
            renderer.update_scene(d, cam)
            frames.append(renderer.render())
            if until():
                break
        return ran

    def save() -> None:
        rows = [frames[i : i + 10] for i in range(0, len(frames), 10)]
        rows = [np.concatenate(r + [np.zeros_like(frames[0])] * (10 - len(r)), axis=1) for r in rows]
        Image.fromarray(np.concatenate(rows, axis=0)).save(out)

    pg.run_fast = chunked
    atexit.register(save)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    worked = tried = 0
    for combo in COMBOS.values():
        if args.only and combo.name not in args.only:
            continue
        t0 = time.monotonic()
        out = run(combo)
        tried += 1
        worked += bool(out.get("ok"))
        print(f"{combo.name} ({time.monotonic() - t0:.0f} s): {combo.sentence}\n  {json.dumps(out)}", flush=True)
    print(f"{worked} of {tried} worked")
    return 0 if worked == tried else 1


if __name__ == "__main__":
    sys.exit(main())
