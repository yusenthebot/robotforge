"""Measure the sensor suite against the truth: distances, weights, contact patterns, noise, latency and cost.

    uv run python scripts/sensor_bench.py [--only truth|drop|wrist|trot|knobs|speed] [--json FILE]

truth  a Unitree Go2 in the warehouse facing a wall 1.8 m away, with a noisy 3D lidar and depth camera: what
       each reads straight ahead against the wall's distance from it (needs offscreen OpenGL for the camera)
drop   an IMU on a block dropped from 1 m: acceleration while it falls and once it rests
wrist  a force-torque sensor at a Franka Panda's wrist: what it says the gripper holds, empty and with a cube
trot   2x2 touch arrays under a trotting Go2's four feet: the stance pattern and where each foot presses
knobs  measured noise against the set noise, and the delay of a delayed GPS against its set latency
speed  headless steps per second of the Go2 in the warehouse with no sensors, an ideal 3D lidar and 10 Hz ones

Library robots download from MuJoCo Menagerie on first use. The numbers go in docs/SENSORS.md.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import tempfile
import time
from typing import Any

import numpy as np

os.environ.setdefault("ROBOTFORGE_HOME", tempfile.mkdtemp(prefix="sensor_bench_"))

from robotforge.playground import Playground  # noqa: E402
from robotforge.sim.render import render_unavailable_reason  # noqa: E402

WALL_FACE_X = -6.9  # the warehouse's side wall: its collision box ends here (10 cm of a 20 cm wall at x = -7)
BOX = {"base": {"shape": "box", "size": [0.2, 0.2, 0.1], "mass": 2.0}}


def warehouse_go2(name: str) -> Playground:
    pg = Playground.with_demo()
    pg.new_world(name)
    pg.load_scene("warehouse", keep_robots=False, starter=False)
    pg.add_robot("go2", library="go2", pos=[-5.0, 2.6], yaw=180)  # faces the side wall, down the empty corner
    return pg


def run(pg: Playground, seconds: float) -> None:
    pg.run_fast(lambda: False, seconds)


def truth() -> dict[str, Any]:
    pg = warehouse_go2("bench_truth")
    pg.add_sensor("go2", {"type": "lidar3d", "noise": 0.02, "rate_hz": 10})
    pg.add_sensor("go2", {"type": "camera", "mode": "depth", "noise": 0.01, "rate_hz": 10})
    run(pg, 1.0)
    m, d = pg.session.model, pg.session.data
    lidar, depth = [], []
    lidar_truth, depth_truth = [], []
    for _ in range(30):
        run(pg, 0.1)
        reading, _ = pg.read_sensors("go2", ["lidar", "depth_camera"])
        origin = d.site_xpos[m.site("go2/rf_lidar").id]
        eye = d.cam_xpos[m.camera("go2/depth_camera").id]
        forward = -d.cam_xmat[m.camera("go2/depth_camera").id].reshape(3, 3)[:, 2]
        lidar.append(reading["sensors"]["lidar"]["ahead"]["distance_m"])
        lidar_truth.append(float(origin[0] - WALL_FACE_X))
        cam = reading["sensors"]["depth_camera"]
        if "center_m" in cam and cam["center_m"] is not None:
            depth.append(cam["center_m"])
            depth_truth.append(float((eye[0] - WALL_FACE_X) / -forward[0]))
    out: dict[str, Any] = {
        "lidar_ahead_m": round(statistics.mean(lidar), 4),
        "lidar_truth_m": round(statistics.mean(lidar_truth), 4),
        "lidar_error_mean_m": round(statistics.mean(np.subtract(lidar, lidar_truth)), 4),
        "lidar_error_std_m": round(float(np.std(np.subtract(lidar, lidar_truth))), 4),
        "lidar_noise_set_m": 0.02,
    }
    if depth:
        err = np.subtract(depth, depth_truth)
        out.update(
            depth_center_m=round(statistics.mean(depth), 4),
            depth_truth_m=round(statistics.mean(depth_truth), 4),
            depth_error_mean_m=round(float(err.mean()), 4),
            depth_error_std_m=round(float(err.std()), 4),
            depth_noise_set_m=0.01,
        )
    else:
        out["depth"] = f"no picture: {render_unavailable_reason()}"
    # One clean scan, every ray toward the wall against the wall plane: the geometry itself, without noise.
    pg.add_sensor("go2", {"type": "lidar3d", "name": "clean"})
    sensor = pg.sensors.suite.sensors["go2/clean"]
    value = sensor.measure(pg.session.model, pg.session.data)
    world = value["rot"] @ sensor.directions.T
    toward = world[0] < -0.2
    plane = (value["origin"][0] - WALL_FACE_X) / -world[0]
    hits = np.isfinite(value["dist"]) & toward & (np.abs(value["dist"] - plane) < 0.05)
    errors = np.abs(value["dist"][hits] - plane[hits])
    out["clean_wall_rays"] = int(hits.sum())
    out["clean_wall_rays_within_1mm"] = round(float((errors < 0.001).mean()), 3)
    out["clean_wall_median_error_m"] = round(float(np.median(errors)), 6)
    return out


def drop() -> dict[str, Any]:
    pg = Playground.with_demo()
    pg.new_world("bench_drop")
    pg.add_robot("block", build=BOX, pos=[0, 0, 1.0])
    pg.add_sensor("block", {"type": "imu", "rate_hz": 200})
    series = []
    for _ in range(150):
        run(pg, 0.01)
        reading, _ = pg.read_sensors("block", ["imu"])
        series.append((round(float(pg.session.data.time), 3), reading["sensors"]["imu"]["accel_m_s2"][2]))
    falling = [a for t, a in series if 0.05 <= t <= 0.4]
    resting = [a for t, a in series if t >= 1.0]
    return {
        "falling_accel_z_mean": round(statistics.mean(falling), 4),
        "falling_accel_z_max_abs": round(max(abs(a) for a in falling), 4),
        "resting_accel_z_mean": round(statistics.mean(resting), 4),
        "landing_s": next(t for t, a in series if a > 1.0),
        "expected_landing_s": round(math.sqrt(2 * (1.0 - 0.05) / 9.81), 3),
    }


def wrist() -> dict[str, Any]:
    pg = Playground.with_demo()
    pg.new_world("bench_wrist")
    pg.add_robot("panda", library="panda")
    cube_mass = 0.1  # a 5 cm cube of 100 g; the Panda's fingers let a 250 g one slip in this model
    pg.add_object("cube", shape="box", size=[0.05, 0.05, 0.05], mass=cube_mass, pos=[0.5, 0.0], color="red")
    pg.add_sensor("panda", {"type": "force_torque", "on": "hand", "name": "wrist"})

    def do(motion) -> None:
        pg.run_fast(lambda: motion.done, 20.0)

    def weigh() -> list[float]:
        out = []
        for _ in range(25):
            run(pg, 0.02)
            out.append(pg.read_sensors("panda", ["wrist"])[0]["sensors"]["wrist"]["payload_kg"])
        return out

    do(pg.behaviors.move_arm("panda", [0.5, 0.0, 0.25], "open"))
    run(pg, 0.5)
    empty = weigh()
    do(pg.behaviors.move_arm("panda", [0.5, 0.0, 0.025], None))
    do(pg.behaviors.set_gripper("panda", "close"))
    do(pg.behaviors.move_arm("panda", [0.5, 0.0, 0.25], None))
    run(pg, 1.0)
    held = weigh()
    cube_z = float(pg.session.data.xpos[pg.session.model.body("cube").id][2])
    return {
        "cube_kg": cube_mass,
        "cube_lifted_to_m": round(cube_z, 3),
        "empty_payload_kg": round(statistics.mean(empty), 4),
        "held_payload_kg": round(statistics.mean(held), 4),
        "held_error_percent": round(100 * (statistics.mean(held) - cube_mass) / cube_mass, 2),
        "held_spread_kg": round(float(np.std(held)), 4),
    }


def trot() -> dict[str, Any]:
    pg = Playground.with_demo()
    pg.new_world("bench_trot")
    pg.add_robot("go2", library="go2")
    feet = ("FL", "FR", "RL", "RR")
    for foot in feet:
        pg.add_sensor("go2", {"type": "touch", "on": foot, "rows": 2, "cols": 2, "name": f"{foot.lower()}_pad"})
    pg.behaviors.move("go2", 0.4, 0.0)
    run(pg, 2.0)  # into its stride
    contact = {f: [] for f in feet}
    truth = {f: [] for f in feet}
    fronts = {f: [] for f in feet}
    m = pg.session.model
    foot_geom = {f: m.geom(f"go2/{f}").id for f in feet}
    for _ in range(300):  # 3 s at 100 Hz
        run(pg, 0.01)
        reading = pg.read_sensors("go2", [f"{f.lower()}_pad" for f in feet])[0]["sensors"]
        d = pg.session.data
        touching = {int(g) for con in d.contact[: d.ncon] for g in (con.geom1, con.geom2)}
        for f in feet:
            pad = reading[f"{f.lower()}_pad"]
            contact[f].append(1.0 if pad["contact"] else 0.0)
            truth[f].append(1.0 if foot_geom[f] in touching else 0.0)
            if pad.get("center"):
                fronts[f].append(pad["center"][0])
    c = {f: np.array(v) for f, v in contact.items()}

    def corr(a: str, b: str) -> float:
        return round(float(np.corrcoef(c[a], c[b])[0, 1]), 3)

    patterns: dict[str, int] = {}
    for i in range(len(c["FL"])):
        key = "".join(f if c[f][i] else "-" * len(f) for f in feet)
        patterns[key] = patterns.get(key, 0) + 1
    top = sorted(patterns.items(), key=lambda kv: -kv[1])[:4]
    return {
        "duty_factor": {f: round(float(c[f].mean()), 3) for f in feet},
        "agrees_with_true_foot_contacts": {f: round(float((c[f] == np.array(truth[f])).mean()), 3) for f in feet},
        "corr_diagonal_FL_RR": corr("FL", "RR"),
        "corr_diagonal_FR_RL": corr("FR", "RL"),
        "corr_side_FL_FR": corr("FL", "FR"),
        "corr_side_FL_RL": corr("FL", "RL"),
        "most_common_stances": [{"feet_down": k, "share": round(n / len(c["FL"]), 3)} for k, n in top],
        "pressure_front_back_range": {
            f: [round(min(v), 2), round(max(v), 2)] if v else None for f, v in fronts.items()
        },
        "walked_m": round(abs(pg.behaviors.pose("go2")["position"][0]), 2),
    }


def knobs() -> dict[str, Any]:
    pg = Playground.with_demo()
    pg.new_world("bench_knobs")
    pg.add_robot("block", build={**BOX, "base": {**BOX["base"], "fixed": True}})
    pg.add_sensor("block", {"type": "imu", "noise": {"accel": 0.05, "gyro": 0.002}, "rate_hz": 100})
    pg.add_sensor("block", {"type": "gps", "noise": 0.03, "rate_hz": 100, "name": "gps"})
    accel, gyro, gps = [], [], []
    for _ in range(1000):
        run(pg, 0.01)
        r = pg.read_sensors("block", ["imu", "gps"])[0]["sensors"]
        accel.append(r["imu"]["accel_m_s2"])
        gyro.append(r["imu"]["gyro_rad_s"])
        gps.append(r["gps"]["position_m"])
    out: dict[str, Any] = {
        "accel_noise_set": 0.05,
        "accel_noise_measured": [round(float(v), 4) for v in np.std(accel, axis=0)],
        "gyro_noise_set": 0.002,
        "gyro_noise_measured": [round(float(v), 5) for v in np.std(gyro, axis=0)],
        "gps_noise_set": 0.03,
        "gps_noise_measured": [round(float(v), 4) for v in np.std(gps, axis=0)],
    }
    # Latency: a dropped block whose GPS answers 0.1 s late, against where the block really was.
    pg2 = Playground.with_demo()
    pg2.new_world("bench_latency")
    pg2.add_robot("block", build=BOX, pos=[0, 0, 3.0])
    pg2.add_sensor("block", {"type": "gps", "latency_s": 0.1, "name": "late"})
    m = pg2.session.model
    site = m.site("block/rf_late").id
    history: list[tuple[float, float]] = []
    lags = []
    for _ in range(350):
        pg2.session.step(1)
        d = pg2.session.data
        history.append((float(d.time) - m.opt.timestep, float(d.site_xpos[site][2])))
        reading = pg2.read_sensors("block", ["late"])[0]["sensors"]["late"]
        if "position_m" not in reading or d.time < 0.2:
            continue
        z = reading["position_m"][2]
        when = min(history, key=lambda h: abs(h[1] - z))[0]
        lags.append(float(d.time) - when)
    out["latency_set_s"] = 0.1
    out["latency_measured_s"] = round(float(np.median(lags)), 4)
    out["latency_note"] = (
        "measured from the clock after a step; MuJoCo's positions then are the step's start, 2 ms back"
    )
    return out


def speed(seconds: float = 2.0, trials: int = 7) -> dict[str, Any]:
    """Steps per second of CPU time (time.thread_time): on a machine shared with other work, wall time measures the
    other work as much as this. Each setup is warmed up once, then the setups take turns, in a new order each trial."""
    setups = {
        "no sensors": [],
        "ideal 3D lidar (read only on demand)": [{"type": "lidar3d"}],
        "10 Hz 2D lidar, 360 rays": [{"type": "lidar", "rate_hz": 10}],
        "10 Hz 3D lidar, 16 x 180 rays": [{"type": "lidar3d", "rate_hz": 10}],
        "10 Hz 3D lidar, 16 x 360 rays": [{"type": "lidar3d", "rate_hz": 10, "rays": 360}],
        "10 Hz 3D lidar + 100 Hz noisy IMU + 4 touch pads": [
            {"type": "lidar3d", "rate_hz": 10},
            {"type": "imu", "rate_hz": 100, "noise": 0.05},
            *[{"type": "touch", "on": f, "name": f"{f.lower()}_pad"} for f in ("FL", "FR", "RL", "RR")],
        ],
    }
    worlds = {}
    steps = int(seconds / 0.002)
    for i, (label, sensors) in enumerate(setups.items()):
        pg = warehouse_go2(f"bench_speed_{i}")
        for s in sensors:
            pg.add_sensor("go2", s)
        pg.session.step(steps // 4)  # warm up
        worlds[label] = pg
    rates: dict[str, list[float]] = {label: [] for label in setups}
    rng = np.random.default_rng(0)
    for _ in range(trials):
        for label in rng.permutation(list(worlds)):
            pg = worlds[str(label)]
            start = time.thread_time()
            pg.session.step(steps)
            rates[str(label)].append(steps / (time.thread_time() - start))
    base = statistics.median(rates["no sensors"])
    out: dict[str, Any] = {"cost_of_one_scan": scan_cost(worlds["10 Hz 3D lidar, 16 x 180 rays"])}
    for label, r in rates.items():
        out[label] = {
            "steps_per_s": round(statistics.median(r)),
            "spread": [round(min(r)), round(max(r))],
            "vs_no_sensors": f"{100 * (statistics.median(r) / base - 1):+.1f}%",
        }
    return out | {"machine_load": round(os.getloadavg()[0], 1), "cores": os.cpu_count()}


def scan_cost(pg: Playground, repeats: int = 15) -> dict[str, Any]:
    """A 3D scan's CPU time against a physics step's, best of several, and what a scan ten times a second adds."""
    sensor = pg.sensors.suite.sensors["go2/lidar"]
    m, d = pg.session.model, pg.session.data
    scans, steps = [], []
    for _ in range(repeats):
        start = time.thread_time()
        sensor.measure(m, d)
        scans.append(time.thread_time() - start)
        start = time.thread_time()
        for _ in range(20):
            mujoco_step(m, d)
        steps.append((time.thread_time() - start) / 20)
    scan, step = min(scans), min(steps)
    per_second = 10 * scan / (step / m.opt.timestep)  # 10 scans against 500 steps of simulated time
    return {
        "scan_ms": round(scan * 1e3, 3),
        "step_ms": round(step * 1e3, 3),
        "added_at_10_hz": f"{100 * per_second:.1f}%",
    }


def mujoco_step(m: Any, d: Any) -> None:
    import mujoco

    mujoco.mj_step(m, d)


TESTS = {"truth": truth, "drop": drop, "wrist": wrist, "trot": trot, "knobs": knobs, "speed": speed}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", choices=list(TESTS), action="append")
    parser.add_argument("--json", help="also write the results to this file")
    args = parser.parse_args()
    results = {}
    for name in args.only or list(TESTS):
        start = time.perf_counter()
        results[name] = TESTS[name]()
        print(f"== {name} ({time.perf_counter() - start:.1f} s)")
        print(json.dumps(results[name], indent=1))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
