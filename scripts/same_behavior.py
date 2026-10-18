"""Does a change leave the robots moving exactly as before? Hash the whole physics state through set scenarios.

    uv run python scripts/same_behavior.py <rev> [--only dog_fresh go2_rl ...]

Every scenario builds a fresh world, sets robots going the way a user or an agent would (standing on a fresh gait,
a shove, a drag by the back, walking, fast turns, a ramp, the Go2 on servos and on raw motors, the learned Go2 and
G1 walks, a hexapod, a humanoid that waves and walks, a balancer, an arm's pick and place and its PD controllers,
a Go2 crossing the warehouse) and records a hash of qpos, qvel and ctrl every 0.25 s of simulated time. It runs
them on this tree and on revision <rev> (its source unpacked in a temp folder), each in its own process, and
reports per scenario whether every hash matched or from when they differ. A speed-up that is meant to change no
behavior must leave all of them identical; exits 1 when one differs. MuJoCo is deterministic on one machine, so
any difference is the change's. Uses the real Menagerie Go2 and G1 (downloaded once into the cache).
Maintainer tool, not part of robotforge.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MARK_EVERY = 0.25  # s of simulated time between hashes


# ---- the scenarios (run inside a worker process) ---------------------------------------------------------------


def _world(name: str, robots: dict[str, dict[str, Any]], terrain: dict[str, Any] | None = None) -> Any:
    from robotforge.playground import Playground

    pg = Playground.with_demo()
    pg.new_world(f"same_{name}")
    if terrain:
        pg.add_terrain(**terrain)
    for robot, source in robots.items():
        pg.add_robot(robot, **source)
    return pg


def _run(pg: Any, seconds: float, marks: list[str]) -> None:
    s = pg.session
    steps = round(seconds / s.model.opt.timestep)
    every = round(MARK_EVERY / s.model.opt.timestep)
    for k in range(steps):
        s.step()
        if (k + 1) % every == 0:
            d = s.data
            marks.append(hashlib.sha256(d.qpos.tobytes() + d.qvel.tobytes() + d.ctrl.tobytes()).hexdigest()[:16])


def _trunk(pg: Any, robot: str) -> tuple[int, int]:
    import mujoco

    m = pg.session.model
    for j in range(m.njnt):
        body = int(m.jnt_bodyid[j])
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE and m.body(body).name.startswith(f"{robot}/"):
            return body, int(m.jnt_dofadr[j])
    raise KeyError(robot)


def _built(spec: str) -> dict[str, Any]:
    sys.path.insert(0, str(ROOT / "tests"))
    import test_builder

    return {"build": getattr(test_builder, spec)}


DOG = {"template": "quadruped"}
GO2 = {"library": "go2"}


def dog_fresh(marks: list[str]) -> None:
    _run(_world("dog_fresh", {"dog": DOG}), 4.0, marks)


def dog_shove(marks: list[str]) -> None:
    pg = _world("dog_shove", {"dog": DOG})
    _run(pg, 2.0, marks)
    pg.session.data.qvel[_trunk(pg, "dog")[1] + 1] += 0.6
    _run(pg, 4.0, marks)


def dog_drag(marks: list[str]) -> None:
    import numpy as np

    pg = _world("dog_drag", {"dog": DOG})
    _run(pg, 2.0, marks)
    body, _ = _trunk(pg, "dog")
    grab = pg.session.data.xpos[body] + np.array([0.0, 0.0, 0.04])
    pg.session.start_drag(body, grab)
    for k in range(1, 21):
        pg.session.move_drag(grab + np.array([0.0, 0.4 * k / 20, 0.0]))
        pg.run_fast(lambda: False, 0.1)
    pg.session.end_drag()
    _run(pg, 3.0, marks)


def dog_walk_turn_stop(marks: list[str]) -> None:
    pg = _world("dog_walk_turn_stop", {"dog": DOG})
    pg.behaviors.move("dog", 0.3, 0.0, walker="gait")
    _run(pg, 3.0, marks)
    pg.behaviors.move("dog", 0.1, 1.0)
    _run(pg, 2.0, marks)
    pg.behaviors.stop("dog")
    _run(pg, 3.0, marks)


def dog_fast_turn(marks: list[str]) -> None:
    pg = _world("dog_fast_turn", {"dog": DOG})
    pg.behaviors.move("dog", 1.0, 2.0, walker="gait")  # beyond its top speed: strides saturate
    _run(pg, 3.0, marks)
    pg.behaviors.move("dog", -1.0, -2.0)
    _run(pg, 2.0, marks)


def dog_ramp(marks: list[str]) -> None:
    ramp = {"type": "ramp", "pos": [0.0, 0.0], "params": {"angle": 15.0, "length": 1.5, "direction": "up"}}
    _run(_world("dog_ramp", {"dog": DOG}, ramp), 4.0, marks)


def two_dogs(marks: list[str]) -> None:
    pg = _world("two_dogs", {"a": DOG, "b": {**DOG, "pos": [0.0, 1.0]}})
    pg.behaviors.move("a", 0.2, 0.3)
    _run(pg, 3.0, marks)


def go2_gait(marks: list[str]) -> None:
    pg = _world("go2_gait", {"go2": GO2})
    _run(pg, 2.0, marks)
    pg.behaviors.move("go2", 0.3, 0.0, walker="gait")
    _run(pg, 3.0, marks)
    pg.behaviors.stop("go2")
    _run(pg, 2.0, marks)


def go2_motors(marks: list[str]) -> None:
    """Menagerie's own Go2 file, raw torque motors: the gait's PD runs every physics step."""
    from robotforge.design import menagerie

    folder = menagerie.local_folder("unitree_go2")
    if folder is None:
        marks.append("skipped: no Go2 in the Menagerie cache")
        return
    pg = _world("go2_motors", {"go2": {"file": str(folder / "go2.xml")}})
    _run(pg, 1.0, marks)
    pg.behaviors.move("go2", 0.3, 0.0, walker="gait")
    _run(pg, 3.0, marks)
    pg.behaviors.stop("go2")
    _run(pg, 2.0, marks)


def go2_rl(marks: list[str]) -> None:
    pg = _world("go2_rl", {"go2": GO2})
    pg.behaviors.move("go2", 0.3, 0.0, walker="rl")
    _run(pg, 3.0, marks)
    pg.behaviors.stop("go2")
    _run(pg, 3.0, marks)


def g1_rl(marks: list[str]) -> None:
    pg = _world("g1_rl", {"g1": {"library": "g1"}})
    _run(pg, 1.0, marks)
    pg.behaviors.move("g1", 0.3, 0.0)
    _run(pg, 3.0, marks)
    pg.behaviors.stop("g1")
    _run(pg, 2.0, marks)


def hexapod(marks: list[str]) -> None:
    pg = _world("hexapod", {"hexapod": _built("HEXAPOD")})
    _run(pg, 1.0, marks)
    pg.behaviors.move("hexapod", 0.15, 0.0)
    _run(pg, 2.0, marks)


def humanoid(marks: list[str]) -> None:
    pg = _world("humanoid", {"h": {"template": "humanoid"}})
    _run(pg, 2.0, marks)
    pg.behaviors.strike_pose("h", "wave")
    _run(pg, 3.0, marks)
    pg.behaviors.move("h", 0.08, 0.3)
    _run(pg, 3.0, marks)
    pg.behaviors.stop("h")
    _run(pg, 2.0, marks)


def balancer(marks: list[str]) -> None:
    pg = _world("balancer", {"b": _built("BALANCER")})
    _run(pg, 2.0, marks)
    pg.behaviors.move("b", 0.3, 0.5)
    _run(pg, 3.0, marks)
    pg.behaviors.stop("b")
    _run(pg, 2.0, marks)


def arm_pick(marks: list[str]) -> None:
    pg = _world("arm_pick", {"arm": {"template": "arm"}})
    pg.add_object("cube", shape="box", size=[0.04, 0.04, 0.04], pos=[0.25, -0.12], color="red")
    pg.add_object("tray", shape="bin", size=[0.14, 0.14, 0.05], pos=[0.05, 0.28], color="blue")
    _arm, info = pg.behaviors.pick_and_place("arm", "cube", "tray")
    _run(pg, 8.0, marks)
    marks.append(f"placed={pg.behaviors.placed('cube', info['region'])['success']}")


def arm_pd(marks: list[str]) -> None:
    from robotforge.control.classic import CartesianPD, JointPD
    from robotforge.control.tasks import make_task

    pg = _world("arm_pd", {"arm": {"template": "arm"}})
    s = pg.session
    task = make_task("reach", s.model, "arm")
    for controller in (JointPD, CartesianPD):
        s.set_controller("arm", controller(task, s.model, s.data))
        _run(pg, 2.0, marks)


def warehouse_trip(marks: list[str]) -> None:
    pg = _world("warehouse_trip", {"go2": GO2})
    pg.load_scene("warehouse")
    pg.behaviors.navigate("go2", "dock")
    _run(pg, 6.0, marks)


SCENARIOS: dict[str, Callable[[list[str]], None]] = {
    f.__name__: f
    for f in (
        dog_fresh,
        dog_shove,
        dog_drag,
        dog_walk_turn_stop,
        dog_fast_turn,
        dog_ramp,
        two_dogs,
        go2_gait,
        go2_motors,
        go2_rl,
        g1_rl,
        hexapod,
        humanoid,
        balancer,
        arm_pick,
        arm_pd,
        warehouse_trip,
    )
}


def worker(names: list[str]) -> dict[str, list[str]]:
    out = {}
    for name in names:
        marks: list[str] = []
        try:
            SCENARIOS[name](marks)
        except Exception as exc:  # recorded as the scenario's last mark, so both sides can be compared
            marks.append(f"error: {type(exc).__name__}: {exc}")
        out[name] = marks
    return out


# ---- the comparison ---------------------------------------------------------------------------------------------


def _hashes(names: list[str], src: Path | None) -> dict[str, list[str]]:
    with tempfile.TemporaryDirectory(prefix="rf-same-") as home:
        env = {**os.environ, "ROBOTFORGE_HOME": home, "ROBOTFORGE_NO_BROWSER": "1"}
        if src is not None:
            env["PYTHONPATH"] = os.pathsep.join([str(src), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
        args = [sys.executable, str(Path(__file__).resolve()), "--worker", *names]
        done = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", env=env)
    if done.returncode != 0:
        raise SystemExit(f"the worker failed:\n{done.stderr[-2000:]}")
    return json.loads(done.stdout.strip().splitlines()[-1])


def compare(now: dict[str, list[str]], before: dict[str, list[str]]) -> dict[str, str]:
    """Per scenario: "identical", or where the hashes first differ."""
    out = {}
    for name, marks in now.items():
        old = before.get(name, [])
        first = next((i for i, (a, b) in enumerate(zip(marks, old, strict=False)) if a != b), None)
        if first is None and len(marks) == len(old):
            out[name] = "identical"
        elif first is None:
            out[name] = f"differs: {len(old)} marks before, {len(marks)} now"
        else:
            out[name] = f"differs from mark {first} ({MARK_EVERY * (first + 1):.2f} s)"
        if marks and marks[-1].startswith(("error", "skipped")):
            out[name] += f" [{marks[-1][:120]}]"
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("rev", nargs="?", help="the revision to compare with (e.g. main, a sha)")
    parser.add_argument("--only", nargs="*", choices=list(SCENARIOS), help="these scenarios only")
    parser.add_argument("--worker", nargs="*", help=argparse.SUPPRESS)  # internal: run scenarios in this process
    args = parser.parse_args(argv)
    if args.worker is not None:
        print(json.dumps(worker(args.worker)))
        return 0
    if not args.rev:
        parser.error("give the revision to compare with")
    sys.path.insert(0, str(ROOT / "scripts"))
    import bench  # its _source_at unpacks a revision's source

    names = args.only or list(SCENARIOS)
    now = _hashes(names, None)
    with bench._source_at(args.rev) as old:
        before = _hashes(names, old)
    verdicts = compare(now, before)
    for name, verdict in verdicts.items():
        print(f"{name:<20} {verdict}  ({len(now[name])} marks)")
    same = all(v.startswith("identical") for v in verdicts.values())
    print("all identical" if same else "DIFFERENT")
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
