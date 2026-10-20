"""G6 demos for scripts/demos.py: an arm wipes a table at a constant force with its force plot in view, puts a peg in a
hole it has wrong by a few millimetres, and picks things of no simple shape by a grasp planned from their shape.

    uv run python scripts/demos.py --only g6_panda_wipe g6_peg_in_hole g6_irregular_pick

`build(Demo, caption)` returns the Demo entries (scripts/demos.py passes its own Demo class and caption function,
so that this file need not import it).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

TABLE_TOP = 0.25
WOOD = "#9c6b3f"


def _tween(
    eye0: tuple[float, ...], at0: tuple[float, ...], eye1: tuple[float, ...], at1: tuple[float, ...], seconds: float
) -> str:
    """Page script: slide the camera from one view to another in `seconds`, easing in and out."""
    return f"""
        window.__cam = Symbol();
        const me = window.__cam, rf = window.__robotforge, t0 = performance.now();
        const a = {list(eye0)}, b = {list(at0)}, c = {list(eye1)}, d = {list(at1)};
        (async () => {{
            while (window.__cam === me) {{
                const x = Math.min(1, (performance.now() - t0) / {seconds * 1000});
                const k = x * x * (3 - 2 * x);
                rf.lookAt(a.map((v, i) => v + (c[i] - v) * k), b.map((v, i) => v + (d[i] - v) * k));
                if (x >= 1) break;
                await new Promise((r) => requestAnimationFrame(r));
            }}
        }})();
    """


def _run_task(robot: str, task: str, method: str, params: dict[str, Any], done: Callable[[Any], bool]) -> Callable:
    """An action that starts `method` on `task` on the live world, as run_controller does, and waits (the timeline
    with it) until `done(task)` or a minute is up."""

    def act(pg: Any) -> None:
        from robotforge.control import make_controller, make_task
        from robotforge.control.compare import EpisodeMonitor

        session = pg.session
        with session.lock:
            model, data = session.model, session.data
            bound = make_task(task, model, robot, params)
            bound.begin(data)
            controller = make_controller(method, bound, model, data)
            bound.begin(data)
            monitor = EpisodeMonitor(bound, model, data)
            session.set_controller(robot, controller)
            session.set_controller("__monitor__", monitor)
        pg.demo_task = bound
        end = time.monotonic() + 90.0
        while not done(bound) and time.monotonic() < end:
            time.sleep(0.05)
        time.sleep(1.0)
        print(f"{task} {method}: {bound.report(session.data)}", flush=True)
        session.set_controller("__monitor__", None)

    return act


def _wipe_world(pg: Any) -> None:
    pg.set_floor("tile")
    pg.add_robot("panda", library="panda", pos=[0.0, 0.0])
    pg.update_robot("panda", end_effector="wiper")
    pg.add_object("desk", shape="table", size=[0.6, 0.6, TABLE_TOP], pos=[0.5, 0.0], color=WOOD)
    pg.add_sensor("panda", {"type": "force_torque", "on": "wiper", "name": "wrist"})


def _peg_world(pg: Any) -> None:
    pg.set_floor("tile")
    pg.add_robot("arm", library="ur5e", mounts=[{"library": "robotiq_2f85"}])
    pg.add_object("socket1", shape="socket", pos=[0.35, -0.2], size=[0.028, 0.12, 0.04], color="#6b7a8a")
    pg.add_object("peg", shape="box", size=[0.024, 0.024, 0.08], pos=[0.45, 0.2], yaw=25.0, mass=0.05, color="orange")
    pg.add_sensor("arm", {"type": "force_torque", "on": "wrist_3_link", "name": "wrist"})


def _hold_peg(pg: Any) -> None:
    """Take hold of the peg and bring it over the socket, with the arm's own motions (move_arm, gripper)."""
    for position, grip in (
        ([0.45, 0.2, 0.2], "open"),
        ([0.45, 0.2, 0.07], "open"),
        (None, "close"),
        ([0.45, 0.2, 0.25], "close"),
        ([0.37, -0.15, 0.25], "close"),
    ):
        if position is None:
            arm = pg.behaviors.set_gripper("arm", "close")
            time.sleep(1.4)
        else:
            arm = pg.behaviors.move_arm("arm", position, grip)
            end = time.monotonic() + 25
            while not arm.done and time.monotonic() < end:
                time.sleep(0.05)


def _irregular_world(pg: Any) -> None:
    pg.set_floor("tile")
    pg.add_robot("arm", library="ur5e", mounts=[{"library": "robotiq_2f85"}])
    pg.add_object("bin", shape="bin", size=[0.25, 0.2, 0.08], pos=[0.0, -0.45], color="#4a78b0")
    pg.add_object("tee", shape="tee", pos=[0.45, 0.15], color="#d9534f")
    pg.add_object("bottle", shape="bottle", pos=[0.62, 0.05], color="#3c9d5d")
    pg.add_object("stone", shape="stone", pos=[0.3, 0.25])


def _pick_each(pg: Any) -> None:
    for name in ("tee", "bottle", "stone"):
        arm, info = pg.behaviors.pick_and_place("arm", name, "bin")
        end = time.monotonic() + 60
        while not arm.done and time.monotonic() < end:
            time.sleep(0.05)
        time.sleep(0.8)
    pg.demo_done = True


def build(Demo: type, caption: Callable[..., str]) -> list:
    """The three demos."""

    def phase(*names: str) -> Callable[[Any], bool]:
        return lambda task: task.plan is not None and task.plan.phase in names

    wipe_cam = (
        "const rf = window.__robotforge; rf.select('panda');"
        "await new Promise((r) => setTimeout(r, 600));"
        "document.getElementById('sensor-views').style.zoom = '1.9';"
        "rf.lookAt([1.45, -1.05, 0.85], [0.3, 0.0, 0.45]);"
    )
    peg_cam = (
        "const rf = window.__robotforge; rf.select('arm');"
        "await new Promise((r) => setTimeout(r, 600));"
        "document.getElementById('sensor-views').style.zoom = '1.6';"
        "rf.lookAt([1.4, -1.15, 0.85], [0.3, -0.05, 0.28]);"
    )
    peg_close = _tween((1.4, -1.15, 0.85), (0.3, -0.05, 0.28), (0.36, -0.95, 0.3), (0.36, -0.2, 0.1), 2.5)
    pick_cam = "const rf = window.__robotforge; rf.select(null); rf.lookAt([1.15, -1.0, 0.95], [0.22, -0.12, 0.22]);"
    return [
        Demo(
            "g6_panda_wipe",
            _wipe_world,
            70.0,
            timeline=[
                (
                    0.4,
                    caption("Wiping at a constant 4 N; the plot is the raw wrist sensor, in the hand's own axes"),
                ),
                (0.8, _run_task("panda", "wipe", "osc", {"force": 4.0, "speed": 0.05}, phase("done"))),
            ],
            page=wipe_cam,
            settle=1.5,
            until=lambda pg: getattr(pg, "demo_task", None) is not None and pg.demo_task.plan.phase == "done",
            tail=2.0,
        ),
        Demo(
            "g6_peg_in_hole",
            _peg_world,
            60.0,
            timeline=[
                (0.4, caption("UR5e picks up the peg and brings it over the socket")),
                (0.8, _hold_peg),
                (1.0, caption("The hole is known only to 6 mm: soft pushing lets the chamfer steer the peg in")),
                (1.1, peg_close),
                (
                    1.2,
                    _run_task("arm", "insert", "osc", {"hole": "socket1", "error": [0.006, 0.0]}, phase("done")),
                ),
            ],
            page=peg_cam,
            settle=1.5,
            until=lambda pg: getattr(pg, "demo_task", None) is not None and pg.demo_task.plan.phase == "done",
            tail=2.5,
        ),
        Demo(
            "g6_irregular_pick",
            _irregular_world,
            75.0,
            timeline=[
                (0.4, caption("Each grasp is planned from the shape: the T by its stem, the bottle, the stone")),
                (1.0, _pick_each),
            ],
            page=pick_cam,
            settle=1.5,
            until=lambda pg: getattr(pg, "demo_done", False),
            tail=2.0,
        ),
    ]
