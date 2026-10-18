"""Measure how well aircraft fly: hover, a step, a square, a landing, a gust, and other airframes.

    uv run python scripts/flight_bench.py [--real]

Runs headless on the same Playground the MCP tools use (move(to=...) is behaviors.navigate). The library aircraft
come from MuJoCo Menagerie (downloaded once into the cache) with --real, else from the offline stand-ins in
tests/fixtures/menagerie. Prints one table; every number is measured, none is assumed. Maintainer tool.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DT = 0.002


def _playground(name: str) -> Any:
    from robotforge.playground import Playground
    from robotforge.sim.session import SimSession

    pg = Playground(session=SimSession.from_xml("<mujoco/>"))
    pg.new_world(name)
    return pg


def _fly(pg: Any, robot: str, to: Any, speed: float | None = None, seconds: float = 60.0) -> dict[str, Any]:
    trip = pg.behaviors.navigate(robot, to, speed, None, seconds)
    pg.run_fast(lambda: trip.done, seconds + 1.0)
    return trip.result()


def _root(pg: Any, robot: str) -> int:
    return pg.session.controllers[robot].frame.root


def _sample(pg: Any, robot: str, seconds: float, every: int = 5) -> np.ndarray:
    """Positions of the robot's root every `every` steps for `seconds`."""
    root, out = _root(pg, robot), []
    for _ in range(int(round(seconds / DT / every))):
        pg.session.step(every)
        out.append(pg.session.data.xpos[root].copy())
    return np.array(out)


def hover(pg: Any, robot: str, z: float = 2.0) -> dict[str, float]:
    _fly(pg, robot, [0.0, 0.0, z])
    pg.run_fast(lambda: False, 2.0)
    p = _sample(pg, robot, 10.0)
    err = np.linalg.norm(p - [0.0, 0.0, z], axis=1)
    return {"hover_rms_mm": 1000 * float(np.sqrt(np.mean(err**2))), "hover_max_mm": 1000 * float(err.max())}


def step(pg: Any, robot: str, distance: float = 2.0) -> dict[str, float]:
    """A 2 m step across from a hover: rise (10 to 90 %), overshoot, settling within 5 % (10 cm)."""
    root = _root(pg, robot)
    start = pg.session.data.xpos[root].copy()
    pg.behaviors.navigate(robot, (start + [distance, 0.0, 0.0]).tolist(), 3.0, None, 30.0)
    x = _sample(pg, robot, 8.0, every=1)[:, 0] - start[0]
    t = np.arange(len(x)) * DT
    rise = float(t[np.argmax(x >= 0.9 * distance)] - t[np.argmax(x >= 0.1 * distance)])
    outside = np.where(np.abs(x - distance) > 0.05 * distance)[0]
    settle = float(t[outside[-1] + 1]) if len(outside) else 0.0
    return {
        "step_rise_s": rise,
        "step_overshoot_pct": 100 * max(0.0, float(x.max()) - distance) / distance,
        "step_settle_s": settle,
    }


def square(pg: Any, robot: str, side: float = 3.0, z: float = 2.0) -> dict[str, float]:
    """A square of `side` m at height z: cross-track error from the four straight legs while it flies them."""
    root = _root(pg, robot)
    here = pg.session.data.xpos[root].copy()
    corners = [here + d for d in ([side, 0, 0], [side, side, 0], [0, side, 0], [0, 0, 0])]
    trip = pg.behaviors.navigate(robot, [c.tolist() for c in corners], None, None, 60.0)
    path = np.array([here, *corners])
    a, b = path[:-1], path[1:]
    ab = b - a
    errors, t0 = [], pg.session.data.time
    while not trip.done and pg.session.data.time - t0 < 60:
        pg.session.step(5)
        p = pg.session.data.xpos[root]
        s = np.clip(np.einsum("ij,ij->i", p - a, ab) / np.einsum("ij,ij->i", ab, ab), 0, 1)
        errors.append(float(np.min(np.linalg.norm(a + ab * s[:, None] - p, axis=1))))
    e = np.array(errors)
    return {
        "square_time_s": float(pg.session.data.time - t0),
        "square_rms_mm": 1000 * float(np.sqrt(np.mean(e**2))),
        "square_max_mm": 1000 * float(e.max()),
    }


def land(pg: Any, robot: str) -> dict[str, float]:
    here = pg.session.data.xpos[_root(pg, robot)]
    result = _fly(pg, robot, [float(here[0]), float(here[1]), 0.0])
    return {"touchdown_m_s": float(result.get("touchdown_m_s", float("nan"))), "landed": float(result.get("landed", 0))}


def gust(pg: Any, robot: str, wind: float = 8.0, seconds: float = 1.5) -> dict[str, float]:
    """A gust from the side while hovering: the furthest it is pushed, and when it is back within 5 cm."""
    from robotforge.behaviors import air

    _fly(pg, robot, [0.0, 0.0, 2.0])
    pg.run_fast(lambda: False, 2.0)
    root = _root(pg, robot)
    hold = pg.session.data.xpos[root].copy()
    air.blow(pg.session, [wind, 0.0], seconds, 1)
    p = _sample(pg, robot, 8.0, every=1)
    d = np.linalg.norm(p - hold, axis=1)
    out = np.where(d > 0.05)[0]
    return {"gust_push_mm": 1000 * float(d.max()), "gust_back_s": float((out[-1] + 1) * DT) if len(out) else 0.0}


def speed(pg: Any, robot: str) -> dict[str, float]:
    """Physics steps per second of computer time while it hovers with its controller running."""
    _fly(pg, robot, [0.0, 0.0, 1.5])
    pg.session.step(500)  # warm up (the first steps pay for setting things up)
    t0, n = time.perf_counter(), 5000
    pg.session.step(n)
    return {"steps_per_s": n / (time.perf_counter() - t0)}


def run(label: str, add: dict[str, Any], tests: list) -> dict[str, Any]:
    row: dict[str, Any] = {"aircraft": label}
    for test in tests:
        pg = _playground(label)
        pg.add_robot("uav", **add)
        row.update(test(pg, "uav"))
    pg = _playground(label)
    pg.add_robot("uav", **add)
    row.update(speed(pg, "uav"))
    row.update(pg.behaviors.robots()[0]["flight"])
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--real", action="store_true", help="the real Menagerie models (downloads them once)")
    args = parser.parse_args()
    os.environ["ROBOTFORGE_HOME"] = tempfile.mkdtemp(prefix="rf-flight-")
    if not args.real:
        os.environ["ROBOTFORGE_MENAGERIE"] = str(ROOT / "tests" / "fixtures" / "menagerie")
    sys.path.insert(0, str(ROOT / "src"))
    from robotforge.design.builder.guide import EXAMPLE_HEXACOPTER

    def mission(pg: Any, robot: str) -> dict[str, float]:  # take off, a square, land: one trip each
        out = {}
        out.update(hover(pg, robot))
        out.update(step(pg, robot))
        pg.run_fast(lambda: False, 3.0)
        out.update(square(pg, robot))
        out.update(land(pg, robot))
        return out

    rows = [
        run("drone template (4 rotors)", {"template": "drone"}, [mission, gust]),
        run("drone template (6 rotors)", {"template": "drone", "params": {"rotors": 6}}, [mission]),
        run("drone template (8 rotors)", {"template": "drone", "params": {"rotors": 8}}, [mission]),
        run("hexacopter built from parts", {"build": EXAMPLE_HEXACOPTER}, [mission, gust]),
        run("Skydio X2", {"library": "skydio_x2"}, [mission, gust]),
        run("Crazyflie 2", {"library": "crazyflie"}, [mission]),
    ]
    keys = [k for k in rows[0] if k not in ("aircraft", "controller")]
    for row in rows:
        print(row["aircraft"])
        for k in keys:
            if k in row:
                v = row[k]
                print(f"  {k:22s} {v:.3f}" if isinstance(v, float) else f"  {k:22s} {v}")


if __name__ == "__main__":
    main()
