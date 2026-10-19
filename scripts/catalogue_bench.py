"""Add open catalogue robots by name for real and find out which load, stand and move (milestone G12a).

    uv run python scripts/catalogue_bench.py [--seed 12] [--only NAME ...] [--out report.json]

Draws 20 robots at random (seeded) across the kinds, adds each to a fresh world the way an agent does
(add_robot(library=<name>), which downloads the robot's folder on first use), and tries what its kind should do:
an arm reaches a point, a legged robot stands and walks (or says why it cannot), a wheeled robot drives, a hand
closes, an aircraft climbs. Prints one line per robot and a table at the end; the JSON keeps the details.
Maintainer tool, not part of robotforge.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
import time
import traceback
from typing import Any

import mujoco
import numpy as np

QUOTAS = (("arm", 5), ("quadruped", 3), ("humanoid", 4), ("wheeled", 3), ("hand", 3), ("aircraft", 1), ("other", 1))
SETTLE_S = 6.0


def draw(seed: int, quotas=QUOTAS) -> list[str]:
    """Catalogue robots chosen at random per kind (by their tags), the same ones for the same seed."""
    from robotforge.design import catalogue

    rng = random.Random(seed)
    chosen: list[str] = []
    for kind, count in quotas:
        pool = sorted(e.id for e in catalogue.search(kind))
        chosen += rng.sample(pool, min(count, len(pool)))
    return chosen


def _base_height(pg: Any, robot: str) -> float:
    return float(pg.session.data.xpos[pg.behaviors._root_body(robot)][2])


def _upright(pg: Any, robot: str) -> bool:
    d = pg.session.data
    return bool(d.xmat[pg.behaviors._root_body(robot)].reshape(3, 3)[2, 2] > 0.7)


def _finite(pg: Any) -> bool:
    return bool(np.isfinite(pg.session.data.qpos).all())


def _span(pg: Any, robot: str) -> float:
    """How tall the robot's bodies stand: lying on its back is a fraction of that."""
    m, d = pg.session.model, pg.session.data
    zs = [
        d.xpos[b][2]
        for b in range(1, m.nbody)
        if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith(f"{robot}/")
    ]
    return max(zs) - min(zs)


def stand(pg: Any, robot: str, floating: bool) -> dict[str, Any]:
    start = _span(pg, robot)
    q0 = pg.session.data.qpos.copy()
    pg.run_fast(lambda: False, SETTLE_S)
    out: dict[str, Any] = {"finite": _finite(pg)}
    if floating:
        kept = _span(pg, robot) / max(start, 1e-6)
        out.update(height_kept=round(kept, 2), upright=_upright(pg, robot))
        out["ok"] = out["finite"] and out["upright"] and kept > 0.7
    else:
        drift = float(np.max(np.abs(pg.session.data.qpos - q0))) if q0.size else 0.0
        out.update(max_joint_drift=round(drift, 3))
        out["ok"] = out["finite"] and drift < 0.35
    return out


def move_arm(pg: Any, robot: str, about: dict[str, Any]) -> dict[str, Any]:
    from robotforge.behaviors.manager import BehaviorError

    d, m = pg.session.data, pg.session.model
    arm = pg.behaviors.arms(robot)[0]  # the first of a two-armed robot
    site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{arm}/tcp")
    home = d.site_xpos[site].copy()
    base = d.xpos[pg.behaviors._root_body(robot)].copy()
    reach = float(about.get("reach_m") or 0.4)
    # points round where the tool starts (a tall robot's shoulder is not at its base), then in front of the base
    around = [(0.2, 0.0, 0.0), (0.15, 0.0, -0.15), (0.25, 0.0, 0.1), (0.1, 0.1, 0.0), (0.1, -0.1, -0.1)]
    targets = [[float(home[i] + f * reach) for i, f in enumerate(step)] for step in around]
    targets += [
        [float(base[0] + f * reach), float(home[1]), float(base[2] + g * reach)]
        for f, g in ((0.55, 0.45), (0.45, 0.6), (0.6, 0.3))
    ]
    targets += [
        [float(base[0] + x), float(home[1]), float(base[2] + z)] for x, z in ((0.6, 0.5), (0.4, 0.3), (0.3, 0.2))
    ]
    last = ""
    for target in targets:
        try:
            motion = pg.behaviors.move_arm(arm, target)
        except BehaviorError as exc:
            last = str(exc)[:160]
            continue
        pg.run_fast(lambda m=motion: m.done, 30.0)
        pg.run_fast(lambda: False, 0.5)
        err = float(np.linalg.norm(d.site_xpos[site] - target))
        if err < 0.03:
            return {"ok": True, "target": [round(v, 3) for v in target], "error_m": round(err, 4)}
        last = f"stopped {err:.2f} m short of {[round(v, 2) for v in target]}"
    return {"ok": False, "why": last or "no reachable point found"}


def move_walker(pg: Any, robot: str, speed: float, seconds: float, kind: str) -> dict[str, Any]:
    from robotforge.behaviors.manager import BehaviorError

    before = pg.behaviors.pose(robot)["position"]
    try:
        pg.behaviors.move(robot, speed, 0.0)
    except BehaviorError as exc:
        return {"ok": False, "why": str(exc)[:200]}
    except ValueError as exc:
        return {"ok": False, "why": str(exc)[:200]}
    pg.run_fast(lambda: False, seconds)
    after = pg.behaviors.pose(robot)
    pg.behaviors.stop(robot)
    went = float(np.hypot(after["position"][0] - before[0], after["position"][1] - before[1]))
    return {
        "ok": bool(after["upright"]) and went > 0.3 * speed * seconds,
        "moved_m": round(went, 2),
        "upright": after["upright"],
    }


def move_hand(pg: Any, robot: str) -> dict[str, Any]:
    m, d = pg.session.model, pg.session.data
    acts = [
        a for a in range(m.nu) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or "").startswith(f"{robot}/")
    ]
    if not acts:
        return {"ok": False, "why": "no actuators"}
    q0 = d.qpos.copy()
    for a in acts:
        low, high = m.actuator_ctrlrange[a] if m.actuator_ctrllimited[a] else (-1.0, 1.0)
        d.ctrl[a] = high if abs(high) >= abs(low) else low
    pg.run_fast(lambda: False, 2.0)
    moved = float(np.max(np.abs(d.qpos - q0)))
    return {"ok": moved > 0.1 and _finite(pg), "max_joint_change": round(moved, 3)}


def move_aircraft(pg: Any, robot: str) -> dict[str, Any]:
    from robotforge.behaviors.manager import BehaviorError

    start = _base_height(pg, robot)
    try:
        pg.behaviors.heave(robot, 0.6)
    except (BehaviorError, ValueError) as exc:
        return {"ok": False, "why": str(exc)[:200]}
    pg.run_fast(lambda: False, 3.0)
    gain = _base_height(pg, robot) - start
    return {"ok": gain > 0.3 and _finite(pg), "climbed_m": round(gain, 2)}


def try_robot(name: str, home: str) -> dict[str, Any]:
    from robotforge.playground import Playground, PlaygroundError

    os.environ["ROBOTFORGE_HOME"] = home
    row: dict[str, Any] = {"name": name}
    pg = Playground.with_demo()
    pg.new_world(f"b_{name}"[:40])
    t = time.time()
    try:
        added = pg.add_robot("r", library=name)
    except (PlaygroundError, ValueError) as exc:
        row.update(load="FAIL", why=str(exc)[:300], add_s=round(time.time() - t, 1))
        return row
    row["add_s"] = round(time.time() - t, 1)
    about = added["catalogue"]
    row.update(load="ok", kind=about["kind"], about=about["about"], can=about["can"])
    doctor = added.get("doctor") or {}
    unfixed = [f"{f['topic']}: {f['problem']}" for f in doctor.get("findings", []) if f.get("status") != "fixed"]
    if unfixed:
        row["doctor_unfixed"] = unfixed
    root = pg.behaviors._root_body("r")
    first = int(pg.session.model.body_jntadr[root])
    floating = pg.session.model.body_jntnum[root] > 0 and pg.session.model.jnt_type[first] == mujoco.mjtJoint.mjJNT_FREE
    row["stand"] = stand(pg, "r", floating)
    kind = about["kind"]
    try:
        if kind == "arm":
            row["move"] = move_arm(pg, "r", about)
        elif kind in ("quadruped", "humanoid"):
            row["move"] = move_walker(pg, "r", 0.2, 6.0, kind) if floating else {"ok": False, "why": "fixed base"}
        elif kind == "wheeled":
            row["move"] = move_walker(pg, "r", 0.3, 4.0, kind)
        elif kind == "hand":
            row["move"] = move_hand(pg, "r")
        elif kind == "aircraft":
            row["move"] = move_aircraft(pg, "r")
        else:
            row["move"] = move_hand(pg, "r")
    except Exception as exc:  # a bench reports what broke and goes on to the next robot
        row["move"] = {"ok": False, "why": f"{type(exc).__name__}: {exc}"[:200], "trace": traceback.format_exc()[-400:]}
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    names = args.only or draw(args.seed)
    home = tempfile.mkdtemp(prefix="rf-cat-")
    rows = []
    for name in names:
        row = try_robot(name, home)
        rows.append(row)
        move = row.get("move", {})
        print(
            f"{name:28s} load={row['load']:4s} kind={row.get('kind', '-'):10s} add={row['add_s']:5.1f}s "
            f"stand={row.get('stand', {}).get('ok', '-')} move={move.get('ok', '-')} "
            f"{row.get('why') or move.get('why') or ''}"[:200],
            flush=True,
        )
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=1)
    loaded = sum(r["load"] == "ok" for r in rows)
    stood = sum(bool(r.get("stand", {}).get("ok")) for r in rows)
    moved = sum(bool(r.get("move", {}).get("ok")) for r in rows)
    print(f"\n{len(rows)} robots: {loaded} load, {stood} stand, {moved} move")


if __name__ == "__main__":
    main()
