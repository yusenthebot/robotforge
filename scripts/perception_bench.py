"""The perception proof: a Go2 enters the warehouse it has never seen, draws its map, finds itself in it and goes to a
named rack by its own map alone. Prints and saves what was measured.

    uv run python scripts/perception_bench.py [--out DIR] [--rack rack_8] [--seconds 150] [--seed N]

Phases, all headless and in simulated time (a few minutes of computer time):
1. explore: map and pose from the lidar and a drifting odometry alone, to the nearest unseen frontier again and again
   inside the warehouse's floor; the map's accuracy against the scene and the pose error are sampled as it goes;
2. localize from nowhere: the map is kept as it is, the particle filter is spread over its free cells and the robot
   walks; how far it walked before the filter held its place, and how far off it was;
3. go to the rack: `move(to=rack, map="own")`, planned on its own map and steered by its own pose estimate; how it
   arrived, in the truth's terms.
The ground truth is read only to score, never to map, localize or plan. Maintainer tool, not part of robotforge.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

AREA = [-7.5, -3.4, 7.5, 5.4]  # the warehouse's floor; its open sides beyond are left alone
START = [-6.2, 2.6]
SAMPLE_EVERY = 5.0  # s of simulated time between samples of the map's accuracy
SEARCH_LOOP = [
    [-3.5, 2.6],
    [1.5, 2.6],
    [3.0, 0.2],
    [-2.5, -0.4],
    [-5.0, 1.0],
]  # a walk for the filter to find itself on


def build(seed: int) -> Any:  # seed picks the warehouse layout
    from robotforge.playground import Playground

    pg = Playground.with_demo()
    pg.new_world("perception_proof")
    pg.load_scene("warehouse", seed=seed, keep_robots=False, starter=False)
    pg.add_robot("go2", library="go2", pos=START, yaw=0)
    return pg


def explore(pg: Any, seconds: float, filter_seed: int = 0) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Explore, sampling the map's accuracy every SAMPLE_EVERY s of simulated time. `filter_seed` seeds the particle
    filter's random draws, which the layout's seed does not."""
    hub = pg.perception
    hub.start("go2", {})
    mapper = hub.mapper("go2")
    mapper.filter.rng = np.random.default_rng(filter_seed)
    rows: list[dict[str, Any]] = []
    next_sample = 0.0

    def sample() -> None:
        nonlocal next_sample
        status = hub.status("go2")
        accuracy = status.get("accuracy", {})
        rows.append(
            {
                "t": status["time"],
                "scans": status["scans"],
                "pose_error_m": status.get("pose_error", {}).get("position_m"),
                "odometry_error_m": status.get("odometry_error", {}).get("position_m"),
                "heading_error_deg": status.get("pose_error", {}).get("heading_deg"),
                "precision": accuracy.get("precision"),
                "wall_error_m": accuracy.get("wall_error_m", {}).get("mean"),
                "explored": accuracy.get("explored"),
            }
        )
        next_sample = status["time"] + SAMPLE_EVERY

    trips = hub.explore_trips("go2", seconds, area=AREA)
    summary: dict[str, Any] = {}
    try:
        while True:
            trip, limit = next(trips)
            budget = limit + 1.0
            while not trip.done and budget > 0:
                step = min(1.0, budget)
                pg.run_fast(lambda t=trip: t.done, step)
                budget -= step
                if float(pg.session.data.time) >= next_sample:
                    sample()
    except StopIteration as done:
        summary = done.value or {}
    sample()
    assert mapper.grid is not None
    return summary, rows


def search(pg: Any) -> dict[str, Any]:
    """Keep the map, forget where the robot is, and find out by walking."""
    hub = pg.perception
    hub.stop("go2")
    hub.localize("go2", "global", 2500)
    mapper = hub.mapper("go2")
    started = float(pg.session.data.time)
    first_held = None  # judged against the truth (error < 0.3 m), to measure; the filter cannot know this
    first_confident = None  # by the filter's own spread alone (sigma < 0.25 m), which is all it can know
    travelled = 0.0
    last = pg.behaviors.pose("go2")["position"][:2]
    errors: list[tuple[float, float, float]] = []
    for point in SEARCH_LOOP:
        trip = pg.behaviors.navigate("go2", point, seconds=40)  # the walk is the experiment's, not its map's
        while not trip.done:
            pg.run_fast(lambda t=trip: t.done, 0.5)
            here = pg.behaviors.pose("go2")["position"][:2]
            travelled += math.dist(here, last)
            last = here
            est = mapper.estimate()
            check = hub.pose_check("go2")
            error = check.get("error_position_m", math.inf)
            errors.append((float(pg.session.data.time) - started, travelled, error))
            moment = (float(pg.session.data.time) - started, travelled, error)
            if first_confident is None and est.sigma_m < 0.25:
                first_confident = moment
            if first_held is None and est.sigma_m < 0.25 and error < 0.3:
                first_held = moment
        if first_held is not None and len(errors) > 10 and errors[-1][1] - first_held[1] > 4.0:
            break
    out: dict[str, Any] = {"particles": 2500, "walked_m": round(travelled, 1)}
    if first_held is not None:
        out.update(held_after_s=round(first_held[0], 1), held_after_m=round(first_held[1], 1))
    if first_confident is not None:  # what the filter itself would have claimed, and how right that was
        out["confident_after_m"] = round(first_confident[1], 1)
        out["error_when_confident_m"] = round(first_confident[2], 3)
    out["held_means"] = "spread under 0.25 m and the true error under 0.3 m (the truth is read to say when)"
    final = errors[-1]
    out["final_error_m"] = round(final[2], 3)
    out["error_after_holding_m"] = {
        "max": round(max(e for _, d, e in errors if first_held and d >= first_held[1]), 3) if first_held else None
    }
    out["trace"] = [[round(t, 1), round(d, 2), round(min(e, 20.0), 3)] for t, d, e in errors[::2]]
    return out


def rack_gap(pg: Any, rack: str) -> float:
    """How far the robot's middle is from the rack's nearest face (m), from the shapes the simulator has."""
    import mujoco

    from robotforge.behaviors import nav_map

    model, data = pg.session.model, pg.session.data
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, rack)
    lows, highs = [], []
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) == body:
            low, high = nav_map.world_extent(model, data, g)
            lows.append(low[:2])
            highs.append(high[:2])
    low, high = np.min(lows, axis=0), np.max(highs, axis=0)
    here = data.xpos[nav_map.root_body(model, "go2")][:2]
    outside = np.maximum(np.maximum(low - here, 0.0), here - high)
    return float(np.hypot(*outside))


def farthest_rack(pg: Any) -> str:
    """The rack furthest from where the robot is: the task is a name, and which one is the proof's choice."""
    import mujoco

    model, data = pg.session.model, pg.session.data
    here = pg.behaviors.pose("go2")["position"][:2]
    racks = [
        (math.dist(data.xpos[b][:2], here), mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b))
        for b in range(model.nbody)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith("rack_")
    ]
    return max(racks)[1]


def reference_trip(seed: int, start: dict[str, Any], rack: str) -> dict[str, Any]:
    """The same trip on the scene's own map and the robot's true pose (what move(to=...) does without map="own"),
    from the pose the own-map trip started at, in a fresh copy of the world."""
    ref = build(seed)
    ref.update_robot("go2", pos=start["position"][:2], yaw=start["heading_deg"])
    trip = ref.behaviors.navigate("go2", rack, seconds=120)
    ref.run_fast(lambda: trip.done, 125.0)
    result = trip.result()
    return {
        "arrived": bool(result["arrived"]),
        "planned_path_m": round(trip.plan.length, 2),
        "walked_m": result.get("distance_m"),
        "time_s": result.get("time_s"),
        "robot_to_rack_face_m": round(rack_gap(ref, rack), 2),
    }


def go_to(pg: Any, rack: str, seed: int) -> dict[str, Any]:
    """Go to the rack on the robot's own map; judge the arrival by the truth, beside the scene-map trip's."""
    hub = pg.perception
    mapper = hub.mapper("go2")
    mapper.mapping = True  # it maps on the way too
    start = pg.behaviors.pose("go2")
    if rack == "farthest":
        rack = farthest_rack(pg)
    trip = pg.behaviors.navigate("go2", rack, seconds=120, own_map=mapper)
    planned = round(trip.plan.length, 2)
    pg.run_fast(lambda: trip.done, 125.0)
    result = trip.result()
    pose = pg.behaviors.pose("go2")
    import mujoco

    body = mujoco.mj_name2id(pg.session.model, mujoco.mjtObj.mjOBJ_BODY, rack)
    toward = pg.session.data.xpos[body][:2] - np.array(pose["position"][:2])
    facing = abs((math.degrees(math.atan2(toward[1], toward[0])) - pose["heading_deg"] + 180) % 360 - 180)
    return {
        "rack": rack,
        "from": [round(v, 2) for v in start["position"][:2]],
        "arrived": bool(result["arrived"]),
        "why": result.get("why"),
        "planned_path_m": planned,
        "walked_m": result.get("distance_m"),
        "time_s": result.get("time_s"),
        "replans": result.get("replans", 0),
        "robot_to_rack_face_m": round(rack_gap(pg, rack), 2),
        "facing_error_deg": round(float(facing), 1),
        "upright": pose["upright"],
        "believed_vs_true": hub.pose_check("go2"),
        "note": result.get("note"),
        "scene_map_trip": reference_trip(seed, start, rack),
    }


def headline(summary: dict[str, Any], accuracy: dict[str, Any], e: np.ndarray, o: np.ndarray) -> dict[str, Any]:
    """The numbers a report states, together (all of them, not the flattering ones): how exploring ended, the map's
    precision and recall against the scene, how much of the floor was seen, how much of what stands in the robot's
    way (at any height) is mapped free, and the pose errors."""
    nav = accuracy.get("navigation", {})
    return {
        "ended": summary.get("ended"),
        "explored_share": accuracy.get("explored"),
        "precision": accuracy.get("precision"),
        "recall": accuracy.get("recall"),
        "f1": accuracy.get("f1"),
        "wall_error_mean_cm": round(100 * accuracy.get("wall_error_m", {}).get("mean", float("nan")), 1),
        "unsafe_free": accuracy.get("unsafe_free"),
        "reachability_found": nav.get("found"),
        "planned_through_blocked": nav.get("through_blocked"),
        "navigation_unscored": nav.get("unscored"),
        "pose_rms_m": round(float(np.sqrt(np.mean(e**2))), 3),
        "pose_max_m": round(float(e.max()), 3),
        "odometry_rms_m": round(float(np.sqrt(np.mean(o**2))), 3),
        "odometry_max_m": round(float(o.max()), 3),
    }


def plot(rows: list[dict[str, Any]], search_trace: list[list[float]], out: Path) -> None:
    """Pose error while exploring (the filter's and the odometry's alone), and while searching from nowhere."""
    from PIL import Image, ImageDraw, ImageFont

    width, height, margin = 1560, 640, 90
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(18)
    small = ImageFont.load_default(14)

    def panel(
        x0: int,
        title: str,
        series: list[tuple[str, tuple[int, int, int], list[tuple[float, float]]]],
        ymax: float,
        xlabel: str,
    ) -> None:
        w, h = 620, height - 2 * margin - 20
        top = margin + 20
        draw.text((x0, margin - 38), title, fill=(29, 32, 38), font=font)
        draw.rectangle([x0, top, x0 + w, top + h], outline=(120, 124, 130))
        xmax = max((p[0] for _, _, pts in series for p in pts), default=1.0) or 1.0
        for k in range(5):
            y = top + h - h * k / 4
            draw.line([(x0, y), (x0 + w, y)], fill=(226, 228, 232))
            draw.text((x0 - 54, y - 8), f"{ymax * k / 4:.2f}", fill=(90, 96, 106), font=small)
        draw.text((x0 + w / 2 - 40, top + h + 14), xlabel, fill=(90, 96, 106), font=small)
        for _, color, pts in series:
            line = [(x0 + w * px / xmax, top + h - h * min(py, ymax) / ymax) for px, py in pts]
            if len(line) > 1:
                draw.line(line, fill=color, width=3)
        for i, (label, color, _) in enumerate(series):
            draw.rectangle([x0 + 16, top + 14 + 24 * i, x0 + 36, top + 28 + 24 * i], fill=color)
            draw.text((x0 + 44, top + 12 + 24 * i), label, fill=(29, 32, 38), font=small)
        draw.text((x0 - 54, margin - 8), "m", fill=(90, 96, 106), font=small)

    blue, grey = (30, 90, 220), (140, 140, 140)
    panel(
        margin,
        "Pose error while it explores (m)",
        [
            (
                "odometry alone",
                grey,
                [(r["t"], r["odometry_error_m"]) for r in rows if r["odometry_error_m"] is not None],
            ),
            ("particle filter", blue, [(r["t"], r["pose_error_m"]) for r in rows if r["pose_error_m"] is not None]),
        ],
        max(1.0, max((r["odometry_error_m"] or 0) for r in rows) * 1.1),
        "simulated seconds",
    )
    if search_trace:
        panel(
            840,
            "Finding itself from nowhere (m)",
            [("position error", blue, [(row[1], row[2]) for row in search_trace])],
            3.0,
            "meters walked",
        )
    image.save(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("perception_proof"))
    parser.add_argument("--rack", default="farthest", help="a rack's name, or farthest from where it ends exploring")
    parser.add_argument("--seconds", type=float, default=150.0)
    parser.add_argument("--seed", type=int, default=0, help="the warehouse layout")
    parser.add_argument("--filter-seed", type=int, default=0, help="the particle filter's random draws")
    args = parser.parse_args()
    os.environ.setdefault("ROBOTFORGE_HOME", tempfile.mkdtemp(prefix="rf-proof-"))
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    pg = build(args.seed)
    summary, rows = explore(pg, args.seconds, args.filter_seed)
    hub = pg.perception
    status = hub.status("go2")
    print(f"explored: {summary} in {time.monotonic() - t0:.0f} s of computer time")
    print(json.dumps({k: status[k] for k in ("map", "pose_error", "odometry_error", "accuracy")}, indent=1))
    errors = hub.errors("go2")
    e = errors["estimate_m"]
    o = errors["odometry_m"]
    print(
        f"pose error over the run: rms {np.sqrt(np.mean(e**2)):.3f} m, max {e.max():.3f} m; "
        f"odometry alone: rms {np.sqrt(np.mean(o**2)):.3f} m, max {o.max():.3f} m"
    )
    mapper = hub.mapper("go2")
    from robotforge.perception import picture

    (args.out / "map_after_exploring.png").write_bytes(picture.map_png(mapper))
    found = search(pg)
    print("from nowhere:", {k: v for k, v in found.items() if k != "trace"})
    (args.out / "map_after_search.png").write_bytes(picture.map_png(mapper))
    reached = go_to(pg, args.rack, args.seed)
    print("to the rack:", json.dumps(reached, indent=1))
    final = hub.status("go2")
    result = {
        "seed": args.seed,
        "filter_seed": args.filter_seed,
        "explore": summary,
        "headline": headline(summary, status["accuracy"], e, o),
        "map": final["map"],
        "accuracy_after_exploring": status["accuracy"],
        "accuracy_at_the_end": final["accuracy"],
        "pose_error_rms_m": round(float(np.sqrt(np.mean(e**2))), 3),
        "pose_error_max_m": round(float(e.max()), 3),
        "heading_error_max_deg": round(float(errors["heading_deg"].max()), 2),
        "odometry_error_rms_m": round(float(np.sqrt(np.mean(o**2))), 3),
        "odometry_error_max_m": round(float(o.max()), 3),
        "odometry_heading_error_max_deg": round(float(errors["odometry_heading_deg"].max()), 2),
        "samples": rows,
        "from_nowhere": found,
        "to_the_rack": reached,
        "computer_seconds": round(time.monotonic() - t0, 1),
    }
    (args.out / "perception_proof.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    plot(rows, found.get("trace", []), args.out / "pose_error.png")
    print(f"saved to {args.out}")
    sys.exit(0 if reached["arrived"] else 1)


if __name__ == "__main__":
    main()
