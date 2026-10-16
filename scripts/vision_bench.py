"""Measure the seeing lesson's color finder: how far its estimate of the red cube is from the truth.

    uv run python scripts/vision_bench.py [--spots 60] [--seed 0] [--picks 12] [--scenes]

Puts the red cube at random spots on the lesson's mat (the other cubes where the lesson starts them), takes a
picture from the overhead camera under each light with the lesson's camera noise, runs the finder with the
settings a learner ends up with, and prints, per light, finder setting and way of locating the cube, how often
the biggest blob was the red cube and how far the estimate was from the cube's true middle. `--picks` also has
the lesson's arm pick the red cube from that many spots it can reach, aiming a set distance off the cube's
middle, to see how far off an estimate may be before the grasp fails. The lesson's text quotes these numbers.
`--scenes` instead runs observe(find="all") in every ready-made scene from every preset view and robot camera,
and prints how far each loose object's estimate is from its true middle, with the flags it came with.
Needs offscreen OpenGL (not the pick test).
"""

from __future__ import annotations

import argparse
import math
import os
import tempfile

import mujoco
import numpy as np

from robotforge.behaviors.arm import ArmMotion
from robotforge.design import scenes
from robotforge.learn import seeing
from robotforge.playground import Playground
from robotforge.sim import vision
from robotforge.sim.render import VIEWS, SnapshotRenderer, render_unavailable_reason

TUNED = {"hue": 0.0, "width": 15.0, "sat": 0.3, "val": 0.2}  # where the mask step leaves the knobs
FINDERS = {
    "as tuned": {"balance": 0.0, "clean": 0.0},
    "balance + clean": {"balance": 1.0, "clean": 1.0},
}
METHODS = {"plane 0 cm": (0.0, 0.0), "plane 2 cm": (2.0, 0.0), "depth": (0.0, 1.0)}
AIM_OFF = (0.0, 0.006, 0.012, 0.02)  # m between the aim and the cube's middle in the pick test


def spots(n: int, seed: int) -> list[tuple[float, float]]:
    """Random spots on the mat clear of the other cubes and the tray, as the lesson's test and shuffle keep them."""
    rng = np.random.default_rng(seed)
    (cx, cy), (sx, sy) = seeing.MAT
    others = [spot for color, spot in seeing.CUBES if f"{color}_cube" != seeing.TARGET]
    out: list[tuple[float, float]] = []
    while len(out) < n:
        x = float(rng.uniform(cx - sx / 2 + seeing.INSET, cx + sx / 2 - seeing.INSET))
        y = float(rng.uniform(cy - sy / 2 + seeing.INSET, cy + sy / 2 - seeing.INSET))
        if math.hypot(x - seeing.TRAY[0], y - seeing.TRAY[1]) < 0.11:
            continue
        if any(math.hypot(x - a, y - b) < seeing.SPACING for a, b in others):
            continue
        out.append((x, y))
    return out


def picks(n: int, seed: int) -> None:
    """Pick the red cube from `n` reachable spots per aim error, from the parked pose into the tray."""
    rng = np.random.default_rng(seed)
    places = [p for p in spots(20 * n, seed) if p[0] >= 0.12 and math.hypot(*p) <= seeing.REACH][:n]
    model = seeing.build_world()[1]
    body = model.body(seeing.TARGET)
    adr = model.jnt_qposadr[body.jntadr[0]]
    print(f"pick test: {len(places)} reachable spots, the aim off the cube's middle by")
    for off in AIM_OFF:
        done = 0
        for x, y in places:
            data = mujoco.MjData(model)
            mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
            arm = ArmMotion(model, "arm", model.key("home").qpos.copy())
            arm.q, _, _ = arm._solve(np.array(seeing.PARK))
            data.qpos[arm.chain.qpos_adr] = arm.q
            data.qpos[adr : adr + 7] = [x, y, seeing.CUBE / 2, 1.0, 0.0, 0.0, 0.0]
            mujoco.mj_forward(model, data)
            angle = rng.uniform(0, 2 * math.pi)
            aim = np.array([x + off * math.cos(angle), y + off * math.sin(angle), seeing.CUBE / 2])
            tray = np.array([seeing.TRAY[0], seeing.TRAY[1], 0.005])
            arm.pick_and_place(aim, seeing.CUBE / 2, tray, seeing.TRAY_SIZE[2])
            steps = round(1 / (arm.rate_hz * model.opt.timestep))
            while not arm.done and data.time < 30:
                arm.update(model, data)
                mujoco.mj_step(model, data, nstep=steps)
            mujoco.mj_step(model, data, nstep=250)
            pos = data.xpos[body.id]
            done += bool(np.all(np.abs(pos[:2] - seeing.TRAY) <= np.array(seeing.TRAY_SIZE[:2]) / 2))
        print(f"  {off * 100:.1f} cm: {done}/{len(places)} in the tray")


def middle(model: mujoco.MjModel, data: mujoco.MjData, body: int) -> np.ndarray:
    """The middle of the box around a body's drawn geoms in the world: what a finder should report."""
    corners = []
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != body or model.geom_group[g] >= 3:  # group 3: collision shapes, not drawn
            continue
        center, half = model.geom_aabb[g, :3], model.geom_aabb[g, 3:]
        signs = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)])
        local = center + signs * half
        corners.append(data.geom_xpos[g] + local @ data.geom_xmat[g].reshape(3, 3).T)
    points = np.concatenate(corners)
    return (points.min(axis=0) + points.max(axis=0)) / 2


FLAGS = ("cut_off", "hidden", "side_on")
FINDABLE = 30  # px of a loose object in a named color's window (cleaned) for the sweep to expect it found


def findable(pg: Playground, view: str, bodies: list[int]) -> set[str]:
    """The loose objects that show at least FINDABLE pixels in some named color's window from `view`."""
    model = pg.session.model
    layers = pg.renderer.layers(pg.session, view, 640, 480, geoms=True, exposure=vision.EXPOSURE)
    hsv = vision.to_hsv(layers.rgb)
    masks = [vision.clean(vision.threshold(hsv, r)) for r in vision.NAMED.values()]
    owner = np.where(layers.geoms >= 0, model.geom_bodyid[np.maximum(layers.geoms, 0)], -1)
    out = set()
    for body in bodies:
        mine = owner == body
        if any((mine & mask).sum() >= FINDABLE for mask in masks):
            out.add(model.body(body).name)
    return out


def scene_sweep() -> None:
    """observe(find="all") in every scene from every view: each loose object's error, by the flags it has, and
    the loose objects in view that it did not list."""
    os.environ["ROBOTFORGE_HOME"] = tempfile.mkdtemp()  # the scenes' worlds are saved here, not in the user's
    rows: list[tuple[str, str, str, float, int, tuple[str, ...], float]] = []
    listed = dropped = seen_loose = 0
    missed: list[tuple[str, str, str]] = []
    for scene in scenes.SCENES:
        pg = Playground.with_demo()
        pg.load_scene(scene)
        pg.simulate(0.5)
        model, data = pg.session.model, pg.session.data
        cameras = [model.camera(i).name for i in range(model.ncam) if model.camera(i).name]
        free = int(mujoco.mjtJoint.mjJNT_FREE)
        loose = [  # top bodies on a free joint, not a robot's ("dog/trunk")
            b
            for b in range(1, model.nbody)
            if model.body_jntnum[b] and model.jnt_type[model.body_jntadr[b]] == free and "/" not in model.body(b).name
        ]
        for view in [*VIEWS, *cameras]:
            found, _ = pg.find(view, "all")
            objects = found["objects"]
            listed += len(objects)
            dropped += found.get("more", 0)
            names = {o["object"] for o in objects if "object" in o}
            names |= {name for o in objects for name in o.get("merged", [])}  # listed, as part of a bigger find
            for name in findable(pg, view, loose) - names:
                missed.append((scene, view, name))
            for o in objects:
                if "object" not in o:
                    continue
                seen_loose += 1
                truth = middle(model, data, model.body(o["object"]).id)
                error = float(np.linalg.norm(np.array(o["position"]) - truth)) * 100
                flags = tuple(f for f in FLAGS if o.get(f))
                rows.append((scene, view, o["object"], error, o["pixels"], flags, o["about_cm"]))
    print(f"{listed} finds listed, {dropped} left out past the list's end, {seen_loose} of them loose objects")
    print(f"{len(missed)} loose objects in view in a findable color were not listed:")
    for scene, view, name in missed:
        print(f"  {scene:16s} {view:22s} {name}")
    print(f"{'scene':16s} {'view':22s} {'object':12s} {'cm':>6s} {'px':>6s} {'about':>6s}  flags")
    for scene, view, name, error, pixels, flags, about in sorted(rows, key=lambda r: -r[3]):
        print(f"{scene:16s} {view:22s} {name:12s} {error:6.2f} {pixels:6d} {about:6.1f}  {' '.join(flags)}")
    clean = np.array([r[3] for r in rows if not r[5]])
    if clean.size:
        ratio = np.array([r[3] / r[6] for r in rows if not r[5]])
        print(
            f"no flag: {clean.size} finds, mean {clean.mean():.2f} cm, median {np.median(clean):.2f}, "
            f"p90 {np.percentile(clean, 90):.2f}, max {clean.max():.2f}; error / about_cm: median "
            f"{np.median(ratio):.2f}, within 1x {np.mean(ratio <= 1):.0%}, within 2x {np.mean(ratio <= 2):.0%}, "
            f"max {ratio.max():.2f}"
        )
    for flag in FLAGS:
        errs = np.array([r[3] for r in rows if flag in r[5]])
        if errs.size:
            print(f"{flag}: {errs.size} finds, median {np.median(errs):.2f} cm, max {errs.max():.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spots", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--picks", type=int, default=0, help="also run the pick test from this many spots")
    parser.add_argument("--scenes", action="store_true", help="measure observe(find='all') in the scenes instead")
    args = parser.parse_args()
    if args.picks:
        picks(args.picks, args.seed)
    reason = render_unavailable_reason()
    if reason is not None:
        raise SystemExit(f"no offscreen OpenGL here: {reason}")
    if args.scenes:
        scene_sweep()
        return
    lesson = seeing.SeeingLesson()
    model = lesson.build()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    arm = ArmMotion(model, "arm", model.key("home").qpos.copy())
    park, _, _ = arm._solve(np.array(seeing.PARK))
    data.qpos[arm.chain.qpos_adr] = park
    body = model.body(seeing.TARGET)
    adr = model.jnt_qposadr[body.jntadr[0]]
    renderer = SnapshotRenderer()
    places = spots(args.spots, args.seed)
    print(
        f"{len(places)} spots, seed {args.seed}; error is the 3D distance to the cube's middle in cm, "
        "across its mean distance on the table alone"
    )
    print(
        f"{'light':14s} {'finder':16s} {'method':11s} {'found':>7s} {'mean':>6s} {'median':>6s} {'p90':>6s} "
        f"{'max':>6s} {'across':>6s}"
    )
    for li, light in enumerate(seeing.LIGHTS):
        seeing.apply_light(model, light)
        noise = np.random.default_rng(args.seed + 100 * li)
        frames = []
        for x, y in places:
            data.qpos[adr : adr + 7] = [x, y, seeing.CUBE / 2, 1.0, 0.0, 0.0, 0.0]
            mujoco.mj_forward(model, data)
            layers = renderer.layers_of(model, data, seeing.CAMERA, seeing.WIDTH, seeing.HEIGHT, 0, geoms=True)
            frames.append((layers, seeing.noisy(layers.rgb, light.noise, noise), np.array([x, y, seeing.CUBE / 2])))
        for finder, switches in FINDERS.items():
            for method, (height, depth) in METHODS.items():
                values = {**TUNED, **switches, "height": height, "depth": depth}
                errors, across = [], []
                found = 0
                for layers, picture, truth in frames:
                    look = seeing.look(layers, picture, values, lesson.owner)
                    if look.estimate is None or look.on[:1] != [seeing.TARGET]:
                        continue
                    found += 1
                    errors.append(float(np.linalg.norm(look.estimate - truth)) * 100)
                    across.append(float(np.linalg.norm(look.estimate[:2] - truth[:2])) * 100)
                e = np.array(errors) if errors else np.array([math.nan])
                print(
                    f"{light.label['en']:14s} {finder:16s} {method:11s} {found:3d}/{len(frames):<3d} "
                    f"{np.mean(e):6.2f} {np.median(e):6.2f} {np.percentile(e, 90):6.2f} {np.max(e):6.2f} "
                    f"{np.mean(across) if across else math.nan:6.2f}"
                )
    renderer.close()


if __name__ == "__main__":
    main()
