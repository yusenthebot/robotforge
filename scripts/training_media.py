"""The training lane's proof: train a robot built from parts to walk and an arm to reach, then film the training view.

    uv run python scripts/training_media.py train --home DIR [--only walk reach_default reach_search walk_sensors ...]
    uv run python scripts/training_media.py rescore --home DIR
    uv run python scripts/training_media.py media --home DIR --out DIR [--only view_shot view_film walk_film]

`train` builds the two worlds in DIR (a quadruped generated from a sentence, an arm with a ball to reach), trains
them with real simulations (two algorithms each, and a search for the arm), and writes DIR/numbers.json: what the
robot does with no learning, and what each algorithm and each trial reached, scored on held-out starts (`rescore`
redoes that from the runs on disk); the `*_sensors` runs learn from the goal and the robot's sensors only.
`media` serves those worlds with the trained policies in them, records the viewer's Training panel (a search
running, a policy playing) and takes the screenshot. Maintainer tool, not part of robotforge; needs Google Chrome
and ffmpeg for `media`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

WALK_SPEED = 0.4
REACH_TARGET = [0.22, 0.08, 0.14]
REACH_TASK = {"target": REACH_TARGET, "jitter": 0.04, "tolerance": 0.03}
SEARCH_SPACE = {
    "noise": {"low": 0.01, "high": 0.3, "log": True},
    "step_size": {"low": 0.01, "high": 0.3, "log": True},
    "init_std": {"low": 0.02, "high": 0.5, "log": True},
}


def build_quad_world(home: Path) -> Any:
    from robotforge.playground import Playground

    os.environ["ROBOTFORGE_HOME"] = str(home)
    pg = Playground.with_demo()
    if (home / "worlds" / "quad_lab").is_dir():  # built by an earlier run: its policies live in it
        pg.open_world("quad_lab")
        return pg
    pg.new_world("quad_lab")
    pg.add_robot("dog", describe="a robot dog")
    pg.add_sensor("dog", {"type": "encoders"})
    return pg


def build_arm_world(home: Path) -> Any:
    from robotforge.playground import Playground

    os.environ["ROBOTFORGE_HOME"] = str(home)
    pg = Playground.with_demo()
    if (home / "worlds" / "arm_lab").is_dir():
        pg.open_world("arm_lab")
        return pg
    pg.new_world("arm_lab")
    pg.add_robot("arm", template="arm")
    pg.add_sensor("arm", {"type": "encoders"})
    pg.add_object("target", shape="sphere", size=[0.03], pos=REACH_TARGET, color="red", fixed=True)
    return pg


def _baseline(env_args: dict[str, Any], episodes: int = 50) -> dict[str, Any]:
    """Doing nothing, scored on the same held-out starts the trained winners are scored on."""
    from robotforge.training.envs import build_env, env_spec
    from robotforge.training.trial import HELD_OUT_SEED, evaluate_policy

    env = build_env(env_spec(**env_args))
    zero = [0.0] * env.action_space.shape[0]
    return evaluate_policy(env, lambda obs: zero, episodes, seed=HELD_OUT_SEED)


def _held_out(folder: Path, trial: str, episodes: int = 50) -> dict[str, Any] | None:
    """One trial's last policy on the held-out starts (the run itself scores only its winner there)."""
    from robotforge.control.rl import store
    from robotforge.control.rl.runtime import MlpPolicy
    from robotforge.training.envs import build_env
    from robotforge.training.trial import HELD_OUT_SEED, evaluate_policy

    path = folder / "trials" / trial / "policy.npz"
    if not path.is_file():
        return None
    env = build_env(store.read_json(folder / "policy.json")["env"])
    try:
        got = evaluate_policy(env, MlpPolicy.load(path), episodes, seed=HELD_OUT_SEED)
    finally:
        env.close()
    return {k: got[k] for k in ("episodes", "mean_return", "success_rate", "mean_result")}


def _summary(folder: Path) -> dict[str, Any]:
    from robotforge.control.rl import store

    status = store.read_json(folder / "status.json")
    meta = store.read_json(folder / "policy.json")
    return {
        "state": status.get("state"),
        "elapsed_s": status.get("elapsed_s"),
        "final": meta.get("final"),
        "observation_numbers": sum(len(s["labels"]) for s in meta.get("observation_layout", [])),
        "observation_sources": {s["source"]: len(s["labels"]) for s in meta.get("observation_layout", [])},
        "trials": [
            {
                "id": t["id"],
                "algo": t["algo"],
                "settings": t["settings"],
                "state": t["state"],
                "steps": t["steps"],
                "score": t.get("score"),
                "held_out": _held_out(folder, t["id"]),
                "curve": t.get("curve", [])[:: max(1, len(t.get("curve", [])) // 12)],
            }
            for t in status.get("trials", [])
        ],
    }


def train(home: Path, only: list[str] | None, walk_steps: int, reach_steps: int, parallel: int) -> None:
    from robotforge.control.rl import store
    from robotforge.training import api

    numbers_path = home / "numbers.json"
    numbers: dict[str, Any] = json.loads(numbers_path.read_text(encoding="utf-8")) if numbers_path.is_file() else {}

    def wanted(name: str) -> bool:
        return not only or name in only

    if wanted("walk"):
        pg = build_quad_world(home)
        world, world_dir = pg.current_world(), pg.store.dir
        args = dict(
            task="walk", world=world, robot="dog", world_dir=world_dir, task_params={"speed": WALK_SPEED},
            action_mode="residual", residual_scale=0.2, episode_seconds=4.0, push=0.5,
        )  # fmt: skip
        base = _baseline(args)
        t0 = time.monotonic()
        api.train(
            algo=["ars", "cem"], steps=walk_steps, parallel=parallel, name="dog_walk", replace=True,
            save_dir=store.policies_home(world_dir), eval_episodes=10, **args,
        )  # fmt: skip
        numbers["walk"] = {
            "task": f"walk at {WALK_SPEED} m/s, residual on the built-in gait, 4 s episodes, shoves 0.5",
            "gait_alone": base,
            "wall_s": round(time.monotonic() - t0),
            **_summary(store.policies_home(world_dir) / "dog_walk"),
        }
        numbers_path.write_text(json.dumps(numbers, indent=1), encoding="utf-8")
        print("walk", json.dumps(numbers["walk"]["final"]))
    if wanted("reach_default"):
        pg = build_arm_world(home)
        world, world_dir = pg.current_world(), pg.store.dir
        args = dict(
            task="reach", world=world, robot="arm", world_dir=world_dir, task_params=REACH_TASK,
            action_mode="direct", episode_seconds=2.0,
        )  # fmt: skip
        base = _baseline(args)
        t0 = time.monotonic()
        api.train(
            algo=["ars", "cem"], steps=reach_steps, parallel=parallel, name="arm_reach_default", replace=True,
            save_dir=store.policies_home(world_dir), eval_episodes=20, **args,
        )  # fmt: skip
        numbers["reach_default"] = {
            "task": "reach a red ball (22, 8, 14 cm) that moves up to 4 cm each episode; success within 3 cm",
            "do_nothing": base,
            "wall_s": round(time.monotonic() - t0),
            **_summary(store.policies_home(world_dir) / "arm_reach_default"),
        }
        numbers_path.write_text(json.dumps(numbers, indent=1), encoding="utf-8")
        print("reach_default", json.dumps(numbers["reach_default"]["final"]))
    if wanted("reach_search"):
        pg = build_arm_world(home)
        world, world_dir = pg.current_world(), pg.store.dir
        args = dict(
            task="reach", world=world, robot="arm", world_dir=world_dir, task_params=REACH_TASK,
            action_mode="direct", episode_seconds=2.0,
        )  # fmt: skip
        t0 = time.monotonic()
        api.train(
            algo=["ars", "cem"], steps=reach_steps, parallel=parallel, name="arm_reach", replace=True,
            save_dir=store.policies_home(world_dir), eval_episodes=20,
            search={"method": "halving", "trials": 9, "eta": 3, "seed": 1, "space": SEARCH_SPACE}, **args,
        )  # fmt: skip
        numbers["reach_search"] = {
            "search": "successive halving, 9 trials over ars and cem, eta 3",
            "wall_s": round(time.monotonic() - t0),
            **_summary(store.policies_home(world_dir) / "arm_reach"),
        }
        numbers_path.write_text(json.dumps(numbers, indent=1), encoding="utf-8")
        print("reach_search", json.dumps(numbers["reach_search"]["final"]))
    walk_extra = dict(task_params={"speed": WALK_SPEED}, action_mode="residual", residual_scale=0.2,
                      episode_seconds=4.0, push=0.5)  # fmt: skip
    reach_extra = dict(task_params=REACH_TASK, action_mode="direct", episode_seconds=2.0)
    sensor_runs = (
        ("walk_sensors", "walk", build_quad_world, "dog", walk_extra, walk_steps, 10),
        ("reach_sensors", "reach", build_arm_world, "arm", reach_extra, reach_steps, 20),
    )
    for name, task, world_fn, robot, args_extra, steps, eval_eps in sensor_runs:
        if not wanted(name):
            continue
        pg = world_fn(home)
        world, world_dir = pg.current_world(), pg.store.dir
        args = dict(task=task, world=world, robot=robot, world_dir=world_dir, observation="sensors", **args_extra)
        t0 = time.monotonic()
        api.train(
            algo=["ars", "cem"], steps=steps, parallel=parallel, name=f"{robot}_{task}_sensors", replace=True,
            save_dir=store.policies_home(world_dir), eval_episodes=eval_eps, **args,
        )  # fmt: skip
        numbers[name] = {
            "task": f"{task} with observation='sensors': the goal and what the robot's own sensors read, nothing else",
            "wall_s": round(time.monotonic() - t0),
            **_summary(store.policies_home(world_dir) / f"{robot}_{task}_sensors"),
        }
        numbers_path.write_text(json.dumps(numbers, indent=1), encoding="utf-8")
        print(name, json.dumps(numbers[name]["final"]))


RUNS = {
    "walk": ("quad_lab", "dog_walk"),
    "walk_sensors": ("quad_lab", "dog_walk_sensors"),
    "reach_default": ("arm_lab", "arm_reach_default"),
    "reach_search": ("arm_lab", "arm_reach"),
    "reach_sensors": ("arm_lab", "arm_reach_sensors"),
}


def rescore(home: Path) -> None:
    """Refresh numbers.json from the runs on disk: every trial's last policy on the held-out starts."""
    numbers_path = home / "numbers.json"
    numbers = json.loads(numbers_path.read_text(encoding="utf-8"))
    for key, (world, run) in RUNS.items():
        folder = home / "worlds" / world / "policies" / run
        if key in numbers and folder.is_dir():
            numbers[key].update(_summary(folder))
            print(key, "rescored", flush=True)
    numbers_path.write_text(json.dumps(numbers, indent=1), encoding="utf-8")


def media(home: Path, out: Path, only: list[str] | None) -> None:
    import demos as D
    import training_films

    out.mkdir(parents=True, exist_ok=True)
    consts = {"reach_task": REACH_TASK, "reach_steps": 200_000, "space": SEARCH_SPACE, "walk_speed": WALK_SPEED}
    for demo in training_films.demos(home, consts):
        if only and demo.name not in only:
            continue
        t0 = time.monotonic()
        path = D.take(demo, out)
        print(f"{demo.name}: {path} ({path.stat().st_size / 1e6:.1f} MB, {time.monotonic() - t0:.0f} s)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    m = sub.add_parser("media")
    m.add_argument("--home", type=Path, required=True)
    m.add_argument("--out", type=Path, required=True)
    m.add_argument("--only", nargs="*")
    r = sub.add_parser("rescore")
    r.add_argument("--home", type=Path, required=True)
    t = sub.add_parser("train")
    t.add_argument("--home", type=Path, required=True)
    t.add_argument("--only", nargs="*")
    t.add_argument("--walk-steps", type=int, default=250_000)
    t.add_argument("--reach-steps", type=int, default=200_000)
    t.add_argument("--parallel", type=int, default=3)
    args = parser.parse_args()
    args.home.mkdir(parents=True, exist_ok=True)
    if args.command == "media":
        media(args.home.resolve(), args.out.resolve(), args.only)
    if args.command == "rescore":
        rescore(args.home.resolve())
    if args.command == "train":
        train(args.home.resolve(), args.only, args.walk_steps, args.reach_steps, args.parallel)


if __name__ == "__main__":
    main()
