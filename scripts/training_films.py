"""The films of the training view (see training_media.py): a search running with the best policy playing, and the dog.

uv run python scripts/training_media.py media --home DIR --out DIR [--only training_search quad_walk]
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import demos as D

from robotforge.control.rl import store
from robotforge.control.rl.jobs import JOBS
from robotforge.training import api
from robotforge.training.live import TrainingFeed
from robotforge.training.runs import start_run


def copy_world(src: Path, name: str, keep_policies: bool) -> Path:
    """A copy of a world in the film's own home, so a film never touches the policies that were trained."""
    dst = Path(os.environ["ROBOTFORGE_HOME"]) / "worlds" / name
    shutil.copytree(src, dst, ignore=None if keep_policies else shutil.ignore_patterns("policies"))
    return dst


def click(selector: str) -> str:
    return f"document.querySelector({json.dumps(selector)}).click();"


def look(eye: list[float], target: list[float]) -> str:
    return f"window.__robotforge.select(null); window.__robotforge.lookAt({eye}, {target});"


def demos(home: Path, consts: dict[str, Any]) -> list[D.Demo]:
    numbers = json.loads((home / "numbers.json").read_text(encoding="utf-8"))
    state: dict[str, float] = {}

    def arm_world(pg: Any) -> None:
        pg.open_world(str(copy_world(home / "worlds" / "arm_lab", "arm_film", keep_policies=False)))
        if "target" not in {o.name for o in pg.current_world().objects}:
            pg.add_object("target", shape="sphere", size=[0.05], pos=consts["reach_task"]["target"], fixed=True)
        else:
            pg.update_object("target", size=[0.05])  # big enough to see; the training does not need it

    def search_and_wait(pg: Any) -> None:
        world, world_dir = pg.current_world(), pg.store.dir
        spec = api.build_spec(
            "reach",
            name="arm_reach",
            world=world,
            robot="arm",
            world_dir=world_dir,
            task_params=consts["reach_task"],
            action_mode="direct",
            episode_seconds=2.0,
            algo=["ars", "cem"],
            steps=consts["reach_steps"],
            parallel=3,
            eval_episodes=20,
            search={"method": "halving", "trials": 9, "eta": 3, "seed": 1, "space": consts["space"]},
        )
        folder = store.policies_home(world_dir) / "arm_reach"
        start_run(spec, folder, replace=True)
        feed, last = TrainingFeed(pg), 0.0
        while JOBS.status(folder).get("state") not in store.FINAL_STATES:
            now = time.monotonic()
            if now - last > 9.0:  # the arm shows what the leading trial can do so far, as the search goes on
                try:
                    feed.play({"run": "arm_reach"})
                    last = now
                except Exception:  # no trial has a policy yet
                    pass
            time.sleep(0.5)
        time.sleep(2.0)  # the panel shows the finished run for a moment before the policy plays
        state["played"] = time.monotonic()

    def arm_done(_pg: Any) -> bool:
        return "played" in state and time.monotonic() - state["played"] > 8.0

    arm = D.Demo(
        "training_search",
        arm_world,
        seconds=300.0,
        page=look([0.7, -0.62, 0.4], [0.14, 0.04, 0.18]),
        settle=2.0,
        timeline=[
            (0.5, D._caption("An arm learns to reach the red ball: nine trials of two algorithms, halved as they go")),
            (1.0, search_and_wait),
            (1.5, D._caption("The best trial plays in the world")),
            (1.6, click('#training-view [data-act="play"].primary-button')),
        ],
        until=arm_done,
        tail=1.0,
    )

    walk = numbers["walk"]
    gait = walk["gait_alone"]["mean_result"]["avg_speed_m_s"]
    scored = walk["final"].get("held_out") or walk["final"]  # the held-out episodes are the fair ones to quote
    learned = scored["mean_result"]["avg_speed_m_s"]
    marks: dict[str, float] = {}

    def quad_world(pg: Any) -> None:
        pg.open_world(str(copy_world(home / "worlds" / "quad_lab", "quad_film", keep_policies=True)))

    def gait_walk(pg: Any) -> None:
        pg.session.reset()
        pg.behaviors.move("dog", consts["walk_speed"], 0.0)
        time.sleep(10.0)
        pg.behaviors.stop("dog")

    def policy_walk(pg: Any) -> None:
        pg.session.reset()
        TrainingFeed(pg).play({"run": "dog_walk"})
        marks["played"] = time.monotonic()

    def quad_done(_pg: Any) -> bool:
        return "played" in marks and time.monotonic() - marks["played"] > 11.5

    ask = consts["walk_speed"]
    algo = walk["final"]["algo"].upper()
    quad = D.Demo(
        "quad_walk",
        quad_world,
        seconds=90.0,
        page=D._follow("dog", (1.5, -1.5, 0.8)),
        settle=2.0,
        timeline=[
            (0.3, D._caption(f"The built-in gait, asked for {ask} m/s, manages {gait:.2f} m/s")),
            (0.5, gait_walk),
            (
                1.0,
                D._caption(
                    f"A policy trained by {algo} on the task's numbers plus the dog's sensors: {learned:.2f} m/s"
                ),
            ),
            (1.2, policy_walk),
        ],
        until=quad_done,
        tail=1.0,
    )
    return [arm, quad]
