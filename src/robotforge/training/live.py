"""What the viewer's Training panel shows and does.

`TrainingFeed.message()` is the viewer stream's "training" message (viewer/server.py): the status of the run that
changed last, sent when it changes. It is the run's status.json (training/runs.py) with a "playable" flag on the run
and on each trial. The panel's buttons come back as "training_play" (run the best or a chosen trial's policy on its
robot in the live world) and "training_stop".
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from robotforge.control.rl import store
from robotforge.control.rl.jobs import JOBS
from robotforge.control.rl.runtime import TrainedPolicy
from robotforge.control.tasks import TaskError
from robotforge.training import play
from robotforge.training.runs import RUN_FILE, TRIALS_DIR
from robotforge.training.trial import POLICY_FILE

SHOW_FOR_S = 12 * 3600  # a finished run older than this is no longer offered
TARGET_NAME = "target"  # an object of this name is where a reach policy reaches to when it plays


class TrainingFeed:
    def __init__(self, playground: Any) -> None:
        self.pg = playground
        self._sent: tuple[str, float] | None = None

    def _homes(self) -> list[Path]:
        world = self.pg.store.dir if self.pg.store is not None else None
        homes = [store.policies_home(world)] if world is not None else []
        homes.append(store.policies_home(None))
        return homes

    def latest(self) -> Path | None:
        """The run whose status changed last (a run of this training package, not older than SHOW_FOR_S)."""
        best: tuple[float, Path] | None = None
        for home in self._homes():
            if not home.is_dir():
                continue
            for folder in home.iterdir():
                try:
                    if not (folder / RUN_FILE).exists():
                        continue
                    changed = (folder / store.STATUS_FILE).stat().st_mtime
                except OSError:
                    continue
                if best is None or changed > best[0]:
                    best = (changed, folder)
        if best is None or time.time() - best[0] > SHOW_FOR_S:
            return None
        return best[1]

    def message(self, force: bool = False) -> dict[str, Any] | None:
        folder = self.latest()
        if folder is None:
            return None
        try:
            status = JOBS.status(folder)
        except TaskError:
            return None
        key = (folder.name, float(status.get("updated", 0) or 0))
        if not force and key == self._sent:
            return None
        self._sent = key
        trials = []
        for trial in status.get("trials", []):
            ready = (folder / TRIALS_DIR / trial["id"] / POLICY_FILE).exists()
            trials.append({**trial, "playable": ready})
        keep = ("state", "name", "task", "robot", "algo", "progress", "elapsed_s", "eta_s", "leader", "search", "error")
        return {
            "type": "training",
            **{k: status[k] for k in keep if k in status},
            "trials": trials,
            "playable": (folder / store.WEIGHTS_FILE).exists(),
        }

    # ---- the panel's buttons ----

    def stop(self, msg: dict[str, Any]) -> dict[str, Any]:
        folder = store.find_policy(str(msg["run"]), self._world_dir())
        status = JOBS.stop(folder)
        return {"state": status.get("state")}

    def play(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Run a policy of the run on its robot in the live world: the best, or the trial `msg["trial"]`."""
        pg = self.pg
        folder = store.find_policy(str(msg["run"]), self._world_dir())
        policy, trial = _policy_to_play(folder, msg.get("trial"))
        spec = play.spec_of(policy.meta)
        if spec is None:
            raise TaskError(
                f"{folder.name!r} was not trained by this pipeline; use run_controller with method='policy:...'"
            )
        session = pg.session
        if spec.robot not in {r["name"] for r in pg.behaviors.robots()}:
            raise TaskError(
                f"the robot {spec.robot!r} is not in this world; open the world it was trained in "
                f"(it is the robot {spec.robot!r} of the world named in {folder.name}'s run)"
            )
        session.reset()
        params = dict(msg.get("task_params") or {})
        with session.lock:
            names = _object_names(session.model)
        if (
            spec.task == "reach"
            and "target" not in params
            and "target" not in spec.task_params
            and TARGET_NAME in names
        ):
            params["target"] = TARGET_NAME
        world = pg.current_world()
        play.attach(session, spec.robot, policy, world=world, task_params=params)
        session.paused = False
        return {"trial": trial, "robot": spec.robot}

    def _world_dir(self) -> Path | None:
        return self.pg.store.dir if self.pg.store is not None else None


def _policy_to_play(folder: Path, trial: str | None) -> tuple[TrainedPolicy, str | None]:
    """The policy to play and the trial it came from: the named trial's, else the run's best, else (a run still
    going) the trial that leads now."""
    from robotforge.control.rl.runtime import MlpPolicy

    meta = store.read_json(folder / store.POLICY_FILE)
    if trial is None and (folder / store.WEIGHTS_FILE).exists():
        return TrainedPolicy.load(folder), (meta.get("final") or {}).get("trial")
    trial = trial or store.read_json(folder / store.STATUS_FILE).get("leader")
    path = folder / TRIALS_DIR / str(trial) / POLICY_FILE
    if not trial or not path.exists():
        raise TaskError(f"{folder.name!r} has no policy yet" if not trial else f"trial {trial!r} has no policy yet")
    return TrainedPolicy(folder, meta, MlpPolicy.load(path)), str(trial)


def _object_names(model: Any) -> set[str]:
    import mujoco

    names = set()
    for obj, count in ((mujoco.mjtObj.mjOBJ_BODY, model.nbody), (mujoco.mjtObj.mjOBJ_GEOM, model.ngeom)):
        names |= {mujoco.mj_id2name(model, obj, i) or "" for i in range(count)}
    return names


def install(playground: Any, hub: Any) -> TrainingFeed:
    """Give the viewer the training stream and its buttons."""
    feed = TrainingFeed(playground)
    hub.training_stream = feed.message
    hub.slow_handlers["training_play"] = feed.play
    hub.slow_handlers["training_stop"] = feed.stop
    return feed
