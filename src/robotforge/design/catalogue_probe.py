"""A short trial of a legged catalogue robot, run when it is added: does it stand, does its gait walk it?

The describe step reads what a robot is from its structure; that says nothing about whether the generic gait the
behaviors find from its legs keeps it up. So the first add of a quadruped or humanoid also runs it alone in a scratch
world for a few seconds of simulated time (headless, well under the time of the import itself): two seconds standing
at home, then six seconds of a slow walk. The result goes into the answer's `catalogue.checked` and shapes what the
description promises. It is kept next to the download, keyed by the robot, its commit and a hash of the code it
depends on, so the same robot is only tried once until that code changes.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import mujoco

from robotforge.design import catalogue, catalogue_fetch, menagerie

log = logging.getLogger(__name__)

SETTLE_S = 2.0  # at home before anything is asked of the robot
WALK_S = 6.0
WALK_SPEED = 0.2  # m/s asked, or the gait's top speed if lower
STEP_S = 0.25  # how often the walk is looked at
STANDS_KEPT = 0.7  # share of its home height a robot that stands has kept after the settle
FELL_SAMPLES = 2  # looks in a row at a robot on its side that make it a fall (one is a stumble)
GOT_ANYWHERE = 0.3  # share of the distance asked that a walker has to cover to have walked
LEGGED = ("quadruped", "humanoid")
WATCHED = ("behaviors", "design/importing")  # folders of code whose change makes an old trial stale
WATCHED_FILES = ("design/mjcf_doctor.py", "design/mjcf_stance.py", "design/catalogue_inspect.py")

_probing: contextvars.ContextVar[bool] = contextvars.ContextVar("catalogue_probing", default=False)


def running() -> bool:
    """Whether the code is inside a trial (its scratch world adds the robot itself and must not try it again)."""
    return _probing.get()


def code_hash() -> str:
    """A short hash of the code a trial's outcome depends on: behaviors, the import doctor and the MJCF repairs."""
    root = Path(__file__).resolve().parents[1]
    files = [p for folder in WATCHED for p in sorted((root / folder).glob("**/*.py"))]
    files += [root / name for name in WATCHED_FILES]
    digest = hashlib.sha1()
    for path in files:
        if path.is_file():
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def _cache_file(entry: catalogue.Entry) -> Path:
    return menagerie.cache_root() / "catalogue" / "trials" / f"{entry.id}-{entry.commit[:12]}-{code_hash()}.json"


def cached(entry: catalogue.Entry) -> dict[str, Any] | None:
    try:
        return json.loads(_cache_file(entry).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def trial(entry: catalogue.Entry, force: bool = False) -> dict[str, Any] | None:
    """The outcome of the trial of `entry` (see `run`), from the cache when it was tried before. None when the
    trial could not be made (it is a courtesy, never a reason for add_robot to fail)."""
    if running():
        return None
    if not force and (found := cached(entry)) is not None:
        return found
    token = _probing.set(True)
    try:
        result = run(entry.id)
    except Exception as exc:  # a trial that breaks says so in the log and the add goes on without it
        log.warning("catalogue: could not try %s: %s", entry.id, exc)
        return None
    finally:
        _probing.reset(token)
    try:
        path = _cache_file(entry)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result), encoding="utf-8")
    except OSError as exc:
        log.debug("catalogue: trial of %s not kept: %s", entry.id, exc)
    return result


def run(catalogue_id: str) -> dict[str, Any]:
    """Add the robot alone to a scratch world and watch it: {stands, height_kept, upright, walk, ...}.

    walk is "walked" (kept its feet for the whole walk), "fell" (tipped over; fell_after_m says how far it got),
    "stalled" (upright but got nowhere), "no gait" (the behaviors have no gait that fits its legs) or "not tried" (it
    did not stand)."""
    from robotforge.design.store import WorldStore
    from robotforge.playground import Playground

    with tempfile.TemporaryDirectory(prefix="rf-trial-") as scratch:
        pg = Playground.with_demo()
        pg.store = WorldStore.create("trial", parent=Path(scratch))
        pg._load_world(pg.store.compile(), created=True)
        pg.add_robot("r", library=catalogue_id)
        return watch(pg, "r")


def watch(pg: Any, robot: str) -> dict[str, Any]:
    """Settle `robot` at home, then walk it slowly; what happened. `pg` is a Playground with the robot in it."""
    out: dict[str, Any] = {"settle_s": SETTLE_S}
    start = _span(pg, robot)
    ran = pg.run_fast(lambda: False, SETTLE_S)
    kept = min(1.0, _span(pg, robot) / start) if start > 0.02 else 1.0  # taller than at the start is standing
    stands = bool(_upright(pg, robot) and kept >= STANDS_KEPT and _finite(pg) and ran >= SETTLE_S)
    out.update(stands=stands, height_kept=round(kept, 2))
    if not stands:
        out["walk"] = "not tried"
        return out
    info = next((r for r in pg.behaviors.robots() if r["name"] == robot), {})
    top = info.get("max_speed")
    if not top:
        out["walk"] = "no gait"
        return out
    speed = min(WALK_SPEED, float(top))
    before = pg.behaviors.pose(robot)["position"]
    pg.behaviors.move(robot, speed, 0.0)
    out["walk_speed"] = round(speed, 2)
    elapsed, went, down = 0.0, 0.0, 0
    while elapsed < WALK_S:
        ran = pg.run_fast(lambda: False, STEP_S)
        elapsed += STEP_S
        pose = pg.behaviors.pose(robot)
        went = float(((pose["position"][0] - before[0]) ** 2 + (pose["position"][1] - before[1]) ** 2) ** 0.5)
        blown = ran < STEP_S - 1e-6  # the simulation blew up and started over
        down = down + 1 if blown or not pose["upright"] or not _finite(pg) else 0
        if blown:
            down = FELL_SAMPLES
        if down >= FELL_SAMPLES:
            out.update(walk="fell", fell_after_m=round(went, 2), fell_after_s=round(elapsed - STEP_S * (down - 1), 1))
            break
    else:
        if down or went < GOT_ANYWHERE * speed * WALK_S:
            out.update(walk="fell" if down else "stalled", walked_m=round(went, 2), walk_s=WALK_S)
            if down:
                out.update(fell_after_m=round(went, 2), fell_after_s=WALK_S)
        else:
            out.update(walk="walked", walked_m=round(went, 2), walk_s=WALK_S)
    pg.behaviors.stop(robot)
    return out


def _span(pg: Any, robot: str) -> float:
    """How tall the robot's bodies stand: lying on its back is a fraction of that."""
    m, d = pg.session.model, pg.session.data
    prefix = f"{robot}/"
    zs = [
        float(d.xpos[b][2])
        for b in range(1, m.nbody)
        if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith(prefix)
    ]
    return max(zs) - min(zs)


def _upright(pg: Any, robot: str) -> bool:
    return bool(pg.behaviors.pose(robot)["upright"])


def _finite(pg: Any) -> bool:
    import numpy as np

    return bool(np.isfinite(pg.session.data.qpos).all())


def describe(checked: dict[str, Any]) -> str:
    """The result of a trial in a sentence for the agent."""
    if not checked["stands"]:
        return f"collapsed within {checked['settle_s']:.0f} s at home (kept {checked['height_kept']:.0%} of its height)"
    walk = checked["walk"]
    if walk == "walked":
        return f"stood, and walked {checked['walked_m']} m upright at {checked['walk_speed']} m/s"
    if walk == "fell":
        return f"stood, but its gait fell over after {checked['fell_after_m']} m at {checked['walk_speed']} m/s"
    if walk == "stalled":
        return (
            f"stood, but its gait got it only {checked['walked_m']} m in {checked['walk_s']:.0f} s "
            f"at {checked['walk_speed']} m/s"
        )
    return "stood; no gait fits it"


__all__ = ["LEGGED", "catalogue_fetch", "describe", "run", "trial", "watch"]
