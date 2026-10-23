"""Local, click-to-place starters. No file paths, downloads or arbitrary builder input."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

if TYPE_CHECKING:
    from robotforge.playground import Playground

STARTERS = ("rover", "quadruped", "arm", "humanoid")
Coordinate = Annotated[float, Field(ge=-100, le=100, allow_inf_nan=False)]


class Placement(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    template: Literal["rover", "quadruped", "arm", "humanoid"]
    pos: list[Coordinate] = Field(min_length=2, max_length=3)
    version: int = Field(ge=1)


def catalog() -> list[dict[str, str]]:
    from robotforge.design.templates import TEMPLATES

    return [{"name": name, "summary": TEMPLATES[name].summary} for name in STARTERS]


def place(pg: Playground, message: dict) -> dict:
    """Called with the playground edit lock held; preflight never publishes a temporary world."""
    import copy

    import mujoco

    from robotforge.design import terrain
    from robotforge.design.world import RobotEntry, compile_world, spawn_height
    from robotforge.playground import PlaygroundError, _unused
    from robotforge.sim import carry, placement
    from robotforge.sim.session import fresh_data

    try:
        request = Placement.model_validate({k: v for k, v in message.items() if k not in {"type", "action", "id"}})
    except ValidationError as exc:
        raise PlaygroundError(f"invalid robot placement: {exc}") from None
    if pg.lessons.lesson is not None:
        raise PlaygroundError("leave the lesson before adding robots")
    store = pg._require_store()
    with pg.session.execution.command("viewer_add_robot"):
        with pg.session.lock:
            if pg.session.version != request.version:
                raise PlaygroundError("the scene changed; choose a spot in the current scene")
            generation = pg.session.generation
        world = store.load()
        taken = {r.name for r in world.robots} | {o.name for o in world.objects} | {p.name for p in world.terrain}
        name = _unused("dog" if request.template == "quadruped" else request.template, taken)
        candidate = world.model_copy(deep=True)
        entry = RobotEntry(name=name, template=request.template, pos=request.pos)
        # The viewer sends a support surface, while RobotEntry's explicit z denotes the robot root.
        if len(request.pos) == 3:
            ground = terrain.Ground.build(world.terrain, world.floor.style)
            on_ground = abs(request.pos[2] - ground.height(*request.pos[:2])) <= 0.001
            # Ground positions keep following later terrain edits; tables retain an explicit height.
            pos = request.pos[:2] if on_ground else [*request.pos[:2], request.pos[2] + spawn_height(entry)]
            entry = entry.model_copy(update={"pos": pos})
        candidate.robots.append(entry)
        compiled = compile_world(candidate, store.dir)
        saved = fresh_data(compiled.model)
        blocked = set(placement.blocked(compiled.model, saved, name))
        # Lessons take their runner lock before the session lock when swapping models.
        with pg.lessons._lock, pg.session.lock:
            if (
                pg.store is not store
                or pg.lessons.lesson is not None
                or pg.session.version != request.version
                or pg.session.generation != generation
            ):
                raise PlaygroundError("the simulation changed; choose a spot again")
            current = copy.copy(saved)
            live = carry.take(pg.session.model, pg.session.data)
            # Unlike carry.restore, preflight must not move an old robot out of a clash.
            # Read live poses at commit: the viewer can keep stepping while compilation runs.
            keep = {r.name for r in world.robots} | {o.name for o in world.objects}
            carry._apply(compiled.model, current, live, lambda joint: carry.owner(joint) in keep)
            mujoco.mj_forward(compiled.model, current)
            blocked.update(placement.blocked(compiled.model, current, name))
            if blocked:
                raise PlaygroundError(f"that spot is occupied by {', '.join(sorted(blocked))}; choose an open spot")
            result = pg.add_robot(name, template=request.template, pos=entry.pos)
            return {"robot": name, "version": pg.session.version, "notes": result["notes"]}
