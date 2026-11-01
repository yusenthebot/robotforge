"""Server-owned row offsets and independent saved/live source qualification."""

from __future__ import annotations

import copy
import math
from typing import TYPE_CHECKING

import mujoco
import numpy as np

from robotforge.playground import PlaygroundError
from robotforge.sim.session import fresh_data
from robotforge.viewer.object_edit import _bounds, _owners

if TYPE_CHECKING:
    from robotforge.design.world import ObjectEntry, World
    from robotforge.playground import Playground
    from robotforge.viewer.object_group import Member


def qualify_sources(pg: Playground, world: World, names: list[str]) -> None:
    """Require every foot to share one external horizontal owner, in both poses."""
    from robotforge.viewer.object_group import _collisions, _feet, _members, _on_top, _pair_enabled, _solid

    model, live = copy.copy(pg.session.model), copy.copy(pg.session.data)
    mujoco.mj_forward(model, live)
    owners, solid, selected = _owners(model, world), _solid(model), set(names)
    common: set[str] | None = None
    for data in (fresh_data(model), live):
        _members(model, data, world, names)
        _collisions(model, data, world, selected)
        for name in names:
            for foot in _feet(model, data, name, owners, solid):
                supports = {
                    owners[geom]
                    for geom in solid
                    if owners[geom] != name and _pair_enabled(model, foot.geom, geom)
                    if _on_top(model, data, foot, geom) is not None
                }
                if supports & selected:
                    raise PlaygroundError("row arrangement needs independent props; unstack the selected props first")
                common = supports if common is None else common & supports
                if not common:
                    raise PlaygroundError(
                        "row arrangement needs one shared external horizontal support in saved and live poses"
                    )


def _x_extent(model: mujoco.MjModel, data: mujoco.MjData, geom: int) -> tuple[float, float]:
    kind, size = int(model.geom_type[geom]), model.geom_size[geom]
    center = float(data.geom_xpos[geom, 0])
    if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        half = float(size[0])
    elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        half = float(size[0] + size[1] * abs(data.geom_xmat[geom].reshape(3, 3)[0, 2]))
    elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        axis = data.geom_xmat[geom].reshape(3, 3)[0]
        half = float(size[0] * math.hypot(axis[0], axis[1]) + size[1] * abs(axis[2]))
    else:
        low, high = _bounds(model, data, geom)
        return float(low[0]), float(high[0])
    return center - half, center + half


def row_offsets(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    members: tuple[Member, ...],
    anchor: tuple[float, float, float],
    gap: float,
) -> tuple[tuple[float, float, float], ...]:
    """Center complete solid X extents in selection order, retaining live headings."""
    extents = []
    for member in members:
        boxes = [
            _x_extent(model, data, geom)
            for geom in member.geoms
            if model.geom_contype[geom] or model.geom_conaffinity[geom]
        ]
        low = min(box[0] for box in boxes) - member.pos[0]
        high = max(box[1] for box in boxes) - member.pos[0]
        extents.append((low, high))
    total = sum(high - low for low, high in extents) + (len(members) - 1) * gap
    cursor = -total / 2
    offsets = []
    for member, (low, high) in zip(members, extents, strict=True):
        offsets.append((cursor - low, 0.0, member.pos[2] - anchor[2]))
        cursor += high - low + gap
    return tuple(offsets)


def unchanged_row(pg: Playground, world: World, targets: dict[str, ObjectEntry]) -> bool:
    """No edit only when targets match authored and complete current geometry poses."""
    from robotforge.viewer.object_group import _members

    model, live = copy.copy(pg.session.model), copy.copy(pg.session.data)
    mujoco.mj_forward(model, live)
    saved = fresh_data(model)
    for member in _members(model, saved, world, list(targets)):
        target = targets[member.entry.name]
        if not np.allclose(target.pos, member.pos, rtol=0, atol=1e-9):
            return False
        if abs(math.remainder(target.yaw - member.yaw, 360)) > 1e-9:
            return False
        for geom in member.geoms:
            if not np.allclose(saved.geom_xpos[geom], live.geom_xpos[geom], rtol=0, atol=1e-9):
                return False
            if not np.allclose(saved.geom_xmat[geom], live.geom_xmat[geom], rtol=0, atol=1e-9):
                return False
    return True
