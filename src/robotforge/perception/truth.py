"""Scoring a map against the scene it was drawn of: the simulator knows where everything is.

The scene's occupancy is rasterized from the compiled model's collision shapes onto the map's own grid (the
same columns the planner uses, behaviors/nav_map.py): a cell is occupied when a shape stands over it between the
heights given (the band the robot's sensor and body occupy). The robot's own shapes are left out. This is used
only to judge a map, never to build it.
"""

from __future__ import annotations

import math
from typing import Any

import mujoco
import numpy as np

from robotforge.behaviors import nav_map
from robotforge.behaviors.nav_grid import clearance
from robotforge.perception.grid import FREE, OCCUPIED, UNKNOWN, OccupancyGrid
from robotforge.sim.geometry import collides

G = mujoco.mjtGeom
TOLERANCE_CELLS = 2  # a mapped wall within this many cells of a real one is right (10 cm at 5 cm cells)
CAP = 1.0  # m: distances beyond are not told apart


def scene_occupancy(
    model: mujoco.MjModel, data: mujoco.MjData, robot: str, grid: OccupancyGrid, low: float, high: float
) -> np.ndarray:
    """Cells of `grid` where a collision shape (not the robot's own, not the floor) stands in the height band
    from `low` to `high` m."""
    own = set(nav_map.robot_geoms(model, robot))
    out = np.zeros(grid.rows * grid.cols, bool)
    layout = grid.grid
    for g in range(model.ngeom):
        if g in own or not collides(model, g) or int(model.geom_type[g]) in (G.mjGEOM_PLANE, G.mjGEOM_HFIELD):
            continue
        columns = nav_map.geom_columns(model, data, g, layout, 0, pad=0.0)
        if columns is None:
            continue
        over = (columns.high > low) & (columns.low < high)
        out[columns.cells[over]] = True
    return out.reshape(grid.rows, grid.cols)


def score(grid: OccupancyGrid, truth: np.ndarray, body: np.ndarray | None = None) -> dict[str, Any]:
    """How well the map agrees with the scene's occupancy `truth` (a boolean grid like the map's).

    precision: the share of mapped obstacle cells within TOLERANCE_CELLS of a real one;
    recall: the share of real obstacle faces that border mapped free space (surfaces the robot saw from the open
    side) that have a mapped obstacle within the tolerance;
    wall_error_m: mean and 95th percentile distance from a mapped obstacle cell to the nearest real one;
    free_correct: the share of mapped free cells that are free in the scene;
    explored: the share of the scene's free cells (inside the area the robot has seen) that the map has seen;
    unsafe_free: given `body` (everything that stands in the robot's way at any height), the share of those cells
    the map calls free, which is how often a path through mapped free space would meet something."""
    state = grid.state()
    mapped = state == OCCUPIED
    free = state == FREE
    res = grid.res
    to_real = clearance(truth, res, CAP) + np.where(truth, 0.0, res / 2)  # center to center, at most CAP + half a cell
    to_mapped = clearance(mapped, res, CAP) + np.where(mapped, 0.0, res / 2)
    near_real = to_real <= TOLERANCE_CELLS * res + 1e-9
    near_mapped = to_mapped <= TOLERANCE_CELLS * res + 1e-9
    out: dict[str, Any] = {"resolution_m": res, "scans": grid.scans}
    n_map = int(mapped.sum())
    out["occupied_cells"] = n_map
    out["free_cells"] = int(free.sum())
    if n_map:
        distances = to_real[mapped]
        out["precision"] = round(float((mapped & near_real).sum() / n_map), 3)
        out["wall_error_m"] = {
            "mean": round(float(distances.mean()), 3),
            "p95": round(float(np.percentile(distances, 95)), 3),
        }
    beside_free = np.zeros_like(free)
    beside_free[1:, :] |= free[:-1, :]
    beside_free[:-1, :] |= free[1:, :]
    beside_free[:, 1:] |= free[:, :-1]
    beside_free[:, :-1] |= free[:, 1:]
    surfaces = truth & beside_free
    if surfaces.any():
        out["recall"] = round(float((surfaces & near_mapped).sum() / surfaces.sum()), 3)
        out["real_surface_cells"] = int(surfaces.sum())
    if free.any():
        out["free_correct"] = round(float((free & ~truth).sum() / free.sum()), 3)
    extent = grid.extent()
    if extent is not None:
        r0, c0 = grid.cell(extent[0], extent[1])
        r1, c1 = grid.cell(extent[2], extent[3])
        seen_box = np.zeros_like(free)
        seen_box[r0 : r1 + 1, c0 : c1 + 1] = True
        scene_free = seen_box & ~truth
        out["explored"] = round(float(((state != UNKNOWN) & scene_free).sum() / max(int(scene_free.sum()), 1)), 3)
    if body is not None and body.any():
        out["unsafe_free"] = round(float((free & body).sum() / body.sum()), 4)
        out["unsafe_free_cells"] = int((free & body).sum())
    if "precision" in out and "recall" in out:
        p, r = out["precision"], out["recall"]
        out["f1"] = round(2 * p * r / (p + r), 3) if p + r else 0.0
    return out


def navigation_agreement(own: Any, scene: Any, start: tuple[float, float]) -> dict[str, Any]:
    """How the robot's own map and the scene's map (behaviors/nav_map.py, from the compiled model) agree on where the
    robot can go. `own` and `scene` are navigation maps (NavMap) on their own grids; the robot is at `start`.

    through_blocked: the share of the cells the robot would plan through on its own map (those it can reach from
    where it stands) that the scene's map says it cannot stand in (a wall, a rack, a pallet, a step);
    found: the share of the cells the scene's map lets it reach, inside the area its own map covers, that its own map
    lets it reach too (what its map lacks is what it has not seen, or took for an obstacle: a ramp is a block to a
    lidar)."""
    from robotforge.behaviors import nav_grid

    out: dict[str, Any] = {}
    cell = own.grid.cell(*start)
    if not own.grid.inside(*cell):
        return out
    mine = nav_grid.reachable(own.grid, own.free, cell)
    rows, cols = np.nonzero(mine)
    if not len(rows):
        return out
    xs = own.grid.x0 + cols * own.grid.res
    ys = own.grid.y0 + rows * own.grid.res
    sr = np.rint((ys - scene.grid.y0) / scene.grid.res).astype(int)
    sc = np.rint((xs - scene.grid.x0) / scene.grid.res).astype(int)
    inside = (sr >= 0) & (sr < scene.grid.rows) & (sc >= 0) & (sc < scene.grid.cols)
    if inside.any():
        blocked = ~scene.free[sr[inside], sc[inside]]
        out["through_blocked"] = round(float(blocked.mean()), 4)
        out["reachable_cells"] = int(inside.sum())
    # the scene's reachable cells within the box the own map has covered
    covered = np.zeros(scene.grid.shape, bool)
    extent_r0, extent_r1 = int(rows.min()), int(rows.max())
    extent_c0, extent_c1 = int(cols.min()), int(cols.max())
    x0, y0 = own.grid.x0 + extent_c0 * own.grid.res, own.grid.y0 + extent_r0 * own.grid.res
    x1, y1 = own.grid.x0 + extent_c1 * own.grid.res, own.grid.y0 + extent_r1 * own.grid.res
    sxs, sys_ = scene.grid.centers()
    covered = (sxs >= x0) & (sxs <= x1) & (sys_ >= y0) & (sys_ <= y1)
    scene_reach = nav_grid.reachable(scene.grid, scene.free, scene.grid.cell(*start)) & covered
    if scene_reach.any():
        orr = np.rint((sys_[scene_reach] - own.grid.y0) / own.grid.res).astype(int)
        occ = np.rint((sxs[scene_reach] - own.grid.x0) / own.grid.res).astype(int)
        ok = (orr >= 0) & (orr < own.grid.rows) & (occ >= 0) & (occ < own.grid.cols)
        found = np.zeros(len(orr), bool)
        found[ok] = mine[orr[ok], occ[ok]]
        out["found"] = round(float(found.mean()), 3)
    return out


def pose_errors(estimate: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Position error (m) and heading error (deg) of an estimated pose against the true one."""
    return {
        "position_m": round(float(math.hypot(estimate[0] - truth[0], estimate[1] - truth[1])), 3),
        "heading_deg": round(abs(math.degrees((estimate[2] - truth[2] + math.pi) % (2 * math.pi) - math.pi)), 2),
    }
