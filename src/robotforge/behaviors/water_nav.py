"""Where a boat can go: water deep enough under its keel, and nothing at the surface in its way.

The same planner and path follower as on land (navigate.py), with another map. A cell is no place for the boat's
middle when the bottom there (the floor, a hill, a quay, a rock) comes up to within its draft plus a margin of the
surface (the shore, a sand bank, a quay wall), or when anything stands in the band from that depth up to the top of
the boat (a dock's posts and deck, a buoy, a floating crate, another boat); a bridge higher than the boat lets it
through. Outside the water's area it is dry land. The room the boat needs around its middle comes from its own
shape, as for a robot on the floor.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from robotforge.behaviors import Kind, nav_map
from robotforge.behaviors import navigate as navigation
from robotforge.behaviors.marine import BoatPilot
from robotforge.behaviors.nav_map import OBSTACLE, Body, NavMap, StaticLayer
from robotforge.sim import water as water_model
from robotforge.sim.geometry import collides, lowest_point

UNDER_KEEL = 0.05  # m of water kept under the boat's lowest point
OVERHEAD = 0.1  # m kept between the top of the boat and a bridge it passes under
SHORE = "the shore"


class WaterPlanner(navigation.Planner):
    """Plans a boat's way across the water (see the module notes)."""

    def __init__(
        self,
        model: mujoco.MjModel,
        robot: str,
        body: Body,
        destination: navigation.Destination,
        water: water_model.Water,
        draft: float,
        freeboard: float,
        layers: dict | None = None,
    ) -> None:
        super().__init__(model, robot, body, destination, None, layers)
        self.water, self.draft, self.freeboard = water, draft, freeboard
        self._bottom: np.ndarray | None = None  # height of the bottom and what stands on it, per cell of the map

    @property
    def keel(self) -> float:
        """Height above the floor below which nothing is in the boat's way."""
        return self.water.level - self.draft - UNDER_KEEL

    def _map(self, layer: StaticLayer, moving: list, blocked: np.ndarray | None, names: list[str]) -> NavMap:
        grid = layer.grid
        names = [*names, SHORE]
        shore = len(names) - 1
        keel, top = self.keel, self.water.level + self.freeboard + OVERHEAD
        lethal = np.zeros(grid.rows * grid.cols, np.int8)
        owner = np.full(grid.rows * grid.cols, -1, np.int32)
        shallow = layer.floor > keel
        if self.water.area is not None:
            xs, ys = grid.centers()
            x0, y0, x1, y1 = self.water.area
            shallow |= ~((xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1)).ravel()
        lethal[shallow] = OBSTACLE
        owner[shallow] = np.where(layer.floor_owner[shallow] >= 0, layer.floor_owner[shallow], shore)
        bottom = layer.floor.copy()  # the bottom with what stands on it (a quay, an island), for _why_not
        for col in [*layer.columns, *moving]:
            hit = (col.low < top) & (col.high > keel)
            lethal[col.cells[hit]] = OBSTACLE
            owner[col.cells[hit]] = col.owner
            rooted = col.low <= layer.floor[col.cells] + 0.05
            np.maximum.at(bottom, col.cells[rooted], col.high[rooted])
        self._bottom = bottom.reshape(grid.shape)
        lethal2 = lethal.reshape(grid.shape)
        if blocked is not None:
            lethal2[blocked] = OBSTACLE
        flat = np.full(grid.shape, self.water.level)
        return NavMap(grid, self.body, flat, lethal2, owner.reshape(grid.shape), names, flat.copy(), None)

    def _why_not(self, nav: NavMap, start: tuple[int, int], goal: np.ndarray) -> navigation.NoWay:
        """A point on land or in shallow water says so; anything else as on land (closed off, too narrow)."""
        here = nav.grid.cell(*self.destination.point)
        a_point = not self.destination.name or self.destination.onto
        mapped = self._bottom is not None and self._bottom.shape == nav.grid.shape
        if a_point and mapped and nav.grid.inside(*here) and self._bottom[here] > self.keel:
            depth = self.water.level - float(self._bottom[here])
            where = "on land" if depth <= 0 else f"in water {depth:.2f} m deep"
            need = self.draft + UNDER_KEEL
            return navigation.NoWay(
                "on_land",
                f"no way to {self._label()} for {self.robot}: it is {where}, and the boat needs {need:.2f} m of "
                "water; send it to a point on the water (a dock or a buoy by name stops it next to them)",
            )
        return super()._why_not(nav, start, goal)


def planner(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    robot: str,
    pilot: BoatPilot,
    destination: navigation.Destination,
    layers: dict | None,
) -> WaterPlanner:
    """A planner for `robot` as it floats now: its footprint, how deep it sits and how high it stands."""
    water = water_model.decode(model)
    if water is None:
        raise ValueError(f"{robot} is a boat, and this world has no water; give it some (set_floor water=1.0)")
    body = nav_map.measure_body(model, data, robot, Kind.BOAT, pilot)
    geoms = [g for g in nav_map.robot_geoms(model, robot) if collides(model, g)]
    bottom = min(float(lowest_point(model, data, g)[2]) for g in geoms)
    tops = [float(nav_map.world_extent(model, data, g)[1][2]) for g in geoms]
    draft = max(water.level - bottom, 0.0)
    freeboard = max(max(tops) - water.level, 0.0)
    # On the water it does not step, nor climb, nor push things aside: whatever it meets is in its way.
    body = Body(body.radius, body.reach, body.front, body.height, body.mass, 0.003, math.radians(5.0), False)
    return WaterPlanner(model, robot, body, destination, water, draft, freeboard, layers)
