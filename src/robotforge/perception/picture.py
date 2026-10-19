"""The map as a picture, for an agent that asks to see it: unknown dark, free light, obstacles amber, the robot's
believed pose a blue arrow, the odometry's own pose a grey one, the particles blue dots. North (+y) is up."""

from __future__ import annotations

import io
import math

import numpy as np

from robotforge.perception.grid import FREE, OCCUPIED
from robotforge.perception.mapper import Mapper

UNKNOWN_RGB, FREE_RGB, OCCUPIED_RGB = (38, 42, 48), (214, 224, 230), (232, 140, 30)
POSE_RGB, ODOMETRY_RGB, PARTICLE_RGB = (30, 90, 220), (120, 120, 120), (60, 120, 230)
MAX_SIDE = 900  # px
MARGIN_CELLS = 10


def map_png(mapper: Mapper) -> bytes:
    from PIL import Image, ImageDraw  # here: Pillow is heavy, and only a picture asked for needs it

    grid = mapper.grid
    if grid is None or grid.extent() is None:
        raise ValueError("nothing is mapped yet: start mapping and move the robot first")
    rows, cols = np.nonzero(grid.seen)
    r0, r1 = max(rows.min() - MARGIN_CELLS, 0), min(rows.max() + MARGIN_CELLS + 1, grid.rows)
    c0, c1 = max(cols.min() - MARGIN_CELLS, 0), min(cols.max() + MARGIN_CELLS + 1, grid.cols)
    state = grid.state()[r0:r1, c0:c1]
    rgb = np.empty(state.shape + (3,), np.uint8)
    rgb[:] = UNKNOWN_RGB
    rgb[state == FREE] = FREE_RGB
    rgb[state == OCCUPIED] = OCCUPIED_RGB
    rgb = rgb[::-1]  # +y up
    scale = max(1, min(8, MAX_SIDE // max(rgb.shape[:2])))
    image = Image.fromarray(rgb).resize((rgb.shape[1] * scale, rgb.shape[0] * scale), Image.NEAREST)
    draw = ImageDraw.Draw(image)
    height = state.shape[0]

    def px(x: float, y: float) -> tuple[float, float]:
        col = (x - grid.x0) / grid.res - c0 + 0.5
        row = height - 1 - ((y - grid.y0) / grid.res - r0) + 0.5
        return col * scale, row * scale

    def arrow(pose: np.ndarray, color: tuple[int, int, int]) -> None:
        x, y = px(pose[0], pose[1])
        length = max(10.0, 0.4 / grid.res * scale)
        tip = (x + length * math.cos(pose[2]), y - length * math.sin(pose[2]))
        draw.line([(x, y), tip], fill=color, width=3)
        draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=color)

    flt = mapper.filter
    if mapper.pose_source == "estimate":
        for xp, yp in zip(flt.x[:300], flt.y[:300], strict=False):
            x, y = px(float(xp), float(yp))
            draw.point((x, y), fill=PARTICLE_RGB)
    if mapper.odometry_pose is not None:
        arrow(mapper.odometry_pose, ODOMETRY_RGB)
    if mapper.estimate_pose is not None:
        arrow(mapper.estimate_pose, POSE_RGB)
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=True)
    return buf.getvalue()
