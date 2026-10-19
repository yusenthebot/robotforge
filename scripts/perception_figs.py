"""Pictures of the perception pieces that are not a viewer page: a depth camera's and a stereo pair's point clouds
side by side (top view and side view), from a Go2 looking down the warehouse aisle.

    uv run python scripts/perception_figs.py [--out DIR]

Maintainer tool, not part of robotforge.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

SIZE = (880, 640)  # px of each panel


def cloud(pg: Any, name: str, count: int) -> np.ndarray:
    reading, _ = pg.read_sensors("go2", [name], points=count)
    return np.array(reading["sensors"][name]["cloud"]["points_m"])


def panel(points: np.ndarray, title: str, note: str) -> Any:
    """A top view (x forward, y left) above a side view (x forward, z up), colored by height."""
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", SIZE, (250, 251, 252))
    draw = ImageDraw.Draw(image)
    big, small = ImageFont.load_default(24), ImageFont.load_default(16)
    draw.text((20, 14), title, fill=(29, 32, 38), font=big)
    draw.text((20, 46), note, fill=(90, 96, 106), font=small)
    x0, x1 = points[:, 0].min() - 0.2, points[:, 0].max() + 0.2
    views = (("top view", 1, (20, 82, 860, 350)), ("side view", 2, (20, 372, 860, 620)))
    z = points[:, 2]
    low, high = float(z.min()), float(max(z.max(), z.min() + 0.5))
    for label, axis, (left, top, right, bottom) in views:
        draw.rectangle([left, top, right, bottom], outline=(200, 204, 210))
        draw.text((left + 8, top + 6), label, fill=(120, 126, 136), font=small)
        v = points[:, axis]
        lo, hi = float(v.min()) - 0.1, float(v.max()) + 0.1
        scale = min((right - left - 20) / (x1 - x0), (bottom - top - 30) / (hi - lo))
        for p, vv, h in zip(points, v, z, strict=True):
            px = left + 10 + (p[0] - x0) * scale
            py = bottom - 10 - (vv - lo) * scale
            k = (h - low) / (high - low)
            color = (int(30 + 200 * k), int(110 - 40 * k), int(220 - 180 * k))
            draw.ellipse([px - 2, py - 2, px + 2, py + 2], fill=color)
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("."))
    args = parser.parse_args()
    os.environ.setdefault("ROBOTFORGE_HOME", tempfile.mkdtemp(prefix="rf-figs-"))
    from PIL import Image

    from robotforge.playground import Playground

    pg = Playground.with_demo()
    pg.new_world("figs")
    pg.load_scene("warehouse", keep_robots=False, starter=False)
    pg.add_robot("go2", library="go2", pos=[-3.6, 2.6], yaw=0)
    pg.add_sensor("go2", {"type": "camera", "mode": "depth", "name": "depth", "width": 320, "height": 240, "range": 12})
    pg.add_sensor(
        "go2",
        {"type": "stereo", "name": "pair", "width": 320, "height": 240, "range": 12, "baseline": 0.06, "noise": 0.01},
    )
    pg.run_fast(lambda: False, 1.0)
    exact = cloud(pg, "depth", 2500)
    stereo = cloud(pg, "pair", 2500)
    left = panel(exact, "Depth camera", f"{len(exact)} points of 76,800 pixels (thinned by voxel); exact depth")
    right = panel(
        stereo,
        "Stereo pair, 6 cm baseline",
        f"{len(stereo)} points; depth from the disparity, 1 cm noise at 1 m, 1/8 px steps",
    )
    sheet = Image.new("RGB", (2 * SIZE[0] + 20, SIZE[1]), (250, 251, 252))
    sheet.paste(left, (0, 0))
    sheet.paste(right, (SIZE[0] + 20, 0))
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / "point_clouds.png"
    sheet.save(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
