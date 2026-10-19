"""A grid of the humanoids that stand, each as it stands a few seconds in, with what it takes (balance_bench.py).

    uv run python scripts/balance_bench.py --robots all --json bench.json
    uv run python scripts/humanoid_grid.py --report bench.json --out humanoid_stand_grid.jpg

Every tile is one robot alone on the floor with whatever stands it up when it is added, rendered offscreen (needs
OpenGL), with its name, its mass, and the largest push (m/s, over the four directions) it was still standing after.
Maintainer tool, not part of robotforge.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

TILE = 480
CAPTION = 58
GAP = 10
MARGIN = 24
TITLE = 70
COLUMNS = 6
SETTLE_S = 4.0


def render(robot: str) -> object:
    import mujoco
    import numpy as np
    from PIL import Image

    from robotforge.design import catalogue_probe
    from robotforge.design.store import WorldStore
    from robotforge.playground import Playground

    with tempfile.TemporaryDirectory(prefix="rf-grid-") as scratch:
        pg = Playground.with_demo()
        pg.store = WorldStore.create("grid", parent=Path(scratch))
        pg._load_world(pg.store.compile(), created=True)
        token = catalogue_probe._probing.set(True)
        try:
            pg.add_robot("r", library=robot)
        finally:
            catalogue_probe._probing.reset(token)
        m, d = pg.session.model, pg.session.data
        pg.run_fast(lambda: False, SETTLE_S)
        roots = (b for b in range(1, m.nbody) if m.body_parentid[b] == 0)
        trunk = next(b for b in roots if (m.body(b).name or "").startswith("r/"))
        top = max(float(d.xpos[b][2]) for b in range(m.nbody) if m.body_rootid[b] == trunk) + 0.15
        centre = np.array([d.xpos[trunk][0], d.xpos[trunk][1], top * 0.5])
        cam = mujoco.MjvCamera()
        cam.lookat[:] = centre
        cam.distance = max(1.3, 1.45 * top)
        cam.azimuth, cam.elevation = 135.0, -12.0
        m.vis.global_.offwidth = max(m.vis.global_.offwidth, TILE)
        m.vis.global_.offheight = max(m.vis.global_.offheight, TILE)
        renderer = mujoco.Renderer(m, TILE, TILE)
        renderer.update_scene(d, camera=cam)
        pixels = renderer.render()
        renderer.close()
        return Image.fromarray(pixels)


def main() -> int:
    from PIL import Image, ImageDraw, ImageFont

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--report", required=True, help="the JSON scripts/balance_bench.py --json wrote")
    parser.add_argument("--out", default="humanoid_stand_grid.jpg")
    parser.add_argument("--max", type=int, default=0, help="at most this many tiles (0: all)")
    args = parser.parse_args()
    rows = [
        r for r in json.loads(Path(args.report).read_text(encoding="utf-8")) if r["stands"] and r["kind"] == "humanoid"
    ]
    rows = list({r["robot"]: r for r in rows}.values())
    rows.sort(key=lambda r: r["robot"])
    if args.max:
        rows = rows[: args.max]
    tiles = []
    for r in rows:
        try:
            tiles.append((r, render(r["robot"])))
        except Exception as exc:  # a robot that will not draw is left out of the grid, said so
            print(f"{r['robot']}: not drawn ({type(exc).__name__}: {str(exc)[:100]})")
    columns = min(COLUMNS, max(len(tiles), 1))
    lines = -(-len(tiles) // columns)
    width = 2 * MARGIN + columns * TILE + (columns - 1) * GAP
    height = TITLE + MARGIN + lines * (TILE + CAPTION) + (lines - 1) * GAP
    sheet = Image.new("RGB", (width, height), (247, 247, 244))
    draw = ImageDraw.Draw(sheet)
    try:
        big = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 30)
        small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 19)
    except OSError:
        big = small = ImageFont.load_default()
    draw.text(
        (MARGIN, 20),
        f"{len(tiles)} humanoids standing, a few seconds after being added (each alone on the floor)",
        font=big,
        fill=(30, 30, 30),
    )
    for k, (r, image) in enumerate(tiles):
        x = MARGIN + (k % columns) * (TILE + GAP)
        y = TITLE + (k // columns) * (TILE + CAPTION + GAP)
        sheet.paste(image, (x, y))
        push = max(r["shove"].values()) if r.get("shove") else 0.0
        load = "" if r.get("load_ok") is None else (", carries 20% load" if r["load_ok"] else "")
        draw.text((x + 6, y + TILE + 6), f"{r['robot']}  ({r['mass']:.0f} kg)", font=small, fill=(30, 30, 30))
        draw.text((x + 6, y + TILE + 30), f"takes a push of {push:.1f} m/s{load}", font=small, fill=(90, 90, 90))
    sheet.save(args.out, quality=88)
    print(f"{args.out}: {len(tiles)} tiles, {width}x{height}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
