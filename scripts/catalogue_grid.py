"""Put the catalogue pictures (scripts/screenshots.py g12_*) in one grid with each robot's name and what it did.

    uv run python scripts/catalogue_grid.py --shots docs/images --report bench.json --out catalogue_grid.jpg

The report is the JSON scripts/catalogue_bench.py --out writes. Maintainer tool, not part of robotforge.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

TILE_W, TILE_H, CAPTION, GAP, MARGIN, TITLE_H, COLUMNS = 640, 360, 66, 12, 28, 78, 5
CROP = (80, 80, 1520, 890)  # of a SAVED_WIDTH (1600 x 1000) picture: the viewer's bars out


def verdict(row: dict[str, Any]) -> str:
    """What the robot did, in a few words."""
    if row["load"] != "ok":
        return "did not load"
    move, stand = row.get("move", {}), row.get("stand", {})
    kind = row["kind"]
    if not stand.get("ok"):
        return "falls over at home" if "upright" in stand else "drifts at home"
    if move.get("ok"):
        if kind == "arm":
            return f"reaches a point to {move['error_m'] * 100:.1f} cm"
        if "moved_m" in move:
            return f"{'walks' if kind in ('quadruped', 'humanoid') else 'drives'} {move['moved_m']:.1f} m upright"
        return "its joints move"
    if "moved_m" in move:
        return f"moved {move['moved_m']:.1f} m, then fell"
    why = move.get("why") or ""
    if "no known kind" in why:
        return "stands; no walker fits its body"
    if "no actuators" in why:
        return "a rigid body: no motors in the file"
    return "stands; " + (why[:36] or "does not move")


def font(size: int) -> Any:
    from PIL import ImageFont

    for name in ("/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/SFNS.ttf", "C:/Windows/Fonts/arial.ttf"):
        if Path(name).is_file():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default(size)


def grid(shots: Path, rows: list[dict[str, Any]], out: Path) -> Path:
    from PIL import Image, ImageDraw

    count = len(rows)
    height = -(-count // COLUMNS)
    sheet = Image.new(
        "RGB",
        (2 * MARGIN + COLUMNS * TILE_W + (COLUMNS - 1) * GAP, 2 * MARGIN + TITLE_H + height * (TILE_H + CAPTION + GAP)),
        (236, 239, 243),
    )
    draw = ImageDraw.Draw(sheet)
    draw.text((MARGIN, MARGIN), "robotforge: 20 open robots, added by name", font=font(34), fill=(29, 32, 38))
    draw.text(
        (MARGIN, MARGIN + 44),
        "add_robot(name, library=<name>)  -  drawn at random from the catalogue (seed 12), real models, real viewer, "
        "each doing what its kind does",
        font=font(19),
        fill=(90, 96, 106),
    )
    for i, row in enumerate(rows):
        r, c = divmod(i, COLUMNS)
        x, y = MARGIN + c * (TILE_W + GAP), MARGIN + TITLE_H + r * (TILE_H + CAPTION + GAP)
        path = shots / f"g12_{row['name']}.jpg"
        if path.is_file():
            tile = Image.open(path).convert("RGB").crop(CROP).resize((TILE_W, TILE_H), Image.Resampling.LANCZOS)
            sheet.paste(tile, (x, y))
        draw.rectangle([x, y + TILE_H, x + TILE_W - 1, y + TILE_H + CAPTION - 1], fill=(255, 255, 255))
        draw.text((x + 12, y + TILE_H + 8), row["name"], font=font(22), fill=(29, 32, 38))
        kind = row.get("kind", "-")
        width = draw.textlength(kind, font=font(18))
        draw.text((x + TILE_W - 12 - width, y + TILE_H + 11), kind, font=font(18), fill=(200, 120, 30))
        draw.text((x + 12, y + TILE_H + 38), verdict(row), font=font(18), fill=(90, 96, 106))
    sheet.save(out, "JPEG", quality=90, optimize=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--shots", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads(args.report.read_text(encoding="utf-8"))
    print(grid(args.shots, rows, args.out))


if __name__ == "__main__":
    main()
