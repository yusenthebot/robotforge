"""Put the catalogue pictures (scripts/screenshots.py g12_*) in one grid with each robot's name and what it did.

    uv run python scripts/catalogue_grid.py --shots docs/images --report bench.json --out catalogue_grid.jpg

The report is g12_report.json, which scripts/screenshots.py writes next to the g12_* pictures (the JSON
scripts/catalogue_bench.py --out writes works too). Maintainer tool, not part of robotforge.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

TILE_W, TILE_H, CAPTION, GAP, MARGIN, TITLE_H, COLUMNS = 640, 360, 66, 12, 28, 78, 5
INSET_W, INSET_H = 176, 99  # the robot as added, in the corner of its tile
CROP = (80, 80, 1520, 890)  # of a SAVED_WIDTH (1600 x 1000) picture: the viewer's bars out


def verdict(row: dict[str, Any]) -> str:
    """What the robot did, in a few words, from the numbers of the bench."""
    if row.get("load", "ok") != "ok":
        return "did not load"
    move, stand = row.get("move", {}), row.get("stand", {})
    kind = row["kind"]
    if not stand.get("ok"):
        if "upright" in stand:
            return f"falls over at home (keeps {stand['height_kept']:.0%} of its height)"
        return "drifts at home"
    if kind == "arm" and move.get("ok"):
        return f"reaches a point, {move['error_m'] * 100:.1f} cm off (point {move['tries']} tried)"
    if "moved_m" in move:
        verb = "walks" if kind in ("quadruped", "humanoid") else "drives"
        if kind == "wheeled":
            return f"{verb} {move['moved_m']:.2f} m of {move['asked_m']:.1f} m asked ({move['speed_got']:.2f} m/s)"
        if move["ok"]:
            return f"{verb} {move['moved_m']:.1f} m upright"
        return (
            f"walked {move['moved_m']:.1f} m, then fell"
            if not move["upright"]
            else f"{verb} only {move['moved_m']:.2f} m"
        )
    if move.get("ok"):
        if "max_joint_change" in move:
            return f"its joints move ({move['max_joint_change']:.1f} rad)"
        return "its joints move"
    why = move.get("why") or ""
    if "no known kind" in why or "cannot walk" in why:
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
        "add_robot(name, library=<name>)  -  drawn at random from the catalogue (seed 12), real models, real viewer; "
        "large picture: the end of the test (reach, walk, drive, grip), corner: as added",
        font=font(19),
        fill=(90, 96, 106),
    )
    for i, row in enumerate(rows):
        r, c = divmod(i, COLUMNS)
        x, y = MARGIN + c * (TILE_W + GAP), MARGIN + TITLE_H + r * (TILE_H + CAPTION + GAP)
        path = shots / f"g12_{row['name']}.jpg"
        if path.is_file():
            tile = Image.open(path).convert("RGB").crop(CROP).resize((TILE_W, TILE_H), Image.Resampling.LANCZOS)
            start = shots / f"g12_{row['name']}_start.jpg"
            if start.is_file():
                inset = Image.open(start).convert("RGB").crop(CROP).resize((INSET_W, INSET_H), Image.Resampling.LANCZOS)
                tile.paste(inset, (8, TILE_H - INSET_H - 8))
                edge = ImageDraw.Draw(tile)
                edge.rectangle([8, TILE_H - INSET_H - 8, 8 + INSET_W - 1, TILE_H - 9], outline=(255, 255, 255), width=3)
                edge.text((14, TILE_H - INSET_H - 4), "as added", font=font(15), fill=(29, 32, 38))
            stamp = f"t = {row['t_end']:.0f} s" if "t_end" in row else ""
            ImageDraw.Draw(tile).text((TILE_W - 100, 10), stamp, font=font(18), fill=(29, 32, 38))
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
    report = json.loads(args.report.read_text(encoding="utf-8"))
    rows = list(report.values()) if isinstance(report, dict) else report
    print(grid(args.shots, rows, args.out))


if __name__ == "__main__":
    main()
