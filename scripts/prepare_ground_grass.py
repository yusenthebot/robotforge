"""Reproduce bounded bundled CC0 grass tiles from verified1k originals, offline.

Download the three source JPGs listed in static/materials/README.md, then run
uv run python scripts/prepare_ground_grass.py SOURCE_DIR --output FILE.
Use the locked Pillow version for byte reproduction.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
from pathlib import Path

import numpy as np
from PIL import Image

INPUTS = {
    "diff": "cfa40bc9d9417d1852db8753a8d5917f110c40101179f63543c382e39bc05e4a",
    "disp": "3c78000885796046cc90cec2a61abea0f65cd8fb6585ac2a7b5987f73c5aa98e",
    "rough": "0c2a1d91aacb8e54a496051b59854e51e456fe858abeade778e308359ef16463",
}
# Actual256-square procedural grass means, preserving the world's authored tint.
TARGET_MEAN = 0.8164235380873729
HEIGHT_MEAN = 0.4333371629901961
ROUGHNESS_MEAN = 0.9797357297411152
MODULE = """// Leafy Grass by Charlotte Baglioni / Poly Haven, CC0-1.0.
// Derived local images, no runtime request to Poly Haven. See materials/README.md.
const COLOR = "COLOR_DATA";
const DETAIL = "DETAIL_DATA";
async function decode(src) {
  const image = new Image();
  image.src = src;
  await image.decode();
  if (image.naturalWidth !==256 || image.naturalHeight !==256) {
    throw new Error("invalid bundled grass dimensions");
  }
  return image;
}
export async function grassTiles() {
  const [color, detail] = await Promise.all([decode(COLOR),decode(DETAIL)]);
  return {color,detail};
}
"""


def linear(value: np.ndarray) -> np.ndarray:
    return np.where(value <= 0.04045, value / 12.92, ((value + 0.055) / 1.055) ** 2.4)


def encoded(value: np.ndarray) -> np.ndarray:
    return np.where(value <= 0.0031308, 12.92 * value, 1.055 * np.maximum(value, 0) ** (1 / 2.4) - 0.055)


def resized(value: np.ndarray) -> np.ndarray:
    return np.asarray(Image.fromarray(value.astype(np.float32)).resize((256, 256), Image.Resampling.BOX), dtype=float)


def generate(source: Path, output: Path) -> None:
    """Verify sources and fixed image/module budgets before writing the lazy module."""
    images: dict[str, Image.Image] = {}
    for channel, expected in INPUTS.items():
        path = source / f"leafy_grass_{channel}_1k.jpg"
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"unexpected source checksum: {path.name}")
        with Image.open(path) as image:
            if image.size != (1024, 1024):
                raise ValueError(f"unexpected source dimensions: {path.name}")
            images[channel] = image.copy()
    rgb = np.asarray(images["diff"].convert("RGB"), dtype=float) / 255
    luminance = resized(linear(rgb) @ np.array([0.2126, 0.7152, 0.0722]))
    tone = TARGET_MEAN * (1 + 0.5 * (luminance / luminance.mean() - 1))
    low, high = -1.0, 1.0
    for _ in range(64):
        bias = (low + high) / 2
        if np.clip(tone + bias, 0, 1).mean() < TARGET_MEAN:
            low = bias
        else:
            high = bias
    gray = np.round(encoded(np.clip(tone + (low + high) / 2, 0, 1)) * 255).astype(np.uint8)
    height = resized(np.asarray(images["disp"].convert("L"), dtype=float) / 255)
    roughness = resized(np.asarray(images["rough"].convert("L"), dtype=float) / 255)
    detail = np.zeros((256, 256, 3), np.uint8)
    detail[:, :, 0] = np.round(np.clip(HEIGHT_MEAN + 0.4 * (height - height.mean()), 0, 1) * 255)
    detail[:, :, 1] = np.round(np.clip(ROUGHNESS_MEAN + 0.1 * (roughness - roughness.mean()), 0, 1) * 255)
    color_file, detail_file = io.BytesIO(), io.BytesIO()
    Image.fromarray(gray).save(color_file, format="JPEG", quality=90, optimize=True)
    Image.fromarray(detail).save(detail_file, format="PNG", optimize=True)
    with Image.open(io.BytesIO(color_file.getvalue())) as decoded:
        mean = linear(np.asarray(decoded, dtype=float) / 255).mean()
    if abs(mean / TARGET_MEAN - 1) >= 0.01:
        raise ValueError("decoded grass albedo mean exceeds the1percent guard")
    if color_file.tell() + detail_file.tell() > 96 * 1024:
        raise ValueError("grass image byte budget exceeded")
    module = MODULE.replace("COLOR_DATA", "data:image/jpeg;base64," + base64.b64encode(color_file.getvalue()).decode())
    module = module.replace("DETAIL_DATA", "data:image/png;base64," + base64.b64encode(detail_file.getvalue()).decode())
    if len(module.encode("utf-8")) > 128 * 1024:
        raise ValueError("grass module byte budget exceeded")
    output.write_text(module, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generate(args.source, args.output)


if __name__ == "__main__":
    main()
