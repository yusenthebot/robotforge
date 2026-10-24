"""Reproduce the bounded bundled CC0 oak tiles from verified1k originals, offline.

Download the three source JPGs listed in static/materials/README.md to a folder,
then run uv run python scripts/prepare_ground_wood.py SOURCE_DIR --output FILE.
No network service or original high-resolution asset is needed by the installed app.
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
    "diff": "f715ed3e4346e94f971ef8dd61221ff6cc774886c20f65381e01e1dc0bb0a634",
    "disp": "ddcf07081436d60c855bd21b530e7e7a3b7e4a84d77a748325149cc2df05ef39",
    "rough": "10ced559145df3c537bf1ccef6dc767ef6c47e134c21a9650bab261fa95ecacf",
}
# Measured linear mean of the previous procedural wood albedo, preserving authored tint.
TARGET_MEAN = 0.806905622181332
MODULE = """// Oak Wood Planks by Dimitrios Savva / Poly Haven, CC0-1.0.
// Derived local images, no runtime request to Poly Haven. See materials/README.md.
const COLOR = "COLOR_DATA";
const DETAIL = "DETAIL_DATA";
async function decode(src, width, height) {
  const image = new Image();
  image.src = src;
  await image.decode();
  if (image.naturalWidth !== width || image.naturalHeight !== height) {
    throw new Error("invalid bundled wood dimensions");
  }
  return image;
}
export async function woodTiles() {
  const [color, detail] = await Promise.all([decode(COLOR,512,512),decode(DETAIL,256,256)]);
  return {color,detail};
}
"""


def linear(value: np.ndarray) -> np.ndarray:
    return np.where(value <= 0.04045, value / 12.92, ((value + 0.055) / 1.055) ** 2.4)


def encoded(value: np.ndarray) -> np.ndarray:
    return np.where(value <= 0.0031308, 12.92 * value, 1.055 * np.maximum(value, 0) ** (1 / 2.4) - 0.055)


def generate(source: Path, output: Path) -> None:
    """Verify the exact sources, area-filter in linear light and write one lazy module."""
    images: dict[str, Image.Image] = {}
    for channel, expected in INPUTS.items():
        path = source / f"oak_wood_planks_{channel}_1k.jpg"
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"unexpected source checksum: {path.name}")
        with Image.open(path) as image:
            if image.size != (1024, 1024):
                raise ValueError(f"unexpected source dimensions: {path.name}")
            images[channel] = image.copy()
    rgb = np.asarray(images["diff"].convert("RGB"), dtype=float) / 255
    luminance = linear(rgb) @ np.array([0.2126, 0.7152, 0.0722])
    resized = np.asarray(
        Image.fromarray(luminance.astype(np.float32)).resize((512, 512), Image.Resampling.BOX), dtype=float
    )
    tone = TARGET_MEAN * (1 + 0.5 * (resized / luminance.mean() - 1))
    low, high = -1.0, 1.0
    for _ in range(64):
        bias = (low + high) / 2
        if np.clip(tone + bias, 0, 1).mean() < TARGET_MEAN:
            low = bias
        else:
            high = bias
    gray = np.round(encoded(np.clip(tone + (low + high) / 2, 0, 1)) * 255).astype(np.uint8)
    height = np.asarray(images["disp"].resize((256, 256), Image.Resampling.BOX), dtype=float) / 255
    roughness = np.asarray(images["rough"].resize((256, 256), Image.Resampling.BOX), dtype=float) / 255
    detail = np.zeros((256, 256, 3), np.uint8)
    detail[:, :, 0] = np.round(np.clip(0.62 + 0.4 * (height - height.mean()), 0, 1) * 255)
    detail[:, :, 1] = np.round(np.clip(0.54 + 0.15 * (roughness - roughness.mean()), 0.3, 0.9) * 255)
    color_file, detail_file = io.BytesIO(), io.BytesIO()
    Image.fromarray(gray).save(color_file, format="JPEG", quality=90, optimize=True)
    Image.fromarray(detail).save(detail_file, format="PNG", optimize=True)
    with Image.open(io.BytesIO(color_file.getvalue())) as decoded:
        mean = linear(np.asarray(decoded, dtype=float) / 255).mean()
    if abs(mean / TARGET_MEAN - 1) >= 0.01:
        raise ValueError("decoded wood albedo mean exceeds the1percent guard")
    if color_file.tell() + detail_file.tell() > 128 * 1024:
        raise ValueError("wood image byte budget exceeded")
    module = MODULE.replace("COLOR_DATA", "data:image/jpeg;base64," + base64.b64encode(color_file.getvalue()).decode())
    module = module.replace("DETAIL_DATA", "data:image/png;base64," + base64.b64encode(detail_file.getvalue()).decode())
    if len(module.encode("utf-8")) > 128 * 1024:
        raise ValueError("wood module byte budget exceeded")
    output.write_text(module, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generate(args.source, args.output)


if __name__ == "__main__":
    main()
