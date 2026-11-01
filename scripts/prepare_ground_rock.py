"""Reproduce the fixed bundled CC0 rock pair from verified official 1k JPEGs.

Run offline with the locked dependencies, as described in static/materials/README.md.
All content, decoded moments, lossless detail and byte guards pass before writing.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

SIZE = 256
ITERATIONS = 64
TARGETS = {
    "color": (0.8168318544161137, 0.08),
    "height": (0.5108787387025122, 0.1611487390382267),
    "roughness": (0.8997186997357539, 0.035),
}

SOURCES = {
    "diff": (927082, "ea48e0e47ad42c8bd312476178b1248ec66f0eb19156a26a1c222f2a40caaf2d"),
    "disp": (264231, "fb053bfd46cef976da9cfee67f045367469cddd69469dcacd67d3fca4bd41545"),
    "rough": (638904, "9e3d9a2fd75244892e7acb6bab2eaed1279299510a7970672c0ee27ff1c19fea"),
}

JPEG = {"quality": 90, "optimize": True}
WEBP = {"lossless": True, "quality": 100, "method": 6, "exact": True}
MODULE = """// Rock Face 03 by Dario Barresi / Rico Cilliers, Poly Haven, CC0-1.0.
const COLOR = COLOR_URL;
const DETAIL = DETAIL_URL;
async function decode(src) {
  const image = new Image();
  image.src = src;
  await image.decode();
  if (image.naturalWidth !== 256 || image.naturalHeight !== 256) {
    throw new Error("invalid bundled rock dimensions");
  }
  return image;
}
export async function rockTiles() {
  const [color, detail] = await Promise.all([decode(COLOR), decode(DETAIL)]);
  return { color, detail };
}
"""


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def linear(value: np.ndarray) -> np.ndarray:
    return np.where(value <= 0.04045, value / 12.92, ((value + 0.055) / 1.055) ** 2.4)


def encoded(value: np.ndarray) -> np.ndarray:
    return np.where(value <= 0.0031308, value * 12.92, 1.055 * value ** (1 / 2.4) - 0.055)


def moments(value: np.ndarray) -> dict:
    if not value.size or not np.isfinite(value).all():
        raise ValueError("nonempty finite channel required")
    return {
        "min": float(value.min()),
        "max": float(value.max()),
        "mean": float(value.mean()),
        "std": float(value.std()),
    }


def solve(value: np.ndarray, target: tuple[float, float]) -> tuple[np.ndarray, dict]:
    if float(value.std()) <= 1e-12:
        raise ValueError("source channel is constant")
    z = (value - value.mean()) / value.std()

    def at(scale: float) -> tuple[np.ndarray, float, np.ndarray]:
        lo, hi = -scale * float(z.max()), 1 - scale * float(z.min())
        for _ in range(ITERATIONS):
            bias = (lo + hi) / 2
            if float(np.clip(scale * z + bias, 0, 1).mean()) < target[0]:
                lo = bias
            else:
                hi = bias
        raw = scale * z + (lo + hi) / 2
        return np.clip(raw, 0, 1), (lo + hi) / 2, raw

    lo, hi = 0.0, 1.0
    if float(at(hi)[0].std()) < target[1]:
        raise ValueError("target outside fixed scale bracket")
    for _ in range(ITERATIONS):
        scale = (lo + hi) / 2
        if float(at(scale)[0].std()) < target[1]:
            lo = scale
        else:
            hi = scale
    scale = (lo + hi) / 2
    result, bias, raw = at(scale)
    return result, {
        "target": {"mean": target[0], "std": target[1]},
        "source256": moments(value),
        "scale": scale,
        "bias": bias,
        "prequantized": moments(result),
        "clippedFraction": float(np.mean((raw < 0) | (raw > 1))),
    }


def decode(data: bytes, size: tuple[int, int], mode: str) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        if image.size != size:
            raise ValueError(f"unexpected dimensions: {image.size}")
        return np.asarray(image.convert(mode), dtype=np.uint8).copy()


EXPECTED = {
    "color.jpg": "1d3584ea9380bc72df015815e45b192aa28c39e2ba97a9e60ce29d6509a035e8",
    "detail.webp": "2b8800b7edfde4969ab11de4c933cf0a7a2cecc4e0cca6d57a8fc0dbfcafa8ee",
}


def generate(source: Path, output: Path) -> None:
    """Verify sources and reproduce exactly one pair before atomically writing a module."""
    fields = {}
    for name, (count, expected) in SOURCES.items():
        filename = f"rock_face_03_{name}_1k.jpg"
        data = (source / filename).read_bytes()
        if len(data) != count or digest(data) != expected:
            raise ValueError(f"unexpected source checksum or size: {filename}")
        array = decode(data, (1024, 1024), "RGB" if name == "diff" else "L").astype(float) / 255
        scalar = linear(array) @ np.array([0.2126, 0.7152, 0.0722]) if name == "diff" else array
        field = np.asarray(
            Image.fromarray(scalar.astype(np.float32)).resize((SIZE, SIZE), Image.Resampling.BOX), dtype=np.float64
        )
        fields[{"diff": "color", "disp": "height", "rough": "roughness"}[name]] = field
    normalized = {}
    for name, field in fields.items():
        normalized[name], info = solve(field, TARGETS[name])
        if info["clippedFraction"] > 0.05 or any(
            abs(info["prequantized"][key] - info["target"][key]) > 1e-10 for key in ("mean", "std")
        ):
            raise ValueError(f"{name}: fixed clipping or prequantization moment guard")
    color = np.rint(encoded(normalized["color"]) * 255).astype(np.uint8)
    detail = np.zeros((SIZE, SIZE, 3), np.uint8)
    detail[:, :, 0] = np.rint(normalized["height"] * 255).astype(np.uint8)
    detail[:, :, 1] = np.rint(normalized["roughness"] * 255).astype(np.uint8)
    images = {}
    for name, array, fmt, options in (("color.jpg", color, "JPEG", JPEG), ("detail.webp", detail, "WEBP", WEBP)):
        stream = io.BytesIO()
        Image.fromarray(array).save(stream, format=fmt, **options)
        images[name] = stream.getvalue()
    rgba = {name: decode(data, (SIZE, SIZE), "RGBA") for name, data in images.items()}
    if any(np.any(value[:, :, 3] != 255) for value in rgba.values()):
        raise ValueError("decoded alpha must be opaque")
    if not np.array_equal(rgba["detail.webp"][:, :, :3], detail) or np.any(rgba["detail.webp"][:, :, 2]):
        raise ValueError("lossless detail must round-trip exactly with B=0")
    if not np.all(rgba["color.jpg"][:, :, :3] == rgba["color.jpg"][:, :, :1]):
        raise ValueError("decoded albedo must remain grayscale")
    decoded = {
        "color": linear(rgba["color.jpg"][:, :, 0].astype(float) / 255),
        "height": rgba["detail.webp"][:, :, 0].astype(float) / 255,
        "roughness": rgba["detail.webp"][:, :, 1].astype(float) / 255,
    }
    for name, field in decoded.items():
        measured = moments(field)
        if abs(measured["mean"] / TARGETS[name][0] - 1) > 0.01 or abs(measured["std"] / TARGETS[name][1] - 1) > 0.10:
            raise ValueError(f"{name}: decoded 1% mean / 10% standard deviation guard")
    for name, expected in EXPECTED.items():
        if digest(images[name]) != expected:
            raise ValueError(f"{name}: encoded bytes differ from frozen recipe; use locked Pillow/libwebp")
    module = (
        MODULE.replace(
            "COLOR_URL", json.dumps("data:image/jpeg;base64," + base64.b64encode(images["color.jpg"]).decode("ascii"))
        )
        .replace(
            "DETAIL_URL",
            json.dumps("data:image/webp;base64," + base64.b64encode(images["detail.webp"]).decode("ascii")),
        )
        .encode("utf-8")
    )
    if sum(map(len, images.values())) > 81920 or len(module) > 114688:
        raise ValueError("fixed image/module byte budget exceeded; no retuning permitted")
    if digest(module) != "b517d61a12e5a85944800978c9fb947dc6f2ae03b5d7a28dd1b93336985653be":
        raise ValueError("loader bytes differ from frozen recipe")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix=".rock-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(module)
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generate(args.source, args.output)
