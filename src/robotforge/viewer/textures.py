"""Bounded native model images, shared by all geoms that use the same pixels."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

MAX_EDGE = 512
MAX_PIXELS = 512 * 512
MAX_IMAGES = 64


@dataclass(frozen=True)
class TexturePayload:
    blob: bytes = b""
    entries: list[dict] = field(default_factory=list)


def build(model: mujoco.MjModel, texture_ids: set[int]) -> TexturePayload:
    """Pack at most one MiB of RGBA8 images; resize only when the shared budget requires it.

    Deduplicate original pixels before budgeting so composite copies retain the same detail.
    RGB/RGBA 2D pictures are supported; other images retain the existing vertex/mean fallback.
    """
    images: dict[tuple, tuple[list[int], np.ndarray]] = {}
    for tex in sorted(texture_ids):
        w, h, channels = int(model.tex_width[tex]), int(model.tex_height[tex]), int(model.tex_nchannel[tex])
        if int(model.tex_type[tex]) != int(mujoco.mjtTexture.mjTEXTURE_2D) or channels not in (3, 4):
            continue
        adr = int(model.tex_adr[tex])
        data = model.tex_data[adr : adr + w * h * channels]
        key = (w, h, channels, hashlib.sha256(data).digest())
        if key in images:
            images[key][0].append(tex)
        elif len(images) < MAX_IMAGES:
            images[key] = ([tex], data.reshape(h, w, channels))
    if not images:
        return TexturePayload()

    sizes = []
    for _, image in images.values():
        h, w = image.shape[:2]
        scale = min(1.0, MAX_EDGE / max(w, h))
        sizes.append((max(1, int(w * scale)), max(1, int(h * scale))))
    scale = min(1.0, math.sqrt(MAX_PIXELS / sum(w * h for w, h in sizes)))
    sizes = [(max(1, int(w * scale)), max(1, int(h * scale))) for w, h in sizes]
    # Very thin pictures must keep a one-pixel short edge, even when the common scale is tiny.
    while sum(w * h for w, h in sizes) > MAX_PIXELS:
        i = max(range(len(sizes)), key=lambda i: sizes[i][0] * sizes[i][1])
        w, h = sizes[i]
        sizes[i] = (max(1, w // 2), max(1, h // 2))

    chunks, entries, offset = [], [], 0
    for (aliases, pixels), (w, h) in zip(images.values(), sizes, strict=True):
        if pixels.shape[:2] != (h, w):
            from PIL import Image

            pixels = np.asarray(Image.fromarray(pixels).resize((w, h), Image.Resampling.BOX))
        if pixels.shape[2] == 3:
            rgba = np.empty((h, w, 4), dtype=np.uint8)
            rgba[:, :, :3], rgba[:, :, 3] = pixels, 255
        else:
            rgba = pixels
        raw = rgba.tobytes()
        entries.append(
            {
                "id": aliases[0],
                "aliases": aliases,
                "width": w,
                "height": h,
                "offset": offset,
                "length": len(raw),
                "alpha": bool(np.any(rgba[:, :, 3] != 255)),
                # Keep the viewer's existing RGB-picture convention instead of lifting
                # untagged imported images into a different, visibly washed-out palette.
                "color_space": "srgb",
                "source_color_spaces": sorted({int(model.tex_colorspace[tex]) for tex in aliases}),
            }
        )
        chunks.append(raw)
        offset += len(raw)
    return TexturePayload(b"".join(chunks), entries)
