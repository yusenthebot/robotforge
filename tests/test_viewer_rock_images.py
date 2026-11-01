"""Frozen bundled rock pixels and portable offline source guards."""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import io
import re
from importlib.resources import files
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def test_bundled_rock_pair_keeps_frozen_bytes_lossless_detail_and_moments():
    module = files("robotforge.viewer").joinpath("static/ground-rock.js").read_bytes()
    assert hashlib.sha256(module).hexdigest() == ("b517d61a12e5a85944800978c9fb947dc6f2ae03b5d7a28dd1b93336985653be")
    assert len(module) <= 114688
    images = re.findall(rb"data:image/(jpeg|webp);base64,([^\"]+)", module)
    assert [kind for kind, _ in images] == [b"jpeg", b"webp"]
    expected = [
        "1d3584ea9380bc72df015815e45b192aa28c39e2ba97a9e60ce29d6509a035e8",
        "2b8800b7edfde4969ab11de4c933cf0a7a2cecc4e0cca6d57a8fc0dbfcafa8ee",
    ]
    decoded, total = {}, 0
    for (kind, encoded), checksum in zip(images, expected, strict=True):
        data = base64.b64decode(encoded, validate=True)
        total += len(data)
        assert hashlib.sha256(data).hexdigest() == checksum
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            assert image.size == (256, 256)
            assert image.mode == ("L" if kind == b"jpeg" else "RGB")
            rgba = np.asarray(image.convert("RGBA"))
            assert np.all(rgba[:, :, 3] == 255)
            decoded[kind] = rgba
    assert total == 78882 and total <= 81920
    assert hashlib.sha256(decoded[b"webp"].tobytes()).hexdigest() == (
        "633e37c9dda80229b48e902374b2f22d5a7c84bbe7816ce0e9730897d062fbae"
    )
    assert not np.any(decoded[b"webp"][:, :, 2])
    assert np.all(decoded[b"jpeg"][:, :, :3] == decoded[b"jpeg"][:, :, :1])
    gray = decoded[b"jpeg"][:, :, 0].astype(float) / 255
    linear = np.where(gray <= 0.04045, gray / 12.92, ((gray + 0.055) / 1.055) ** 2.4)
    for channel, mean, std in (
        (linear, 0.8168318544161137, 0.08),
        (decoded[b"webp"][:, :, 0] / 255, 0.5108787387025122, 0.1611487390382267),
        (decoded[b"webp"][:, :, 1] / 255, 0.8997186997357539, 0.035),
    ):
        assert abs(channel.mean() / mean - 1) <= 0.01
        assert abs(channel.std() / std - 1) <= 0.10


@pytest.mark.parametrize("existing", [False, True])
def test_offline_rock_rejects_unverified_sources_before_writing(tmp_path, existing):
    script = Path(__file__).resolve().parents[1] / "scripts/prepare_ground_rock.py"
    spec = importlib.util.spec_from_file_location("prepare_ground_rock", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / "rock_face_03_diff_1k.jpg").write_bytes(b"unverified")
    output = tmp_path / "output.js"
    if existing:
        output.write_text("preserve existing output")
    with pytest.raises(ValueError, match="unexpected source checksum or size"):
        module.generate(tmp_path, output)
    if existing:
        assert output.read_text() == "preserve existing output"
    else:
        assert not output.exists()
