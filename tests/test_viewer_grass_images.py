"""Check shipped asset integrity and fail-closed offline source validation."""

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


def test_bundled_grass_pair_has_verified_bytes_dimensions_and_bounded_tint():
    module = files("robotforge.viewer").joinpath("static/ground-grass.js").read_text()
    assert len(module.encode()) <= 128 * 1024
    images = re.findall(r"data:image/(jpeg|png);base64,([^\"]+)", module)
    assert [kind for kind, _ in images] == ["jpeg", "png"]
    expected = [
        "5c76596cd17b2b516514a8ec4068e88ea8b0d07d4988af88c47b6574a5fde1ed",
        "b0db64526e3b0f76c7308ebf1d8e92f6943ea154c3977ccc053a7dc5ced108f8",
    ]
    total = 0
    for (kind, encoded), checksum in zip(images, expected, strict=True):
        data = base64.b64decode(encoded, validate=True)
        assert hashlib.sha256(data).hexdigest() == checksum
        total += len(data)
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            assert image.size == (256, 256)
            if kind == "jpeg":
                assert image.mode == "L"
                value = np.asarray(image, dtype=float) / 255
                linear = np.where(value <= 0.04045, value / 12.92, ((value + 0.055) / 1.055) ** 2.4)
                assert abs(linear.mean() / 0.8164235380873729 - 1) < 0.01
            else:
                assert image.mode == "RGB"
                assert not np.any(np.asarray(image)[:, :, 2])
    assert total <= 96 * 1024


@pytest.mark.parametrize("existing", [False, True])
def test_offline_preparation_rejects_unverified_source_before_writing(tmp_path, existing):
    script = Path(__file__).resolve().parents[1] / "scripts/prepare_ground_grass.py"
    spec = importlib.util.spec_from_file_location("prepare_ground_grass", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / "leafy_grass_diff_1k.jpg").write_bytes(b"unverified source")
    output = tmp_path / "output.js"
    if existing:
        output.write_text("preserve existing output")
    with pytest.raises(ValueError, match="unexpected source checksum"):
        module.generate(tmp_path, output)
    if existing:
        assert output.read_text() == "preserve existing output"
    else:
        assert not output.exists()
